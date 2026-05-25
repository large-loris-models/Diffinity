#!/usr/bin/env python3
"""
Test PLAID diffusion model with unconditional and guided generation.

This script performs generation using the PLAID model with optional
gradient-based automaton guidance for constrained generation.

Usage:
    # Unconditional generation
    python test_diffusion_plaid.py --weights_path=/path/to/weights/

    # Guided generation (constrain output to match regex)
    python test_diffusion_plaid.py --weights_path=/path/to/weights/ \
        --guidance --regex_pattern='.*restaurant.*' --guidance_scale=1.0

    # Compare guidance targets
    python test_diffusion_plaid.py --guidance --guidance_target=x_reconst  # PLAID-style
    python test_diffusion_plaid.py --guidance --guidance_target=z          # Diffusion-LM style
"""

import argparse
import sys
import os
import time
import torch
import torch.nn.functional as F
import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAID_ROOT = os.environ.get('PLAID_ROOT')

if not PLAID_ROOT:
    raise RuntimeError(
        "Set PLAID_ROOT to your local PLAID checkout before running Diffinity."
    )

PLAID_ROOT = os.path.abspath(os.path.expanduser(PLAID_ROOT))
if not os.path.isdir(PLAID_ROOT):
    raise RuntimeError(f"PLAID_ROOT does not point to a directory: {PLAID_ROOT}")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PLAID_ROOT not in sys.path:
    sys.path.insert(0, PLAID_ROOT)

import lib.datasets
import lib.models
import lib.utils


def main():
    parser = argparse.ArgumentParser(description='Test PLAID unconditional generation')

    # Configurable parameters
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to PLAID weights directory')
    parser.add_argument('--n_samples', type=int, default=8,
                        help='Number of samples to generate (default: 8)')
    parser.add_argument('--seq_len', type=int, default=64,
                        help='Sequence length (default: 64)')
    parser.add_argument('--sampling_timesteps', type=int, default=256,
                        help='Number of diffusion timesteps (default: 256)')
    parser.add_argument('--seed', type=int, default=37,
                        help='Random seed for reproducibility (default: 37)')

    # Fixed PLAID architecture parameters (for 1B model)
    parser.add_argument('--dim', type=int, default=2048,
                        help='Model dimension (default: 2048)')
    parser.add_argument('--n_blocks', type=int, default=24,
                        help='Number of transformer blocks (default: 24)')
    parser.add_argument('--n_heads', type=int, default=32,
                        help='Number of attention heads (default: 32)')
    parser.add_argument('--vocab_size', type=int, default=32768,
                        help='Vocabulary size (default: 32768)')
    parser.add_argument('--embed_dim', type=int, default=16,
                        help='Embedding dimension (default: 16)')

    # Fixed diffusion parameters
    parser.add_argument('--gamma_0', type=float, default=-3.0,
                        help='Noise schedule start (default: -3.0)')
    parser.add_argument('--gamma_1', type=float, default=6.0,
                        help='Noise schedule end (default: 6.0)')
    parser.add_argument('--initial_noise_scale', type=float, default=1.0,
                        help='Initial noise scaling (default: 1.0)')
    parser.add_argument('--score_temp', type=float, default=0.9,
                        help='Score temperature (default: 0.9)')
    parser.add_argument('--ddim_sampler', action='store_true',
                        help='Use DDIM sampler (faster)')

    # Guidance parameters
    parser.add_argument('--guidance', action='store_true',
                        help='Enable gradient-based automaton guidance')
    parser.add_argument('--regex_pattern', type=str, default='.*restaurant.*',
                        help='Regex pattern for guidance (default: .*restaurant.*)')
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                        help='Strength of gradient guidance (default: 1.0)')
    parser.add_argument('--guidance_target', type=str, default='x_reconst',
                        choices=['x_reconst', 'z'],
                        help='Where to apply guidance: x_reconst (PLAID-style) or z (Diffusion-LM style)')
    parser.add_argument('--guidance_interval', type=int, default=1,
                        help='Apply guidance every N steps (default: 1)')
    parser.add_argument('--scoring_mode', type=str, default='neg_squared',
                        choices=['neg_squared', 'inverse', 'gaussian'],
                        help='Distance-to-score conversion (default: neg_squared)')
    parser.add_argument('--scale_schedule', type=str, default='constant',
                        choices=['constant', 'linear_ramp', 'inverse_linear', 'quadratic_ramp', 'cosine'],
                        help='Additional schedule on top of guidance (default: constant). Both modes use PLAID native noise-schedule scaling.')
    parser.add_argument('--gradient_clip', type=float, default=None,
                        help='Max gradient norm for clipping (default: None = no clipping)')
    parser.add_argument('--softmax_temp', type=float, default=1.0,
                        help='Softmax temperature for scoring (lower = sharper, try 0.1 for PLAID)')
    parser.add_argument('--no_noise_scaling', action='store_true',
                        help='Disable noise-schedule scaling (use constant guidance throughout)')
    parser.add_argument('--log_score', action='store_true',
                        help='Use log(score) gradients instead of score gradients (stronger for low scores)')
    parser.add_argument('--guidance_start_frac', type=float, default=0.0,
                        help='Fraction of (noisy) timesteps to skip before starting guidance. '
                             '0.0 = guide all steps (default), 0.5 = guide only last 50%% of steps, '
                             '0.75 = guide only last 25%% of steps. '
                             'Timesteps go from t=1 (noise) to t=0 (clean).')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Process samples in mini-batches of this size to reduce peak GPU memory. '
                             '0 = all at once, -1 = auto (estimate from automaton size)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import numpy as np
    np.random.seed(args.seed)

    # Print configuration
    print("=" * 80)
    if args.guidance:
        print("PLAID Guided Generation Test")
    else:
        print("PLAID Unconditional Generation Test")
    print("=" * 80)
    print(f"Weights: {args.weights_path}")
    print(f"Architecture: dim={args.dim}, n_blocks={args.n_blocks}, n_heads={args.n_heads}")
    print(f"Generation: n_samples={args.n_samples}, seq_len={args.seq_len}")
    print(f"Sampling: timesteps={args.sampling_timesteps}, ddim={args.ddim_sampler}")
    print(f"Embedding dimension: {args.embed_dim}")
    print(f"Random seed: {args.seed}")
    if args.guidance:
        print()
        print("Guidance Configuration:")
        print(f"  Pattern: '{args.regex_pattern}'")
        print(f"  Target: {args.guidance_target}")
        print(f"  Scale: {args.guidance_scale}")
        print(f"  Interval: every {args.guidance_interval} steps")
        print(f"  Scoring mode: {args.scoring_mode}")
        print(f"  Schedule: {args.scale_schedule}")
        print(f"  Gradient clip: {args.gradient_clip}")
        if args.guidance_start_frac > 0:
            skip_steps = int(args.guidance_start_frac * args.sampling_timesteps)
            print(f"  Guidance start: after {skip_steps}/{args.sampling_timesteps} steps "
                  f"(t < {1.0 - args.guidance_start_frac:.2f})")
    print("=" * 80)
    print()

    # Setup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float64)

    # Helper function for log(1-exp(x))
    def log1mexp(x):
        return torch.log(-torch.expm1(x))

    # PLAID loads tokenizer resources from paths relative to its checkout.
    print("Loading tokenizer...")
    original_dir = os.getcwd()
    try:
        os.chdir(PLAID_ROOT)
        tokenizer = lib.datasets.openwebtext2_tokenizer()
    finally:
        os.chdir(original_dir)
    print(f"✓ Tokenizer loaded (vocab_size: {args.vocab_size})")
    print()

    # Load model components
    print("Loading model...")

    # Helper function to create modules (needed for mup base shapes)
    def create_modules(dim, n_heads):
        return {
            'noise_schedule': lib.models.NoiseSchedule().float(),
            'gamma_bounds': lib.models.GammaBounds(args.gamma_0, args.gamma_1).float(),
            'embedding_matrix': lib.models.EmbeddingMatrix(args.vocab_size, args.embed_dim).float(),
            'model': lib.models.DiffusionModel(dim, args.embed_dim, args.n_blocks, n_heads, args.vocab_size).float()
        }

    # Create main modules and base shapes for mup
    modules = create_modules(args.dim, args.n_heads)
    base_modules = create_modules(256, 4)
    delta_modules = create_modules(128, 2)

    # Set base shapes for mup (required for PLAID)
    import mup
    for key in modules:
        main, base, delta = modules[key], base_modules[key], delta_modules[key]
        mup.set_base_shapes(main, base, delta=delta)
        main.cuda()

    # Load weights
    if args.weights_path:
        print(f"Loading weights from {args.weights_path}")
        for key in modules:
            path = os.path.join(args.weights_path, f'{key}.pt')
            if os.path.exists(path):
                modules[key].load_state_dict(torch.load(path))
                print(f"✓ Loaded {key}")
            else:
                print(f"⚠ Warning: {path} not found, using random initialization")

    # Set to eval mode and keep pretrained PLAID weights frozen.
    for module in modules.values():
        module.eval()
        module.requires_grad_(False)

    print()
    print("Model loaded successfully!")

    # Count parameters
    total_params = sum(p.numel() for module in modules.values() for p in module.parameters())
    print(f"Total parameters: {total_params:,}")
    print()

    # Initialize guidance if enabled
    guidance_handler = None
    if args.guidance:
        print("Initializing gradient guidance...")
        from plaid_gradient_guidance import PLAIDGradientGuidance, get_plaid_tokenizer_dict

        # Get tokenizer dict
        print("  Building tokenizer vocabulary...")
        tokenizer_dict = get_plaid_tokenizer_dict(args.vocab_size)
        print(f"  Tokenizer vocab size: {len(tokenizer_dict)}")

        # Get embedding matrix (detach to only compute grads w.r.t. latent)
        embedding_matrix_for_guidance = modules['embedding_matrix']().detach()

        # Create guidance handler
        guidance_handler = PLAIDGradientGuidance(
            regex_pattern=args.regex_pattern,
            token_embeddings=embedding_matrix_for_guidance,
            tokenizer_dict=tokenizer_dict,
            guidance_scale=args.guidance_scale,
            guidance_target=args.guidance_target,
            guidance_interval=args.guidance_interval,
            scoring_mode=args.scoring_mode,
            scale_schedule=args.scale_schedule,
            gradient_clip=args.gradient_clip,
            total_steps=args.sampling_timesteps,
            temperature=args.softmax_temp,
            use_noise_scaling=not args.no_noise_scaling,
            use_log_score=args.log_score,
            verbose=True
        )

        # Check satisfiability: BFS over token automaton to find shortest
        # accepting path. If > seq_len tokens, no sequence can ever match.
        from collections import deque
        automaton = guidance_handler.automaton
        visited = {automaton.initial_state}
        queue = deque([(automaton.initial_state, 0)])
        min_tokens = None
        while queue:
            state, depth = queue.popleft()
            if automaton.is_final(state):
                min_tokens = depth
                break
            if depth >= args.seq_len:
                continue
            for next_state in set(automaton.get_transitions(state).values()):
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, depth + 1))
        if min_tokens is None or min_tokens > args.seq_len:
            print(f"UNSATISFIABLE: shortest accepting path requires "
                  f"{min_tokens if min_tokens is not None else '>'+str(args.seq_len)} tokens "
                  f"but seq_len={args.seq_len}")
            sys.exit(1)
        print(f"  Min tokens to satisfy regex: {min_tokens}")
        print()

    # Generate samples
    print("=" * 80)
    if args.guidance:
        print(f"Generating guided samples (pattern: '{args.regex_pattern}')...")
    else:
        print("Generating unconditional samples...")
    print("=" * 80)

    # Determine batch size for mini-batching
    batch_size = args.batch_size
    if batch_size == -1 and guidance_handler is not None:
        # Auto: estimate from automaton size
        from plaid_gradient_guidance import recommend_batch_size
        num_states = len(guidance_handler.automaton.state_list)
        num_transitions = len(guidance_handler.automaton.transitions)
        batch_size = recommend_batch_size(args.n_samples, args.seq_len, num_states, num_transitions)
        print(f"Auto batch size: {batch_size} (for {args.n_samples} total samples)")
        print(f"BATCH_SIZE: {batch_size}")
    elif batch_size <= 0:
        batch_size = args.n_samples
    else:
        print(f"BATCH_SIZE: {batch_size}")

    n_batches = (args.n_samples + batch_size - 1) // batch_size
    if n_batches > 1:
        print(f"Mini-batching: {n_batches} batches of up to {batch_size} samples")

    # Use torch.no_grad() for unconditional, but need grads for guidance
    context_manager = torch.enable_grad() if args.guidance else torch.no_grad()

    # Pre-generate all initial noise with the original seed so that results
    # are identical regardless of batch_size (each sample gets the same z_init).
    torch.manual_seed(args.seed)
    all_z_init = torch.randn((args.n_samples, args.seq_len, args.embed_dim), device='cuda') * args.initial_noise_scale

    all_x_samples = []
    generation_start_time = time.time()

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_n = min(batch_size, args.n_samples - batch_start)

        if n_batches > 1:
            print(f"\n--- Batch {batch_idx+1}/{n_batches} ({batch_n} samples) ---")

        with context_manager:
            embedding_matrix = modules['embedding_matrix']()
            gamma_0, gamma_1 = modules['gamma_bounds']()

            # Slice pre-generated noise for this batch
            z = all_z_init[batch_start:batch_start + batch_n].clone()
            x_selfcond = torch.zeros_like(z).float()

            # Reset guidance step counter if using guidance
            if guidance_handler is not None:
                guidance_handler.reset()

            # Diffusion sampling loop
            for i, t in enumerate(tqdm.tqdm(torch.linspace(1., 0., args.sampling_timesteps))):
                t = t[None].cuda()
                s = t - 1. / args.sampling_timesteps

                # Compute noise schedule
                gamma_s = modules['noise_schedule'](s).double()
                gamma_t = modules['noise_schedule'](t).double()
                gamma_s = gamma_0 + (gamma_1 - gamma_0) * gamma_s
                gamma_t = gamma_0 + (gamma_1 - gamma_0) * gamma_t

                # Compute alphas and sigmas
                alpha_squared_s = torch.sigmoid(-gamma_s)
                alpha_squared_t = torch.sigmoid(-gamma_t)
                alpha_s = alpha_squared_s.sqrt()
                alpha_t = alpha_squared_t.sqrt()
                sigma_squared_s = torch.sigmoid(gamma_s)
                sigma_squared_t = torch.sigmoid(gamma_t)
                sigma_s = sigma_squared_s.sqrt()
                sigma_t = sigma_squared_t.sqrt()

                # Model forward pass
                guidance_active_this_step = (
                    guidance_handler is not None
                    and i >= int(args.guidance_start_frac * args.sampling_timesteps)
                )
                if guidance_handler is not None:
                    z_for_forward = z.float().requires_grad_(True)
                else:
                    z_for_forward = z.float()

                logits, x_reconst = modules['model'](
                    z=z_for_forward,
                    gamma=gamma_t.float(),
                    embedding_matrix=embedding_matrix,
                    bias_scale=1.,
                    x_selfcond=x_selfcond
                )

                x_selfcond = x_reconst.clone().detach()
                x_reconst = x_reconst.double()

                # ============================================================
                # APPLY GRADIENT GUIDANCE (if enabled)
                # ============================================================
                if guidance_active_this_step:
                    x_reconst_for_guidance = x_reconst.clone().detach()

                    x_reconst, z_modified = guidance_handler.apply(
                        z=z_for_forward,
                        x_reconst=x_reconst_for_guidance,
                        sigma_squared_t=sigma_squared_t,
                        alpha_squared_t=alpha_squared_t,
                        logits=logits
                    )

                    if args.guidance_target == 'z':
                        z = z_modified.detach().double()
                    else:
                        z = z.detach()
                elif guidance_handler is not None:
                    # Guidance not yet active but requires_grad was set —
                    # detach to free the computation graph and prevent
                    # memory accumulation from unused forward passes.
                    z = z.detach()
                    x_reconst = x_reconst.detach()
                # ============================================================

                # Compute predicted noise
                epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
                epsilon_pred /= args.score_temp
                x_reconst = (z - (sigma_t * epsilon_pred)) / alpha_t

                # Update z (if not final step)
                if t > 0:
                    if args.ddim_sampler:
                        z = (alpha_s * x_reconst) + (sigma_s * epsilon_pred)
                    else:
                        c = -torch.expm1(gamma_s - gamma_t)
                        z *= (1 - c) * alpha_squared_s.sqrt() / alpha_squared_t.sqrt()
                        z += c * (alpha_squared_s.sqrt() * x_reconst.double())
                        z += (c * (1 - alpha_squared_s)).sqrt() * torch.randn_like(z)

            # Final decoding for this batch
            logits, _ = modules['model'](
                z=z.float(),
                gamma=gamma_t.float(),
                embedding_matrix=embedding_matrix,
                bias_scale=1.,
                x_selfcond=x_selfcond
            )
            batch_samples = logits.argmax(dim=-1)
            all_x_samples.append(batch_samples)

    x_samples = torch.cat(all_x_samples, dim=0)

    generation_elapsed = time.time() - generation_start_time
    print(f"\nGENERATION_TIME_SECONDS: {generation_elapsed:.2f}")

    # Print generated samples with parseable markers
    print()
    print("=" * 80)
    print("Generated Samples:")
    print("=" * 80)

    for idx, x in enumerate(x_samples):
        decoded = tokenizer.decode(x.tolist(), skip_special_tokens=False)
        # Replace newlines with ↵ symbol for cleaner printing
        decoded_display = decoded.replace("\n", "↵")
        print(f"\nSample {idx + 1}:")
        print(decoded_display)
        print("-" * 80)

    # Print samples in parseable format at the end
    print()
    print("=" * 80)
    print("PARSEABLE_OUTPUT_START")
    print("=" * 80)
    for idx, x in enumerate(x_samples):
        decoded = tokenizer.decode(x.tolist(), skip_special_tokens=False)
        # Use markers that are easy to parse
        print(f"<SAMPLE id={idx}>")
        print(decoded)
        print("</SAMPLE>")
    print("=" * 80)
    print("PARSEABLE_OUTPUT_END")
    print("=" * 80)

    print()
    print("✓ Generation complete!")


if __name__ == '__main__':
    main()
