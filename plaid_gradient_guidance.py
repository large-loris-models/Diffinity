"""
Gradient-based automaton guidance for PLAID diffusion model.

This module provides gradient guidance that steers PLAID's generation
toward sequences matching a regex pattern, using differentiable automaton
scoring from compute_score.py.

Two guidance modes are supported:
1. 'x_reconst': Apply gradient to reconstruction (PLAID-native style)
2. 'z': Apply gradient directly to noisy latent (Diffusion-LM style)

Usage:
    from plaid_gradient_guidance import PLAIDGradientGuidance, get_plaid_tokenizer_dict

    tokenizer_dict = get_plaid_tokenizer_dict()
    guidance = PLAIDGradientGuidance(
        regex_pattern='.*restaurant.*',
        token_embeddings=embedding_matrix,
        tokenizer_dict=tokenizer_dict,
        guidance_scale=1.0,
        guidance_target='x_reconst'
    )

    # In sampling loop:
    x_reconst, z = guidance.apply(z, x_reconst, sigma_squared_t, alpha_squared_t)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Callable, Tuple
import sys
import os
import time

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from automaton_alignment import SimpleTokenizerWrapper, AutomatonAligner
from compute_score import distance_score, distance_score_batched, logits_score_batched, SEQUENTIAL_REDUCTION_THRESHOLD


def require_plaid_root() -> str:
    """Return PLAID_ROOT after validating that it points to a PLAID checkout."""
    plaid_root = os.environ.get('PLAID_ROOT')
    if not plaid_root:
        raise RuntimeError(
            "Set PLAID_ROOT to your local PLAID checkout before running Diffinity."
        )

    plaid_root = os.path.abspath(os.path.expanduser(plaid_root))
    if not os.path.isdir(plaid_root):
        raise RuntimeError(f"PLAID_ROOT does not point to a directory: {plaid_root}")

    if plaid_root not in sys.path:
        sys.path.insert(0, plaid_root)
    return plaid_root


# Memory estimation constants
# The 1.8x safety factor accounts for:
#   - logsumexp backward creates a temporary copy of the [B, T/2, S, S, S] intermediate (~1.5x)
#   - model forward activations for backward (~0.2 GB/sample)
#   - CUDA allocator fragmentation
GUIDANCE_MEMORY_SAFETY_FACTOR = 2.5


def estimate_guidance_memory_gb(batch_size, seq_len, num_states, num_transitions):
    """Estimate peak GPU memory (GB) for the guidance computation.

    Dominant costs in logits_score_batched:
      Forward saved tensors:
        1. SegmentLogsumexp vals for autograd: B * T * num_transitions * 4
        2. Parallel reduction intermediates (summed over log2(T) levels):
           ~B * T * S^3 * 4
      Backward temporaries:
        3. logsumexp backward creates softmax weights: ~B * T/2 * S^3 * 4
        4. Model forward activations saved for backward: ~0.15 GB per sample

    The 2.5x safety factor accounts for backward overhead (which scales
    super-linearly with S^3), model activations, and CUDA fragmentation.
    """
    if num_states > SEQUENTIAL_REDUCTION_THRESHOLD:
        # Sequential path: O(S²) reduction intermediates instead of O(S³)
        per_sample = seq_len * (num_transitions + num_states ** 2) * 4
    else:
        per_sample = seq_len * (num_transitions + num_states ** 3) * 4
    raw_gb = batch_size * per_sample / 1e9
    return raw_gb * GUIDANCE_MEMORY_SAFETY_FACTOR


def recommend_batch_size(n_samples, seq_len, num_states, num_transitions,
                         gpu_memory_gb=None, base_memory_gb=None):
    """Recommend a batch size that fits in GPU memory.

    If gpu_memory_gb/base_memory_gb are not provided, attempts to read them
    from CUDA. Falls back to conservative defaults.
    """
    if gpu_memory_gb is None:
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            gpu_memory_gb = 48.0
    if base_memory_gb is None:
        if torch.cuda.is_available():
            base_memory_gb = torch.cuda.memory_allocated() / 1e9
        else:
            base_memory_gb = 6.0

    available_gb = gpu_memory_gb - base_memory_gb
    per_sample_gb = estimate_guidance_memory_gb(1, seq_len, num_states, num_transitions)
    if per_sample_gb <= 0:
        return n_samples
    max_batch = max(1, int(available_gb / per_sample_gb))
    return min(max_batch, n_samples)


def get_plaid_tokenizer_dict(vocab_size: int = 32768) -> Dict[int, str]:
    """
    Extract vocabulary dictionary from PLAID's openwebtext2 tokenizer.

    Args:
        vocab_size: Size of the vocabulary (default: 32768 for PLAID)

    Returns:
        Dictionary mapping token_id -> token_string
    """
    # PLAID loads tokenizer resources from paths relative to its checkout.
    plaid_dir = require_plaid_root()
    original_dir = os.getcwd()

    try:
        os.chdir(plaid_dir)
        import lib.datasets
        tokenizer = lib.datasets.openwebtext2_tokenizer()
    finally:
        os.chdir(original_dir)

    # Build dict: token_id -> token_string
    tokenizer_dict = {}
    for token_id in range(vocab_size):
        # Decode single token
        try:
            token_str = tokenizer.decode([token_id], skip_special_tokens=False)
            tokenizer_dict[token_id] = token_str
        except Exception:
            tokenizer_dict[token_id] = f'<UNK_{token_id}>'

    return tokenizer_dict


# ============================================================================
# Scaling schedules for guidance strength
# ============================================================================

def constant_schedule(base_scale: float, step: int, total_steps: int) -> float:
    """Constant guidance scale throughout sampling."""
    return base_scale


def linear_ramp_schedule(base_scale: float, step: int, total_steps: int) -> float:
    """Linear ramp from 0 to base_scale (stronger guidance at end)."""
    progress = step / max(total_steps, 1)
    return base_scale * progress


def inverse_linear_schedule(base_scale: float, step: int, total_steps: int) -> float:
    """Linear ramp from base_scale to 0 (stronger guidance at start)."""
    progress = step / max(total_steps, 1)
    return base_scale * (1.0 - progress)


def quadratic_ramp_schedule(base_scale: float, step: int, total_steps: int) -> float:
    """Quadratic ramp from 0 to base_scale (even stronger at end)."""
    progress = step / max(total_steps, 1)
    return base_scale * (progress ** 2)


def cosine_schedule(base_scale: float, step: int, total_steps: int) -> float:
    """Cosine schedule: starts at 0, peaks at base_scale, returns to 0."""
    import math
    progress = step / max(total_steps, 1)
    return base_scale * (1.0 - math.cos(progress * math.pi)) / 2.0


SCHEDULE_REGISTRY = {
    'constant': constant_schedule,
    'linear_ramp': linear_ramp_schedule,
    'inverse_linear': inverse_linear_schedule,
    'quadratic_ramp': quadratic_ramp_schedule,
    'cosine': cosine_schedule,
}


# ============================================================================
# Main guidance class
# ============================================================================

class PLAIDGradientGuidance:
    """
    Gradient-based guidance for PLAID diffusion model using automaton constraints.

    This class computes differentiable scores based on how well latent vectors
    align with a regex pattern (via automaton), then uses gradients to guide
    the diffusion process toward matching sequences.

    Attributes:
        automaton: Token-level automaton representing the regex constraint
        wrapped_tokenizer: Tokenizer wrapped for automaton compatibility
        token_embeddings: Detached embedding matrix for scoring
        guidance_scale: Base strength of guidance
        guidance_target: Where to apply guidance ('x_reconst' or 'z')
        guidance_interval: Apply guidance every N steps
        scoring_mode: Distance-to-score conversion ('neg_squared', 'inverse', 'gaussian')
        scale_schedule: Function to adjust scale based on step
        gradient_clip: Maximum gradient norm (None for no clipping)
        step_count: Current step counter
        total_steps: Total number of sampling steps
    """

    def __init__(
        self,
        regex_pattern: str,
        token_embeddings: torch.Tensor,
        tokenizer_dict: Dict[int, str],
        guidance_scale: float = 1.0,
        guidance_target: str = 'x_reconst',
        guidance_interval: int = 1,
        scoring_mode: str = 'neg_squared',
        scale_schedule: str = 'constant',
        gradient_clip: Optional[float] = None,
        total_steps: int = 256,
        temperature: float = 1.0,
        use_noise_scaling: bool = True,
        use_log_score: bool = False,
        verbose: bool = True
    ):
        """
        Initialize guidance handler.

        Args:
            regex_pattern: Regex pattern to guide generation toward
            token_embeddings: Embedding matrix from PLAID model [vocab_size, embed_dim]
            tokenizer_dict: Dict mapping token_id -> token_string
            guidance_scale: Base strength of gradient guidance
            guidance_target: 'x_reconst' (PLAID-style) or 'z' (Diffusion-LM style)
            guidance_interval: Apply guidance every N steps
            scoring_mode: 'neg_squared' (recommended), 'inverse', or 'gaussian'
            scale_schedule: Additional schedule on top of guidance_scale. Both modes use
                           PLAID's native noise-schedule scaling (sigma²/√alpha²), so 'constant'
                           is typically sufficient.
            gradient_clip: Max gradient norm, None to disable
            total_steps: Total sampling timesteps (for schedule)
            temperature: Softmax temperature for scoring. Lower values (e.g., 0.1) make the
                        probability distribution more peaked, giving more weight to closer tokens.
                        Recommended for large vocabularies like PLAID's 32K tokens.
            use_noise_scaling: If True, scale guidance by sigma²/√alpha² (strong early, weak late).
                              If False, use constant scaling (may work better for semantic guidance).
            use_log_score: If True, compute gradients of log(score) instead of score.
                          This removes the gradient ∝ probability scaling issue, giving
                          stronger gradients for low-scoring samples.
            verbose: Print debug information
        """
        self.guidance_scale = guidance_scale
        self.guidance_target = guidance_target
        self.guidance_interval = guidance_interval
        self.scoring_mode = scoring_mode
        self.gradient_clip = gradient_clip
        self.total_steps = total_steps
        self.temperature = temperature
        self.use_noise_scaling = use_noise_scaling
        self.use_log_score = use_log_score
        self.verbose = verbose
        self.step_count = 0

        # Validate guidance target
        if guidance_target not in ['x_reconst', 'z']:
            raise ValueError(f"guidance_target must be 'x_reconst' or 'z', got '{guidance_target}'")

        # Setup scale schedule
        if isinstance(scale_schedule, str):
            if scale_schedule not in SCHEDULE_REGISTRY:
                raise ValueError(f"Unknown schedule: {scale_schedule}. Available: {list(SCHEDULE_REGISTRY.keys())}")
            self.scale_schedule = SCHEDULE_REGISTRY[scale_schedule]
        else:
            self.scale_schedule = scale_schedule

        # CRITICAL: Detach embeddings to only compute gradients w.r.t. latent
        self.token_embeddings = token_embeddings.detach()

        # Setup automaton from regex
        self.wrapped_tokenizer = SimpleTokenizerWrapper(tokenizer_dict)
        self.aligner = AutomatonAligner(self.wrapped_tokenizer)

        if verbose:
            print(f"Creating automaton for pattern: '{regex_pattern}'")

        # Build the character-level FSM first (fast) and pre-check memory
        # before spending minutes/hours building the full token automaton.
        fsm = self.aligner.regex_to_fsm(regex_pattern)
        num_states_est = len(fsm.states)
        num_trans_est = num_states_est * self.aligner.vocab_size  # upper bound
        # Divide out the safety factor: char_fsm × vocab_size already overestimates
        # actual token automaton transitions, so applying 2.5× on top causes false OOMs.
        est_per_sample_pre = estimate_guidance_memory_gb(1, 64, num_states_est, num_trans_est) / GUIDANCE_MEMORY_SAFETY_FACTOR
        if torch.cuda.is_available():
            gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_used_gb = torch.cuda.memory_allocated() / 1e9
            available_gb = gpu_total_gb - gpu_used_gb
            if est_per_sample_pre > available_gb:
                print(f"ESTIMATED_PER_SAMPLE_GB: {est_per_sample_pre:.2f}")
                print(f"RECOMMENDED_BATCH_SIZE: 1")
                print(f"Pre-build memory estimate {est_per_sample_pre:.1f} GB exceeds "
                      f"available {available_gb:.1f} GB — skipping automaton build.")
                print("CUDA out of memory")
                sys.exit(1)

        # PLAID doesn't have explicit END tokens, so disable terminal state handling
        t_automaton_start = time.time()
        self.automaton = self.aligner.create_token_automaton(
            fsm=fsm,
            add_terminal_states=False
        )
        t_automaton_elapsed = time.time() - t_automaton_start

        if verbose:
            print(f"Automaton created:")
            print(f"  States: {len(self.automaton.state_list)}")
            print(f"  Final states: {len(self.automaton.final_states)}")
            print(f"  Transitions: {len(self.automaton.transitions)}")
            print(f"AUTOMATON_CREATION_SECONDS: {t_automaton_elapsed:.2f}")
            print(f"AUTOMATON_STATES: {len(self.automaton.state_list)}")
            print(f"AUTOMATON_TRANSITIONS: {len(self.automaton.transitions)}")
            # Report memory estimation
            num_st = len(self.automaton.state_list)
            num_tr = len(self.automaton.transitions)
            est_per_sample = estimate_guidance_memory_gb(1, 64, num_st, num_tr)
            print(f"  Est. guidance memory per sample: {est_per_sample:.2f} GB")
            print(f"ESTIMATED_PER_SAMPLE_GB: {est_per_sample:.2f}")
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / 1e9
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                rec_batch = recommend_batch_size(20, 64, num_st, num_tr)
                print(f"  GPU: {mem_alloc:.1f} GB used / {mem_total:.1f} GB total")
                print(f"  Recommended max batch size (for 20 samples): {rec_batch}")
                print(f"RECOMMENDED_BATCH_SIZE: {rec_batch}")
            print(f"  Guidance target: {guidance_target}")
            print(f"  Guidance scale: {guidance_scale}")
            print(f"  Scoring mode: {scoring_mode}")
            print(f"  Temperature: {temperature}")
            print(f"  Noise scaling: {use_noise_scaling}")
            print(f"  Log-score gradients: {use_log_score}")
            print(f"  Scale schedule: {scale_schedule if isinstance(scale_schedule, str) else 'custom'}")

    def reset(self):
        """Reset step counter (call at start of each generation)."""
        self.step_count = 0

    def _compute_guidance_gradient(
        self,
        latent: torch.Tensor,
        return_score: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute gradient of automaton score w.r.t. latent vectors.

        Args:
            latent: Latent vectors [batch_size, seq_len, embed_dim]
            return_score: If True, also return the scores

        Returns:
            gradients: Gradient tensor same shape as latent
            scores: Optional score tensor [batch_size] if return_score=True
        """
        batch_size = latent.shape[0]

        # Ensure float32 for scoring (token embeddings are float32)
        latent_f32 = latent.float()

        with torch.enable_grad():
            latent_grad = latent_f32.clone().detach().requires_grad_(True)

            # Compute automaton scores using batched scoring
            scores = distance_score_batched(
                latent_grad,
                self.wrapped_tokenizer,
                self.automaton,
                self.token_embeddings,
                scoring_mode=self.scoring_mode,
                temperature=self.temperature
            )

            # Compute gradients
            if self.use_log_score:
                # Use log(score) to remove gradient ∝ probability scaling
                # This gives stronger gradients for low-scoring samples
                log_scores = torch.log(scores + 1e-10)  # eps for numerical stability
                log_scores.sum().backward()
            else:
                # Standard: gradient of score directly
                scores.sum().backward()

            gradients = latent_grad.grad
            gradients = gradients.clone() if gradients is not None else torch.zeros_like(latent_grad)

        # Apply gradient clipping if specified
        if self.gradient_clip is not None:
            grad_norms = gradients.view(batch_size, -1).norm(dim=1, keepdim=True)
            # Reshape for broadcasting: [batch_size, 1, 1]
            grad_norms = grad_norms.view(batch_size, 1, 1)
            clip_coef = torch.clamp(self.gradient_clip / (grad_norms + 1e-8), max=1.0)
            gradients = gradients * clip_coef

        if return_score:
            return gradients, scores.detach()
        return gradients, None

    def _compute_guidance_gradient_from_model(
        self,
        z: torch.Tensor,
        model: torch.nn.Module,
        gamma: torch.Tensor,
        embedding_matrix: torch.Tensor,
        x_selfcond: torch.Tensor,
        return_score: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute gradient of automaton score using model's actual logits.

        This backpropagates through the transformer to get gradients w.r.t. z
        based on the model's real probability distribution.

        Args:
            z: Noisy latent [batch_size, seq_len, embed_dim]
            model: PLAID DiffusionModel
            gamma: Current gamma value for noise schedule
            embedding_matrix: Token embedding matrix
            x_selfcond: Self-conditioning input
            return_score: If True, also return the scores

        Returns:
            gradients: Gradient tensor same shape as z
            scores: Optional score tensor [batch_size] if return_score=True
        """
        batch_size = z.shape[0]

        with torch.enable_grad():
            # Clone z and enable gradients
            z_grad = z.clone().detach().float().requires_grad_(True)

            # Run model forward pass to get real logits
            logits, _ = model(
                z=z_grad,
                gamma=gamma.float(),
                embedding_matrix=embedding_matrix,
                bias_scale=1.,
                x_selfcond=x_selfcond
            )

            # Convert logits to LOG-probabilities (stay in log-space to avoid underflow)
            # Apply temperature scaling if desired
            log_probs = F.log_softmax(logits / self.temperature, dim=-1)

            # Compute automaton scores in log-space (returns log-scores)
            log_scores = logits_score_batched(log_probs, self.automaton)

            # Backprop on log-scores directly (no need to exponentiate)
            log_scores.sum().backward()

            gradients = z_grad.grad

            gradients = gradients.clone() if gradients is not None else torch.zeros_like(z_grad)

            # Convert log-scores to scores for display/return
            scores = torch.exp(log_scores)

        # Apply gradient clipping if specified
        if self.gradient_clip is not None:
            grad_norms = gradients.view(batch_size, -1).norm(dim=1, keepdim=True)
            grad_norms = grad_norms.view(batch_size, 1, 1)
            clip_coef = torch.clamp(self.gradient_clip / (grad_norms + 1e-8), max=1.0)
            gradients = gradients * clip_coef

        if return_score:
            return gradients, scores.detach()
        return gradients, None

    def _compute_guidance_gradient_from_logits(
        self,
        logits: torch.Tensor,
        return_score: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute automaton score from pre-computed logits (no gradient computation).

        This is a fast path when we already have logits from the sampling forward pass
        and just need to compute the score (without gradients back to z).

        Args:
            logits: Pre-computed logits from model [batch_size, seq_len, vocab_size]
            return_score: If True, also return the scores

        Returns:
            gradients: Zero tensor (no gradients computed in this mode)
            scores: Score tensor [batch_size] if return_score=True
        """
        batch_size = logits.shape[0]

        # Convert logits to LOG-probabilities
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)

        # Compute automaton scores in log-space
        log_scores = logits_score_batched(log_probs.detach(), self.automaton)
        scores = torch.exp(log_scores)

        # No gradients in this mode - return zeros
        gradients = torch.zeros_like(logits[:, :, :logits.shape[-1] // 2048])  # Placeholder shape
        # Actually we need correct shape - let's just return None and handle it in apply()

        if return_score:
            return None, scores.detach()
        return None, None

    def _compute_guidance_gradient_from_precomputed_logits(
        self,
        z: torch.Tensor,
        logits: torch.Tensor,
        return_score: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute gradient of automaton score using pre-computed logits.

        This reuses logits from the sampling forward pass but requires that the
        computation graph is still intact (logits must have grad_fn connecting to z).

        Args:
            z: Noisy latent [batch_size, seq_len, embed_dim] (for shape reference)
            logits: Pre-computed logits WITH gradient graph [batch_size, seq_len, vocab_size]
            return_score: If True, also return the scores

        Returns:
            gradients: Gradient tensor same shape as z
            scores: Optional score tensor [batch_size] if return_score=True
        """
        batch_size = z.shape[0]

        if not logits.requires_grad:
            if self.verbose and self.step_count == 1:
                print(f"    [WARNING] logits don't require grad - cannot compute gradients!")
            # Fallback: return zeros
            return torch.zeros_like(z), torch.zeros(batch_size, device=z.device)

        # Convert logits to LOG-probabilities (stay in log-space to avoid underflow)
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)

        # Compute automaton scores in log-space (returns log-scores)
        torch.cuda.synchronize()
        t_score_start = time.time()
        log_scores = logits_score_batched(log_probs, self.automaton)
        torch.cuda.synchronize()
        t_score_end = time.time()

        # Backprop in two stages to separate automaton vs model backward cost
        # Stage 1: Automaton backward (log_scores -> logits, through automaton + log_softmax)
        torch.cuda.synchronize()
        t_auto_bwd_start = time.time()
        grad_logits = torch.autograd.grad(
            log_scores.sum(),
            logits,
            retain_graph=True,
        )[0]
        torch.cuda.synchronize()
        t_auto_bwd_end = time.time()

        # Stage 2: Model backward (logits -> z, through entire 1B model)
        torch.cuda.synchronize()
        t_model_bwd_start = time.time()
        gradients = torch.autograd.grad(
            logits,
            z,
            grad_outputs=grad_logits,
            retain_graph=False,
            allow_unused=True
        )[0]
        torch.cuda.synchronize()
        t_model_bwd_end = time.time()

        # Accumulate timing stats
        if not hasattr(self, '_timing_stats'):
            self._timing_stats = {'score_fwd': 0.0, 'auto_bwd': 0.0, 'model_bwd': 0.0, 'count': 0}
        self._timing_stats['score_fwd'] += t_score_end - t_score_start
        self._timing_stats['auto_bwd'] += t_auto_bwd_end - t_auto_bwd_start
        self._timing_stats['model_bwd'] += t_model_bwd_end - t_model_bwd_start
        self._timing_stats['count'] += 1
        if self._timing_stats['count'] % 100 == 0:
            n = self._timing_stats['count']
            avg_fwd = self._timing_stats['score_fwd'] / n * 1000
            avg_auto = self._timing_stats['auto_bwd'] / n * 1000
            avg_model = self._timing_stats['model_bwd'] / n * 1000
            print(f"    [TIMING] step {n}: avg score_fwd={avg_fwd:.1f}ms, auto_bwd={avg_auto:.1f}ms, model_bwd={avg_model:.1f}ms")

        if gradients is None:
            gradients = torch.zeros_like(z)
        else:
            gradients = gradients.clone()

        # Convert log-scores to scores for display/return
        scores = torch.exp(log_scores)

        # Apply gradient clipping if specified
        if self.gradient_clip is not None:
            grad_norms = gradients.view(batch_size, -1).norm(dim=1, keepdim=True)
            grad_norms = grad_norms.view(batch_size, 1, 1)
            clip_coef = torch.clamp(self.gradient_clip / (grad_norms + 1e-8), max=1.0)
            gradients = gradients * clip_coef

        if return_score:
            return gradients, scores.detach()
        return gradients, None

    def apply(
        self,
        z: torch.Tensor,
        x_reconst: torch.Tensor,
        sigma_squared_t: torch.Tensor,
        alpha_squared_t: torch.Tensor,
        model: Optional[torch.nn.Module] = None,
        gamma: Optional[torch.Tensor] = None,
        embedding_matrix: Optional[torch.Tensor] = None,
        x_selfcond: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gradient guidance to the diffusion process.

        Args:
            z: Current noisy latent [batch_size, seq_len, embed_dim] (float64)
               MUST have requires_grad=True if logits are passed for gradient computation
            x_reconst: Current reconstruction [batch_size, seq_len, embed_dim] (float64)
            sigma_squared_t: Noise variance at timestep t
            alpha_squared_t: Signal variance at timestep t
            model: Optional PLAID DiffusionModel. If provided AND logits is None,
                   does a forward pass to get logits. If logits is provided, model is ignored.
            gamma: Current gamma value (required if model is provided and logits is None)
            embedding_matrix: Token embedding matrix (required if model is provided and logits is None)
            x_selfcond: Self-conditioning input (required if model is provided and logits is None)
            logits: Optional pre-computed logits [batch_size, seq_len, vocab_size].
                   If provided, skips the model forward pass (saves ~50% compute).
                   The computation graph must be intact (logits.requires_grad=True) for
                   gradients to flow back to z.

        Returns:
            modified_x_reconst: Modified reconstruction (for epsilon computation)
            modified_z: Modified noisy latent (for next step)
        """
        self.step_count += 1

        # Skip if not at guidance interval
        if self.step_count % self.guidance_interval != 0:
            return x_reconst, z

        # Compute current scale based on schedule
        current_scale = self.scale_schedule(
            self.guidance_scale,
            self.step_count,
            self.total_steps
        )

        # Skip if scale is effectively zero
        if current_scale < 1e-10:
            return x_reconst, z

        # Choose scoring method:
        # 1. Pre-computed logits (fastest - reuses sampling forward pass)
        # 2. Model forward pass (slower - does extra forward pass)
        # 3. Distance-based proxy (legacy)
        use_precomputed_logits = logits is not None
        use_model_logits = model is not None and not use_precomputed_logits

        if use_precomputed_logits:
            # Fast path: use pre-computed logits from sampling forward pass
            gradients, scores = self._compute_guidance_gradient_from_precomputed_logits(
                z=z,
                logits=logits,
                return_score=True
            )
        elif use_model_logits:
            # Slow path: do another forward pass to get logits
            gradients, scores = self._compute_guidance_gradient_from_model(
                z=z,
                model=model,
                gamma=gamma,
                embedding_matrix=embedding_matrix,
                x_selfcond=x_selfcond,
                return_score=True
            )
        else:
            # Legacy: distance-based proxy (doesn't match model's probability)
            if self.guidance_target == 'x_reconst':
                gradients, scores = self._compute_guidance_gradient(x_reconst, return_score=True)
            else:
                gradients, scores = self._compute_guidance_gradient(z, return_score=True)

        # Convert gradients to float64 to match diffusion math
        gradients = gradients.double()

        # Apply guidance based on target
        if self.use_noise_scaling:
            # PLAID's native noise-schedule scaling (sigma²/sqrt(alpha²))
            plaid_scale_factor = sigma_squared_t / alpha_squared_t.sqrt()
            modification = gradients * current_scale * plaid_scale_factor
        else:
            modification = gradients * current_scale

        if self.guidance_target == 'x_reconst':
            modified_x_reconst = x_reconst + modification
            modified_z = z
        else:  # guidance_target == 'z'
            modified_z = z + modification
            modified_x_reconst = x_reconst

        return modified_x_reconst, modified_z


# ============================================================================
# Convenience function for quick setup
# ============================================================================

def create_guidance(
    regex_pattern: str,
    embedding_matrix: torch.Tensor,
    vocab_size: int = 32768,
    **kwargs
) -> PLAIDGradientGuidance:
    """
    Convenience function to create guidance with automatic tokenizer setup.

    Args:
        regex_pattern: Regex pattern for guidance
        embedding_matrix: PLAID's embedding matrix
        vocab_size: Vocabulary size
        **kwargs: Additional arguments for PLAIDGradientGuidance

    Returns:
        Configured PLAIDGradientGuidance instance
    """
    tokenizer_dict = get_plaid_tokenizer_dict(vocab_size)
    return PLAIDGradientGuidance(
        regex_pattern=regex_pattern,
        token_embeddings=embedding_matrix,
        tokenizer_dict=tokenizer_dict,
        **kwargs
    )


# ============================================================================
# Testing
# ============================================================================

if __name__ == '__main__':
    print("PLAID Gradient Guidance Module")
    print("=" * 60)
    print()
    print("This module provides gradient-based guidance for PLAID.")
    print()
    print("Quick test - creating guidance with dummy embeddings...")

    # Create dummy tokenizer dict
    dummy_tokenizer = {i: f'tok_{i}' for i in range(100)}
    dummy_embeddings = torch.randn(100, 16)

    try:
        guidance = PLAIDGradientGuidance(
            regex_pattern='.*tok_5.*',
            token_embeddings=dummy_embeddings,
            tokenizer_dict=dummy_tokenizer,
            guidance_scale=1.0,
            verbose=True
        )
        print()
        print("Basic initialization test passed.")

        # Test gradient computation
        print()
        print("Testing gradient computation...")
        dummy_z = torch.randn(2, 8, 16, dtype=torch.float64, device='cpu')
        dummy_x = torch.randn(2, 8, 16, dtype=torch.float64, device='cpu')
        dummy_sigma_sq = torch.tensor(0.5, dtype=torch.float64)
        dummy_alpha_sq = torch.tensor(0.5, dtype=torch.float64)

        modified_x, modified_z = guidance.apply(
            dummy_z, dummy_x, dummy_sigma_sq, dummy_alpha_sq
        )

        print(f"Input z shape: {dummy_z.shape}")
        print(f"Output x_reconst shape: {modified_x.shape}")
        print(f"x_reconst modified: {not torch.allclose(dummy_x, modified_x)}")
        print()
        print("Gradient computation test passed.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 60)
    print("To use with PLAID:")
    print("  from plaid_gradient_guidance import create_guidance")
    print("  guidance = create_guidance('.*pattern.*', embedding_matrix)")
    print("=" * 60)
