# Diffinity: Gradient-Based Guidance for Discrete Diffusion Language Models

This repository contains code for the paper Continuous Diffusion Models Can Obey Formal Syntax.

## Prerequisites

1. **Install PLAID**: Clone PLAID from its [GitHub repository](https://github.com/igul222/plaid) and install its dependencies. Diffinity expects the environment variable `PLAID_ROOT` to point to that checkout.

2. **Download PLAID weights**: Follow the PLAID README instructions to download pretrained weights (e.g., the 1B-parameter model).

3. **Install our extra dependency**:
   ```bash
   pip install interegular
   ```

4. **Set `PLAID_ROOT`**:
   ```bash
   export PLAID_ROOT=/path/to/plaid
   ```
## Usage

### Guided generation (regex-constrained)

```bash
PLAID_ROOT=/path/to/plaid python test_diffusion_plaid.py \
    --weights_path=/path/to/plaid-weights/ \
    --guidance \
    --regex_pattern='[A-Za-z]+ more [A-Za-z .,]*' \
    --guidance_scale=1.0
```

You will need over 12GB of GPU to run the model. Experiments were performed on a 48GB NVIDIA A6000.

### Key options

- `--regex_pattern`: Regex pattern to constrain against
- `--n_samples`: Number of samples to generate (default: 8)
- `--seq_len`: Sequence length (default: 64)
- `--sampling_timesteps`: Number of diffusion timesteps (default: 256)
- `--guidance_scale`: Strength of guidance (default: 1.0)
- `--batch_size`: Mini-batch size for guidance memory control (`-1` = auto)
- `--guidance_start_frac`: Fraction of noisy timesteps to skip before applying guidance

## File descriptions

| File | Description |
|------|-------------|
| `test_diffusion_plaid.py` | Entry point for unconditional and guided generation |
| `plaid_gradient_guidance.py` | `PLAIDGradientGuidance` class that hooks into the PLAID denoising loop |
| `automaton_alignment.py` | Tokenizer wrapper, automaton aligner, and `TokenAutomaton` for mapping character-level regex to token-level constraints |
| `compute_score.py` | Scoring functions (`distance_score`, `logits_score_batched`) used by the guidance |
