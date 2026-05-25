"""
Score function for automaton-guided diffusion.

This module implements differentiable score functions that can be used
to guide the diffusion process based on automaton constraints.
"""

import torch as th
from automaton_alignment import TokenAutomaton

# When num_states exceeds this threshold, use sequential vector-matrix
# multiplication (O(S²) memory) instead of parallel tree reduction (O(S³) memory).
SEQUENTIAL_REDUCTION_THRESHOLD = 256


def compute_token_score(distance, eps=1e-10, scoring_mode='inverse'):
    """
    Compute a score from the distance between latent vector and token embedding.

    This function can be replaced with different scoring methods for experimentation.

    Args:
        distance: L2 distance between latent vector and token embedding
                 Can be a scalar or a tensor of distances
        eps: Small constant for numerical stability
        scoring_mode: One of:
            - 'inverse': 1/(d+eps) - weak gradients
            - 'neg_squared': -d^2 - stronger gradients
            - 'gaussian': exp(-d^2) - normalized, strong gradients

    Returns:
        score: Higher values indicate the latent vector is closer to the token
              Same shape as input distance
    """
    if scoring_mode == 'inverse':
        return 1.0 / (distance + eps)
    elif scoring_mode == 'neg_squared':
        return -distance ** 2
    elif scoring_mode == 'gaussian':
        return th.exp(-distance ** 2)
    else:
        raise ValueError(f"Unknown scoring_mode: {scoring_mode}")


def distance_score(latent_vector, tokenizer, token_automaton, token_embeddings, scoring_mode='inverse', temperature=1.0):
    """
    Compute a differentiable score based on automaton constraints.

    This function takes a latent vector from the diffusion process and computes
    a score indicating how well it aligns with the automaton constraints.
    The function is differentiable to enable gradient-based guidance.

    Note: This function operates on a single sequence (no batch dimension).
    For batched inputs, call this function separately for each item in the batch.

    Args:
        latent_vector: Latent representation from diffusion process
                      Shape: [seq_len, embed_dim]
        tokenizer: Tokenizer for the model (used to decode/interpret tokens)
        token_automaton: TokenAutomaton instance from automaton_alignment.py
                        defining the constraints
        token_embeddings: Token embedding matrix from the model
                         Shape: [vocab_size, embed_dim]
        scoring_mode: How to convert distances to scores ('inverse', 'neg_squared', 'gaussian')
        temperature: Temperature for softmax normalization. Lower values (e.g., 0.1) make
                    the distribution more peaked, giving more probability to closer tokens.
                    Useful for large vocabularies where softmax becomes too flat.

    Returns:
        score: Differentiable scalar tensor (higher scores indicate better alignment)
    """
    # Get sequence length
    seq_len = latent_vector.shape[0]

    # Get state ordering from the automaton
    num_states = len(token_automaton.state_list)
    initial_state_idx = token_automaton.state_to_idx[token_automaton.initial_state]

    # Initialize score_vector: probability 1 at initial state, 0 elsewhere
    # Shape: [num_states]
    score_vector = th.zeros(num_states, dtype=latent_vector.dtype, device=latent_vector.device)
    score_vector[initial_state_idx] = 1.0

    # Epsilon for numerical stability
    eps = 1e-10

    # Pre-convert all transition data to tensors ONCE (not in the loop!)
    # These tensors are constant across all sequence positions
    state_token_tensors = []
    state_dest_tensors = []
    state_embeddings = []
    for state_idx in range(num_states):
        valid_token_list, dest_indices_list = token_automaton.state_transitions[state_idx]
        if len(valid_token_list) > 0:
            token_ids = th.tensor(valid_token_list, dtype=th.long, device=latent_vector.device)
            state_token_tensors.append(token_ids)
            state_dest_tensors.append(
                th.tensor(dest_indices_list, dtype=th.long, device=latent_vector.device)
            )
            # Pre-compute embeddings (don't depend on current_latent, so compute once!)
            state_embeddings.append(token_embeddings[token_ids])
        else:
            state_token_tensors.append(None)
            state_dest_tensors.append(None)
            state_embeddings.append(None)

    # Loop through each position in the sequence
    for pos in range(seq_len):
        # Get latent vector at current position
        current_latent = latent_vector[pos]  # Shape: [embed_dim]

        # Compute transition matrix based on current_latent
        # transition_matrix[i][j] = probability of transitioning from state i to state j
        transition_matrix = th.zeros(
            (num_states, num_states),
            dtype=latent_vector.dtype,
            device=latent_vector.device
        )

        # For each state, compute transition probabilities
        for state_idx in range(num_states):
            # Get pre-computed tensors (reuse across all positions!)
            dest_indices = state_dest_tensors[state_idx]
            valid_embeddings = state_embeddings[state_idx]

            if valid_embeddings is None:
                # No valid transitions from this state
                continue

            # VECTORIZED: Compute all distances at once using broadcasting
            # current_latent: [embed_dim]
            # valid_embeddings: [num_valid, embed_dim]
            # Result: [num_valid] distances
            distances = th.norm(
                current_latent.unsqueeze(0) - valid_embeddings,  # Broadcasting: [1, embed_dim] - [num_valid, embed_dim]
                p=2,
                dim=1
            )  # [num_valid]

            # VECTORIZED: Compute all scores at once
            token_scores = compute_token_score(distances, eps, scoring_mode)  # [num_valid]

            # Normalize to get probabilities
            # For negative scores (neg_squared), we need to use softmax normalization
            # For positive scores (inverse, gaussian), simple normalization works
            if scoring_mode == 'neg_squared':
                # Softmax normalization with temperature: exp(score/T) / sum(exp(score/T))
                # Lower temperature makes distribution more peaked (more probability on closest tokens)
                token_probs = th.softmax(token_scores / temperature, dim=0)  # [num_valid]
            else:
                # Simple normalization for positive scores (temperature applied as power)
                token_probs = token_scores / (token_scores.sum() + eps)  # [num_valid]

            # VECTORIZED: Fill transition matrix using scatter_add (no Python loop!)
            # Accumulate probabilities at destination state indices
            transition_matrix[state_idx].scatter_add_(0, dest_indices, token_probs)

        # Update score_vector: new distribution over states after seeing this token
        # score_vector[j] = Σᵢ score_vector[i] × transition_matrix[i][j]
        score_vector = score_vector @ transition_matrix

    # Compute final score: sum of probabilities for all final states
    # Higher score means higher probability of being accepted by the automaton
    final_state_indices = [token_automaton.state_to_idx[s] for s in token_automaton.final_states]

    if len(final_state_indices) > 0:
        # Index into score_vector and sum - this preserves gradients properly
        final_score = score_vector[final_state_indices].sum()
    else:
        # No final states (edge case) - return zero with proper gradient connection
        final_score = score_vector.sum() * 0.0

    return final_score


def distance_score_batched(latent_vectors, tokenizer, token_automaton, token_embeddings, scoring_mode='inverse', temperature=1.0):
    """
    Batched version of distance_score for improved GPU utilization.

    Compute differentiable scores for a batch of sequences based on automaton constraints.
    Processes all samples in parallel for better performance.

    Args:
        latent_vectors: Batched latent representations from diffusion process
                       Shape: [batch_size, seq_len, embed_dim]
        tokenizer: Tokenizer for the model (used to decode/interpret tokens)
        token_automaton: TokenAutomaton instance from automaton_alignment.py
                        defining the constraints
        token_embeddings: Token embedding matrix from the model
                         Shape: [vocab_size, embed_dim]
        scoring_mode: How to convert distances to scores ('inverse', 'neg_squared', 'gaussian')
        temperature: Temperature for softmax normalization. Lower values (e.g., 0.1) make
                    the distribution more peaked, giving more probability to closer tokens.
                    Useful for large vocabularies where softmax becomes too flat.

    Returns:
        scores: Differentiable score tensor, one score per sample
               Shape: [batch_size]
    """
    batch_size = latent_vectors.shape[0]
    seq_len = latent_vectors.shape[1]

    # Get state ordering from the automaton
    num_states = len(token_automaton.state_list)
    initial_state_idx = token_automaton.state_to_idx[token_automaton.initial_state]

    # Initialize score_vectors: probability 1 at initial state, 0 elsewhere
    # Shape: [batch_size, num_states]
    score_vectors = th.zeros(
        batch_size, num_states,
        dtype=latent_vectors.dtype,
        device=latent_vectors.device
    )
    score_vectors[:, initial_state_idx] = 1.0

    # Epsilon for numerical stability
    eps = 1e-10

    # Pre-convert all transition data to tensors ONCE (not in the loop!)
    # These tensors are constant across all sequence positions and batch items
    state_token_tensors = []
    state_dest_tensors = []
    state_embeddings = []
    for state_idx in range(num_states):
        valid_token_list, dest_indices_list = token_automaton.state_transitions[state_idx]
        if len(valid_token_list) > 0:
            token_ids = th.tensor(valid_token_list, dtype=th.long, device=latent_vectors.device)
            state_token_tensors.append(token_ids)
            state_dest_tensors.append(
                th.tensor(dest_indices_list, dtype=th.long, device=latent_vectors.device)
            )
            # Pre-compute embeddings (don't depend on current_latent, so compute once!)
            state_embeddings.append(token_embeddings[token_ids])
        else:
            state_token_tensors.append(None)
            state_dest_tensors.append(None)
            state_embeddings.append(None)

    # Loop through each position in the sequence
    for pos in range(seq_len):
        # Get latent vectors at current position for all batch items
        current_latents = latent_vectors[:, pos, :]  # Shape: [batch_size, embed_dim]

        # Compute transition matrices for all batch items based on current_latents
        # transition_matrices[b][i][j] = probability of transitioning from state i to state j for batch item b
        transition_matrices = th.zeros(
            (batch_size, num_states, num_states),
            dtype=latent_vectors.dtype,
            device=latent_vectors.device
        )

        # For each state, compute transition probabilities for all batch items
        for state_idx in range(num_states):
            # Get pre-computed tensors (reuse across all positions and batch items!)
            dest_indices = state_dest_tensors[state_idx]
            valid_embeddings = state_embeddings[state_idx]

            if valid_embeddings is None:
                # No valid transitions from this state
                continue

            # BATCHED + VECTORIZED: Compute all distances at once
            # current_latents: [batch_size, embed_dim]
            # valid_embeddings: [num_valid, embed_dim]
            # Result: [batch_size, num_valid] distances
            distances = th.cdist(
                current_latents.unsqueeze(1),  # [batch_size, 1, embed_dim]
                valid_embeddings.unsqueeze(0),  # [1, num_valid, embed_dim]
                p=2
            ).squeeze(1)  # [batch_size, num_valid]

            # BATCHED + VECTORIZED: Compute all scores at once
            token_scores = compute_token_score(distances, eps, scoring_mode)  # [batch_size, num_valid]

            # Normalize to get probabilities
            # For negative scores (neg_squared), we need to use softmax normalization
            # For positive scores (inverse, gaussian), simple normalization works
            if scoring_mode == 'neg_squared':
                # Softmax normalization with temperature: exp(score/T) / sum(exp(score/T))
                # Lower temperature makes distribution more peaked (more probability on closest tokens)
                token_probs = th.softmax(token_scores / temperature, dim=1)  # [batch_size, num_valid]
            else:
                # Simple normalization for positive scores
                token_probs = token_scores / (token_scores.sum(dim=1, keepdim=True) + eps)  # [batch_size, num_valid]

            # BATCHED: Fill transition matrices using vectorized scatter_add
            # Expand dest_indices for batch dimension: [num_valid] -> [batch_size, num_valid]
            dest_indices_batch = dest_indices.unsqueeze(0).expand(batch_size, -1)

            # Scatter token_probs into transition matrices
            # transition_matrices[:, state_idx, :] has shape [batch_size, num_states]
            # token_probs has shape [batch_size, num_valid]
            # dest_indices_batch has shape [batch_size, num_valid]
            transition_matrices[:, state_idx, :].scatter_add_(1, dest_indices_batch, token_probs)

        # Update score_vectors: new distribution over states after seeing this token
        # score_vectors[b][j] = Σᵢ score_vectors[b][i] × transition_matrices[b][i][j]
        # Shape: [batch_size, num_states] @ [batch_size, num_states, num_states]
        #      = [batch_size, num_states]
        score_vectors = th.bmm(
            score_vectors.unsqueeze(1),  # [batch_size, 1, num_states]
            transition_matrices  # [batch_size, num_states, num_states]
        ).squeeze(1)  # [batch_size, num_states]

    # Compute final scores: sum of probabilities for all final states
    # Higher score means higher probability of being accepted by the automaton
    final_state_indices = [token_automaton.state_to_idx[s] for s in token_automaton.final_states]

    if len(final_state_indices) > 0:
        # Index into score_vectors and sum along state dimension - this preserves gradients properly
        # Shape: [batch_size, num_final_states] -> [batch_size]
        final_scores = score_vectors[:, final_state_indices].sum(dim=1)
    else:
        # No final states (edge case) - return zeros with proper gradient connection
        final_scores = score_vectors.sum(dim=1) * 0.0

    return final_scores


class _FullSegmentLogsumexpFn(th.autograd.Function):
    """Gather + logsumexp with fully scatter-free backward.

    Forward: gather log_probs by token IDs, then segment_reduce logsumexp.
    Backward:
      1. Logsumexp grad: softmax within each (src,dest) group (gathers only)
      2. Gather grad: permute to token-sorted order, segment_reduce sum,
         index-assign to vocab — no scatter_add collisions anywhere.
    """

    @staticmethod
    def forward(ctx, log_probs, concat_token_ids, lengths, group_ids,
                token_sort_perm, unique_tokens, token_group_lengths):
        batch_size = log_probs.shape[0]
        num_groups = lengths.shape[0]

        vals = log_probs[:, concat_token_ids]

        batch_lengths = lengths.repeat(batch_size)
        flat_vals = vals.reshape(-1)
        max_vals = th.segment_reduce(flat_vals, 'max', lengths=batch_lengths, unsafe=True)
        max_vals = max_vals.view(batch_size, num_groups)
        exp_shifted = th.exp(vals - max_vals[:, group_ids])
        flat_exp = exp_shifted.reshape(-1)
        sum_exp = th.segment_reduce(flat_exp, 'sum', lengths=batch_lengths, unsafe=True)
        sum_exp = sum_exp.view(batch_size, num_groups)
        result = max_vals + th.log(sum_exp + 1e-45)

        ctx.save_for_backward(vals, result, group_ids,
                              token_sort_perm, unique_tokens, token_group_lengths)
        ctx.vocab_size = log_probs.shape[1]

        return result

    @staticmethod
    def backward(ctx, grad_output):
        (vals, result, group_ids,
         token_sort_perm, unique_tokens, token_group_lengths) = ctx.saved_tensors
        vocab_size = ctx.vocab_size
        batch_size = vals.shape[0]

        # Logsumexp gradient: softmax within each group
        result_expanded = result[:, group_ids]
        grad_expanded = grad_output[:, group_ids]
        grad_vals = grad_expanded * th.exp(vals - result_expanded)

        # Gather gradient: sort by token, segment_reduce sum, index-assign
        sorted_grad = grad_vals[:, token_sort_perm]
        num_unique = unique_tokens.shape[0]
        batch_token_lengths = token_group_lengths.repeat(batch_size)
        grad_per_token = th.segment_reduce(
            sorted_grad.reshape(-1), 'sum',
            lengths=batch_token_lengths, unsafe=True
        ).view(batch_size, num_unique)

        grad_log_probs = th.zeros(batch_size, vocab_size,
                                  dtype=vals.dtype, device=vals.device)
        grad_log_probs[:, unique_tokens] = grad_per_token

        return grad_log_probs, None, None, None, None, None, None


def _get_segment_cache(token_automaton, device):
    """Get or compute cached segment data for fast logsumexp."""
    cache_key = f'_segment_cache_{device}'
    if hasattr(token_automaton, cache_key):
        return getattr(token_automaton, cache_key)

    num_states = len(token_automaton.state_list)

    # Group transitions by (src, dest) pair
    pair_to_tokens = {}
    for state_idx in range(num_states):
        valid_token_list, dest_indices_list = token_automaton.state_transitions[state_idx]
        for tok, dest in zip(valid_token_list, dest_indices_list):
            key = (state_idx, dest)
            if key not in pair_to_tokens:
                pair_to_tokens[key] = []
            pair_to_tokens[key].append(tok)

    if not pair_to_tokens:
        setattr(token_automaton, cache_key, None)
        return None

    all_ids, lengths, src_indices, dest_indices = [], [], [], []
    for (src, dest), token_list in pair_to_tokens.items():
        all_ids.extend(token_list)
        lengths.append(len(token_list))
        src_indices.append(src)
        dest_indices.append(dest)

    concat_token_ids = th.tensor(all_ids, dtype=th.long, device=device)
    lengths = th.tensor(lengths, dtype=th.long, device=device)
    src_indices = th.tensor(src_indices, dtype=th.long, device=device)
    dest_indices = th.tensor(dest_indices, dtype=th.long, device=device)

    # Group IDs: element-to-group mapping
    group_ids = th.repeat_interleave(
        th.arange(lengths.shape[0], device=device), lengths
    )

    # Token sort: for scatter-free gather backward
    token_sort_perm = th.argsort(concat_token_ids, stable=True)
    sorted_ids = concat_token_ids[token_sort_perm]
    unique_tokens, token_group_lengths = th.unique_consecutive(
        sorted_ids, return_counts=True
    )

    cache = (concat_token_ids, lengths, src_indices, dest_indices,
             group_ids, token_sort_perm, unique_tokens, token_group_lengths,
             num_states)
    setattr(token_automaton, cache_key, cache)
    return cache


def logits_score_batched(log_probs, token_automaton):
    """
    Compute automaton alignment score in log-space to avoid numerical underflow.

    Uses segment_reduce with a custom autograd backward for efficient
    gradient computation (no scatter collisions in forward or backward).

    Args:
        log_probs: Log-probabilities from model's log_softmax(logits)
                   Shape: [batch_size, seq_len, vocab_size]
        token_automaton: TokenAutomaton instance from automaton_alignment.py
                        defining the constraints

    Returns:
        log_scores: Differentiable LOG-score tensor (not exponentiated)
                   Shape: [batch_size]
                   To get actual scores: torch.exp(log_scores)
    """
    batch_size = log_probs.shape[0]
    seq_len = log_probs.shape[1]

    num_states = len(token_automaton.state_list)
    initial_state_idx = token_automaton.state_to_idx[token_automaton.initial_state]

    NEG_INF = -1e32
    log_score_vectors = th.full(
        (batch_size, num_states), NEG_INF,
        dtype=log_probs.dtype, device=log_probs.device
    )
    log_score_vectors[:, initial_state_idx] = 0.0

    cache = _get_segment_cache(token_automaton, log_probs.device)
    if cache is None:
        return th.full((batch_size,), NEG_INF, dtype=log_probs.dtype, device=log_probs.device)

    (concat_token_ids, lengths, src_indices, dest_indices,
     group_ids, token_sort_perm, unique_tokens, token_group_lengths,
     num_states) = cache

    # Batch positions' segment_logsumexp, chunking to limit peak memory.
    # During forward, ~3 copies of [flat_rows, num_transitions] exist simultaneously.
    num_transitions = concat_token_ids.shape[0]
    MAX_ELEMENTS = 500_000_000  # keep each copy under ~2GB (float32)
    chunk_positions = max(1, MAX_ELEMENTS // (batch_size * num_transitions))

    if chunk_positions >= seq_len:
        flat_log_probs = log_probs.reshape(batch_size * seq_len, -1)
        all_group_logsumexp = _FullSegmentLogsumexpFn.apply(
            flat_log_probs, concat_token_ids, lengths, group_ids,
            token_sort_perm, unique_tokens, token_group_lengths
        ).reshape(batch_size, seq_len, -1)
    else:
        chunks = []
        for start in range(0, seq_len, chunk_positions):
            end = min(start + chunk_positions, seq_len)
            chunk = log_probs[:, start:end, :].reshape(batch_size * (end - start), -1)
            chunk_result = _FullSegmentLogsumexpFn.apply(
                chunk, concat_token_ids, lengths, group_ids,
                token_sort_perm, unique_tokens, token_group_lengths
            ).reshape(batch_size, end - start, -1)
            chunks.append(chunk_result)
        all_group_logsumexp = th.cat(chunks, dim=1)

    # Build all log_T matrices at once
    all_log_T = th.full((batch_size, seq_len, num_states, num_states), NEG_INF,
                        dtype=log_probs.dtype, device=log_probs.device)
    all_log_T[:, :, src_indices, dest_indices] = all_group_logsumexp

    if num_states > SEQUENTIAL_REDUCTION_THRESHOLD:
        # Sequential vector-matrix multiply: O(S²) memory per step.
        # Propagate log_score_vectors [B, S] through each position's
        # transition matrix [B, S, S] one at a time.
        for pos in range(seq_len):
            log_T = all_log_T[:, pos, :, :]  # [B, S, S] view
            # log_score_vectors[:, i] + log_T[:, i, j] -> logsumexp over i -> [B, j]
            log_contributions = log_score_vectors.unsqueeze(2) + log_T  # [B, S, S]
            log_score_vectors = th.logsumexp(log_contributions, dim=1)  # [B, S]
    else:
        # Parallel reduction: compute product of all transition matrices in
        # O(log(seq_len)) steps using log-semiring matrix multiplication.
        # log-semiring matmul: result[i,j] = logsumexp_k(A[i,k] + B[k,j])
        matrices = all_log_T  # [B, T, S, S]
        while matrices.shape[1] > 1:
            n = matrices.shape[1]
            if n % 2 == 1:
                remainder = matrices[:, -1:]
                matrices = matrices[:, :-1]
            else:
                remainder = None
            left = matrices[:, 0::2]   # [B, N/2, S, S]
            right = matrices[:, 1::2]  # [B, N/2, S, S]
            # [B, N/2, S, S, 1] + [B, N/2, 1, S, S] -> logsumexp over k -> [B, N/2, S, S]
            matrices = th.logsumexp(left.unsqueeze(-1) + right.unsqueeze(-3), dim=-2)
            if remainder is not None:
                matrices = th.cat([matrices, remainder], dim=1)

        # product_matrix[i,j] = log P(state i -> state j over all positions)
        product_matrix = matrices[:, 0]  # [B, S, S]

        # Since initial vector is one-hot at initial_state (0 there, -inf elsewhere):
        # v_final[j] = product_matrix[initial_state, j]
        log_score_vectors = product_matrix[:, initial_state_idx, :]  # [B, S]

    final_state_indices = [token_automaton.state_to_idx[s] for s in token_automaton.final_states]

    if len(final_state_indices) > 0:
        final_log_scores = th.logsumexp(
            log_score_vectors[:, final_state_indices], dim=1
        )
    else:
        final_log_scores = th.full(
            (batch_size,), NEG_INF,
            dtype=log_probs.dtype, device=log_probs.device
        )

    return final_log_scores
