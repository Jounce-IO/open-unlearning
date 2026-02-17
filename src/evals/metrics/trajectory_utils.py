"""
Utilities for computing trajectory tensors from logits and fixation steps.

This module provides functions to:
- Stack logits history into a tensor
- Compute four trajectory types (steps, fixation_start, fixation_end, fixation_ratio)
- trajectories_from_logits: model-free entry-point (logits + fixation → trajectory tensors)
- Extract logits at specific steps
- Decode logits to text
"""

import torch
from typing import List, Optional, Union


def stack_logits_history(logits_history: List[torch.Tensor]) -> torch.Tensor:
    """
    Stack logits history list into a tensor.

    Supports any batch size B; no samples are discarded.

    Args:
        logits_history: List of [B, L, V] tensors (one per step)

    Returns:
        R: [B, V, L, S] tensor (stacked logits, one per sample)
    """
    if not logits_history:
        raise ValueError("logits_history cannot be empty")

    # Stack along new dimension: [S, B, L, V] -> permute to [B, V, L, S]
    stacked = torch.stack(logits_history, dim=0)
    S, B, L, V = stacked.shape
    R = stacked.permute(1, 3, 2, 0)  # [B, V, L, S]
    return R


def compute_trajectories(
    R: torch.Tensor, F: torch.Tensor, S: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute four trajectory tensors from logits R and fixation steps F.

    Supports batched input: R [B, V, L, S], F [B, L]; returns four [B, V, L, S] tensors.
    Also accepts single-sample R [V, L, S], F [L] (B=1); returns four [V, L, S] tensors.

    All fixation trajectories satisfy: first (s=0) = R step 0, last (s=S-1) = R step F[l].

    Args:
        R: [B, V, L, S] or [V, L, S] logits tensor
        F: [B, L] or [L] fixation steps (step index where each position was fixed)
        S: Number of diffusion steps (must be > 1)

    Returns:
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio: each [B, V, L, S] or [V, L, S]
    """
    if R.dim() == 3:
        R = R.unsqueeze(0)
        F = F.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False

    B, V, L, S_actual = R.shape
    assert S == S_actual, f"S mismatch: {S} != {S_actual}"
    assert F.shape == (B, L), f"F shape mismatch: {F.shape} != ({B}, {L})"
    assert S > 1, f"S must be > 1, got {S}"

    device = R.device

    # Steps trajectory: Direct copy of R
    T_steps = R.clone()  # [B, V, L, S]

    s_vec = torch.arange(S, device=device, dtype=torch.long)
    F_exp = F.unsqueeze(2)  # [B, L, 1]
    s_exp = s_vec.view(1, 1, S)  # [1, 1, S]

    # Fixation start: T[b,v,l,s] = R[b,v,l, min(s, F[b,l])]
    source_step = torch.minimum(s_exp, F_exp)
    source_step = torch.clamp(source_step, 0, S - 1)
    index = source_step.unsqueeze(1).expand(B, V, L, S)
    T_fixation_start = torch.gather(R, dim=3, index=index)

    # Fixation end: T[b,v,l,s] = R[b,v,l, max(0, F[b,l]-(S-1)+s)]
    source_step = F_exp - (S - 1) + s_exp
    source_step = torch.clamp(source_step, 0, S - 1)
    index = source_step.unsqueeze(1).expand(B, V, L, S)
    T_fixation_end = torch.gather(R, dim=3, index=index)

    # Fixation ratio: T[b,v,l,s] = R[b,v,l, floor(F[b,l]*s/(S-1))]
    ratio_step = (F_exp * s_exp) // (S - 1)
    ratio_step = torch.clamp(ratio_step, 0, S - 1)
    index = ratio_step.unsqueeze(1).expand(B, V, L, S)
    T_fixation_ratio = torch.gather(R, dim=3, index=index)

    if squeeze_out:
        T_steps = T_steps.squeeze(0)
        T_fixation_start = T_fixation_start.squeeze(0)
        T_fixation_end = T_fixation_end.squeeze(0)
        T_fixation_ratio = T_fixation_ratio.squeeze(0)

    return T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio


def compute_fixation_start_trajectory(
    raw: torch.Tensor, step_index: int, fixation_indices: torch.Tensor
) -> torch.Tensor:
    """
    Compute fixation_start trajectory logits at a single step (on-demand).

    At trajectory step s, position l uses raw at diffusion step min(s, F[l]).

    Args:
        raw: [V, L, S] logits for one sample (one slice of R).
        step_index: Trajectory step s in 0 .. S-1.
        fixation_indices: [L] fixation step per position (same as F[b]).

    Returns:
        trajectory_step: [V, L] logits at the specified step.
    """
    V, L, S = raw.shape
    assert 0 <= step_index < S, f"step_index {step_index} out of range [0, {S-1}]"
    assert fixation_indices.shape == (L,), f"fixation_indices shape {fixation_indices.shape} != (L={L},)"
    device = raw.device
    s_t = torch.tensor(step_index, device=device, dtype=torch.long)
    source_step = torch.minimum(s_t, fixation_indices)
    source_step = torch.clamp(source_step, 0, S - 1)
    index = source_step.view(1, L, 1).expand(V, L, 1)
    return torch.gather(raw, dim=2, index=index).squeeze(2)


def compute_fixation_end_trajectory(
    raw: torch.Tensor, step_index: int, fixation_indices: torch.Tensor
) -> torch.Tensor:
    """
    Compute fixation_end trajectory logits at a single step (on-demand).

    At trajectory step s, position l uses raw at diffusion step max(0, F[l]-(S-1)+s).

    Args:
        raw: [V, L, S] logits for one sample (one slice of R).
        step_index: Trajectory step s in 0 .. S-1.
        fixation_indices: [L] fixation step per position (same as F[b]).

    Returns:
        trajectory_step: [V, L] logits at the specified step.
    """
    V, L, S = raw.shape
    assert 0 <= step_index < S, f"step_index {step_index} out of range [0, {S-1}]"
    assert fixation_indices.shape == (L,), f"fixation_indices shape {fixation_indices.shape} != (L={L},)"
    source_step = fixation_indices - (S - 1) + step_index
    source_step = torch.clamp(source_step, 0, S - 1)
    index = source_step.view(1, L, 1).expand(V, L, 1)
    return torch.gather(raw, dim=2, index=index).squeeze(2)


def compute_fixation_ratio_trajectory(
    raw: torch.Tensor, step_index: int, fixation_indices: torch.Tensor
) -> torch.Tensor:
    """
    Compute fixation_ratio trajectory logits at a single step (on-demand).

    At trajectory step s, position l uses raw at diffusion step floor(F[l]*s/(S-1)).

    Args:
        raw: [V, L, S] logits for one sample (one slice of R).
        step_index: Trajectory step s in 0 .. S-1.
        fixation_indices: [L] fixation step per position (same as F[b]).

    Returns:
        trajectory_step: [V, L] logits at the specified step.
    """
    V, L, S = raw.shape
    assert 0 <= step_index < S, f"step_index {step_index} out of range [0, {S-1}]"
    assert fixation_indices.shape == (L,), f"fixation_indices shape {fixation_indices.shape} != (L={L},)"
    if S <= 1:
        return raw[:, :, 0].clone()
    ratio_step = (fixation_indices * step_index) // (S - 1)
    ratio_step = torch.clamp(ratio_step, 0, S - 1)
    index = ratio_step.view(1, L, 1).expand(V, L, 1)
    return torch.gather(raw, dim=2, index=index).squeeze(2)


def trajectories_from_logits(
    logits_history: List[torch.Tensor],
    fixation_steps: torch.Tensor,
    prompt_lens: Union[List[int], torch.Tensor],
    return_trajectory_tensors: bool = True,
) -> dict:
    """
    Compute the four trajectory tensors from raw logits and fixation data (model-free).

    Tensor-in, tensor-out: no model or sampler dependency. Callers can pass data from
    a saved file (logits_history, fixation_steps, prompt_lens) to get deterministic
    trajectory tensors. Useful for testing and offline analysis.

    Supports two contracts (detected by shape):

    - **Full-sequence:** logits_history entries are [B, L_full, V] with L_full = prompt
      + generated; fixation_steps is [B, L_full]. We slice to the generated region using
      prompt_lens (R = stacked logits without prompt positions; F = fixation slice).
    - **Generated-only:** logits_history entries are [B, L_gen, V] (generated region
      only); fixation_steps is still [B, T_full] with T_full > L_gen. We use R = stack
      as-is (no slice) and slice fixation_steps to the generated region for F.
      Detection: when logits_history[0].shape[1] < fixation_steps.shape[1].

    Args:
        logits_history: List of S tensors. Full-sequence: each [B, L_full, V].
            Generated-only: each [B, L_gen, V].
        fixation_steps: [B, T_full] long tensor; step index where each position was fixed.
        prompt_lens: Length-B list (or 1-d tensor) of prompt lengths per sample.
        return_trajectory_tensors: If True (default), return trajectory tensors; if False,
            return only R, F, S, L.

    Returns:
        If return_trajectory_tensors=True: dict with "steps", "fixation_start",
        "fixation_end", "fixation_ratio", "S", "L".
        If return_trajectory_tensors=False: dict with "R", "F", "S", "L" only.
    """
    if not logits_history:
        raise ValueError("logits_history cannot be empty")
    if fixation_steps.dim() != 2:
        raise ValueError(
            f"fixation_steps must be 2-d [B, T_full], got shape {fixation_steps.shape}"
        )

    R_full = stack_logits_history(logits_history)  # [B, V, T_full, S]
    B, V, T_full, S = R_full.shape
    if isinstance(prompt_lens, torch.Tensor):
        prompt_lens = prompt_lens.tolist()
    if len(prompt_lens) != B:
        raise ValueError(
            f"prompt_lens length {len(prompt_lens)} must match batch size B={B}"
        )

    L_logits = logits_history[0].shape[1]
    T_fixation = fixation_steps.shape[1]
    max_prompt_len = max(prompt_lens)

    if L_logits < T_fixation:
        # Generated-only: logits are [B, L_gen, V]; fixation_steps is [B, T_full].
        R = R_full
        L = L_logits
        F_list = []
        for b in range(B):
            F_full = fixation_steps[b]
            slice_F = F_full[max_prompt_len : max_prompt_len + L]
            if slice_F.shape[0] >= L:
                F_b = slice_F[:L]
            else:
                F_b = torch.cat(
                    [
                        slice_F,
                        torch.full(
                            (L - slice_F.shape[0],),
                            S - 1,
                            dtype=torch.long,
                            device=F_full.device,
                        ),
                    ]
                )
            F_list.append(F_b)
        F = torch.stack(F_list, dim=0)  # [B, L]
    else:
        # Full-sequence: slice R and F to generated region.
        generated_len = T_full - max_prompt_len
        R = R_full[:, :, max_prompt_len : max_prompt_len + generated_len, :]  # [B, V, L, S]
        _, _, L, _ = R.shape

        F_list = []
        for b in range(B):
            F_full = fixation_steps[b]
            if F_full.shape[0] > max_prompt_len:
                slice_F = F_full[max_prompt_len : max_prompt_len + L]
                if slice_F.shape[0] >= L:
                    F_b = slice_F[:L]
                else:
                    F_b = torch.cat(
                        [
                            slice_F,
                            torch.full(
                                (L - slice_F.shape[0],),
                                S - 1,
                                dtype=torch.long,
                                device=F_full.device,
                            ),
                        ]
                    )
            else:
                F_b = torch.full(
                    (L,), S - 1, dtype=torch.long, device=F_full.device
                )
            F_list.append(F_b)
        F = torch.stack(F_list, dim=0)  # [B, L]

    if not return_trajectory_tensors:
        return {"R": R, "F": F, "S": S, "L": L}

    T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(
        R, F, S
    )
    return {
        "steps": T_steps,
        "fixation_start": T_fixation_start,
        "fixation_end": T_fixation_end,
        "fixation_ratio": T_fixation_ratio,
        "S": S,
        "L": L,
    }


def effective_lengths_from_eos(
    sequences: torch.Tensor,
    prompt_lens: Union[List[int], torch.Tensor],
    L: int,
    eos_token_id: Optional[int],
) -> List[int]:
    """
    Compute per-sample effective length (positions 0..L_eff-1 include first EOS).

    For each sample b, the generated region is sequences[b, prompt_lens[b]:prompt_lens[b]+L].
    L_eff[b] = (first index of eos_token_id in that region) + 1, or L if no EOS.
    If eos_token_id is None, returns [L] * B (no EOS trimming).

    Args:
        sequences: [B, T_full] token ids (full sequence including prompt + generated).
        prompt_lens: Length-B list or 1-d tensor of prompt lengths per sample.
        L: Generated length (number of positions in the generated region).
        eos_token_id: Token ID for EOS, or None to use full length for all samples.

    Returns:
        List of B integers: effective length per sample (1-indexed count including EOS position).
    """
    if not isinstance(sequences, torch.Tensor):
        sequences = torch.tensor(sequences, dtype=torch.long)
    B = sequences.shape[0]
    if isinstance(prompt_lens, torch.Tensor):
        prompt_lens = prompt_lens.tolist()
    if eos_token_id is None:
        return [L] * B
    out: List[int] = []
    for b in range(B):
        pl = prompt_lens[b]
        gen_slice = sequences[b, pl : pl + L]
        eq = gen_slice == eos_token_id
        if not isinstance(eq, torch.Tensor):
            eq = torch.tensor([eq], dtype=torch.bool, device=gen_slice.device if isinstance(gen_slice, torch.Tensor) else None)
        match = eq.nonzero(as_tuple=True)[0]
        if match.numel() == 0:
            out.append(L)
        else:
            first_eos_idx = int(match[0].item())
            out.append(first_eos_idx + 1)
    return out


def extract_logits_at_step(trajectory: torch.Tensor, step: int) -> torch.Tensor:
    """
    Extract logits at a specific step from trajectory.
    
    Args:
        trajectory: [V, L, S] trajectory tensor
        step: Step index (0 to S-1)
    
    Returns:
        logits: [V, L] logits at the specified step
    """
    V, L, S = trajectory.shape
    assert 0 <= step < S, f"Step {step} out of range [0, {S-1}]"
    return trajectory[:, :, step]  # [V, L]


def decode_logits_to_text(
    logits: torch.Tensor,
    tokenizer,
    input_ids: torch.Tensor,
    prompt_len: int,
) -> List[str]:
    """
    Convert logits to tokens via argmax, then decode to text.
    
    Args:
        logits: [V, L] logits tensor (or [1, L, V] for batch format)
        tokenizer: Tokenizer to use for decoding
        input_ids: [B, T] original input token IDs
        prompt_len: Length of prompt (tokens before generation region)
    
    Returns:
        texts: List of decoded text strings (one per sample)
    """
    # Handle different input shapes
    if logits.dim() == 2:
        # [V, L] -> need to transpose to [L, V] for argmax
        logits = logits.transpose(0, 1)  # [L, V]
        # Add batch dimension
        logits = logits.unsqueeze(0)  # [1, L, V]
    elif logits.dim() == 3:
        # [B, L, V] - already in correct format
        pass
    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")
    
    B, L, V = logits.shape
    
    # Get predicted tokens via argmax
    predicted_tokens = torch.argmax(logits, dim=-1)  # [B, L]
    
    # Decode tokens to text
    texts = []
    for b in range(B):
        # Only decode the generation portion (after prompt)
        gen_tokens = predicted_tokens[b, :]  # [L]
        # Decode
        text = tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True)
        texts.append(text)
    
    return texts
