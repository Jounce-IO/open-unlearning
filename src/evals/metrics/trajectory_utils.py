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
from typing import List, Union


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


def trajectories_from_logits(
    logits_history: List[torch.Tensor],
    fixation_steps: torch.Tensor,
    prompt_lens: Union[List[int], torch.Tensor],
) -> dict:
    """
    Compute the four trajectory tensors from raw logits and fixation data (model-free).

    Tensor-in, tensor-out: no model or sampler dependency. Callers can pass data from
    a saved file (logits_history, fixation_steps, prompt_lens) to get deterministic
    trajectory tensors. Useful for testing and offline analysis.

    Args:
        logits_history: List of S tensors, each [B, L_full, V] (one per diffusion step).
        fixation_steps: [B, T_full] long tensor; step index where each position was fixed.
        prompt_lens: Length-B list (or 1-d tensor) of prompt lengths per sample.
            Used to slice the generated region (after max(prompt_lens)).

    Returns:
        Dict with keys:
            - "steps": [B, V, L, S] tensor
            - "fixation_start": [B, V, L, S] tensor
            - "fixation_end": [B, V, L, S] tensor
            - "fixation_ratio": [B, V, L, S] tensor
            - "S": int (number of diffusion steps)
            - "L": int (generated length)
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

    max_prompt_len = max(prompt_lens)
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
