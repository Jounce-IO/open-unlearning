"""
Utilities for computing trajectory tensors from logits and fixation steps.

This module provides functions to:
- Stack logits history into a tensor
- Compute four trajectory types (steps, fixation_start, fixation_end, fixation_ratio)
- Extract logits at specific steps
- Decode logits to text
"""

import torch
from typing import List


def stack_logits_history(logits_history: List[torch.Tensor]) -> torch.Tensor:
    """
    Stack logits history list into a tensor.
    
    Args:
        logits_history: List of [B, L, V] tensors (one per step)
    
    Returns:
        R: [V, L, S] tensor (stacked logits)
        For single sample (B=1), transposes to [V, L, S]
        For multiple samples, takes first sample or averages
    """
    if not logits_history:
        raise ValueError("logits_history cannot be empty")
    
    # Stack along new dimension: [S, B, L, V]
    stacked = torch.stack(logits_history, dim=0)
    S, B, L, V = stacked.shape
    
    # For now, handle single sample case (B=1)
    # TODO: Support batched case (average or select first)
    if B == 1:
        # Transpose to [V, L, S]
        R = stacked[0].transpose(0, 1).transpose(1, 2)  # [L, V] -> [V, L] then add S
        # Actually, we want [V, L, S], so:
        R = stacked[0].permute(2, 1, 0)  # [B, L, V] -> [V, L] then stack gives [V, L, S]
        # Wait, stacked is [S, B, L, V], so:
        R = stacked.squeeze(1).permute(2, 1, 0)  # [S, L, V] -> [V, L, S]
    else:
        # For batched, take first sample for now
        # TODO: Support averaging or per-sample trajectories
        R = stacked[:, 0, :, :].permute(2, 1, 0)  # [S, L, V] -> [V, L, S]
    
    return R


def compute_trajectories(
    R: torch.Tensor, F: torch.Tensor, S: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute four trajectory tensors from logits R and fixation steps F.
    
    All fixation trajectories satisfy: first (s=0) = R step 0, last (s=S-1) = R step F[l].
    
    Args:
        R: [V, L, S] logits tensor
        F: [L] fixation steps tensor (step index where each position was fixed)
        S: Number of diffusion steps (must be > 1)
    
    Returns:
        T_steps: [V, L, S] steps trajectory (direct copy of R)
        T_fixation_start: [V, L, S] fixation start trajectory (from step 0 to fixation)
        T_fixation_end: [V, L, S] fixation end trajectory (from step 0 to fixation)
        T_fixation_ratio: [V, L, S] fixation ratio trajectory (linear interpolation from 0 to F[l])
    """
    V, L, S_actual = R.shape
    assert S == S_actual, f"S mismatch: {S} != {S_actual}"
    assert F.shape[0] == L, f"F length mismatch: {F.shape[0]} != {L}"
    assert S > 1, f"S must be > 1, got {S}"
    
    device = R.device
    dtype = R.dtype
    
    # Steps trajectory: Direct copy of R
    T_steps = R.clone()  # [V, L, S]
    
    # Fixation start trajectory: From step 0 to fixation
    # T_fixation_start[v, l, s] = R[v, l, min(s, F[l])]
    T_fixation_start = torch.zeros_like(R)
    for l in range(L):
        fix_step = int(F[l].item())
        for s in range(S):
            source_step = min(s, fix_step)
            source_step = max(0, min(source_step, S - 1))  # Clamp to [0, S-1]
            T_fixation_start[:, l, s] = R[:, l, source_step]
    
    # Fixation end trajectory: From step 0 to fixation
    # T_fixation_end[v, l, s] = R[v, l, max(0, F[l]-(S-1)+s)]
    T_fixation_end = torch.zeros_like(R)
    for l in range(L):
        fix_step = int(F[l].item())
        for s in range(S):
            source_step = max(0, fix_step - (S - 1) + s)
            source_step = max(0, min(source_step, S - 1))  # Clamp to [0, S-1]
            T_fixation_end[:, l, s] = R[:, l, source_step]
    
    # Fixation ratio trajectory: Linear interpolation from step 0 to fixation
    # T_fixation_ratio[v, l, s] = R[v, l, floor(F[l]*s/(S-1))]
    # Assumes S > 1 (asserted above)
    T_fixation_ratio = torch.zeros_like(R)
    for l in range(L):
        fix_step = int(F[l].item())
        for s in range(S):
            ratio_step = int(fix_step * s / (S - 1))
            ratio_step = max(0, min(ratio_step, S - 1))  # Clamp to [0, S-1]
            T_fixation_ratio[:, l, s] = R[:, l, ratio_step]
    
    return T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio


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
