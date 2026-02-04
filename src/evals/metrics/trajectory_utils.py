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

    # Steps trajectory: R itself (memory optimization; callers must not modify T_steps).
    T_steps = R  # [B, V, L, S]

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


# Trajectory type names for compute_one_trajectory (must match keys returned by compute_trajectories).
TRAJECTORY_NAMES = ("steps", "fixation_start", "fixation_end", "fixation_ratio")


def get_logits_at_trajectory_step(
    R: torch.Tensor,
    F: torch.Tensor,
    S: int,
    traj_name: str,
    b: int,
    step: int,
) -> torch.Tensor:
    """
    Return logits for sample b at trajectory step `step` for the given trajectory type.
    Does not build the full [B, V, L, S] trajectory tensor.
    """
    B, V, L, S_actual = R.shape
    assert S == S_actual and F.shape == (B, L) and 0 <= b < B and 0 <= step < S and S > 1
    if traj_name not in TRAJECTORY_NAMES:
        raise ValueError(f"traj_name must be one of {TRAJECTORY_NAMES}, got {traj_name!r}")
    device = R.device
    R_b = R[b]
    F_b = F[b]
    if traj_name == "steps":
        return R_b[:, :, step].clone()
    if traj_name == "fixation_start":
        source_step = torch.minimum(torch.tensor(step, device=device, dtype=torch.long), F_b)
    elif traj_name == "fixation_end":
        source_step = F_b - (S - 1) + step
    elif traj_name == "fixation_ratio":
        source_step = (F_b * step) // (S - 1)
    else:
        raise ValueError(f"traj_name must be one of {TRAJECTORY_NAMES}, got {traj_name!r}")
    source_step = torch.clamp(source_step, 0, S - 1)
    index = source_step.unsqueeze(0).unsqueeze(-1).expand(V, L, 1)
    return torch.gather(R_b, dim=2, index=index).squeeze(-1)


def compute_one_trajectory(
    R: torch.Tensor, F: torch.Tensor, S: int, traj_name: str
) -> torch.Tensor:
    """
    Compute a single trajectory tensor from R and F (memory-efficient: one [B,V,L,S] at a time).

    Args:
        R: [B, V, L, S] logits tensor
        F: [B, L] fixation steps
        S: number of diffusion steps
        traj_name: one of "steps", "fixation_start", "fixation_end", "fixation_ratio"

    Returns:
        Single tensor [B, V, L, S] for the requested trajectory type.
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

    if traj_name == "steps":
        out = R  # T_steps is R (callers must not modify)
    else:
        F_exp = F.unsqueeze(2)  # [B, L, 1]
        s_vec = torch.arange(S, device=device, dtype=torch.long)
        s_exp = s_vec.view(1, 1, S)  # [1, 1, S]

        if traj_name == "fixation_start":
            source_step = torch.minimum(s_exp, F_exp)
            source_step = torch.clamp(source_step, 0, S - 1)
            index = source_step.unsqueeze(1).expand(B, V, L, S)
            out = torch.gather(R, dim=3, index=index)
        elif traj_name == "fixation_end":
            source_step = F_exp - (S - 1) + s_exp
            source_step = torch.clamp(source_step, 0, S - 1)
            index = source_step.unsqueeze(1).expand(B, V, L, S)
            out = torch.gather(R, dim=3, index=index)
        elif traj_name == "fixation_ratio":
            ratio_step = (F_exp * s_exp) // (S - 1)
            ratio_step = torch.clamp(ratio_step, 0, S - 1)
            index = ratio_step.unsqueeze(1).expand(B, V, L, S)
            out = torch.gather(R, dim=3, index=index)
        else:
            raise ValueError(
                f"traj_name must be one of {TRAJECTORY_NAMES}, got {traj_name!r}"
            )

    if squeeze_out:
        out = out.squeeze(0)
    return out


def prepare_R_from_logits(
    logits_history: List[torch.Tensor],
    fixation_steps: torch.Tensor,
    prompt_lens: Union[List[int], torch.Tensor],
) -> dict:
    """
    Stack logits and slice to generated region; return R, F, S, L for one-at-a-time trajectory computation.

    Use with compute_one_trajectory(R, F, S, traj_name) to hold at most one trajectory tensor at a time.

    Returns:
        Dict with "R", "F", "S", "L" (R [B,V,L,S], F [B,L], S int, L int).
    """
    if not logits_history:
        raise ValueError("logits_history cannot be empty")
    if fixation_steps.dim() != 2:
        raise ValueError(
            f"fixation_steps must be 2-d [B, T_full], got shape {fixation_steps.shape}"
        )

    B, L_full, V = logits_history[0].shape
    S = len(logits_history)
    T_full = fixation_steps.shape[1]
    if isinstance(prompt_lens, torch.Tensor):
        prompt_lens = prompt_lens.tolist()
    if len(prompt_lens) != B:
        raise ValueError(
            f"prompt_lens length {len(prompt_lens)} must match batch size B={B}"
        )
    if L_full < T_full:
        generated_len = L_full
        max_prompt_len = T_full - L_full
    else:
        max_prompt_len = max(prompt_lens)
        generated_len = L_full - max_prompt_len
    L = generated_len
    device = logits_history[0].device
    dtype = logits_history[0].dtype
    R = torch.empty((B, V, L, S), device=device, dtype=dtype)
    for s in range(S):
        t = logits_history[s]
        slice_t = t[:, max_prompt_len : max_prompt_len + L, :] if L_full > L else t
        R[:, :, :, s] = slice_t.permute(0, 2, 1)

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

    return {"R": R, "F": F, "S": S, "L": L}


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

    Returns:
        Dict with keys:
            - "steps": [B, V, L, S] tensor
            - "fixation_start": [B, V, L, S] tensor
            - "fixation_end": [B, V, L, S] tensor
            - "fixation_ratio": [B, V, L, S] tensor
            - "S": int (number of diffusion steps)
            - "L": int (generated length)
    """
    prepared = prepare_R_from_logits(logits_history, fixation_steps, prompt_lens)
    R = prepared["R"]
    F = prepared["F"]
    S = prepared["S"]
    L = prepared["L"]
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
