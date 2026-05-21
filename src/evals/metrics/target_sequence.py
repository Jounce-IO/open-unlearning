"""Target-sequence helpers for trajectory probability and related metrics."""

from __future__ import annotations

from typing import Any, FrozenSet, Mapping, Optional

import torch

from data.utils import IGNORE_INDEX


def parse_exclude_suffix_template_tokens(trajectory_config: Optional[Mapping[str, Any]]) -> bool:
    """Return whether to drop chat-template suffix tokens from prob target labels."""
    if not trajectory_config:
        return False
    raw = trajectory_config.get("target_sequence")
    if raw is None:
        return False
    if isinstance(raw, Mapping):
        return bool(raw.get("exclude_suffix_template_tokens", False))
    return False


def suffix_template_token_ids(tokenizer: Any) -> FrozenSet[int]:
    """Token ids treated as start of post-answer chat template suffix."""
    ids: set[int] = set()
    candidates: list[str] = [
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|endoftext|>",
    ]
    eot = getattr(tokenizer, "eot_token", None)
    if isinstance(eot, str) and eot:
        candidates.append(eot)
    eos = getattr(tokenizer, "eos_token", None)
    if isinstance(eos, str) and eos:
        candidates.append(eos)
    for text in candidates:
        if not text:
            continue
        enc = tokenizer.encode(text, add_special_tokens=False)
        ids.update(int(t) for t in enc)
    return frozenset(ids)


def first_suffix_label_index(
    labels: torch.Tensor,
    suffix_ids: FrozenSet[int],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> Optional[int]:
    """Index of the first non-ignore label id in ``suffix_ids``, or None."""
    flat = labels.reshape(-1)
    for i in range(flat.numel()):
        y = int(flat[i].item())
        if y == ignore_index or y < 0:
            continue
        if y in suffix_ids:
            return i
    return None


def mask_suffix_template_labels(
    labels: torch.Tensor,
    tokenizer: Any,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Mask labels from the first suffix-template token through the end of the slice."""
    suffix_ids = suffix_template_token_ids(tokenizer)
    idx = first_suffix_label_index(labels, suffix_ids, ignore_index=ignore_index)
    if idx is None:
        return labels
    out = labels.clone()
    out.reshape(-1)[idx:] = ignore_index
    return out


def maybe_mask_suffix_template_labels(
    labels: torch.Tensor,
    tokenizer: Any,
    *,
    exclude_suffix_template_tokens: bool,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    if not exclude_suffix_template_tokens or tokenizer is None:
        return labels
    return mask_suffix_template_labels(
        labels, tokenizer, ignore_index=ignore_index
    )
