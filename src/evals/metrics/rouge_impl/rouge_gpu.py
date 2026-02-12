"""
GPU-accelerated ROUGE helpers (PyTorch).

Token lists are encoded to integer IDs (vocab from batch); ROUGE-1 and LCS
are computed on GPU with batched or per-pair ops.
"""

from typing import List, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _fmeasure(precision: float, recall: float) -> float:
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def build_vocab_from_token_lists(ref_token_lists: List[list], pred_token_lists: List[list]) -> Tuple[dict, int]:
    """
    Build token -> id mapping from all tokens in ref and pred lists.
    Sorted for determinism. Returns (token_to_id, vocab_size).
    """
    all_tokens = set()
    for tlist in ref_token_lists:
        all_tokens.update(tlist)
    for tlist in pred_token_lists:
        all_tokens.update(tlist)
    sorted_tokens = sorted(all_tokens)
    token_to_id = {t: i for i, t in enumerate(sorted_tokens)}
    return token_to_id, len(sorted_tokens)


def token_lists_to_padded_ids(
    token_lists: List[list],
    token_to_id: dict,
    pad_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode list of token lists to padded tensor and lengths.

    Args:
        token_lists: List of lists of token strings.
        token_to_id: Mapping token -> int id (0..V-1).
        pad_id: Integer used for padding (e.g. V or -1; must be consistent with bincount usage).
        device: Target device.

    Returns:
        ids: (B, max_len) long tensor, padded with pad_id.
        lengths: (B,) long tensor of actual lengths.
    """
    B = len(token_lists)
    lengths = torch.tensor([len(tlist) for tlist in token_lists], dtype=torch.long, device=device)
    max_len = lengths.max().item()
    if max_len == 0:
        ids = torch.full((B, 1), pad_id, dtype=torch.long, device=device)
        return ids, lengths
    ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    for b, tlist in enumerate(token_lists):
        if tlist:
            encoded = [token_to_id[t] for t in tlist]
            ids[b, : len(encoded)] = torch.tensor(encoded, dtype=torch.long, device=device)
    return ids, lengths


def rouge1_from_token_ids_gpu(
    ref_ids: torch.Tensor,
    pred_ids: torch.Tensor,
    ref_len: int,
    pred_len: int,
    vocab_size: int,
    pad_id: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    ROUGE-1 recall, precision, F1 for one pair from token ID tensors (1D).
    ref_ids and pred_ids are 1D; only first ref_len and pred_len elements are used (rest may be padding).
    """
    if ref_len <= 0 or pred_len <= 0:
        return (0.0, 0.0, 0.0)
    ref_valid = ref_ids[:ref_len]
    pred_valid = pred_ids[:pred_len]
    ref_counts = torch.bincount(ref_valid, minlength=vocab_size)
    pred_counts = torch.bincount(pred_valid, minlength=vocab_size)
    intersection = (ref_counts.min(pred_counts)).sum().item()
    ref_total = ref_len
    pred_total = pred_len
    recall = intersection / ref_total
    precision = intersection / pred_total
    f1 = _fmeasure(precision, recall)
    return (recall, precision, f1)


def rouge1_batch_gpu(
    ref_ids: torch.Tensor,
    pred_ids: torch.Tensor,
    len_ref: torch.Tensor,
    len_pred: torch.Tensor,
    vocab_size: int,
    device: torch.device,
) -> List[Tuple[float, float, float]]:
    """
    ROUGE-1 (recall, precision, f1) for each pair in the batch.
    ref_ids (B, max_R), pred_ids (B, max_C); pad_id must be >= vocab_size so bincount ignores it.
    """
    B = ref_ids.size(0)
    pad_id = vocab_size
    results = []
    for b in range(B):
        r, p, f = rouge1_from_token_ids_gpu(
            ref_ids[b], pred_ids[b],
            len_ref[b].item(), len_pred[b].item(),
            vocab_size, pad_id, device,
        )
        results.append((r, p, f))
    return results


def lcs_lengths_batched_gpu(
    ref_ids: torch.Tensor,
    pred_ids: torch.Tensor,
    len_ref: torch.Tensor,
    len_pred: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Batched LCS length: one scalar per pair.
    ref_ids (B, max_R), pred_ids (B, max_C). Only positions within length are used (mask).
    """
    B, max_R, max_C = ref_ids.size(0), ref_ids.size(1), pred_ids.size(1)
    table = torch.zeros((B, max_R + 1, max_C + 1), dtype=torch.long, device=device)
    for i in range(1, max_R + 1):
        for j in range(1, max_C + 1):
            match = (ref_ids[:, i - 1] == pred_ids[:, j - 1])
            valid = (i <= len_ref) & (j <= len_pred)
            diag = table[:, i - 1, j - 1] + 1
            up = table[:, i - 1, j]
            left = table[:, i, j - 1]
            new_val = torch.where(
                match & valid,
                diag,
                torch.maximum(up, left),
            )
            table[:, i, j] = new_val
    lcs_lengths = torch.zeros(B, dtype=torch.long, device=device)
    for b in range(B):
        r, c = len_ref[b].item(), len_pred[b].item()
        lcs_lengths[b] = table[b, r, c]
    return lcs_lengths
