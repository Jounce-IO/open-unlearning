"""
Step-wise score provider abstraction for generalized unlearning metrics.

Step-wise score at position ell is the probability of the target token at ell,
evaluated at the fixation step for that position. For AR: fixation step = ell,
score = next-token probability. For dLLM: fixation step F[ell] and alignment
(causal vs same_position) determine which logit slice is used.

Metrics (Probability, Truth Ratio, Min-K, ES) consume per-position scores
or their geometric mean; they are agnostic to model type or alignment.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np
import torch

from data.utils import IGNORE_INDEX

logger = logging.getLogger("evaluator")


class StepWiseScoreProvider(Protocol):
    """Protocol for computing per-position step-wise scores (and optional fixation steps)."""

    def get_per_position_scores(
        self,
        model_or_logits: Any,
        batch: dict[str, Any],
        *,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = IGNORE_INDEX,
    ) -> list[tuple[list[float], Optional[list[int]]]]:
        """
        Return per-position scores for each sample in the batch.

        Args:
            model_or_logits: Model (for AR) or precomputed logits/trajectory data (for Fixation).
            batch: Dict with input_ids, labels (if not passed separately), etc.
            labels: Optional override for labels (e.g. for truth_ratio correct/wrong).
            ignore_index: Label value that indicates positions to skip.

        Returns:
            List of (scores, fixation_steps) per sample. scores: list of float,
            one per non-ignore position (probability of target token at that position).
            fixation_steps: list of int (step index when each position was fixed) or None.
            For AR, fixation_steps can be None or [0,1,...,T-1].
        """
        ...


def sequence_probability_from_scores(scores: list[float]) -> float:
    """Geometric mean of per-position scores = length-normalized sequence probability."""
    if not scores:
        return 0.0
    return float(np.exp(np.mean(np.log(np.array(scores, dtype=np.float64) + 1e-12))))


def evaluate_probability_via_provider(
    provider: StepWiseScoreProvider,
    model: Any,
    batch: dict[str, Any],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> list[dict[str, Any]]:
    """
    Compute probability (and avg_loss) per sample using a step-wise score provider.

    Returns the same format as evaluate_probability: list of {"prob", "avg_loss"}
    for use with run_batchwise_evals and truth_ratio pre_compute.
    """
    results = provider.get_per_position_scores(
        model, batch, ignore_index=ignore_index
    )
    out = []
    for i, (scores, _) in enumerate(results):
        if not scores:
            logger.info(
                "pre_compute probability (via provider): empty scores for sample %s — "
                "provider returned no per-position scores (no valid positions or L_use=0)",
                i,
            )
            out.append({"prob": None, "avg_loss": None})
            continue
        prob = sequence_probability_from_scores(scores)
        avg_loss = float(-np.log(prob + 1e-12))
        out.append({"prob": prob, "avg_loss": avg_loss})
    return out


class ARStepWiseScoreProvider:
    """
    Step-wise score provider for autoregressive (causal) models.

    Uses next-token log-probs: score at position ell = p(a_ell | q, a_<ell).
    Always causal alignment; fixation step for position ell is ell.
    Uses the same positions as evaluate_probability (all non-ignore in shifted labels).
    """

    def get_per_position_scores(
        self,
        model_or_logits: Any,
        batch: dict[str, Any],
        *,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = IGNORE_INDEX,
    ) -> list[tuple[list[float], Optional[list[int]]]]:
        if labels is not None:
            batch = {**batch, "labels": labels}
        batch = {k: v.to(model_or_logits.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            output = model_or_logits(**batch)
        logits = output.logits
        lab = batch["labels"]
        shifted_labels = lab[..., 1:].contiguous()
        logits_shifted = logits[..., :-1, :].contiguous()
        log_probs = torch.nn.functional.log_softmax(logits_shifted.float(), dim=-1)
        bsz, seq_len, _ = logits_shifted.shape
        results: list[tuple[list[float], Optional[list[int]]]] = []
        for i in range(bsz):
            probs_list: list[float] = []
            for pos in range(seq_len):
                if shifted_labels[i, pos].item() == ignore_index:
                    continue
                y = shifted_labels[i, pos].item()
                lp = log_probs[i, pos, int(y)].item()
                probs_list.append(float(np.exp(lp)))
            fixation_steps = list(range(len(probs_list))) if probs_list else None
            results.append((probs_list, fixation_steps))
        return results


def build_fixation_logits_from_R_F(
    R: torch.Tensor, F: torch.Tensor
) -> torch.Tensor:
    """
    Build [B, L, V] fixation logits from R [B, V, L, S] and F [B, L].

    fixation_logits[b, ell, :] = R[b, :, ell, F[b, ell]] (logits at position ell
    at the step when ell was fixed). For a single sample, R is [V, L, S], F is [L].
    """
    if R.dim() == 3:
        R = R.unsqueeze(0)
        F = F.unsqueeze(0)
    B, V, L, S = R.shape
    _ = R.device  # device, reserved
    F_clamped = F.clamp(0, S - 1)
    index = F_clamped.view(B, 1, L, 1).expand(B, V, L, 1).long()
    gathered = torch.gather(R, dim=3, index=index).squeeze(3)
    return gathered.permute(0, 2, 1)


def build_effective_step_fixation_logits_from_history(
    lh: list[torch.Tensor], F: torch.Tensor, report_step: int
) -> torch.Tensor:
    """Same semantics as :func:`build_effective_step_fixation_logits` using list-backed logits.

    ``lh[s]`` is ``[B, L, V]`` per diffusion step ``s``. ``F`` is ``[B, L]`` or ``[L]`` (single row).
    """
    if F.dim() == 1:
        F = F.unsqueeze(0)
    S = len(lh)
    if S <= 0:
        raise ValueError("logits history must be non-empty")
    B, L, _V = lh[0].shape
    report_step_clamped = min(max(0, report_step), S - 1) if S > 0 else 0
    s_eff = torch.minimum(
        torch.full_like(F, report_step_clamped, device=F.device, dtype=F.dtype),
        F.clamp(0, S - 1),
    )
    out = torch.empty(B, L, lh[0].shape[2], device=lh[0].device, dtype=lh[0].dtype)
    for s in range(S):
        mask = s_eff == s
        if mask.any():
            contrib = lh[s]
            out[mask] = contrib[mask]
    return out


def build_fixation_logits_from_history(
    lh: list[torch.Tensor], F: torch.Tensor, b: int
) -> torch.Tensor:
    """Fixation logits ``[L, V]`` for sample row ``b`` from list-backed history.

    Matches :func:`build_fixation_logits_from_R_F` for a single packed sample:
    position ``ell`` uses logits at diffusion step ``F[ell]`` (clamped to ``[0, S-1]``).
    Each ``lh[s]`` is ``[B, L, V]``.
    """
    if F.dim() > 1:
        F_row = F[int(b)]
    else:
        F_row = F
    S = len(lh)
    if S <= 0:
        raise ValueError("logits history must be non-empty")
    L = int(F_row.shape[0])
    device = F_row.device
    dtype = lh[0].dtype
    V = lh[0].shape[-1]
    out = torch.empty(L, V, device=device, dtype=dtype)
    bb = int(b)
    for ell in range(L):
        s_fix = int(F_row[ell].clamp(0, S - 1).item())
        out[ell] = lh[s_fix][bb, ell, :]
    return out


def build_effective_step_fixation_logits(
    R: torch.Tensor, F: torch.Tensor, report_step: int
) -> torch.Tensor:
    """
    Build [B, L, V] logits at report step s using effective step per position.

    Effective step for position ell at report step s: s_eff(ell, s) = min(s, F[ell]).
    If position ell is not fixed yet at s (F[ell] > s), use latest step s.
    fixation_logits[b, ell, :] = R[b, :, ell, s_eff(b, ell)] with s_eff clamped to [0, S-1].
    For a single sample, R is [V, L, S], F is [L].
    """
    if R.dim() == 3:
        R = R.unsqueeze(0)
        F = F.unsqueeze(0)
    B, V, L, S = R.shape
    report_step_clamped = min(max(0, report_step), S - 1) if S > 0 else 0
    s_eff = torch.minimum(
        torch.full_like(F, report_step_clamped, device=R.device, dtype=F.dtype),
        F.clamp(0, S - 1),
    )
    index = s_eff.view(B, 1, L, 1).expand(B, V, L, 1).long()
    gathered = torch.gather(R, dim=3, index=index).squeeze(3)
    return gathered.permute(0, 2, 1)


class FixationStepWiseScoreProvider:
    """
    Step-wise score provider for trajectory/fixation logits (dLLM).

    Uses fixation_logits [B, L, V] where fixation_logits[b, ell, :] is the logit
    vector at the step when position ell was fixed. Supports logit_alignment:
    - causal: logits at index i predict token at i+1; score at ell uses slice ell-1.
    - same_position: logits at index i predict token at i; score at ell uses slice ell.
    """

    def __init__(self, logit_alignment: str = "causal"):
        if logit_alignment not in ("causal", "same_position"):
            raise ValueError(
                f"logit_alignment must be 'causal' or 'same_position', got {logit_alignment!r}"
            )
        self.logit_alignment = logit_alignment

    def get_per_position_scores(
        self,
        model_or_logits: Any,
        batch: dict[str, Any],
        *,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = IGNORE_INDEX,
    ) -> list[tuple[list[float], Optional[list[int]]]]:
        if labels is not None:
            batch = {**batch, "labels": labels}
        lab = batch["labels"]
        if lab.dim() == 1:
            lab = lab.unsqueeze(0)
        if isinstance(model_or_logits, dict):
            fixation_logits = model_or_logits.get("fixation_logits")
            report_step = model_or_logits.get("report_step")
            if fixation_logits is None or report_step is not None:
                R = model_or_logits["R"]
                F = model_or_logits["F"]
                if report_step is not None:
                    fixation_logits = build_effective_step_fixation_logits(
                        R, F, int(report_step)
                    )
                else:
                    fixation_logits = build_fixation_logits_from_R_F(R, F)
        else:
            fixation_logits = model_or_logits
        fixation_logits = fixation_logits.float()
        B, L, V = fixation_logits.shape
        L_lab = lab.shape[1]
        # Trajectory R/F can have one more position than labels (e.g. final step); cap to avoid index error.
        L_use = min(L, L_lab)
        log_probs = torch.nn.functional.log_softmax(fixation_logits, dim=-1)
        results: list[tuple[list[float], Optional[list[int]]]] = []
        for b in range(B):
            probs_list: list[float] = []
            fixation_list: list[int] = []
            for ell in range(L_use):
                if lab[b, ell].item() == ignore_index:
                    continue
                if self.logit_alignment == "causal":
                    logit_idx = max(0, ell - 1)
                else:
                    logit_idx = ell
                y = int(lab[b, ell].item())
                lp = log_probs[b, logit_idx, y].item()
                probs_list.append(float(np.exp(lp)))
                fixation_list.append(ell)
            results.append((probs_list, fixation_list if probs_list else None))
        return results


def extraction_strength_from_fixation(
    fixation_logits: torch.Tensor,
    labels: torch.Tensor,
    F: torch.Tensor,
    S: int,
    logit_alignment: str = "causal",
    ignore_index: int = IGNORE_INDEX,
) -> float:
    """Compute dLLM extraction strength: 1 minus the smallest fraction of fixation steps to drop so the rest match target.

    fixation_logits: [L, V] logits at each position at its fixation step.
    labels: [L] target token ids.
    F: [L] fixation step index per position.
    S: total number of steps.
    Returns value in [0, 1]; AR path unchanged (use prefix-based ES elsewhere).
    """
    fixation_logits = fixation_logits.float()
    L, V = fixation_logits.shape
    if L == 0 or S <= 0:
        return 0.0
    L_lab = int(labels.shape[0])
    L_use = min(L, L_lab)
    if L_use == 0:
        return 0.0
    device = fixation_logits.device
    ells = torch.arange(L, device=device, dtype=torch.long)
    if logit_alignment == "causal":
        logit_idx = (ells - 1).clamp(min=0)
    else:
        logit_idx = ells
    logits_rows = fixation_logits[logit_idx]
    preds = logits_rows.argmax(dim=-1)

    lab = labels.reshape(-1).to(device=device)
    valid = (lab != ignore_index) & (lab >= 0) & (lab < V)
    if valid.sum().item() == 0:
        return 0.0

    Fr = F.reshape(-1).to(device=device)
    if Fr.numel() == L:
        F_row = Fr
    else:
        F_row = torch.broadcast_to(Fr, (L,))

    preds_use = preds[:L_use]
    lab_use = lab[:L_use]
    valid_use = valid[:L_use]
    F_use = F_row[:L_use]

    ts = torch.arange(S, device=device, dtype=torch.long).view(S, 1)
    required = valid_use.unsqueeze(0) & (F_use.unsqueeze(0) >= ts)
    mismatch = required & (preds_use.unsqueeze(0) != lab_use.unsqueeze(0))
    any_mismatch = mismatch.any(dim=1)
    ok = ~any_mismatch
    if ok.any():
        best_t = int(torch.argmax(ok.to(torch.float32)).item())
    else:
        best_t = S
    return float(1.0 - (best_t / S))


def _default_max_ce_logits_rows(vocab: int) -> int:
    """Upper bound on shifted-token rows per ``cross_entropy`` forward (float32 ``[N,V]`` logits).

    Bounds peak GPU memory for large ``V`` (e.g. LLaDA vocab ~128k) when many segments are
    packed. Override with env ``TRAJECTORY_PACKED_CE_BUDGET_BYTES`` (default ~380MB logits).
    """
    try:
        budget = int(os.environ.get("TRAJECTORY_PACKED_CE_BUDGET_BYTES", "380000000"))
    except ValueError:
        budget = 380_000_000
    v = max(int(vocab), 1)
    per_vocab = max(1, budget // (4 * v))
    return max(64, min(8192, per_vocab))


def _cross_entropy_packed_shifted_rows(
    flat_logits_parts: List[torch.Tensor],
    flat_labels_parts: List[torch.Tensor],
    ignore_index: int,
    max_rows: int,
) -> torch.Tensor:
    """Run ``cross_entropy`` on row chunks without concatenating all logits into one tensor."""
    max_rows = max(1, int(max_rows))
    all_chunks: list[torch.Tensor] = []
    buf_l: list[torch.Tensor] = []
    buf_y: list[torch.Tensor] = []
    cur = 0

    def flush() -> None:
        nonlocal cur, buf_l, buf_y
        if cur == 0:
            return
        logits_cat = torch.cat(buf_l, dim=0)
        labels_cat = torch.cat(buf_y, dim=0)
        losses = torch.nn.functional.cross_entropy(
            logits_cat,
            labels_cat,
            ignore_index=ignore_index,
            reduction="none",
        )
        all_chunks.append(losses)
        del logits_cat, labels_cat
        buf_l.clear()
        buf_y.clear()
        cur = 0

    for pl, pyl in zip(flat_logits_parts, flat_labels_parts, strict=True):
        r = 0
        n = pl.shape[0]
        while r < n:
            if cur >= max_rows:
                flush()
            space = max_rows - cur
            need = min(n - r, space)
            if need <= 0:
                flush()
                continue
            buf_l.append(pl[r : r + need])
            buf_y.append(pyl[r : r + need])
            cur += need
            r += need
    flush()
    return torch.cat(all_chunks, dim=0)


def compute_prob_packed_shifted_segments(
    segment_fixation_logits: Sequence[torch.Tensor],
    segment_labels: Sequence[torch.Tensor],
    device: torch.device,
    ignore_index: int = IGNORE_INDEX,
    *,
    max_ce_logits_rows: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Per-segment ``prob`` / ``avg_loss`` from packed shifted-token cross-entropy.

    Each segment is one sample (or one view of one sample): ``segment_fixation_logits[i]``
    is ``[1, T_i, V]`` and ``segment_labels[i]`` is ``[1, T_i]``, already view-sliced
    (e.g. full length ``L`` or eos prefix). Segments may have different ``T_i`` (ragged).

    Implements the same shift, ``min(T_logits, T_labels)``, dtype promotion, and
    ``ignore_index`` handling as :func:`compute_prob_from_fixation_logits`, then runs
    ``cross_entropy`` over row chunks (bounded by ``max_ce_logits_rows`` or an automatic
    budget from ``V``) so peak memory stays ~``O(max_rows * V)`` instead of ``O(T_total * V)``.

    Returns one dict per segment, same contract as ``compute_prob_from_fixation_logits``.
    """
    if len(segment_fixation_logits) != len(segment_labels):
        raise ValueError(
            "segment_fixation_logits and segment_labels must have the same length; "
            f"got {len(segment_fixation_logits)} vs {len(segment_labels)}"
        )
    if not segment_fixation_logits:
        return []

    flat_logits_parts: list[torch.Tensor] = []
    flat_labels_parts: list[torch.Tensor] = []
    seg_shift_lens: list[int] = []

    with torch.no_grad():
        for log, lab in zip(segment_fixation_logits, segment_labels, strict=True):
            if log.dim() != 3 or lab.dim() != 2:
                raise ValueError(
                    "Each segment expects fixation_logits [1, T, V] and labels [1, T]; "
                    f"got logits {tuple(log.shape)}, labels {tuple(lab.shape)}"
                )
            if log.device != device:
                log = log.to(device)
            if lab.device != device:
                lab = lab.to(device)
            t_fl = log.shape[1]
            t_lab = lab.shape[1]
            t = min(t_fl, t_lab)
            log = log[:, :t, :].contiguous()
            lab = lab[:, :t].contiguous()
            shifted_logits = log[:, :-1, :].contiguous()
            shifted_labels = lab[:, 1:].contiguous()
            n_shift = shifted_logits.shape[1]
            if shifted_logits.dtype in (torch.bfloat16, torch.float16):
                shifted_logits = shifted_logits.float()
            if n_shift == 0:
                seg_shift_lens.append(0)
                continue
            v = shifted_logits.shape[-1]
            flat_logits_parts.append(shifted_logits.reshape(n_shift, v))
            flat_labels_parts.append(shifted_labels.reshape(n_shift))
            seg_shift_lens.append(n_shift)

        if not flat_logits_parts:
            return [{"prob": 1.0, "avg_loss": 0.0} for _ in seg_shift_lens]

        v0 = int(flat_logits_parts[0].shape[-1])
        row_cap = (
            max(1, int(max_ce_logits_rows))
            if max_ce_logits_rows is not None
            else _default_max_ce_logits_rows(v0)
        )
        token_losses = _cross_entropy_packed_shifted_rows(
            flat_logits_parts,
            flat_labels_parts,
            ignore_index,
            row_cap,
        )
        labels_cat = torch.cat(flat_labels_parts, dim=0)

        out: List[Dict[str, float]] = []
        offset = 0
        for n_shift in seg_shift_lens:
            if n_shift == 0:
                out.append({"prob": 1.0, "avg_loss": 0.0})
                continue
            chunk = token_losses[offset : offset + n_shift]
            chunk_y = labels_cat[offset : offset + n_shift]
            sum_loss = chunk.sum()
            num_tok = (chunk_y != ignore_index).sum().clamp(min=1)
            avg_loss = sum_loss / num_tok.to(dtype=sum_loss.dtype)
            prob = torch.exp(-avg_loss)
            out.append(
                {
                    "prob": float(prob.detach().cpu()),
                    "avg_loss": float(avg_loss.detach().cpu()),
                }
            )
            offset += n_shift

        return out


def compute_prob_from_fixation_logits(
    fixation_logits: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    ignore_index: int = IGNORE_INDEX,
) -> List[Dict[str, float]]:
    """Per-sample ``prob`` / ``avg_loss`` from batch fixation logits (trajectory step helper).

    Shared by trajectory ``_call_metric_at_step`` (probability) and tests. Uses shifted
    logits/labels CE (same as legacy batch path) so results align with
    ``FixationStepWiseScoreProvider`` when the first label position is ignored (standard LM).
    """
    bsz = fixation_logits.shape[0]
    segs_log = [fixation_logits[b : b + 1] for b in range(bsz)]
    segs_lab = [labels[b : b + 1] for b in range(bsz)]
    return compute_prob_packed_shifted_segments(segs_log, segs_lab, device, ignore_index)


def trajectory_step_logits_to_prob_batch(logits_vl: torch.Tensor) -> torch.Tensor:
    """Convert trajectory step logits ``[V, L]`` to batch ``[1, L, V]`` for sequence probability.

    Use with ``compute_prob_from_fixation_logits`` on fixation-aligned rows so results match
    ``FixationStepWiseScoreProvider`` / non-traj generalized probability (no extra row shuffle).
    """
    return logits_vl.t().unsqueeze(0)


def evaluate_probability_confidence_ordered_from_fixation_logits(
    fixation_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    logit_alignment: str = "causal",
    ignore_index: int = IGNORE_INDEX,
) -> List[Dict[str, Any]]:
    """Confidence-ordered sequence probability from fixation logits ``[B, L, V]`` (dLLM).

    For each sample, collects per-position probability of the target token using the same
    index rule as ``FixationStepWiseScoreProvider``, sorts descending, geometric mean.
    """
    fixation_logits = fixation_logits.float()
    log_probs = torch.nn.functional.log_softmax(fixation_logits, dim=-1)
    B, L, V = fixation_logits.shape
    results: List[Dict[str, Any]] = []
    for b in range(B):
        pos_probs: list[float] = []
        for ell in range(L):
            if labels[b, ell].item() == ignore_index:
                continue
            logit_idx = max(0, ell - 1) if logit_alignment == "causal" else ell
            y = int(labels[b, ell].item())
            if y < 0 or y >= V:
                continue
            lp = log_probs[b, logit_idx, y].item()
            pos_probs.append(float(np.exp(lp)))
        if not pos_probs:
            logger.info(
                "confidence_ordered (fixation): no valid target positions for sample %s",
                b,
            )
            results.append({"prob": None, "avg_loss": None})
            continue
        pos_arr = np.array(pos_probs, dtype=np.float64)
        pos_probs_sorted = np.sort(pos_arr)[::-1]
        geom_mean = float(np.exp(np.mean(np.log(pos_probs_sorted + 1e-12))))
        avg_loss = float(-np.log(geom_mean + 1e-12))
        results.append({
            "prob": geom_mean,
            "avg_loss": avg_loss,
            "confidence_ordered_probs": pos_probs_sorted.tolist(),
        })
    return results


def diffusion_fixation_logits_for_probability(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """Sampler-based fixation logits [B, T, V] for non-traj generalized probability (dLLM).

    Delegates to ``DiffusionModelAdapter._fixation_logits_from_sampler`` so scoring matches
    trajectory generalized probability (``trajectory_step_logits_to_prob_batch`` +
    ``compute_prob_from_fixation_logits``) without running the full ``trajectory_metrics`` loop.
    """
    try:
        from dllm.integrations.open_unlearning_adapter import (
            DiffusionModelAdapter,
        )
    except ImportError as exc:
        raise ImportError(
            "non-trajectory generalized sequence probability for diffusion models requires "
            "dllm.integrations.open_unlearning_adapter.DiffusionModelAdapter"
        ) from exc
    if not isinstance(model, DiffusionModelAdapter):
        raise TypeError(
            "use_generalized_sequence_probability=True with diffusion sampling requires "
            f"DiffusionModelAdapter, got {type(model).__name__}"
        )
    return model._fixation_logits_from_sampler(
        input_ids, attention_mask, labels, ignore_index
    )
