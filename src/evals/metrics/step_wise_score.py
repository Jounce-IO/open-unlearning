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

from typing import Any, Optional, Protocol

import numpy as np
import torch

from data.utils import IGNORE_INDEX


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
    for scores, _ in results:
        if not scores:
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
    device = R.device
    F_clamped = F.clamp(0, S - 1)
    index = F_clamped.view(B, 1, L, 1).expand(B, V, L, 1).long()
    gathered = torch.gather(R, dim=3, index=index).squeeze(3)
    return gathered.permute(0, 2, 1)


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
        log_probs = torch.nn.functional.log_softmax(fixation_logits, dim=-1)
        results: list[tuple[list[float], Optional[list[int]]]] = []
        for b in range(B):
            probs_list: list[float] = []
            fixation_list: list[int] = []
            for ell in range(L):
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
    preds = torch.zeros(L, dtype=torch.long, device=fixation_logits.device)
    for ell in range(L):
        logit_idx = max(0, ell - 1) if logit_alignment == "causal" else ell
        preds[ell] = torch.argmax(fixation_logits[logit_idx, :], dim=-1)
    valid = (labels != ignore_index) & (labels >= 0) & (labels < V)
    if valid.sum().item() == 0:
        return 0.0
    F_np = F.cpu().numpy() if F.dim() > 0 else np.array([F.item()])
    if F_np.size != L:
        F_np = np.broadcast_to(F_np, (L,))
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    valid_np = valid.cpu().numpy()
    best_t = S
    for t in range(S):
        match = True
        for ell in range(L):
            if not valid_np[ell]:
                continue
            if F_np[ell] >= t:
                if preds_np[ell] != labels_np[ell]:
                    match = False
                    break
        if match:
            best_t = t
            break
    return float(1.0 - (best_t / S))
