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
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple

import numpy as np
import torch

from data.utils import IGNORE_INDEX
from evals.metrics.utils import _tensor_to_list_of_floats

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


def _es_audit_prefix(ctx: Optional[Mapping[str, Any]]) -> str:
    if not ctx:
        return ""
    parts = []
    for k in ("sample_idx", "traj_name", "view", "step"):
        v = ctx.get(k)
        if v is not None:
            parts.append(f"{k}={v}")
    return ("[" + " ".join(parts) + "] ") if parts else ""


def extraction_strength_from_fixation(
    fixation_logits: torch.Tensor,
    labels: torch.Tensor,
    F: torch.Tensor,
    S: int,
    logit_alignment: str = "causal",
    ignore_index: int = IGNORE_INDEX,
    *,
    audit: bool = False,
    audit_ctx: Optional[Mapping[str, Any]] = None,
) -> float:
    """Compute dLLM extraction strength: 1 minus the smallest fraction of fixation steps to drop so the rest match target.

    fixation_logits: [L, V] logits at each position at its fixation step.
    labels: [L] target token ids.
    F: [L] fixation step index per position.
    S: total number of steps.
    Returns value in [0, 1]; AR path unchanged (use prefix-based ES elsewhere).

    When ``audit`` is True and the evaluator logger is DEBUG-enabled, emits
    ``EXTRACTION_STRENGTH_AUDIT`` lines (branches, per-t pass/fail, ``best_t``, logits snippet, optional decode).
    """
    audit_on = bool(audit) and logger.isEnabledFor(logging.DEBUG)
    ap = _es_audit_prefix(audit_ctx)

    def _alog(msg: str, *args: Any) -> None:
        if audit_on:
            logger.debug("EXTRACTION_STRENGTH_AUDIT " + ap + msg, *args)

    fixation_logits = fixation_logits.float()
    L, V = fixation_logits.shape
    _alog(
        "enter L=%s V=%s S=%s logit_alignment=%s ignore_index=%s fixation_logits_shape=%s labels_shape=%s F_shape=%s",
        L,
        V,
        S,
        logit_alignment,
        ignore_index,
        tuple(fixation_logits.shape),
        tuple(labels.shape),
        tuple(F.shape),
    )
    if L == 0 or S <= 0:
        _alog("branch early_exit: L==0 or S<=0 (L=%s S=%s) -> return 0.0", L, S)
        return 0.0
    L_lab = labels.shape[0]
    L_use = min(L, L_lab)
    if L_use == 0:
        _alog("branch early_exit: L_use==0 (L=%s L_lab=%s) -> return 0.0", L, L_lab)
        return 0.0
    preds = torch.zeros(L, dtype=torch.long, device=fixation_logits.device)
    for ell in range(L):
        logit_idx = max(0, ell - 1) if logit_alignment == "causal" else ell
        preds[ell] = torch.argmax(fixation_logits[logit_idx, :], dim=-1)
    valid = (labels != ignore_index) & (labels >= 0) & (labels < V)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        _alog("branch early_exit: no valid label positions (valid_count=0) -> return 0.0")
        return 0.0
    F_np = F.cpu().numpy() if F.dim() > 0 else np.array([F.item()])
    if F_np.size != L:
        _alog("branch F_broadcast: F_np.size=%s != L=%s -> broadcast_to (L,)", F_np.size, L)
        F_np = np.broadcast_to(F_np, (L,))
    else:
        _alog("branch F_shape: F_np.size == L=%s (no broadcast)", L)
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    valid_np = valid.cpu().numpy()
    mismatch_valid = int(
        sum(
            1
            for ell in range(L_use)
            if valid_np[ell] and preds_np[ell] != labels_np[ell]
        )
    )
    _alog(
        "preds_summary L_use=%s valid_positions=%s argmax_mismatch_on_valid=%s",
        L_use,
        n_valid,
        mismatch_valid,
    )
    max_chars = 8000
    if audit_ctx and audit_ctx.get("max_decode_chars") is not None:
        try:
            max_chars = int(audit_ctx["max_decode_chars"])
        except (TypeError, ValueError):
            max_chars = 8000
    tok = audit_ctx.get("tokenizer") if audit_ctx else None
    if tok is not None and audit_on:
        try:
            pred_ids = preds[:L_use].detach().cpu().tolist()
            lab_ids = labels[:L_use].detach().cpu().tolist()
            pred_txt = tok.decode(pred_ids, skip_special_tokens=True)
            lab_txt = tok.decode(lab_ids, skip_special_tokens=True)
            if len(pred_txt) > max_chars:
                pred_txt = pred_txt[:max_chars] + "…"
            if len(lab_txt) > max_chars:
                lab_txt = lab_txt[:max_chars] + "…"
            _alog("text_decode pred_argmax_seq=%r labels_seq=%r", pred_txt, lab_txt)
        except Exception as e:
            _alog("text_decode failed: %s", e)
    # Logit snapshot: up to 5 valid positions — logit row used, pred/label ids, log-prob of label token
    if audit_on:
        logged = 0
        with torch.no_grad():
            for ell in range(L_use):
                if not valid_np[ell]:
                    continue
                logit_idx = max(0, ell - 1) if logit_alignment == "causal" else ell
                row = fixation_logits[logit_idx].float()
                lp = torch.log_softmax(row, dim=-1)
                yi = int(labels_np[ell])
                pi = int(preds_np[ell])
                max_logit = float(row.max().item())
                logp_lab = float(lp[yi].item()) if 0 <= yi < V else float("nan")
                _alog(
                    "logit_pos ell=%s logit_idx=%s pred_id=%s label_id=%s F=%s max_logit=%.4f logp_label=%.4f",
                    ell,
                    logit_idx,
                    pi,
                    yi,
                    int(F_np[ell]) if ell < len(F_np) else "?",
                    max_logit,
                    logp_lab,
                )
                logged += 1
                if logged >= 5:
                    break
    best_t = S
    for t in range(S):
        match = True
        first_mismatch: Optional[Tuple[int, int, int, int]] = None
        for ell in range(L_use):
            if not valid_np[ell]:
                continue
            if F_np[ell] >= t:
                if preds_np[ell] != labels_np[ell]:
                    match = False
                    first_mismatch = (
                        ell,
                        int(preds_np[ell]),
                        int(labels_np[ell]),
                        int(F_np[ell]),
                    )
                    break
        if match:
            _alog(
                "loop_t t=%s PASS (all constrained positions match) -> best_t=%s break",
                t,
                t,
            )
            best_t = t
            break
        if first_mismatch is not None:
            _alog(
                "loop_t t=%s FAIL first_mismatch ell=%s pred_id=%s label_id=%s F[ell]=%s",
                t,
                first_mismatch[0],
                first_mismatch[1],
                first_mismatch[2],
                first_mismatch[3],
            )
    es = float(1.0 - (best_t / S))
    _alog(
        "result best_t=%s S=%s es=1-best_t/S=%s (best_t==S means no t in [0,S-1] passed)",
        best_t,
        S,
        es,
    )
    return es


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
    with torch.no_grad():
        fixation_logits = fixation_logits.to(device)
        labels = labels.to(device)
        t_fl = fixation_logits.shape[1]
        t_lab = labels.shape[1]
        t = min(t_fl, t_lab)
        fixation_logits = fixation_logits[:, :t, :].contiguous()
        labels = labels[:, :t].contiguous()
        shifted_logits = fixation_logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        if shifted_logits.dtype in (torch.bfloat16, torch.float16):
            shifted_logits = shifted_logits.float()
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
        losses = loss_fn(shifted_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
        num_token_gt = (shifted_labels != ignore_index).sum(dim=-1).clamp(min=1)
        avg_losses = losses / num_token_gt
        normalized_probs = torch.exp(-avg_losses)
        avg_losses_list = _tensor_to_list_of_floats(avg_losses)
        normalized_probs_list = _tensor_to_list_of_floats(normalized_probs)
        return [
            {"prob": prob, "avg_loss": avg_loss}
            for prob, avg_loss in zip(normalized_probs_list, avg_losses_list)
        ]


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
