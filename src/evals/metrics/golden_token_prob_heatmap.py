"""Golden-token softmax probability heatmaps over (diffusion step, token index).

Used by ``trajectory_metrics`` when ``golden_token_prob_heatmap`` is enabled. The
registered metric is a stub; aggregation and tensor math live in the trajectory loop.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from evals.metrics.base import unlearning_metric

logger = logging.getLogger("evaluator")


def _softmax_pick(logits_v: np.ndarray, idx: int) -> float:
    """Return softmax(logits_v)[idx] in float64 (numerically stable)."""
    x = logits_v.astype(np.float64, copy=False)
    m = float(np.max(x))
    e = np.exp(x - m)
    s = float(np.sum(e))
    if s <= 0.0:
        return 0.0
    return float(e[idx] / s)


def _logit_column_for_alignment(ell: int, *, logit_alignment: str) -> int:
    if logit_alignment == "causal":
        return max(0, ell - 1)
    if logit_alignment == "same_position":
        return ell
    raise ValueError(
        f"logit_alignment must be causal|same_position, got {logit_alignment!r}"
    )


def log_golden_token_heatmap_sample_diagnostics(
    *,
    sample_idx: int,
    idx_str: str,
    traj_name: str,
    logits_by_step: Mapping[int, torch.Tensor],
    gen_labels: torch.Tensor,
    steps_to_use: Sequence[int],
    logit_alignment: str,
    ignore_index: int,
    L_gen: int,
    L_eff: int,
) -> None:
    """Verbose INFO logs: softmax mass on gold vs entropy, key token indices, duplicate logit columns.

    Intended for K8s / short runs when ``trajectory_config.golden_token_heatmap_verbose_log`` is true.
    """
    if not steps_to_use:
        return
    gl = gen_labels.detach().long().reshape(-1)[:L_gen]
    # Token positions to probe (generation indices 0..L_gen-1)
    probe_ells = sorted(
        {
            0,
            1,
            2,
            3,
            4,
            5,
            min(10, L_gen - 1),
            min(20, L_gen - 1),
            L_gen // 2,
            max(0, L_gen - 2),
            max(0, L_gen - 1),
        }
    )
    probe_ells = [e for e in probe_ells if 0 <= e < L_gen]

    # Which diffusion reporting steps to print (first / mid / last)
    pick = sorted({0, len(steps_to_use) // 2, len(steps_to_use) - 1})
    picked_steps = [int(steps_to_use[i]) for i in pick if 0 <= i < len(steps_to_use)]

    # Causal: ell=0 and ell=1 share logit_idx 0 — log once per sample/traj
    if logit_alignment == "causal":
        li0 = _logit_column_for_alignment(0, logit_alignment=logit_alignment)
        li1 = _logit_column_for_alignment(1, logit_alignment=logit_alignment)
        if li0 == li1:
            logger.info(
                "[GOLDEN_HM_DIAG] causal_alignment_note sample=%s idx=%s traj=%s "
                "ell=0 and ell=1 both use logit column %s (matches FixationStepWiseScoreProvider / packed shifted CE rule)",
                sample_idx,
                idx_str,
                traj_name,
                li0,
            )

    for st in picked_steps:
        sl = logits_by_step.get(int(st))
        if sl is None or sl.dim() != 2:
            logger.info(
                "[GOLDEN_HM_DIAG] skip step=%s (missing or bad logits shape=%s)",
                st,
                None if sl is None else tuple(sl.shape),
            )
            continue
        V, Lw = int(sl.shape[0]), int(sl.shape[1])
        slf = sl[:, : min(Lw, L_gen)].detach().float()
        Lc = int(slf.shape[1])
        logger.info(
            "[GOLDEN_HM_DIAG] sample=%s idx=%s traj=%s report_step=%s V=%s L_logits=%s L_gen=%s L_eff=%s align=%s",
            sample_idx,
            idx_str,
            traj_name,
            st,
            V,
            Lw,
            L_gen,
            L_eff,
            logit_alignment,
        )
        for ell in probe_ells:
            if ell >= Lc:
                continue
            y = int(gl[ell].item())
            if y == ignore_index or y < 0 or y >= V:
                logger.info(
                    "[GOLDEN_HM_DIAG]   ell=%s gold=%s SKIP (ignore or oov V=%s)",
                    ell,
                    y,
                    V,
                )
                continue
            li = _logit_column_for_alignment(ell, logit_alignment=logit_alignment)
            if li >= Lc:
                continue
            logits_row = slf[:, li]
            dist = torch.softmax(logits_row, dim=0)
            p_gold = float(dist[y].item())
            ent = float((-(dist * (dist.clamp_min(1e-30)).log())).sum().item())
            top_i = int(torch.argmax(dist).item())
            p_top = float(dist[top_i].item())
            logger.info(
                "[GOLDEN_HM_DIAG]   ell=%s gold=%s logit_col=%s p_gold=%.6f H_softmax=%.4f "
                "top1_id=%s p_top1=%.6f top1_eq_gold=%s",
                ell,
                y,
                li,
                p_gold,
                ent,
                top_i,
                p_top,
                str(top_i == y),
            )

    # Per-step summary on **steps** traj only: mean p_gold and mean entropy over valid ells (first reporting step)
    if traj_name == "steps" and picked_steps:
        st0 = picked_steps[0]
        sl0 = logits_by_step.get(int(st0))
        if sl0 is not None and sl0.dim() == 2:
            slf0 = sl0[:, : min(int(sl0.shape[1]), L_gen)].detach().float()
            Lc0 = int(slf0.shape[1])
            p_list: list[float] = []
            h_list: list[float] = []
            for ell in range(Lc0):
                y = int(gl[ell].item())
                if y == ignore_index or y < 0 or y >= int(sl0.shape[0]):
                    continue
                li = _logit_column_for_alignment(ell, logit_alignment=logit_alignment)
                if li >= Lc0:
                    continue
                d = torch.softmax(slf0[:, li], dim=0)
                p_list.append(float(d[y].item()))
                h_list.append(float((-(d * (d.clamp_min(1e-30)).log())).sum().item()))
            if p_list:
                logger.info(
                    "[GOLDEN_HM_DIAG] summary sample=%s idx=%s traj=steps report_step=%s "
                    "n_valid_ell=%s mean_p_gold=%.6f min_p_gold=%.6f max_p_gold=%.6f mean_H=%.4f",
                    sample_idx,
                    idx_str,
                    st0,
                    len(p_list),
                    float(np.mean(p_list)),
                    float(np.min(p_list)),
                    float(np.max(p_list)),
                    float(np.mean(h_list)),
                )


def compute_golden_token_prob_heatmap_row(
    logits_vl: torch.Tensor,
    gen_labels: torch.Tensor,
    *,
    logit_alignment: str,
    ignore_index: int,
) -> np.ndarray:
    """Per-token golden probability for one diffusion step (one trajectory slice).

    ``logits_vl`` is ``[V, L]`` (vocab, positions). ``gen_labels`` length ``L`` aligned
    with generation positions. Returns length-``L`` float64 vector; ``nan`` where label
    is ``ignore_index`` or out-of-vocab.
    """
    if logits_vl.dim() != 2:
        raise ValueError(f"logits_vl must be [V, L], got {tuple(logits_vl.shape)}")
    device = logits_vl.device
    lab = gen_labels.detach().to(device=device).long().reshape(-1)
    L = int(logits_vl.shape[1])
    if lab.numel() < L:
        raise ValueError(f"gen_labels length {lab.numel()} < L={L}")
    lab = lab[:L]
    sl = logits_vl.detach().float().cpu().numpy()
    out = np.full((L,), np.nan, dtype=np.float64)
    V = sl.shape[0]
    for ell in range(L):
        y = int(lab[ell].item())
        if y == ignore_index or y < 0 or y >= V:
            continue
        logit_idx = _logit_column_for_alignment(ell, logit_alignment=logit_alignment)
        if logit_idx >= L:
            continue
        logits_row = sl[:, logit_idx]
        out[ell] = _softmax_pick(logits_row, y)
    return out


class GoldenTokenHeatmapAccumulator:
    """Running mean of golden-token probs over samples (per view / trajectory type)."""

    def __init__(
        self,
        *,
        step_indices: Sequence[int],
        L_gen: int,
        trajectory_names: Sequence[str],
        views: Sequence[str],
    ) -> None:
        self._step_indices = [int(s) for s in step_indices]
        self._n_steps = len(self._step_indices)
        self._L_gen = int(L_gen)
        self._trajectory_names = list(trajectory_names)
        self._views = list(views)
        self._sum: Dict[str, Dict[str, np.ndarray]] = {
            v: {
                t: np.zeros((self._n_steps, self._L_gen), dtype=np.float64)
                for t in self._trajectory_names
            }
            for v in self._views
        }
        self._cnt: Dict[str, Dict[str, np.ndarray]] = {
            v: {
                t: np.zeros((self._n_steps, self._L_gen), dtype=np.int64)
                for t in self._trajectory_names
            }
            for v in self._views
        }

    def add_traj(
        self,
        *,
        traj_name: str,
        logits_by_step: Mapping[int, torch.Tensor],
        gen_labels: torch.Tensor,
        logit_alignment: str,
        ignore_index: int,
        L_eff: Optional[int] = None,
    ) -> None:
        """Accumulate one sample for ``traj_name`` (all diffusion steps in ``logits_by_step``)."""
        if traj_name not in self._sum["full"]:
            return
        for view in self._views:
            if view == "full":
                L_use = self._L_gen
                labels_use = gen_labels[: self._L_gen]
            elif view == "eos":
                if L_eff is None:
                    continue
                L_use = min(int(L_eff), self._L_gen)
                labels_use = gen_labels[:L_use]
            else:
                continue
            for i_step, st in enumerate(self._step_indices):
                sl = logits_by_step.get(int(st))
                if sl is None or sl.dim() != 2:
                    continue
                sl_t = sl[:, :L_use]
                row = compute_golden_token_prob_heatmap_row(
                    sl_t,
                    labels_use,
                    logit_alignment=logit_alignment,
                    ignore_index=ignore_index,
                )
                tgt_sum = self._sum[view][traj_name]
                tgt_cnt = self._cnt[view][traj_name]
                for ell in range(L_use):
                    val = row[ell]
                    if np.isnan(val):
                        continue
                    tgt_sum[i_step, ell] += val
                    tgt_cnt[i_step, ell] += 1

    def finalize_agg_value(self) -> Dict[str, Any]:
        """Nested ``view`` → ``traj`` → matrix + axes (JSON-serializable)."""
        out: Dict[str, Any] = {}
        x_token_index = list(range(self._L_gen))
        for view in self._views:
            out[view] = {}
            for traj_name in self._trajectory_names:
                s = self._sum[view][traj_name]
                c = self._cnt[view][traj_name]
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = np.divide(s, np.maximum(c, 1))
                out[view][traj_name] = {
                    "matrix_mean": mean.tolist(),
                    "matrix_count": c.astype(int).tolist(),
                    "y_step_index": list(self._step_indices),
                    "x_token_index": x_token_index,
                }
        return out


@unlearning_metric(name="golden_token_prob_heatmap")
def golden_token_prob_heatmap(model, **kwargs) -> Dict[str, Any]:
    """Stub: real values are produced inside ``trajectory_metrics`` when enabled."""
    _ = model, kwargs
    return {"agg_value": None, "value_by_index": {}}
