"""Golden-token softmax probability heatmaps over (diffusion step, token index).

Used by ``trajectory_metrics`` when ``golden_token_prob_heatmap`` is enabled. The
registered metric is a stub; aggregation and tensor math live in the trajectory loop.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from evals.metrics.base import unlearning_metric


def _softmax_pick(logits_v: np.ndarray, idx: int) -> float:
    """Return softmax(logits_v)[idx] in float64 (numerically stable)."""
    x = logits_v.astype(np.float64, copy=False)
    m = float(np.max(x))
    e = np.exp(x - m)
    s = float(np.sum(e))
    if s <= 0.0:
        return 0.0
    return float(e[idx] / s)


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
        if logit_alignment == "causal":
            logit_idx = max(0, ell - 1)
        elif logit_alignment == "same_position":
            logit_idx = ell
        else:
            raise ValueError(
                f"logit_alignment must be causal|same_position, got {logit_alignment!r}"
            )
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
