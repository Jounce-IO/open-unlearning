"""Persist derived golden-token / argmax diagnostics for prob-metric investigation.

Writes ``prob_trajectory_dump/`` under the eval output directory when
``trajectory_config.prob_trajectory_dump.enabled`` is true.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from data.utils import IGNORE_INDEX
from evals.metrics.golden_token_prob_heatmap import (
    _logit_column_for_alignment,
    compute_golden_token_prob_heatmap_row,
)
from evals.metrics.step_wise_score import (
    FixationStepWiseScoreProvider,
    build_effective_step_fixation_logits,
    build_effective_step_fixation_logits_from_history,
    compute_prob_packed_shifted_segments,
    sequence_probability_from_scores,
    trajectory_step_logits_to_prob_batch,
)

logger = logging.getLogger("evaluator")


def parse_prob_trajectory_dump_config(
    trajectory_config: Optional[Mapping[str, Any]],
) -> tuple[bool, int]:
    """Return (enabled, max_samples) from ``trajectory_config.prob_trajectory_dump``."""
    if not trajectory_config:
        return False, 0
    raw = trajectory_config.get("prob_trajectory_dump")
    if raw is None:
        return False, 0
    if isinstance(raw, DictConfig):
        raw = OmegaConf.to_container(raw, resolve=True) or {}
    if not isinstance(raw, dict):
        return False, 0
    enabled = bool(raw.get("enabled", False))
    max_samples = max(0, int(raw.get("max_samples", 5)))
    return enabled, max_samples


def derive_position_arrays(
    logits_vl: torch.Tensor,
    gen_labels: torch.Tensor,
    *,
    logit_alignment: str,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Golden prob, argmax id, and gold id per generation position (length L).

    Invalid positions use ``nan`` for golden_probs and ``-1`` for ids.
    """
    if logits_vl.dim() != 2:
        raise ValueError(f"logits_vl must be [V, L], got {tuple(logits_vl.shape)}")
    lab = gen_labels.detach().long().reshape(-1)
    L = int(logits_vl.shape[1])
    if lab.numel() < L:
        lab = torch.cat(
            [
                lab,
                torch.full((L - lab.numel(),), ignore_index, dtype=lab.dtype, device=lab.device),
            ]
        )
    lab = lab[:L]
    slf = logits_vl.detach().float()
    V = int(slf.shape[0])
    golden = np.full((L,), np.nan, dtype=np.float64)
    argmax = np.full((L,), -1, dtype=np.int64)
    gold_ids = np.full((L,), -1, dtype=np.int64)
    for ell in range(L):
        y = int(lab[ell].item())
        if y == ignore_index or y < 0 or y >= V:
            continue
        li = _logit_column_for_alignment(ell, logit_alignment=logit_alignment)
        if li >= L:
            continue
        dist = torch.softmax(slf[:, li], dim=0)
        golden[ell] = float(dist[y].item())
        argmax[ell] = int(torch.argmax(dist).item())
        gold_ids[ell] = y
    return golden, argmax, gold_ids


def geom_mean_from_golden_probs(golden_probs: np.ndarray) -> float:
    """Geometric mean over finite positive golden probs (matches sequence_probability_from_scores)."""
    valid = golden_probs[np.isfinite(golden_probs) & (golden_probs > 0)]
    if valid.size == 0:
        return 0.0
    return sequence_probability_from_scores(valid.tolist())


def prob_packed_shifted_from_step_logits(
    logits_vl: torch.Tensor,
    gen_labels: torch.Tensor,
    device: torch.device,
    ignore_index: int = IGNORE_INDEX,
) -> float:
    """Scalar prob from post-loop packed shifted CE on step snapshot ``[V,L]``."""
    logits_b = trajectory_step_logits_to_prob_batch(logits_vl).to(device=device)
    lab_b = gen_labels.detach().long().reshape(1, -1).to(device=device)
    out = compute_prob_packed_shifted_segments([logits_b], [lab_b], device, ignore_index)
    if not out or out[0].get("prob") is None:
        return float("nan")
    return float(out[0]["prob"])


def prob_fixation_provider_from_sample_traj(
    sample_traj: Dict[str, Any],
    gen_labels: torch.Tensor,
    report_step: int,
    *,
    logit_alignment: str,
    ignore_index: int = IGNORE_INDEX,
) -> float:
    """Geometric mean of FixationStepWiseScoreProvider scores at ``report_step``."""
    lab = gen_labels.detach().long().reshape(-1)
    batch_prov = {"labels": lab.unsqueeze(0)}
    provider = FixationStepWiseScoreProvider(logit_alignment=logit_alignment)
    F = sample_traj["F"]
    if sample_traj.get("lh") is not None:
        lh = sample_traj["lh"]
        b = int(sample_traj["b"])
        F_1 = F.unsqueeze(0) if F.dim() == 1 else F
        lh_1 = [t[b : b + 1] for t in lh]
        fixation_logits = build_effective_step_fixation_logits_from_history(
            lh_1, F_1, int(report_step)
        )
    else:
        R = sample_traj["R"]
        F_1 = F.unsqueeze(0) if F.dim() == 1 else F
        fixation_logits = build_effective_step_fixation_logits(R.unsqueeze(0), F_1, int(report_step))
    model_or_logits = {"fixation_logits": fixation_logits}
    results = provider.get_per_position_scores(
        model_or_logits, batch_prov, ignore_index=ignore_index
    )
    if not results or not results[0][0]:
        return 0.0
    return sequence_probability_from_scores(results[0][0])


def _prob_reported_from_step_values(
    step_values_by_view: Dict[str, Any],
    *,
    view: str,
    traj_name: str,
    step: int,
    batch_pos: int,
) -> float:
    """Read packed/post-loop probability list entry for one batch position."""
    try:
        vals = step_values_by_view[view][traj_name][step]["probability"]
        if batch_pos < len(vals):
            v = vals[batch_pos]
            return float(v) if v is not None else float("nan")
    except (KeyError, TypeError, IndexError):
        pass
    return float("nan")


class ProbTrajectoryDumpCollector:
    """Accumulate per-sample derived diagnostics (cap at ``max_samples``)."""

    def __init__(
        self,
        *,
        max_samples: int,
        trajectory_names: Sequence[str],
        include_views: Sequence[str],
        logit_alignment: str,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.max_samples = int(max_samples)
        self.trajectory_names = list(trajectory_names)
        self.include_views = list(include_views)
        self.logit_alignment = logit_alignment
        self.ignore_index = ignore_index
        self._samples: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []

    def __len__(self) -> int:
        return len(self._order)

    def add_from_batch(
        self,
        *,
        batch_pos: int,
        idx_str: str,
        sample_traj: Dict[str, Any],
        gen_labels: torch.Tensor,
        steps_to_use: Sequence[int],
        effective_length: int,
        fixation_F: torch.Tensor,
        committed_ids: Optional[torch.Tensor],
        step_values_by_view: Dict[str, Any],
        prompt_text: str,
        answer_text: str,
        get_logits_vl_for_step: Any,
    ) -> None:
        """Record one sample after packed probability has filled ``step_values_by_view``."""
        if len(self._order) >= self.max_samples:
            return
        if idx_str in self._samples:
            return
        L = int(gen_labels.numel())
        F_np = fixation_F.detach().cpu().numpy().astype(np.int64).reshape(-1)[:L]
        comm_np: Optional[np.ndarray] = None
        if committed_ids is not None and committed_ids.numel() >= L:
            comm_np = committed_ids.detach().cpu().numpy().astype(np.int64).reshape(-1)[:L]

        traj_data: Dict[str, Any] = {}
        for traj_name in self.trajectory_names:
            traj_data[traj_name] = {}
            for view in self.include_views:
                view_rec: Dict[str, Any] = {"steps": []}
                for step in steps_to_use:
                    logits_vl = get_logits_vl_for_step(traj_name, int(step))
                    if logits_vl is None:
                        continue
                    gl = gen_labels
                    lv = logits_vl
                    if view == "eos":
                        Ls = min(int(effective_length), L)
                        gl = gl[:Ls]
                        lv = lv[:, :Ls]
                    golden, argmax, gold_ids = derive_position_arrays(
                        lv,
                        gl,
                        logit_alignment=self.logit_alignment,
                        ignore_index=self.ignore_index,
                    )
                    device = lv.device
                    prob_packed = prob_packed_shifted_from_step_logits(
                        lv, gl, device, self.ignore_index
                    )
                    prob_provider = prob_fixation_provider_from_sample_traj(
                        sample_traj,
                        gl,
                        int(step),
                        logit_alignment=self.logit_alignment,
                        ignore_index=self.ignore_index,
                    )
                    prob_geom = geom_mean_from_golden_probs(golden)
                    prob_reported = _prob_reported_from_step_values(
                        step_values_by_view,
                        view=view,
                        traj_name=traj_name,
                        step=int(step),
                        batch_pos=batch_pos,
                    )
                    view_rec["steps"].append(
                        {
                            "step": int(step),
                            "golden_probs": golden,
                            "argmax_ids": argmax,
                            "gold_ids": gold_ids,
                            "prob_packed_shifted": prob_packed,
                            "prob_fixation_provider": prob_provider,
                            "prob_geom_mean_golden": prob_geom,
                            "prob_reported": prob_reported,
                        }
                    )
                traj_data[traj_name][view] = view_rec

        self._samples[idx_str] = {
            "idx_str": idx_str,
            "batch_pos": int(batch_pos),
            "prompt_text": prompt_text,
            "answer_text": answer_text,
            "L_gen": L,
            "L_eff": int(effective_length),
            "fixation_F": F_np,
            "committed_ids": comm_np,
            "logit_alignment": self.logit_alignment,
            "trajectories": traj_data,
        }
        self._order.append(idx_str)

    def write(
        self,
        output_dir: Path,
        *,
        manifest_extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Write ``manifest.json`` and ``sample_XXX.npz`` files; return dump root."""
        root = Path(output_dir) / "prob_trajectory_dump"
        root.mkdir(parents=True, exist_ok=True)
        manifest: Dict[str, Any] = {
            "schema_version": 1,
            "n_samples": len(self._order),
            "sample_indices": list(self._order),
            "trajectory_names": self.trajectory_names,
            "include_views": self.include_views,
        }
        if manifest_extra:
            manifest.update(manifest_extra)
        for i, idx_str in enumerate(self._order):
            rec = self._samples[idx_str]
            npz_path = root / f"sample_{i:03d}.npz"
            flat: Dict[str, Any] = {
                "idx_str": np.array(idx_str),
                "prompt_text": np.array(rec["prompt_text"]),
                "answer_text": np.array(rec["answer_text"]),
                "L_gen": np.int64(rec["L_gen"]),
                "L_eff": np.int64(rec["L_eff"]),
                "fixation_F": rec["fixation_F"],
                "logit_alignment": np.array(rec["logit_alignment"]),
            }
            if rec.get("committed_ids") is not None:
                flat["committed_ids"] = rec["committed_ids"]
            for traj_name, views in rec["trajectories"].items():
                for view, vrec in views.items():
                    steps_list = vrec.get("steps") or []
                    step_arr = np.array([s["step"] for s in steps_list], dtype=np.int64)
                    prefix = f"{traj_name}__{view}"
                    flat[f"{prefix}__step_indices"] = step_arr
                    if steps_list:
                        L = int(rec["L_gen"])
                        if view == "eos":
                            L = min(int(rec["L_eff"]), L)
                        n_st = len(steps_list)
                        golden_stack = np.stack(
                            [s["golden_probs"] for s in steps_list], axis=0
                        )
                        argmax_stack = np.stack([s["argmax_ids"] for s in steps_list], axis=0)
                        gold_stack = np.stack([s["gold_ids"] for s in steps_list], axis=0)
                        flat[f"{prefix}__golden_probs"] = golden_stack
                        flat[f"{prefix}__argmax_ids"] = argmax_stack
                        flat[f"{prefix}__gold_ids"] = gold_stack
                        flat[f"{prefix}__prob_packed_shifted"] = np.array(
                            [s["prob_packed_shifted"] for s in steps_list],
                            dtype=np.float64,
                        )
                        flat[f"{prefix}__prob_fixation_provider"] = np.array(
                            [s["prob_fixation_provider"] for s in steps_list],
                            dtype=np.float64,
                        )
                        flat[f"{prefix}__prob_geom_mean_golden"] = np.array(
                            [s["prob_geom_mean_golden"] for s in steps_list],
                            dtype=np.float64,
                        )
                        flat[f"{prefix}__prob_reported"] = np.array(
                            [s["prob_reported"] for s in steps_list],
                            dtype=np.float64,
                        )
            np.savez_compressed(npz_path, **flat)
            manifest[f"sample_{i:03d}_npz"] = npz_path.name
            manifest[f"sample_{i:03d}_idx"] = idx_str
        with open(root / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(
            "prob_trajectory_dump: wrote %s samples to %s",
            len(self._order),
            root,
        )
        return root
