"""
Layout-specific truth_ratio pre_compute: nested probability must use trajectory-sliced logits
when ``traj_name`` is set (forget loop), so correct probs (and TR) differ across steps /
fixation_start / fixation_end / fixation_ratio when R varies by diffusion step.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from data.utils import IGNORE_INDEX


def _synthetic_rf_and_labels(*, V: int = 8, L: int = 6, S: int = 5):
    """
    R[v,l,s]: for s<=2 favor target token 1; for s>=3 favor token 2.
    F ascending so fixation_start at report s=3 mixes low-s (token 1) and high-s (token 2) columns.
    """
    R = torch.full((V, L, S), -10.0)
    for s in range(S):
        if s <= 2:
            R[1, :, s] = 10.0
        else:
            R[2, :, s] = 10.0
    F = torch.tensor([0, 1, 2, 3, 4, 4], dtype=torch.long)
    sample_traj = {"R": R, "F": F, "S": S, "L": L}
    labels_correct = torch.full((1, L), IGNORE_INDEX, dtype=torch.long)
    labels_correct[0, 1:] = 1
    labels_wrong = [
        torch.full((1, L), IGNORE_INDEX, dtype=torch.long),
    ]
    labels_wrong[0][0, 1:] = 2
    batch_template = {
        "input_ids": torch.zeros(1, L, dtype=torch.long),
        "labels": labels_correct.clone(),
        "labels_correct": labels_correct,
        "labels_wrong": labels_wrong,
    }
    return sample_traj, batch_template


@pytest.mark.parametrize("step", [3])
def test_pre_compute_correct_prob_differs_across_traj_names(step: int):
    from omegaconf import OmegaConf

    from evals.metrics.trajectory_metrics import (
        _compute_pre_compute_metrics_at_step,
        _get_logits_at_step,
    )

    sample_traj, batch_template = _synthetic_rf_and_labels()
    trajectory_config = OmegaConf.create(
        {
            "use_generalized_sequence_probability": True,
            "logit_alignment": "causal",
        }
    )
    pre_compute_config = OmegaConf.create(
        {
            "correct": {
                "access_key": "correct",
                "labels_field": "labels_correct",
                "handler": "probability",
            },
            "wrong": {
                "access_key": "wrong",
                "labels_field": "labels_wrong",
                "handler": "probability",
            },
        }
    )
    traj_names = ("steps", "fixation_start", "fixation_end", "fixation_ratio")
    correct_probs = []
    for traj_name in traj_names:
        logits_vl = _get_logits_at_step(sample_traj, traj_name, step)
        logits_1lv = logits_vl.t().unsqueeze(0)
        results = _compute_pre_compute_metrics_at_step(
            pre_compute_config=pre_compute_config,
            logits=logits_1lv,
            batch_template=batch_template,
            tokenizer=None,
            sample_labels=batch_template["labels"],
            sample_input_ids=batch_template["input_ids"],
            sample_prompt_len=0,
            sample_idx="0",
            trajectory_config=trajectory_config,
            sample_traj=sample_traj,
            step=step,
            traj_name=traj_name,
        )
        p = results["correct"]["value_by_index"]["0"]["prob"]
        assert p is not None, traj_name
        correct_probs.append(float(p))

    assert len(set(round(p, 6) for p in correct_probs)) > 1, correct_probs


def test_truth_ratio_agg_value_differs_across_traj_names():
    from omegaconf import OmegaConf

    from evals.metrics import METRICS_REGISTRY
    from evals.metrics.trajectory_metrics import (
        _call_metric_at_step,
        _get_logits_at_step,
    )

    sample_traj, batch_template = _synthetic_rf_and_labels()
    trajectory_config = OmegaConf.create(
        {
            "use_generalized_sequence_probability": True,
            "logit_alignment": "causal",
        }
    )
    metric_cfg = OmegaConf.create(
        {
            "aggregator": "closer_to_1_better",
            "pre_compute": {
                "correct": {
                    "handler": "probability",
                    "access_key": "correct",
                    "labels_field": "labels_correct",
                },
                "wrong": {
                    "handler": "probability",
                    "access_key": "wrong",
                    "labels_field": "labels_wrong",
                },
            },
        }
    )
    truth_metric = METRICS_REGISTRY["truth_ratio"]
    step = 3
    agg_values = []
    for traj_name in ("steps", "fixation_start", "fixation_end", "fixation_ratio"):
        logits_vl = _get_logits_at_step(sample_traj, traj_name, step)
        out = _call_metric_at_step(
            metric=truth_metric,
            logits=logits_vl,
            batch_template=batch_template,
            tokenizer=None,
            sample_labels=batch_template["labels"],
            sample_input_ids=batch_template["input_ids"],
            sample_prompt_len=0,
            metric_config=metric_cfg,
            sample_idx="0",
            trajectory_config=trajectory_config,
            sample_traj=sample_traj,
            step=step,
            traj_name=traj_name,
        )
        assert out.get("agg_value") is not None
        assert not math.isnan(out["agg_value"])
        agg_values.append(float(out["agg_value"]))

    assert len(set(round(v, 5) for v in agg_values)) > 1, agg_values


def test_without_traj_name_still_uses_generalized_rf_path():
    """When traj_name is omitted, nested probability uses FixationStepWiseScoreProvider (full R,F)."""
    from omegaconf import OmegaConf

    from evals.metrics.trajectory_metrics import _compute_pre_compute_metrics_at_step

    sample_traj, batch_template = _synthetic_rf_and_labels()
    trajectory_config = OmegaConf.create(
        {
            "use_generalized_sequence_probability": True,
            "logit_alignment": "causal",
        }
    )
    pre_compute_config = OmegaConf.create(
        {
            "correct": {
                "access_key": "correct",
                "labels_field": "labels_correct",
                "handler": "probability",
            },
        }
    )
    step = 3
    results = _compute_pre_compute_metrics_at_step(
        pre_compute_config=pre_compute_config,
        logits=torch.zeros(1, 6, 8),
        batch_template=batch_template,
        tokenizer=None,
        sample_labels=batch_template["labels"],
        sample_input_ids=batch_template["input_ids"],
        sample_prompt_len=0,
        sample_idx="0",
        trajectory_config=trajectory_config,
        sample_traj=sample_traj,
        step=step,
    )
    p = results["correct"]["value_by_index"]["0"]["prob"]
    assert p is not None
    assert p > 0
