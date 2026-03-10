"""Unit tests for _scalar_metrics_for_wandb (W&B log only scalar aggregates)."""

import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from trainer.base import _scalar_metrics_for_wandb


def test_simple_agg_value():
    """Dict with scalar agg_value is logged as key -> float."""
    metrics = {
        "forget_Q_A_Prob": {"agg_value": 0.42, "value_by_index": {"0": 0.4, "1": 0.44}},
    }
    out = _scalar_metrics_for_wandb(metrics)
    assert out == {"forget_Q_A_Prob": 0.42}


def test_nested_trajectory_agg_value():
    """Dict with nested agg_value (view -> traj -> metric -> array) reduces to mean over steps."""
    metrics = {
        "trajectory_forget_Q_A_Prob": {
            "agg_value": {
                "full": {
                    "steps": {"probability": np.array([0.1, 0.2, 0.3, 0.4])},
                },
            },
            "value_by_index": {},
            "step_distribution": {},
        },
    }
    out = _scalar_metrics_for_wandb(metrics)
    assert "trajectory_forget_Q_A_Prob" in out
    assert out["trajectory_forget_Q_A_Prob"] == 0.25  # mean(0.1, 0.2, 0.3, 0.4)


def test_lmeval_flat_scalars():
    """LMEval-style top-level scalars pass through."""
    metrics = {
        "mmlu/acc": 0.65,
        "truthfulqa/mc1": 0.72,
    }
    out = _scalar_metrics_for_wandb(metrics)
    assert out == {"mmlu/acc": 0.65, "truthfulqa/mc1": 0.72}


def test_skip_keys_excluded():
    """config, run_info, trajectory_step_metadata, mia_min_k_by_step, forget_truth_ratio_by_step are not logged."""
    metrics = {
        "config": {"model": "test"},
        "run_info": {"world_size": 1},
        "trajectory_step_metadata": {"agg_value": None, "trajectory_step_metadata": {}},
        "mia_min_k_by_step": {"0": {"agg_value": 0.5}},
        "forget_truth_ratio_by_step": {"0": {"value_by_index": {}}},
        "forget_truth_ratio": {"agg_value": 0.8},
    }
    out = _scalar_metrics_for_wandb(metrics)
    assert "config" not in out
    assert "run_info" not in out
    assert "trajectory_step_metadata" not in out
    assert "mia_min_k_by_step" not in out
    assert "forget_truth_ratio_by_step" not in out
    assert out["forget_truth_ratio"] == 0.8


def test_no_nested_structures_or_skip_keys_in_output():
    """Output contains only flat key -> float; no value_by_index or step_distribution."""
    metrics = {
        "a": {"agg_value": 1.0, "value_by_index": {"0": 1.0}, "step_distribution": {"mean": [0.5]}},
        "b": {"agg_value": {"full": {"steps": {"m": np.array([2.0, 4.0])}}}, "value_by_index": {}},
    }
    out = _scalar_metrics_for_wandb(metrics)
    assert out == {"a": 1.0, "b": 3.0}
    for v in out.values():
        assert isinstance(v, float)
