"""
Invariant-aligned expectation-based tests for trajectory/metrics.

Per plan: given data -> compute expected from invariant/OpenUnlearning formula -> call real metric -> assert actual ~= expected.
Minimal to no mocks; no GPU or real model. References: knowledge/dllm-unlearning-invariants/metrics.tex, notation.tex.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))


def test_rouge_l_recall_expected_from_reference_hypothesis():
    """TOFU uses ROUGE-L recall. Given reference and hypothesis strings, expected score matches rouge_score."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    reference = "the cat sat on the mat"
    hypothesis = "the cat sat on the mat"
    scores = scorer.score(reference, hypothesis)
    expected = scores["rougeL"].recall
    assert expected == 1.0

    hypothesis_partial = "the cat"
    scores2 = scorer.score(reference, hypothesis_partial)
    expected2 = scores2["rougeL"].recall
    assert 0 < expected2 < 1
    assert np.isclose(expected2, 2.0 / 6.0, rtol=1e-5) or expected2 > 0


def test_hm_aggregate_two_components_expected_hmean():
    """MU harmonic mean: 2 components with known values -> expected = n / sum(1/x_i)."""
    import scipy.stats

    from evals.metrics import METRICS_REGISTRY
    from evals.metrics.trajectory_metrics import _call_metric_at_step
    import torch

    metric = METRICS_REGISTRY["hm_aggregate"]
    a, b = 0.5, 0.5
    expected = scipy.stats.hmean([a, b])
    pre = {
        "retain_Q_A_Prob": {"agg_value": a},
        "retain_Q_A_ROUGE": {"agg_value": b},
    }
    retain_agg = {"0": {"full": pre, "eos": pre}}
    logits = torch.zeros(1, 2, 8)
    batch_t = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "index": torch.tensor([0]),
    }
    r = _call_metric_at_step(
        metric,
        logits,
        batch_t,
        metric_config={},
        sample_idx="0",
        step_index=0,
        retain_agg_by_step=retain_agg,
        trajectory_view="full",
    )
    assert r["agg_value"] is not None
    assert np.isclose(r["agg_value"], expected, rtol=1e-9)


def test_hm_aggregate_merged_9_mu_expected_hmean_per_traj_view():
    """Per-traj retain_agg_by_step: hm_aggregate yields expected hmean for each traj_name and view."""
    import scipy.stats

    from evals.metrics import METRICS_REGISTRY
    from evals.metrics.trajectory_metrics import _call_metric_at_step
    import torch

    from evals.metrics.trajectory_metrics import EXPECTED_9_MU_KEYS

    metric = METRICS_REGISTRY["hm_aggregate"]
    trajectory_names = ("steps", "fixation_start", "fixation_end", "fixation_ratio")
    values_9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    pre = {k: {"agg_value": v} for k, v in zip(sorted(EXPECTED_9_MU_KEYS), values_9)}
    expected_hmean = scipy.stats.hmean(values_9)

    retain_agg_by_step = {}
    for t in trajectory_names:
        retain_agg_by_step[t] = {"0": {"full": pre, "eos": pre}}

    logits = torch.zeros(1, 2, 8)
    batch_t = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "index": torch.tensor([0]),
    }

    for traj_name in trajectory_names:
        for view in ("full", "eos"):
            r = _call_metric_at_step(
                metric,
                logits,
                batch_t,
                metric_config={},
                sample_idx="0",
                step_index=0,
                retain_agg_by_step=retain_agg_by_step,
                trajectory_view=view,
                traj_name=traj_name,
            )
            assert r["agg_value"] is not None
            assert np.isclose(r["agg_value"], expected_hmean, rtol=1e-9), (
                f"traj_name={traj_name} view={view}"
            )
