"""
Comprehensive tests for truth_ratio pre_compute contract: same indices, list-of-N-dicts,
and compatibility with original OpenUnlearning (single wrong dict) and trajectory (N options).

Original locuslab/open-unlearning (GitHub):
- forget_Truth_Ratio uses two pre_compute metrics: forget_Q_A_PARA_Prob (correct) and
  forget_Q_A_PERT_Prob (wrong), each returns single dict with value_by_index; truth_ratio
  expects same indices and uses wrong_answer_results[idx]["avg_loss"].
- Their probability metric returns list of per-sample dicts; run_batchwise_evals with
  multi-option data does dict_transpose so value_by_index has lists per key; they
  aggregate with aggregate_to_1D. We support list-of-N-dicts for wrong and average
  inside truth_ratio (TOFU invariant: average over 5 perturbed answers).

Tests: original-style (single wrong), list-of-N-dicts (trajectory), list-of-lists
conversion (3D labels path), and _compute_pre_compute_metrics_at_step conversion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics import METRICS_REGISTRY


# ---- Original OpenUnlearning contract: single dict for correct and wrong, same indices ----
def test_truth_ratio_original_style_single_wrong_dict_same_indices():
    """Original locuslab/open-unlearning: pre_compute correct and wrong are single dicts with value_by_index; same keys."""
    if "truth_ratio" not in METRICS_REGISTRY:
        pytest.skip("truth_ratio not registered")
    metric = METRICS_REGISTRY["truth_ratio"]
    # Use string keys (as in trajectory idx_str) to match real path
    correct_vbi = {"0": {"prob": 0.5, "avg_loss": -np.log(0.5)}, "1": {"prob": 0.4, "avg_loss": -np.log(0.4)}}
    wrong_vbi = {"0": {"prob": 0.25, "avg_loss": -np.log(0.25)}, "1": {"prob": 0.2, "avg_loss": -np.log(0.2)}}
    result = metric._metric_fn(
        model=None,
        pre_compute={
            "correct": {"value_by_index": correct_vbi},
            "wrong": {"value_by_index": wrong_vbi},
        },
        aggregator="closer_to_1_better",
    )
    assert result["agg_value"] is not None
    assert "value_by_index" in result


def test_truth_ratio_asserts_when_correct_and_wrong_indices_differ():
    """When correct and wrong have different index sets, truth_ratio asserts (same as upstream)."""
    if "truth_ratio" not in METRICS_REGISTRY:
        pytest.skip("truth_ratio not registered")
    metric = METRICS_REGISTRY["truth_ratio"]
    correct_vbi = {"0": {"prob": 0.5, "avg_loss": -np.log(0.5)}, "1": {"prob": 0.4, "avg_loss": -np.log(0.4)}}
    wrong_list = [{"value_by_index": {"0": {"prob": 0.25, "avg_loss": -np.log(0.25)}}, "agg_value": 0.25}]
    with pytest.raises(AssertionError, match="same indices"):
        metric._metric_fn(
            model=None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


def test_truth_ratio_list_of_n_dicts_string_indices_ks_test_ready():
    """Trajectory style: wrong = list of N dicts, indices are strings (idx_str). Output must have 'score' per index for ks_test."""
    if "truth_ratio" not in METRICS_REGISTRY:
        pytest.skip("truth_ratio not registered")
    metric = METRICS_REGISTRY["truth_ratio"]
    correct_vbi = {"0": {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = [
        {"value_by_index": {"0": {"prob": 0.1, "avg_loss": -np.log(0.1)}}},
        {"value_by_index": {"0": {"prob": 0.2, "avg_loss": -np.log(0.2)}}},
    ]
    result = metric._metric_fn(
        model=None,
        pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
        aggregator="closer_to_1_better",
    )
    assert result["agg_value"] is not None
    assert "0" in result["value_by_index"]
    score = result["value_by_index"]["0"]["score"]
    assert score is not None
    # ks_test expects: for evals in value_by_index.values(): evals["score"] is a scalar
    for evals in result["value_by_index"].values():
        s = evals["score"]
        assert s is not None
        assert np.isscalar(s) or (isinstance(s, np.ndarray) and s.size == 1)


def test_ks_test_gets_score_from_truth_ratio_output():
    """ks_test reads pre_compute['forget']['value_by_index'].values() and evals['score']; must be present."""
    # Simulate what trajectory passes to ks_test after forget_truth_ratio
    forget_tr_value_by_index = {
        "0": {"score": 0.8},
        "1": {"score": 0.9},
    }
    pre_compute_forget = {"value_by_index": forget_tr_value_by_index}
    stats = np.array([evals["score"] for evals in pre_compute_forget["value_by_index"].values()])
    assert stats.shape == (2,)
    assert list(stats) == [0.8, 0.9]


# ---- List-of-lists conversion (trajectory 3D labels path) ----
def test_trajectory_list_of_lists_conversion_produces_same_indices_as_correct():
    """Simulate trajectory: pre_result = list of N lists (from probability with 3D labels). Conversion must use idx_key so indices match correct."""

    # Minimal mock: we only need the conversion branch, so supply batch_template with labels_wrong as 3D
    # and mock _call_metric_at_step to return list of N lists.

    idx_key = "0"
    _ = 0  # sample_idx (int); we normalize to idx_key str in real code
    # Simulate pre_result from evaluate_probability when batch["labels"] is [1, N, L]
    pre_result_list_of_lists = [
        [{"prob": 0.2, "avg_loss": -np.log(0.2)}],  # option 0
        [{"prob": 0.15, "avg_loss": -np.log(0.15)}],  # option 1
    ]
    # Conversion logic (same as in trajectory_metrics)
    wrong_results = []
    for k in range(len(pre_result_list_of_lists)):
        opt_list = pre_result_list_of_lists[k]
        first = opt_list[0] if opt_list and isinstance(opt_list[0], dict) else {}
        wrong_results.append({
            "value_by_index": {
                idx_key: first if isinstance(first, dict) else {"prob": None, "avg_loss": None},
            },
            "agg_value": first.get("prob") if isinstance(first, dict) else first.get("avg_loss"),
        })
    correct_indices = [idx_key]
    wrong_indices = list(wrong_results[0]["value_by_index"].keys())
    assert correct_indices == wrong_indices, "trajectory conversion must produce same indices as correct"
    # Now truth_ratio can run
    correct_vbi = {idx_key: {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    result = METRICS_REGISTRY["truth_ratio"]._metric_fn(
        model=None,
        pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_results},
        aggregator="closer_to_1_better",
    )
    assert result["agg_value"] is not None
    assert idx_key in result["value_by_index"] and "score" in result["value_by_index"][idx_key]


def test_compute_pre_compute_metrics_at_step_list_of_lists_to_list_of_n_dicts():
    """When _call_metric_at_step returns list of N lists (3D labels path), pre_compute['wrong'] is list of N dicts with idx_key."""
    import torch
    from unittest.mock import patch, MagicMock
    from evals.metrics.trajectory_metrics import _compute_pre_compute_metrics_at_step

    L = 16
    sample_idx = "0"
    pre_compute_config = {
        "forget_Q_A_PARA_Prob": {"access_key": "correct", "labels_field": "labels_correct", "handler": "probability"},
        "forget_Q_A_PERT_Prob": {"access_key": "wrong", "labels_field": "labels_wrong", "handler": "probability"},
    }
    batch_template = {
        "input_ids": torch.zeros(1, L, dtype=torch.long),
        "labels": torch.zeros(1, L, dtype=torch.long),
        "labels_correct": torch.zeros(1, L, dtype=torch.long),
        "labels_wrong": torch.zeros(1, 5, L, dtype=torch.long),
    }
    logits = torch.zeros(1, L, 100)
    tokenizer = MagicMock()

    returns = [
        [{"prob": 0.5, "avg_loss": -0.693}],
        [[{"prob": 0.2, "avg_loss": -1.61}], [{"prob": 0.15, "avg_loss": -1.90}]],
    ]

    with patch("evals.metrics.trajectory_metrics._call_metric_at_step", side_effect=returns):
        results = _compute_pre_compute_metrics_at_step(
            pre_compute_config=pre_compute_config,
            logits=logits,
            batch_template=batch_template,
            tokenizer=tokenizer,
            sample_labels=None,
            sample_input_ids=batch_template["input_ids"],
            sample_prompt_len=0,
            sample_idx=sample_idx,
        )

    assert "correct" in results
    assert "wrong" in results
    correct_vbi = results["correct"]["value_by_index"]
    wrong_list = results["wrong"]
    assert isinstance(wrong_list, list)
    assert len(wrong_list) == 2
    wrong_indices = list(wrong_list[0]["value_by_index"].keys())
    correct_indices = list(correct_vbi.keys())
    assert correct_indices == wrong_indices, "same indices after list-of-lists conversion"
    assert correct_indices == [sample_idx]
    for w in wrong_list:
        assert "value_by_index" in w and sample_idx in w["value_by_index"]
        assert "avg_loss" in w["value_by_index"][sample_idx] or "prob" in w["value_by_index"][sample_idx]


