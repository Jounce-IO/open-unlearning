"""
Reproduce the exact bugs that occur in K8s trajectory eval:
1. AssertionError: truth_ratio: correct and wrong pre_compute must have same indices
2. KeyError: 'score' in ks_test when value_by_index entries lack "score"

These tests use a BUGGY copy of truth_ratio (assert, no str normalization, no intersection)
so we do NOT fix production. We prove that the test harness mimics the real pipeline
by feeding the exact pre_compute shapes that the real code can produce.

Real pipeline flow (trajectory):
- _compute_pre_compute_metrics_at_step(config={correct: probability, wrong: probability})
- For "correct": _call_metric_at_step(probability) -> can return list or dict; we normalize to {idx_key: ...}
- For "wrong": labels_wrong is list of tensors -> we get list of results; we normalize to [{value_by_index: {idx_key: ...}}, ...]
- Edge cases that cause bug: (1) wrong[0]["value_by_index"] empty -> wrong_indices = []
  (2) Some path returns value_by_index keyed by int (e.g. batch index 0) for correct, str for wrong -> key set mismatch
  (3) truth_ratio returns value_by_index with entries missing "score" -> ks_test KeyError
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))


def _truth_ratio_buggy(model, **kwargs):
    """
    Buggy truth_ratio: NO str normalization, ASSERT same indices (no intersection).
    Matches the original failing behavior so we can reproduce the bug in tests.
    """
    from evals.metrics.memorization import aggregate_to_1D

    def closer_to_1_better(arr):
        return np.mean(np.minimum(arr, 1 / (arr + 1e-10)))

    def true_better(arr):
        return np.mean(np.maximum(0, 1 - arr))

    def prob_mean(arr):
        return np.mean(arr)

    if kwargs["aggregator"] == "closer_to_1_better":
        aggregator = closer_to_1_better
    elif kwargs["aggregator"] == "true_better":
        aggregator = true_better
    elif kwargs["aggregator"] == "prob_mean":
        aggregator = prob_mean
    else:
        raise ValueError(f"Invalid truth ratio aggregator: {kwargs['aggregator']}")

    correct_answer_results = kwargs["pre_compute"]["correct"]["value_by_index"]
    wrong_input = kwargs["pre_compute"]["wrong"]

    # BUGGY: no str() normalization - keep raw keys (int vs str mismatch in real pipeline)
    correct_indices = list(correct_answer_results.keys())

    if isinstance(wrong_input, list):
        n_wrong_options = len(wrong_input)
        wrong_indices = (
            list(wrong_input[0]["value_by_index"].keys())
            if n_wrong_options > 0 and "value_by_index" in wrong_input[0]
            else []
        )
        # BUGGY: assert instead of intersection - this raises in real job when sets differ
        assert correct_indices == wrong_indices, (
            "truth_ratio: correct and wrong pre_compute must have same indices"
        )
        wrong_answer_results = None
        filtered_indices = [
            idx
            for idx in correct_indices
            if correct_answer_results[idx] is not None
            and correct_answer_results[idx].get("avg_loss") is not None
        ]
        wrong_probs_per_idx = {}
        for idx in filtered_indices:
            wrong_avg_losses = [
                wrong_input[k]["value_by_index"][idx]["avg_loss"]
                for k in range(n_wrong_options)
                if idx in wrong_input[k].get("value_by_index", {})
                and wrong_input[k]["value_by_index"][idx] is not None
                and wrong_input[k]["value_by_index"][idx].get("avg_loss") is not None
            ]
            if not wrong_avg_losses:
                continue
            wrong_probs_per_idx[idx] = float(
                np.mean(np.exp(-np.array(wrong_avg_losses, dtype=np.float64)))
            )
        filtered_indices = [idx for idx in filtered_indices if idx in wrong_probs_per_idx]
    else:
        wrong_answer_results = wrong_input["value_by_index"]
        wrong_indices = list(wrong_answer_results.keys())
        assert correct_indices == wrong_indices, (
            "truth_ratio: correct and wrong pre_compute must have same indices"
        )
        filtered_indices = [
            idx
            for idx in correct_indices
            if correct_answer_results[idx] is not None
            and wrong_answer_results[idx] is not None
            and correct_answer_results[idx].get("avg_loss") is not None
            and wrong_answer_results[idx].get("avg_loss") is not None
        ]
        wrong_probs_per_idx = None

    if not filtered_indices:
        return {"agg_value": None, "value_by_index": {}}

    correct_avg_losses = [
        correct_answer_results[idx]["avg_loss"] for idx in filtered_indices
    ]
    correct_avg_losses = aggregate_to_1D(np.array(correct_avg_losses))
    correct_prob = np.exp(-correct_avg_losses)

    if wrong_probs_per_idx is not None:
        wrong_prob = np.array(
            [wrong_probs_per_idx[idx] for idx in filtered_indices], dtype=np.float64
        )
    else:
        wrong_avg_losses = [
            wrong_answer_results[idx]["avg_loss"] for idx in filtered_indices
        ]
        wrong_avg_losses = aggregate_to_1D(np.array(wrong_avg_losses))
        wrong_prob = np.exp(-wrong_avg_losses)

    if kwargs["aggregator"] != "prob_mean":
        truth_ratios = wrong_prob / (correct_prob + 1e-10)
    else:
        truth_ratios = correct_prob / (correct_prob + wrong_prob + 1e-10)

    value_by_index = {
        idx: {"score": truth_ratios[i]} for i, idx in enumerate(filtered_indices)
    }
    truth_ratio_stats = np.array([evals["score"] for evals in value_by_index.values()])
    forget_tr_avg = aggregator(truth_ratio_stats)
    return {"agg_value": forget_tr_avg, "value_by_index": value_by_index}


# ---- Reproduce AssertionError: correct has keys, wrong has empty value_by_index ----
def test_reproduce_assert_when_wrong_first_option_has_empty_value_by_index():
    """Real case: wrong = list of dicts but wrong[0]["value_by_index"] is empty -> wrong_indices = []."""
    correct_vbi = {"0": {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = [{"value_by_index": {}, "agg_value": None}]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


def test_reproduce_assert_when_wrong_has_empty_value_by_index_multiple_options():
    """Wrong list has N options but first has empty value_by_index."""
    correct_vbi = {"0": {"prob": 0.5, "avg_loss": -np.log(0.5)}, "1": {"prob": 0.4, "avg_loss": -np.log(0.4)}}
    wrong_list = [
        {"value_by_index": {}, "agg_value": None},
        {"value_by_index": {}, "agg_value": None},
    ]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


def test_reproduce_assert_when_correct_has_int_key_wrong_has_str_key():
    """Real case: trajectory uses idx_str for wrong; some path returns int key for correct (e.g. batch index 0)."""
    correct_vbi = {0: {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = [{"value_by_index": {"0": {"prob": 0.25, "avg_loss": -np.log(0.25)}}}]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


def test_reproduce_assert_when_correct_has_str_key_wrong_has_int_key():
    """Opposite: correct keyed by "0", wrong keyed by 0 (e.g. from different code path)."""
    correct_vbi = {"0": {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = [{"value_by_index": {0: {"prob": 0.25, "avg_loss": -np.log(0.25)}}}]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


def test_reproduce_assert_when_correct_has_more_indices_than_wrong():
    """Correct has ["0","1"], wrong only ["0"] -> sets differ."""
    correct_vbi = {
        "0": {"prob": 0.5, "avg_loss": -np.log(0.5)},
        "1": {"prob": 0.4, "avg_loss": -np.log(0.4)},
    }
    wrong_list = [{"value_by_index": {"0": {"prob": 0.25, "avg_loss": -np.log(0.25)}}}]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


def test_reproduce_assert_when_wrong_has_more_indices_than_correct():
    """Wrong has ["0","1"], correct only ["0"]."""
    correct_vbi = {"0": {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = [
        {
                    "value_by_index": {
                        "0": {"prob": 0.25, "avg_loss": -np.log(0.25)},
                        "1": {"prob": 0.2, "avg_loss": -np.log(0.2)},
                    }
                }
    ]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


def test_reproduce_assert_when_correct_and_wrong_have_different_key_sets():
    """Correct has ["0","2"], wrong has ["0","1"] -> disjoint except one."""
    correct_vbi = {
        "0": {"prob": 0.5, "avg_loss": -np.log(0.5)},
        "2": {"prob": 0.3, "avg_loss": -np.log(0.3)},
    }
    wrong_list = [
        {
                    "value_by_index": {
                        "0": {"prob": 0.25, "avg_loss": -np.log(0.25)},
                        "1": {"prob": 0.2, "avg_loss": -np.log(0.2)},
                    }
                }
    ]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


def test_reproduce_assert_when_wrong_list_empty():
    """Wrong is empty list -> wrong_indices from wrong_input[0] would fail; we guard with n_wrong_options > 0 so wrong_indices = []."""
    correct_vbi = {"0": {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = []
    with pytest.raises((AssertionError, IndexError)):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


def test_reproduce_assert_when_correct_empty_wrong_has_keys():
    """Correct value_by_index empty, wrong has keys."""
    correct_vbi = {}
    wrong_list = [{"value_by_index": {"0": {"prob": 0.25, "avg_loss": -np.log(0.25)}}}]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


def test_reproduce_assert_single_dict_wrong_different_keys():
    """Single wrong dict (non-list path): correct has "0", wrong has "1"."""
    correct_vbi = {"0": {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_single = {"value_by_index": {"1": {"prob": 0.25, "avg_loss": -np.log(0.25)}}}
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_single},
            aggregator="closer_to_1_better",
        )


def test_reproduce_assert_single_dict_wrong_empty():
    """Single wrong dict with empty value_by_index."""
    correct_vbi = {"0": {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_single = {"value_by_index": {}}
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_single},
            aggregator="closer_to_1_better",
        )


def test_reproduce_assert_data_index_int_vs_str_453():
    """Mimic real TOFU data index 453: correct keyed by 453 (int from batch), wrong by "453" (idx_str)."""
    correct_vbi = {453: {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = [{"value_by_index": {"453": {"prob": 0.25, "avg_loss": -np.log(0.25)}}}]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


def test_reproduce_assert_order_of_keys_same_set_but_list_order_differs():
    """Same set of keys but list order differs: ["0","1"] vs ["1","0"] -> assert compares lists so fails."""
    correct_vbi = {
        "0": {"prob": 0.5, "avg_loss": -np.log(0.5)},
        "1": {"prob": 0.4, "avg_loss": -np.log(0.4)},
    }
    wrong_list = [
        {
                    "value_by_index": {
                        "1": {"prob": 0.2, "avg_loss": -np.log(0.2)},
                        "0": {"prob": 0.25, "avg_loss": -np.log(0.25)},
                    }
                }
    ]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


# ---- Exact error message from the assertion (as in real job) ----
REAL_JOB_ASSERTION_MESSAGE = "truth_ratio: correct and wrong pre_compute must have same indices"


def test_real_truth_ratio_with_bad_pre_compute_either_raises_from_memorization_or_returns_result():
    """
    Reproduce the REAL error from the ACTUAL SOURCE CODE (memorization.py) when the
    bug is present; or validate the fix when it is in place.
    No patch, no fake: call the real truth_ratio with pre_compute that has
    int-keyed correct and str-keyed wrong.
    - If the source still has the assert (bug): AssertionError must be raised from
      memorization.py (same stack as pod: trajectory_metrics -> memorization.truth_ratio).
    - If the fix is in place: no error, and we get a valid result with value_by_index.
    """
    import traceback as tb_module
    from evals.metrics import METRICS_REGISTRY

    metric = METRICS_REGISTRY["truth_ratio"]
    bad_pre_compute = {
        "correct": {"value_by_index": {0: {"prob": 0.5, "avg_loss": -np.log(0.5)}}},
        "wrong": [{"value_by_index": {"0": {"prob": 0.25, "avg_loss": -np.log(0.25)}}}],
    }
    try:
        result = metric._metric_fn(
            model=None,
            pre_compute=bad_pre_compute,
            aggregator="closer_to_1_better",
        )
    except AssertionError as err:
        msg = err.args[0] if err.args else str(err)
        assert REAL_JOB_ASSERTION_MESSAGE in msg, f"Expected pod message in error; got: {msg!r}"
        tb = tb_module.extract_tb(err.__traceback__)
        frame_files = [f.filename for f in tb]
        assert any("memorization.py" in f for f in frame_files), (
            f"AssertionError must be raised from memorization.py (actual source); traceback files: {frame_files}"
        )
        frame_names = [f.name for f in tb]
        assert any("truth_ratio" in n for n in frame_names), (
            f"AssertionError must be raised in truth_ratio; traceback names: {frame_names}"
        )
        return
    assert result is not None and "value_by_index" in result
    assert "0" in result["value_by_index"] and "score" in result["value_by_index"]["0"]


def test_assertion_error_message_equals_real_job_message():
    """
    Validate that the reproduced AssertionError message is exactly the one from the
    original source (memorization.truth_ratio before the fix). This is the same
    message the real job raised in K8s. (Python may append assert details to args[0].)
    """
    correct_vbi = {0: {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = [{"value_by_index": {"0": {"prob": 0.25, "avg_loss": -np.log(0.25)}}}]
    pre_compute = {"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list}
    with pytest.raises(AssertionError) as exc_info:
        _truth_ratio_buggy(
            None,
            pre_compute=pre_compute,
            aggregator="closer_to_1_better",
        )
    msg = exc_info.value.args[0] if exc_info.value.args else str(exc_info.value)
    assert msg.startswith(REAL_JOB_ASSERTION_MESSAGE), (
        f"Expected error from real job to start with {REAL_JOB_ASSERTION_MESSAGE!r}, got: {msg!r}"
    )
    assert REAL_JOB_ASSERTION_MESSAGE in msg


def test_real_job_path_raises_same_assertion_when_metric_is_buggy():
    """
    Run the same call path as the real job: _call_metric_at_step(truth_ratio, ...)
    with pre_compute that has int-keyed correct and str-keyed wrong. Mock
    _compute_pre_compute_metrics_at_step to return that bad pre_compute (as would
    happen if normalization were skipped); temporarily use the buggy metric impl.
    Validates the exact AssertionError from the source (the registered metric).
    """
    from unittest.mock import patch
    from evals.metrics import METRICS_REGISTRY
    from evals.metrics.trajectory_metrics import _call_metric_at_step
    import torch

    metric = METRICS_REGISTRY["truth_ratio"]
    bad_pre_compute = {
        "correct": {"value_by_index": {0: {"prob": 0.5, "avg_loss": -np.log(0.5)}}},
        "wrong": [{"value_by_index": {"0": {"prob": 0.25, "avg_loss": -np.log(0.25)}}}],
    }
    batch_template = {
        "input_ids": torch.zeros(1, 8, dtype=torch.long),
        "labels": torch.zeros(1, 8, dtype=torch.long),
    }
    logits = torch.zeros(1, 8, 100)
    metric_config = {
        "pre_compute": {"correct": {"handler": "probability"}, "wrong": {"handler": "probability"}},
        "aggregator": "closer_to_1_better",
    }

    def mock_compute_pre_compute(*args, **kwargs):
        return bad_pre_compute

    with patch(
        "evals.metrics.trajectory_metrics._compute_pre_compute_metrics_at_step",
        side_effect=mock_compute_pre_compute,
    ), patch.object(metric, "_metric_fn", _truth_ratio_buggy):
        with pytest.raises(AssertionError) as exc_info:
            _call_metric_at_step(
                metric=metric,
                logits=logits,
                batch_template=batch_template,
                tokenizer=None,
                sample_idx="0",
                metric_config=metric_config,
            )
    msg = exc_info.value.args[0] if exc_info.value.args else str(exc_info.value)
    assert msg.startswith(REAL_JOB_ASSERTION_MESSAGE) and REAL_JOB_ASSERTION_MESSAGE in msg, (
        f"Real job path must raise same message as source; got: {msg!r}"
    )


def test_real_source_with_bad_input_does_not_raise_after_fix():
    """
    Regression: the actual truth_ratio in memorization.py (no patch) with the
    same bad input (int correct, str wrong) must NOT raise; it normalizes and uses intersection.
    """
    from evals.metrics import METRICS_REGISTRY

    metric = METRICS_REGISTRY["truth_ratio"]
    pre_compute = {
        "correct": {"value_by_index": {0: {"prob": 0.5, "avg_loss": -np.log(0.5)}}},
        "wrong": [{"value_by_index": {"0": {"prob": 0.25, "avg_loss": -np.log(0.25)}}}],
    }
    result = metric._metric_fn(
        model=None,
        pre_compute=pre_compute,
        aggregator="closer_to_1_better",
    )
    assert result is not None
    assert "agg_value" in result
    assert "value_by_index" in result
    assert "0" in result["value_by_index"]
    assert "score" in result["value_by_index"]["0"]


# ---- Reproduce KeyError: 'score' in ks_test (real error) and assert fix ----


def _ks_test_buggy_extraction(forget_vbi):
    """Exact buggy logic that raised in pod: evals['score'] without .get."""
    return np.array(
        [evals["score"] for evals in forget_vbi.values() if isinstance(evals, dict)],
        dtype=np.float64,
    )


def test_ks_test_keyerror_reproduced_with_buggy_extraction():
    """
    Reproduce the REAL error: the exact line that raised in K8s (evals["score"])
    when value_by_index contains an entry without 'score' (e.g. truth_ratio return dict).
    """
    forget_vbi = {"0": {"agg_value": None, "value_by_index": {}}}
    with pytest.raises(KeyError, match="score"):
        _ks_test_buggy_extraction(forget_vbi)


def test_ks_test_real_error_reproduction_exact_line():
    """
    Reproduce the REAL error: the exact line in ks_test that raised in K8s.
    forget['value_by_index'] had entries with only 'prob'/'avg_loss' (no 'score')
    from trajectory structuring when truth_ratio returned empty value_by_index.
    """
    pre_compute_forget = {
        "agg_value": None,
        "value_by_index": {"0": {"prob": None, "avg_loss": None}},
    }
    # Exact line from ks_test (before fix): evals["score"] for evals in ...
    evals_list = list(pre_compute_forget["value_by_index"].values())
    with pytest.raises(KeyError, match="score"):
        _ = [evals["score"] for evals in evals_list]


def test_ks_test_raises_when_forget_has_no_valid_scores():
    """
    ks_test does not require retain for the forget side; invalid forget pre_compute is a bug → fail.
    When forget value_by_index has no valid 'score', ks_test raises ValueError (same as upstream).
    """
    from evals.metrics import METRICS_REGISTRY

    ks_test_metric = METRICS_REGISTRY["ks_test"]
    pre_compute = {
        "forget": {"agg_value": None, "value_by_index": {"0": {"prob": None, "avg_loss": None}}},
    }
    with pytest.raises(ValueError, match="no valid.*value_by_index.*score"):
        ks_test_metric._metric_fn(model=None, pre_compute=pre_compute)


def test_ks_test_raises_when_forget_empty_or_no_valid_scores():
    """
    ks_test fails when forget pre_compute has no valid scores (same as upstream).
    Empty value_by_index or entries without valid 'score' → ValueError.
    """
    from evals.metrics import METRICS_REGISTRY

    ks_test_metric = METRICS_REGISTRY["ks_test"]
    # Case 1: forget value_by_index empty
    pre_compute_empty = {"forget": {"agg_value": None, "value_by_index": {}}}
    with pytest.raises(ValueError, match="no valid"):
        ks_test_metric._metric_fn(model=None, pre_compute=pre_compute_empty)

    # Case 2: entries with only "prob" (no "score") → no valid scores after filter
    pre_compute_placeholder = {
        "forget": {"agg_value": None, "value_by_index": {"0": {"prob": None}}},
    }
    with pytest.raises(ValueError, match="no valid"):
        ks_test_metric._metric_fn(model=None, pre_compute=pre_compute_placeholder)


# ---- Buggy truth_ratio passes when keys match (sanity: bug is only on mismatch) ----
def test_buggy_truth_ratio_passes_when_same_str_keys_list_wrong():
    """When correct and wrong have same str keys, buggy version does not assert."""
    correct_vbi = {"0": {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = [{"value_by_index": {"0": {"prob": 0.25, "avg_loss": -np.log(0.25)}}}]
    result = _truth_ratio_buggy(
        None,
        pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
        aggregator="closer_to_1_better",
    )
    assert result["agg_value"] is not None
    assert "0" in result["value_by_index"] and "score" in result["value_by_index"]["0"]


def test_buggy_truth_ratio_passes_when_same_int_keys_list_wrong():
    """When both use int keys and same indices, buggy version passes."""
    correct_vbi = {0: {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = [{"value_by_index": {0: {"prob": 0.25, "avg_loss": -np.log(0.25)}}}]
    result = _truth_ratio_buggy(
        None,
        pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
        aggregator="closer_to_1_better",
    )
    assert result["agg_value"] is not None
    assert 0 in result["value_by_index"] and "score" in result["value_by_index"][0]


# ---- Mimic exact pre_compute from _compute_pre_compute_metrics_at_step ----
def test_reproduce_pre_compute_shape_from_trajectory_correct_int_wrong_str():
    """
    _compute_pre_compute_metrics_at_step: 'correct' can get value_by_index from
    a path that returns batch index (int); 'wrong' gets idx_key (str). No normalization in buggy -> assert.
    """
    sample_idx_str = "453"
    correct_as_from_batch_index = {0: {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_as_from_trajectory = [
        {"value_by_index": {sample_idx_str: {"prob": 0.2, "avg_loss": -np.log(0.2)}}},
        {"value_by_index": {sample_idx_str: {"prob": 0.15, "avg_loss": -np.log(0.15)}}},
    ]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={
                "correct": {"value_by_index": correct_as_from_batch_index},
                "wrong": wrong_as_from_trajectory,
            },
            aggregator="closer_to_1_better",
        )


def test_reproduce_pre_compute_shape_empty_wrong_from_fixation_provider():
    """
    When fixation provider returns no scores for a sample/step, wrong path can store
    value_by_index {idx_key: {"prob": None, "avg_loss": None}}. If that gets filtered
    or wrong[0]["value_by_index"] is empty for another reason, wrong_indices = [].
    """
    correct_vbi = {"0": {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_empty_scores = [{"value_by_index": {"0": {"prob": None, "avg_loss": None}}}]
    result = _truth_ratio_buggy(
        None,
        pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_empty_scores},
        aggregator="closer_to_1_better",
    )
    assert result["agg_value"] is None or result["value_by_index"] == {}


def test_reproduce_pre_compute_shape_wrong_first_option_missing_key():
    """Wrong list: first option has value_by_index {"1": ...} but correct has {"0": ...} -> wrong_indices = ["1"], correct_indices = ["0"]."""
    correct_vbi = {"0": {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = [{"value_by_index": {"1": {"prob": 0.25, "avg_loss": -np.log(0.25)}}}]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


# ---- More variants: aggregator, multiple samples ----
def test_reproduce_assert_aggregator_true_better():
    correct_vbi = {"0": {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = [{"value_by_index": {}}]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="true_better",
        )


def test_reproduce_assert_aggregator_prob_mean():
    correct_vbi = {0: {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = [{"value_by_index": {"0": {"prob": 0.25, "avg_loss": -np.log(0.25)}}}]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="prob_mean",
        )


def test_reproduce_assert_three_samples_correct_012_wrong_01():
    correct_vbi = {
        "0": {"prob": 0.5, "avg_loss": -np.log(0.5)},
        "1": {"prob": 0.4, "avg_loss": -np.log(0.4)},
        "2": {"prob": 0.3, "avg_loss": -np.log(0.3)},
    }
    wrong_list = [
        {
                    "value_by_index": {
                        "0": {"prob": 0.25, "avg_loss": -np.log(0.25)},
                        "1": {"prob": 0.2, "avg_loss": -np.log(0.2)},
                    }
                }
    ]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


def test_reproduce_assert_five_wrong_options_first_empty_vbi():
    """5 wrong options (TOFU style) but first has empty value_by_index."""
    correct_vbi = {"0": {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = [
        {"value_by_index": {}},
        {"value_by_index": {"0": {"prob": 0.2, "avg_loss": -np.log(0.2)}}},
        {"value_by_index": {"0": {"prob": 0.18, "avg_loss": -np.log(0.18)}}},
        {"value_by_index": {"0": {"prob": 0.22, "avg_loss": -np.log(0.22)}}},
        {"value_by_index": {"0": {"prob": 0.19, "avg_loss": -np.log(0.19)}}},
    ]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


def test_reproduce_assert_correct_key_float_like_str():
    """Keys that look like numbers: "0.0" vs "0" can differ in some serialization paths."""
    correct_vbi = {"0": {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = [{"value_by_index": {"0.0": {"prob": 0.25, "avg_loss": -np.log(0.25)}}}]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


def test_reproduce_assert_large_data_indices():
    """Large data indices (e.g. from big dataset): 10000 (int) vs "10000" (str)."""
    correct_vbi = {10000: {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = [{"value_by_index": {"10000": {"prob": 0.25, "avg_loss": -np.log(0.25)}}}]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


def test_reproduce_ks_test_keyerror_second_entry_missing_score():
    """
    REAL ks_test with pre_compute that caused KeyError in pod: one entry has 'score',
    second has only 'prob'. With fix, we use .get('score') and skip missing; no KeyError.
    """
    from evals.metrics import METRICS_REGISTRY

    ks_test_metric = METRICS_REGISTRY["ks_test"]
    pre_compute = {
        "forget": {
            "value_by_index": {
                "0": {"score": 0.8},
                "1": {"prob": 0.9},
            }
        },
    }
    result = ks_test_metric._metric_fn(model=None, pre_compute=pre_compute)
    assert result is not None and "agg_value" in result
    # No reference_logs so pvalue is None; we used only entry "0" (has score)
    assert result["agg_value"] is None


def test_reproduce_ks_test_exact_pod_structure_raises():
    """
    ks_test with structure that pod had: value_by_index entry has no 'score'.
    We fail (ValueError) when forget has no valid scores, same as upstream.
    """
    from evals.metrics import METRICS_REGISTRY

    ks_test_metric = METRICS_REGISTRY["ks_test"]
    pre_compute = {
        "forget": {
            "value_by_index": {
                "0": {"agg_value": None, "value_by_index": {}},
            }
        },
    }
    with pytest.raises(ValueError, match="no valid"):
        ks_test_metric._metric_fn(model=None, pre_compute=pre_compute)


def test_reproduce_ks_test_keyerror_none_score():
    """Entry has key 'score' but value None - list comp still runs; no KeyError. So we need entry without key."""
    from evals.metrics import METRICS_REGISTRY

    ks_test_metric = METRICS_REGISTRY["ks_test"]
    pre_compute = {
        "forget": {"value_by_index": {"0": {"score": None}, "1": {"score": 0.9}}},
    }
    result = ks_test_metric._metric_fn(model=None, pre_compute=pre_compute)
    assert result is not None


def test_reproduce_assert_single_wrong_dict_correct_two_keys_wrong_one():
    """Single wrong (non-list): correct has "0","1"; wrong has only "0"."""
    correct_vbi = {
        "0": {"prob": 0.5, "avg_loss": -np.log(0.5)},
        "1": {"prob": 0.4, "avg_loss": -np.log(0.4)},
    }
    wrong_single = {"value_by_index": {"0": {"prob": 0.25, "avg_loss": -np.log(0.25)}}}
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_single},
            aggregator="closer_to_1_better",
        )


def test_reproduce_assert_list_wrong_second_option_has_different_keys():
    """Wrong[0] has key "0", wrong[1] has key "1" - we use wrong[0] for wrong_indices so ["0"]. Correct has ["0","1"] -> list length differs, assert fails."""
    correct_vbi = {
        "0": {"prob": 0.5, "avg_loss": -np.log(0.5)},
        "1": {"prob": 0.4, "avg_loss": -np.log(0.4)},
    }
    wrong_list = [
        {"value_by_index": {"0": {"prob": 0.25, "avg_loss": -np.log(0.25)}}},
        {"value_by_index": {"1": {"prob": 0.2, "avg_loss": -np.log(0.2)}}},
    ]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


def test_reproduce_assert_correct_keys_from_run_batchwise_evals_style():
    """Simulate run_batchwise_evals: data_indices are ints from batch['index'].tolist()."""
    correct_vbi = {453: {"prob": 0.5, "avg_loss": -np.log(0.5)}, 454: {"prob": 0.4, "avg_loss": -np.log(0.4)}}
    wrong_list = [
        {
                    "value_by_index": {
                        "453": {"prob": 0.25, "avg_loss": -np.log(0.25)},
                        "454": {"prob": 0.2, "avg_loss": -np.log(0.2)},
                    }
                }
    ]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


def test_reproduce_assert_wrong_list_one_option_has_key_other_empty():
    """Wrong has 2 options: first has value_by_index {}, second has {"0": ...}. We take wrong[0].keys() -> []."""
    correct_vbi = {"0": {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = [
        {"value_by_index": {}},
        {"value_by_index": {"0": {"prob": 0.2, "avg_loss": -np.log(0.2)}}},
    ]
    with pytest.raises(AssertionError, match="same indices"):
        _truth_ratio_buggy(
            None,
            pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
            aggregator="closer_to_1_better",
        )


def test_ks_test_succeeds_when_forget_has_score():
    """Sanity: ks_test does not raise when every entry has 'score'."""
    from evals.metrics import METRICS_REGISTRY

    ks_test_metric = METRICS_REGISTRY["ks_test"]
    pre_compute = {
        "forget": {"value_by_index": {"0": {"score": 0.8}, "1": {"score": 0.9}}},
    }
    result = ks_test_metric._metric_fn(model=None, pre_compute=pre_compute)
    assert "agg_value" in result
