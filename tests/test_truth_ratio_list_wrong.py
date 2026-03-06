"""
Unit tests for truth_ratio when pre_compute["wrong"] is a list of N result dicts (average over N options).
"""

from pathlib import Path
import sys
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics import METRICS_REGISTRY


def test_truth_ratio_list_of_wrong_dicts_averages():
    if "truth_ratio" not in METRICS_REGISTRY:
        return
    metric = METRICS_REGISTRY["truth_ratio"]
    correct_vbi = {
        0: {"prob": 0.5, "avg_loss": -np.log(0.5)},
        1: {"prob": 0.4, "avg_loss": -np.log(0.4)},
    }
    wrong_opt0 = {0: {"prob": 0.1, "avg_loss": -np.log(0.1)}, 1: {"prob": 0.2, "avg_loss": -np.log(0.2)}}
    wrong_opt1 = {0: {"prob": 0.2, "avg_loss": -np.log(0.2)}, 1: {"prob": 0.1, "avg_loss": -np.log(0.1)}}
    wrong_opt2 = {0: {"prob": 0.15, "avg_loss": -np.log(0.15)}, 1: {"prob": 0.15, "avg_loss": -np.log(0.15)}}
    wrong_list = [
        {"agg_value": 0.15, "value_by_index": wrong_opt0},
        {"agg_value": 0.15, "value_by_index": wrong_opt1},
        {"agg_value": 0.15, "value_by_index": wrong_opt2},
    ]
    result = metric._metric_fn(
        model=None,
        pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
        aggregator="closer_to_1_better",
    )
    assert result["agg_value"] is not None
    assert isinstance(result["value_by_index"], dict)
    for i, idx in enumerate([0, 1]):
        wrong_prob_avg = np.mean([
            np.exp(-wrong_opt0[idx]["avg_loss"]),
            np.exp(-wrong_opt1[idx]["avg_loss"]),
            np.exp(-wrong_opt2[idx]["avg_loss"]),
        ])
        correct_prob = np.exp(-correct_vbi[idx]["avg_loss"])
        expected_tr = wrong_prob_avg / (correct_prob + 1e-10)
        key = str(idx)  # truth_ratio normalizes keys to str
        assert key in result["value_by_index"]
        reported = result["value_by_index"][key]["score"]
        if isinstance(reported, np.ndarray):
            reported = float(reported.flat[i]) if reported.size > i else None
        assert reported is not None
        np.testing.assert_allclose(reported, expected_tr, rtol=1e-5)


def test_truth_ratio_single_wrong_dict_unchanged():
    if "truth_ratio" not in METRICS_REGISTRY:
        return
    metric = METRICS_REGISTRY["truth_ratio"]
    correct_vbi = {0: {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_vbi = {0: {"prob": 0.25, "avg_loss": -np.log(0.25)}}
    result = metric._metric_fn(
        model=None,
        pre_compute={
            "correct": {"value_by_index": correct_vbi},
            "wrong": {"value_by_index": wrong_vbi},
        },
        aggregator="closer_to_1_better",
    )
    assert result["agg_value"] is not None
    expected_tr = 0.25 / (0.5 + 1e-10)
    reported = result["value_by_index"]["0"]["score"]  # truth_ratio normalizes keys to str
    if isinstance(reported, np.ndarray):
        reported = float(reported.flat[0])
    np.testing.assert_allclose(reported, expected_tr, rtol=1e-5)
