"""Tests for per-index trajectory probability persistence helpers."""

from __future__ import annotations

import numpy as np

from evals.metrics.trajectory_metrics import (
    _build_trajectory_value_by_index,
    _init_step_index_keys_by_view,
    _record_step_index_key,
    _should_persist_value_by_index,
    _subset_value_by_index_for_metrics,
)


def test_should_persist_for_guided_tr_pass() -> None:
    assert _should_persist_value_by_index(None, "retain__guided_tr_para")
    assert _should_persist_value_by_index(None, "retain__guided_tr_pert")
    assert _should_persist_value_by_index(None, "ra__guided_tr_correct")
    assert _should_persist_value_by_index(None, "forget__guided_prob")
    assert not _should_persist_value_by_index(None, "forget__unguided")
    assert not _should_persist_value_by_index({"persist_value_by_index": False}, "retain__guided_tr_para")


def test_build_value_by_index_matches_batch_nanmean() -> None:
    include_views = ("full",)
    trajectory_names = ("steps",)
    step_values = {
        "full": {
            "steps": {
                0: {"probability": [0.2, 0.8]},
                1: {"probability": [0.4, 0.6]},
            }
        }
    }
    keys = _init_step_index_keys_by_view(include_views, trajectory_names)
    _record_step_index_key(keys, "full", "steps", 0, "probability", "a")
    _record_step_index_key(keys, "full", "steps", 0, "probability", "b")
    _record_step_index_key(keys, "full", "steps", 1, "probability", "a")
    _record_step_index_key(keys, "full", "steps", 1, "probability", "b")
    vbi = _build_trajectory_value_by_index(
        step_values, keys, include_views, trajectory_names, [0, 1]
    )
    assert vbi["a"]["full"]["steps"]["probability"] == [0.2, 0.4]
    assert vbi["b"]["full"]["steps"]["probability"] == [0.8, 0.6]
    assert np.nanmean([0.2, 0.8]) == 0.5
    assert np.nanmean([0.4, 0.6]) == 0.5


def test_subset_value_by_index_for_metrics() -> None:
    vbi = {
        "0": {
            "full": {
                "steps": {
                    "probability": [0.1, 0.2],
                    "probability_wrong_sum": [0.9, 0.8],
                }
            }
        }
    }
    sub = _subset_value_by_index_for_metrics(vbi, ("probability",))
    assert "probability_wrong_sum" not in sub["0"]["full"]["steps"]
    assert sub["0"]["full"]["steps"]["probability"] == [0.1, 0.2]
