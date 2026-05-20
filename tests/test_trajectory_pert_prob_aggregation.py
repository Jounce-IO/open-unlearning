"""PERT trajectory prob aggregation: mean (TR) vs sum (norm denominator)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.trajectory_metrics import (
    _build_trajectory_value_by_index,
    _init_step_index_keys_by_view,
    _merge_pert_option_probability_layers,
    _record_step_index_key,
    _sync_pert_wrong_sum_index_keys,
)


def test_merge_pert_three_options_mean_and_sum():
    opts = [
        {"full": {"steps": {0: {"probability": [0.2]}}}},
        {"full": {"steps": {0: {"probability": [0.4]}}}},
        {"full": {"steps": {0: {"probability": [0.6]}}}},
    ]
    main: dict = {"full": {"steps": {}}}
    _merge_pert_option_probability_layers(main, opts, emit_wrong_sum=True)
    assert main["full"]["steps"][0]["probability"] == pytest.approx([0.4])
    assert main["full"]["steps"][0]["probability_wrong_sum"] == pytest.approx([1.2])


def test_merge_pert_single_option():
    opts = [{"full": {"steps": {1: {"probability": [0.7, 0.8]}}}}]
    main: dict = {"full": {"steps": {}}}
    _merge_pert_option_probability_layers(main, opts, emit_wrong_sum=True)
    assert main["full"]["steps"][1]["probability"] == pytest.approx([0.7, 0.8])
    assert main["full"]["steps"][1]["probability_wrong_sum"] == pytest.approx([0.7, 0.8])


def test_merge_pert_appends_across_dataloader_batches_for_value_by_index():
    """PERT merge must extend per-step lists (not replace) to match appended index keys."""
    include_views = ("full",)
    trajectory_names = ("steps",)
    main: dict = {"full": {"steps": {}}}
    keys = _init_step_index_keys_by_view(include_views, trajectory_names)

    def merge_batch(
        per_option_rows: list[list[float]],
        index_ids: list[str],
    ) -> None:
        opts = [
            {"full": {"steps": {0: {"probability": list(row)}}}} for row in per_option_rows
        ]
        _merge_pert_option_probability_layers(main, opts, emit_wrong_sum=True)
        for idx in index_ids:
            _record_step_index_key(keys, "full", "steps", 0, "probability", idx)

    # Batch 0: two samples, two perturbed options → mean [0.4, 0.6]
    merge_batch([[0.2, 0.4], [0.6, 0.8]], ["0", "1"])
    # Batch 1: append mean [0.3, 0.5] (would fail if merge replaced instead of extended)
    merge_batch([[0.1, 0.3], [0.5, 0.7]], ["2", "3"])
    _sync_pert_wrong_sum_index_keys(keys, main, include_views, trajectory_names)

    assert main["full"]["steps"][0]["probability"] == pytest.approx([0.4, 0.6, 0.3, 0.5])
    vbi = _build_trajectory_value_by_index(
        main,
        keys,
        include_views,
        trajectory_names,
        [0],
        metrics=("probability", "probability_wrong_sum"),
    )
    assert len(vbi) == 4
    assert vbi["0"]["full"]["steps"]["probability"] == pytest.approx([0.4])
    assert vbi["1"]["full"]["steps"]["probability"] == pytest.approx([0.6])
    assert vbi["2"]["full"]["steps"]["probability"] == pytest.approx([0.3])
    assert vbi["3"]["full"]["steps"]["probability"] == pytest.approx([0.5])
    assert vbi["0"]["full"]["steps"]["probability_wrong_sum"] == pytest.approx([0.8])
    assert vbi["1"]["full"]["steps"]["probability_wrong_sum"] == pytest.approx([1.2])
    assert vbi["2"]["full"]["steps"]["probability_wrong_sum"] == pytest.approx([0.6])
    assert vbi["3"]["full"]["steps"]["probability_wrong_sum"] == pytest.approx([1.0])


def test_merge_pert_nan_step_propagates():
    import math

    opts = [
        {"full": {"steps": {0: {"probability": [float("nan")]}}}},
        {"full": {"steps": {0: {"probability": [0.4]}}}},
    ]
    main: dict = {"full": {"steps": {}}}
    _merge_pert_option_probability_layers(main, opts, emit_wrong_sum=True)
    assert math.isnan(main["full"]["steps"][0]["probability"][0])
