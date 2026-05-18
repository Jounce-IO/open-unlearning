"""PERT trajectory prob aggregation: mean (TR) vs sum (norm denominator)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.trajectory_metrics import _merge_pert_option_probability_layers


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


def test_merge_pert_nan_step_propagates():
    import math

    opts = [
        {"full": {"steps": {0: {"probability": [float("nan")]}}}},
        {"full": {"steps": {0: {"probability": [0.4]}}}},
    ]
    main: dict = {"full": {"steps": {}}}
    _merge_pert_option_probability_layers(main, opts, emit_wrong_sum=True)
    assert math.isnan(main["full"]["steps"][0]["probability"][0])
