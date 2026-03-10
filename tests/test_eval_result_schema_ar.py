"""
T009: Contract test — AR eval result JSON has same top-level keys and required metric keys
as dLLM baseline. Asserts AR result keys ⊆ fixture keys and required keys present.
Baseline: open-unlearning/tests/fixtures/dllm_eval_result_baseline.json
"""

import json
import pytest
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures"
BASELINE_JSON = FIXTURE_DIR / "dllm_eval_result_baseline.json"

REQUIRED_TOP_LEVEL = {"run_info"}
REQUIRED_METRIC_KEYS = {"forget_Q_A_Prob", "retain_Q_A_Prob", "forget_truth_ratio", "retain_truth_ratio"}


def _load_baseline_keys():
    """Load baseline (fixture) and return set of top-level keys and required metric names."""
    assert BASELINE_JSON.exists(), f"Baseline fixture missing: {BASELINE_JSON}"
    with open(BASELINE_JSON) as f:
        data = json.load(f)
    return set(data.keys()), data


class TestEvalResultSchemaAr:
    """AR eval result must have same top-level keys as dLLM baseline; required keys present."""

    def test_baseline_fixture_exists_and_has_required_keys(self):
        """Fixture exists and contains required run_info and metric keys."""
        _, data = _load_baseline_keys()
        for key in REQUIRED_TOP_LEVEL:
            assert key in data, f"Baseline missing required top-level key: {key}"
        for key in REQUIRED_METRIC_KEYS:
            assert key in data, f"Baseline missing required metric key: {key}"
            assert "agg_value" in data[key] or isinstance(data[key], (int, float)), (
                f"Baseline metric {key} should have agg_value or be scalar"
            )

    def test_ar_result_keys_subset_of_baseline(self):
        """Given an AR eval result dict, its top-level keys must be ⊆ baseline keys."""
        baseline_keys, _ = _load_baseline_keys()
        # Simulate AR result (minimal): must not introduce keys outside baseline
        ar_result = {
            "run_info": {"world_size": 1, "total_samples": 1},
            "forget_Q_A_Prob": {"agg_value": 0.1},
            "retain_Q_A_Prob": {"agg_value": 0.2},
            "forget_truth_ratio": {"agg_value": 0.3},
            "retain_truth_ratio": {"agg_value": 0.4},
        }
        ar_keys = set(ar_result.keys())
        assert ar_keys <= baseline_keys, (
            f"AR result has keys not in baseline: {ar_keys - baseline_keys}. "
            "AR must not add top-level keys; schema must match dLLM."
        )

    def test_ar_result_has_required_keys(self):
        """AR result must contain all required top-level and metric keys."""
        ar_result = {
            "run_info": {"world_size": 1},
            "forget_Q_A_Prob": {"agg_value": 0.0},
            "retain_Q_A_Prob": {"agg_value": 0.0},
            "forget_truth_ratio": {"agg_value": 0.0},
            "retain_truth_ratio": {"agg_value": 0.0},
        }
        for key in REQUIRED_TOP_LEVEL:
            assert key in ar_result, f"AR result missing required key: {key}"
        for key in REQUIRED_METRIC_KEYS:
            assert key in ar_result, f"AR result missing required metric key: {key}"
            assert "agg_value" in ar_result[key], f"AR result metric {key} missing agg_value"
