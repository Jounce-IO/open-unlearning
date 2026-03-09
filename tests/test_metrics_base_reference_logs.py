"""Tests for reference_logs loading in evals.metrics.base (retain trajectory by_step keys)."""

import pytest
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.base import UnlearningMetric
from unittest.mock import Mock, patch


def _dummy_metric_fn(**kwargs):
    return {"agg_value": 0.5}


class TestReferenceLogsByStepLoading:
    """When loading reference_logs from file, mia_min_k_by_step and forget_truth_ratio_by_step are exposed as retain_mia_by_step and retain_forget_tr_by_step."""

    def test_prepare_kwargs_adds_retain_mia_by_step_and_retain_forget_tr_by_step(self):
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        mock_logs = {
            "some_key": {"agg_value": 0.1},
            "mia_min_k_by_step": {"0": {"agg_value": 0.2}, "1": {"agg_value": 0.3}},
            "forget_truth_ratio_by_step": {
                "0": {"value_by_index": {"0": {"score": 0.5}}},
                "1": {"value_by_index": {"0": {"score": 0.6}}},
            },
        }
        with patch.object(metric, "load_logs_from_file", return_value=mock_logs):
            kwargs = metric.prepare_kwargs_evaluate_metric(
                Mock(),
                "test_metric",
                reference_logs={
                    "retain_model_logs": {
                        "path": "/tmp/fake_retain.json",
                        "include": {"some_key": {"access_key": "some_key"}},
                    }
                },
            )
        assert "reference_logs" in kwargs
        ref = kwargs["reference_logs"].get("retain_model_logs") or {}
        assert "retain_mia_by_step" in ref
        assert ref["retain_mia_by_step"] == mock_logs["mia_min_k_by_step"]
        assert "retain_forget_tr_by_step" in ref
        assert ref["retain_forget_tr_by_step"] == mock_logs["forget_truth_ratio_by_step"]
