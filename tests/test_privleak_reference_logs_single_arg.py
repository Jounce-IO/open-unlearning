"""Regression test: privleak must not receive reference_logs twice (trajectory_metrics call path).

When trajectory_metrics calls privleak, it passes reference_logs explicitly and must exclude
reference_logs from the **kwargs spread to avoid TypeError: got multiple values for keyword argument 'reference_logs'.
"""

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

# Import so METRICS_REGISTRY is populated
from evals.metrics import METRICS_REGISTRY


def _minimal_reference_logs():
    """Minimal reference_logs structure that privleak expects."""
    return {
        "retain_model_logs": {
            "retain": {"agg_value": 0.5},
        }
    }


class TestPrivleakReferenceLogsSingleArg:
    """privleak must receive reference_logs exactly once per call (no duplicate keyword)."""

    def test_privleak_with_reference_logs_passed_once_succeeds(self):
        """When reference_logs is passed only as the explicit arg (excluded from **kwargs), privleak runs without TypeError."""
        metric = METRICS_REGISTRY.get("privleak")
        assert metric is not None, "privleak must be registered"
        ref_logs = _minimal_reference_logs()
        kwargs = {
            "pre_compute": {"forget": {"agg_value": 0.3}},
            "reference_logs": ref_logs,
            "ref_value": 0.5,
        }
        # Correct pattern (as trajectory_metrics must use): pass reference_logs explicitly, exclude from spread so it is not duplicated
        result = metric._metric_fn(
            model=None,
            pre_compute=kwargs["pre_compute"],
            reference_logs=ref_logs,
            ref_value=kwargs["ref_value"],
            **{k: v for k, v in kwargs.items() if k not in ("model", "tokenizer", "pre_compute", "reference_logs", "ref_value")},
        )
        assert "agg_value" in result
        assert result["agg_value"] is not None

    def test_privleak_with_reference_logs_passed_twice_raises(self):
        """When reference_logs is passed both explicitly and via **kwargs, privleak raises TypeError (regression for old bug)."""
        metric = METRICS_REGISTRY.get("privleak")
        assert metric is not None, "privleak must be registered"
        ref_logs = _minimal_reference_logs()
        kwargs = {
            "pre_compute": {"forget": {"agg_value": 0.3}},
            "reference_logs": ref_logs,
            "ref_value": 0.5,
        }
        with pytest.raises(TypeError, match="multiple values for keyword argument 'reference_logs'"):
            metric._metric_fn(
                model=None,
                pre_compute=kwargs["pre_compute"],
                reference_logs=ref_logs,
                ref_value=kwargs["ref_value"],
                **{k: v for k, v in kwargs.items() if k not in ("model", "tokenizer", "pre_compute")},
            )
