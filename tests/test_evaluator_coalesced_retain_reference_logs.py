"""Test that coalesced trajectory path passes reference_logs to the metric.

When coalesce_trajectory_metrics=True, the evaluator calls the first metric once
with merged_args. If reference_logs (from the first metric config) is not included
in merged_args, the metric never loads retain_model_logs and forget_quality/privleak
warn "retain_model_logs not provided in reference_logs". This test asserts the
evaluator passes reference_logs so the bug is caught.
"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from omegaconf import OmegaConf

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


def test_coalesced_trajectory_passes_reference_logs_to_metric():
    """With coalesce_trajectory_metrics=True and retain_logs_path set, metric must receive reference_logs with retain_model_logs."""
    from evals.base import Evaluator
    from evals import get_evaluators

    with __retain_fixture_file__() as retain_path:
        eval_cfg = OmegaConf.create({
            "handler": "TOFUEvaluator",
            "output_dir": "/tmp/test_coalesced_retain",
            "overwrite": True,
            "coalesce_trajectory_metrics": True,
            "retain_logs_path": str(retain_path),
            "metrics": {
                "trajectory_all": {
                    "handler": "trajectory_metrics",
                    "reference_logs": {
                        "retain_model_logs": {
                            "path": str(retain_path),
                            "include": {
                                "mia_min_k": {"access_key": "retain"},
                                "forget_truth_ratio": {"access_key": "retain"},
                            },
                        },
                    },
                    "datasets": {},
                    "collators": {},
                    "metrics": ["privleak", "truth_ratio"],
                    "metric_display_names": ["trajectory_privleak", "trajectory_forget_quality"],
                },
                "trajectory_b": {
                    "handler": "trajectory_metrics",
                    "reference_logs": {
                        "retain_model_logs": {
                            "path": str(retain_path),
                            "include": {"mia_min_k": {"access_key": "retain"}},
                        },
                    },
                    "datasets": {},
                    "collators": {},
                    "metrics": ["privleak"],
                    "metric_display_names": ["trajectory_privleak_b"],
                },
            },
        })

        mock_metric = Mock(return_value={
            "trajectory_all": {"agg_value": 0.5},
            "trajectory_b": {"agg_value": 0.5},
        })
        with patch("evals.base.get_metrics") as get_metrics_mock:
            get_metrics_mock.return_value = {
                "trajectory_all": mock_metric,
                "trajectory_b": Mock(return_value={"agg_value": 0.5}),
            }
            evaluators = get_evaluators({"tofu_trajectory": eval_cfg})
        ev = evaluators["tofu_trajectory"]
        mock_model = MagicMock()
        with (
            patch.object(ev, "load_logs_from_file", return_value={}),
            patch.object(ev, "save_logs"),
            patch.object(ev, "prepare_model", return_value=mock_model),
        ):
            ev.evaluate(mock_model)

        mock_metric.assert_called_once()
        call_kwargs = mock_metric.call_args[1]
        assert "reference_logs" in call_kwargs, (
            "Coalesced path must pass reference_logs to the metric so retain_model_logs is loaded; "
            "otherwise forget_quality/privleak warn and return None."
        )
        ref_logs = call_kwargs["reference_logs"]
        assert "retain_model_logs" in ref_logs, (
            "reference_logs must contain retain_model_logs (from first metric config) when retain_logs_path is set."
        )


class __retain_fixture_file__:
    """Context manager that creates a minimal retain JSON and yields its path."""

    def __init__(self):
        self.tmp = None

    def __enter__(self):
        import tempfile
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump({
            "mia_min_k": {"retain": {"agg_value": 0.1}},
            "forget_truth_ratio": {"retain": {"value_by_index": {"0": {"score": 0.5}}}},
        }, self.tmp)
        self.tmp.close()
        return self.tmp.name

    def __exit__(self, *args):
        if self.tmp and Path(self.tmp.name).exists():
            Path(self.tmp.name).unlink(missing_ok=True)
