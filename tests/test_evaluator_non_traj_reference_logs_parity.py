"""Non-trajectory Evaluator passes reference_logs only when retain path is usable (trajectory parity).

When ``retain_logs_path`` is null, YAML still has ``reference_logs.retain_model_logs.path`` resolved
to null. Passing that shell into ``ks_test`` caused RetainReferenceValidationError; trajectory
already omitted reference_logs in that case.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from omegaconf import OmegaConf

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


def _canonical_retain_json_path(tmp_path: Path) -> Path:
    p = tmp_path / "retain_ref.json"
    p.write_text(
        json.dumps(
            {
                "mia_min_k": {"agg_value": 0.1},
                "forget_truth_ratio": {
                    "value_by_index": {"0": {"score": 0.5}},
                    "agg_value": 0.5,
                },
            }
        )
    )
    return p


class TestReferenceLogsPathHelper:
    """Unit tests for evals.base.reference_logs_has_usable_retain_path."""

    def test_null_container_false(self) -> None:
        from evals.base import reference_logs_has_usable_retain_path

        assert reference_logs_has_usable_retain_path(None) is False
        assert reference_logs_has_usable_retain_path({}) is False

    def test_shell_path_null_false(self) -> None:
        from evals.base import reference_logs_has_usable_retain_path

        shell = {
            "retain_model_logs": {
                "path": None,
                "include": {
                    "forget_truth_ratio": {"access_key": "retain_ftr"},
                },
            },
        }
        assert reference_logs_has_usable_retain_path(shell) is False

    def test_string_none_false(self) -> None:
        from evals.base import reference_logs_has_usable_retain_path

        assert (
            reference_logs_has_usable_retain_path(
                {"retain_model_logs": {"path": "none", "include": {}}}
            )
            is False
        )

    def test_real_path_true(self, tmp_path: Path) -> None:
        from evals.base import reference_logs_has_usable_retain_path

        p = _canonical_retain_json_path(tmp_path)
        assert (
            reference_logs_has_usable_retain_path(
                {
                    "retain_model_logs": {
                        "path": str(p),
                        "include": {"forget_truth_ratio": {"access_key": "retain_ftr"}},
                    },
                }
            )
            is True
        )


class TestPrepareKwargsOmitsShellReferenceLogs:
    """prepare_kwargs must not inject YAML shells (defense in depth)."""

    def test_ks_test_prepare_kwargs_no_reference_when_path_null(self) -> None:
        from evals.metrics import METRICS_REGISTRY

        metric = METRICS_REGISTRY["ks_test"]
        forget_pre = {
            "value_by_index": {"0": {"score": 0.3}},
            "agg_value": 0.3,
        }
        out = metric.prepare_kwargs_evaluate_metric(
            model=None,
            metric_name="forget_quality",
            cache={"forget_truth_ratio": forget_pre},
            tokenizer=None,
            template_args=None,
            eval_cfg={},
            pre_compute={"forget_truth_ratio": {"access_key": "forget"}},
            reference_logs={
                "retain_model_logs": {
                    "path": None,
                    "include": {"forget_truth_ratio": {"access_key": "retain_ftr"}},
                },
            },
        )
        assert "reference_logs" not in out

    def test_ks_test_prepare_kwargs_loads_when_path_set(self, tmp_path: Path) -> None:
        from evals.metrics import METRICS_REGISTRY

        p = _canonical_retain_json_path(tmp_path)
        metric = METRICS_REGISTRY["ks_test"]
        forget_pre = {
            "value_by_index": {"0": {"score": 0.3}},
            "agg_value": 0.3,
        }
        out = metric.prepare_kwargs_evaluate_metric(
            model=None,
            metric_name="forget_quality",
            cache={"forget_truth_ratio": forget_pre},
            tokenizer=None,
            template_args=None,
            eval_cfg={},
            pre_compute={"forget_truth_ratio": {"access_key": "forget"}},
            reference_logs={
                "retain_model_logs": {
                    "path": str(p),
                    "include": {"forget_truth_ratio": {"access_key": "retain_ftr"}},
                },
            },
        )
        assert "reference_logs" in out
        rml = out["reference_logs"]["retain_model_logs"]
        assert rml.get("retain_ftr") is not None

    def test_ks_test_prepare_kwargs_passes_evaluator_cached_payload(self) -> None:
        """Evaluator injects load_and_validate_reference output (no path); must reach ks_test."""
        from evals.metrics import METRICS_REGISTRY

        metric = METRICS_REGISTRY["ks_test"]
        forget_pre = {
            "value_by_index": {"0": {"score": 0.3}},
            "agg_value": 0.3,
        }
        cached = {
            "retain_model_logs": {
                "retain_ftr": {
                    "value_by_index": {"0": {"score": 0.5}},
                    "agg_value": 0.5,
                },
            }
        }
        out = metric.prepare_kwargs_evaluate_metric(
            model=None,
            metric_name="forget_quality",
            cache={"forget_truth_ratio": forget_pre},
            tokenizer=None,
            template_args=None,
            eval_cfg={},
            pre_compute={"forget_truth_ratio": {"access_key": "forget"}},
            reference_logs=cached,
        )
        assert out.get("reference_logs") == cached


class TestNonTrajectoryEvaluatorReferenceLogsParity:
    """Evaluator per-metric loop: omit reference_logs when path is null (like trajectory)."""

    def test_forget_quality_not_passed_reference_logs_when_path_null(
        self, tmp_path: Path
    ) -> None:
        from evals import get_evaluators

        fq_cfg = {
            "handler": "ks_test",
            "pre_compute": {"forget_truth_ratio": {"access_key": "forget"}},
            "reference_logs": {
                "retain_model_logs": {
                    "path": None,
                    "include": {"forget_truth_ratio": {"access_key": "retain_ftr"}},
                },
            },
        }
        eval_cfg = OmegaConf.create(
            {
                "handler": "TOFUEvaluator",
                "output_dir": str(tmp_path / "out"),
                "overwrite": True,
                "retain_reference_mode": True,
                "retain_logs_path": None,
                "samples": 2,
                "metrics": {"forget_quality": fq_cfg},
            }
        )
        mock_metric = Mock(return_value={"agg_value": None})
        with patch("evals.base.get_metrics") as get_metrics_mock:
            get_metrics_mock.return_value = {"forget_quality": mock_metric}
            evaluators = get_evaluators({"tofu": eval_cfg})
        ev = evaluators["tofu"]
        mock_model = MagicMock()
        with (
            patch.object(ev, "load_logs_from_file", return_value={}),
            patch.object(ev, "save_logs"),
            patch.object(ev, "prepare_model", return_value=mock_model),
        ):
            ev.evaluate(mock_model)

        mock_metric.assert_called_once()
        call_kw = mock_metric.call_args[1]
        assert "reference_logs" not in call_kw, (
            "YAML reference_logs shell must be stripped when retain_logs_path is null"
        )

    def test_forget_quality_passes_loaded_reference_when_path_set(
        self, tmp_path: Path
    ) -> None:
        from evals import get_evaluators

        ref_path = _canonical_retain_json_path(tmp_path)
        fq_cfg = {
            "handler": "ks_test",
            "pre_compute": {"forget_truth_ratio": {"access_key": "forget"}},
            "reference_logs": {
                "retain_model_logs": {
                    "path": str(ref_path),
                    "include": {
                        "mia_min_k": {"access_key": "retain"},
                        "forget_truth_ratio": {"access_key": "retain_ftr"},
                    },
                },
            },
        }
        eval_cfg = OmegaConf.create(
            {
                "handler": "TOFUEvaluator",
                "output_dir": str(tmp_path / "out2"),
                "overwrite": True,
                "retain_logs_path": str(ref_path),
                "samples": 2,
                "metrics": {"forget_quality": fq_cfg},
            }
        )
        mock_metric = Mock(return_value={"agg_value": 0.42})
        with patch("evals.base.get_metrics") as get_metrics_mock:
            get_metrics_mock.return_value = {"forget_quality": mock_metric}
            evaluators = get_evaluators({"tofu": eval_cfg})
        ev = evaluators["tofu"]
        mock_model = MagicMock()
        load_data = {
            "mia_min_k": {"agg_value": 0.1},
            "forget_truth_ratio": {
                "value_by_index": {"0": {"score": 0.5}},
                "agg_value": 0.5,
            },
        }
        with (
            patch.object(ev, "load_logs_from_file", return_value=load_data),
            patch.object(ev, "save_logs"),
            patch.object(ev, "prepare_model", return_value=mock_model),
        ):
            ev.evaluate(mock_model)

        mock_metric.assert_called_once()
        call_kw = mock_metric.call_args[1]
        assert "reference_logs" in call_kw
        rml = call_kw["reference_logs"]["retain_model_logs"]
        assert rml.get("retain_ftr") is not None
