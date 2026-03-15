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
                                    "forget_truth_ratio": {"access_key": "retain_ftr"},
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
            "reference_logs must contain retain_model_logs (loaded at start when path is set)."
        )
        rml = ref_logs["retain_model_logs"]
        assert rml.get("retain") is not None, "retain slot must be set from mia_min_k"
        assert rml.get("retain_ftr") is not None, "retain_ftr slot must be set from forget_truth_ratio"


def test_coalesced_trajectory_first_metric_without_reference_logs_completes():
    """Coalesced path when first metric has no reference_logs: evaluate completes and metric is called (reference_logs may be absent)."""
    from evals import get_evaluators

    eval_cfg = OmegaConf.create({
        "handler": "TOFUEvaluator",
        "output_dir": "/tmp/test_coalesced_no_ref",
        "overwrite": True,
        "coalesce_trajectory_metrics": True,
        "metrics": {
            "trajectory_all": {
                "handler": "trajectory_metrics",
                "datasets": {},
                "collators": {},
                "metrics": ["privleak"],
                "metric_display_names": ["trajectory_privleak"],
            },
            "trajectory_b": {
                "handler": "trajectory_metrics",
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
    assert "reference_logs" not in call_kwargs or call_kwargs.get("reference_logs") in (None, {}), (
        "When first metric has no reference_logs config, we must not inject it."
    )


def test_coalesced_trajectory_empty_reference_logs_passed_as_empty_dict():
    """Coalesced path when first metric has reference_logs: {} — metric receives reference_logs as empty dict."""
    from evals import get_evaluators

    eval_cfg = OmegaConf.create({
        "handler": "TOFUEvaluator",
        "output_dir": "/tmp/test_coalesced_empty_ref",
        "overwrite": True,
        "coalesce_trajectory_metrics": True,
        "metrics": {
            "trajectory_all": {
                "handler": "trajectory_metrics",
                "reference_logs": {},
                "datasets": {},
                "collators": {},
                "metrics": ["privleak"],
                "metric_display_names": ["trajectory_privleak"],
            },
            "trajectory_b": {
                "handler": "trajectory_metrics",
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
    # When config has empty reference_logs (no path), we do not pass reference_logs so the metric
    # does not require step-matched reference (e.g. retain_reference_mode run).
    assert "reference_logs" not in call_kwargs


def test_per_metric_path_passes_reference_logs():
    """Per-metric path (no coalescing): single metric still receives reference_logs from its config."""
    from evals import get_evaluators

    with __retain_fixture_file__() as retain_path:
        eval_cfg = OmegaConf.create({
            "handler": "TOFUEvaluator",
            "output_dir": "/tmp/test_per_metric_ref",
            "overwrite": True,
            "coalesce_trajectory_metrics": False,
            "metrics": {
                "trajectory_all": {
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
                    "metric_display_names": ["trajectory_privleak"],
                },
            },
        })
        mock_metric = Mock(return_value={"trajectory_all": {"agg_value": 0.5}})
        with patch("evals.base.get_metrics") as get_metrics_mock:
            get_metrics_mock.return_value = {"trajectory_all": mock_metric}
            evaluators = get_evaluators({"tofu_trajectory": eval_cfg})
        ev = evaluators["tofu_trajectory"]
        mock_model = MagicMock()
        with (
            patch.object(ev, "save_logs"),
            patch.object(ev, "prepare_model", return_value=mock_model),
        ):
            ev.evaluate(mock_model)
        mock_metric.assert_called_once()
        call_kwargs = mock_metric.call_args[1]
        assert "reference_logs" in call_kwargs
        rml = call_kwargs["reference_logs"].get("retain_model_logs")
        assert rml is not None and rml.get("retain") is not None, "Loaded reference must have retain slot"


def test_coalesced_trajectory_three_metrics_still_passes_first_reference_logs():
    """Coalesced path with three metrics: first metric's reference_logs is still passed."""
    from evals import get_evaluators

    with __retain_fixture_file__() as retain_path:
        eval_cfg = OmegaConf.create({
            "handler": "TOFUEvaluator",
            "output_dir": "/tmp/test_coalesced_three",
            "overwrite": True,
            "coalesce_trajectory_metrics": True,
            "retain_logs_path": str(retain_path),
            "metrics": {
                "trajectory_all": {
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
                    "metric_display_names": ["a"],
                },
                "trajectory_b": {
                    "handler": "trajectory_metrics",
                    "datasets": {},
                    "collators": {},
                    "metrics": ["privleak"],
                    "metric_display_names": ["b"],
                },
                "trajectory_c": {
                    "handler": "trajectory_metrics",
                    "datasets": {},
                    "collators": {},
                    "metrics": ["privleak"],
                    "metric_display_names": ["c"],
                },
            },
        })
        mock_metric = Mock(return_value={
            "trajectory_all": {"agg_value": 0.5},
            "trajectory_b": {"agg_value": 0.5},
            "trajectory_c": {"agg_value": 0.5},
        })
        with patch("evals.base.get_metrics") as get_metrics_mock:
            get_metrics_mock.return_value = {
                "trajectory_all": mock_metric,
                "trajectory_b": Mock(return_value={"agg_value": 0.5}),
                "trajectory_c": Mock(return_value={"agg_value": 0.5}),
            }
            evaluators = get_evaluators({"tofu_trajectory": eval_cfg})
        ev = evaluators["tofu_trajectory"]
        mock_model = MagicMock()
        with (
            patch.object(ev, "save_logs"),
            patch.object(ev, "prepare_model", return_value=mock_model),
        ):
            ev.evaluate(mock_model)
        mock_metric.assert_called_once()
        call_kwargs = mock_metric.call_args[1]
        assert "reference_logs" in call_kwargs
        rml = call_kwargs["reference_logs"].get("retain_model_logs")
        assert rml is not None and rml.get("retain") is not None


def test_coalesced_reference_logs_path_resolved_from_omegaconf():
    """When first metric's reference_logs uses OmegaConf interpolation, path is resolved in merged_args."""
    from evals import get_evaluators

    with __retain_fixture_file__() as retain_path:
        eval_cfg = OmegaConf.create({
            "retain_logs_path": str(retain_path),
            "handler": "TOFUEvaluator",
            "output_dir": "/tmp/test_coalesced_resolve",
            "overwrite": True,
            "coalesce_trajectory_metrics": True,
            "metrics": {
                "trajectory_all": {
                    "handler": "trajectory_metrics",
                    "reference_logs": {
                        "retain_model_logs": {
                            "path": "${retain_logs_path}",
                            "include": {"mia_min_k": {"access_key": "retain"}},
                        },
                    },
                    "datasets": {},
                    "collators": {},
                    "metrics": ["privleak"],
                    "metric_display_names": ["a"],
                },
                "trajectory_b": {
                    "handler": "trajectory_metrics",
                    "datasets": {},
                    "collators": {},
                    "metrics": ["privleak"],
                    "metric_display_names": ["b"],
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
            patch.object(ev, "save_logs"),
            patch.object(ev, "prepare_model", return_value=mock_model),
        ):
            ev.evaluate(mock_model)
        mock_metric.assert_called_once()
        call_kwargs = mock_metric.call_args[1]
        assert "reference_logs" in call_kwargs
        ref = call_kwargs["reference_logs"].get("retain_model_logs")
        assert ref is not None, "reference_logs must contain retain_model_logs (loaded at start)."
        assert ref.get("retain") is not None, (
            "Path was resolved and file loaded; retain slot must be set from canonical mia_min_k."
        )


def test_coalesced_reference_logs_fallback_when_to_container_raises():
    """When OmegaConf.to_container raises in the reference_logs block, fallback to dict() so metric still receives reference_logs."""
    from omegaconf import OmegaConf as OC
    from evals import get_evaluators

    with __retain_fixture_file__() as retain_path:
        eval_cfg = OmegaConf.create({
            "handler": "TOFUEvaluator",
            "output_dir": "/tmp/test_coalesced_fallback",
            "overwrite": True,
            "coalesce_trajectory_metrics": True,
            "retain_logs_path": str(retain_path),
            "metrics": {
                "trajectory_all": {
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
                    "metric_display_names": ["a"],
                },
                "trajectory_b": {
                    "handler": "trajectory_metrics",
                    "datasets": {},
                    "collators": {},
                    "metrics": ["privleak"],
                    "metric_display_names": ["b"],
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
        real_to_container = OC.to_container
        # First to_container call is for logs["config"] (eval_cfg), second is for reference_logs in coalesced block.
        side_effects = [
            lambda *a, **kw: real_to_container(*a, **kw),
            RuntimeError("simulated"),
        ]

        with (
            patch.object(ev, "load_logs_from_file", return_value={}),
            patch.object(ev, "save_logs"),
            patch.object(ev, "prepare_model", return_value=mock_model),
        ):
            with patch("omegaconf.OmegaConf.to_container", side_effect=side_effects):
                ev.evaluate(mock_model)
        mock_metric.assert_called_once()
        call_kwargs = mock_metric.call_args[1]
        assert "reference_logs" in call_kwargs
        assert "retain_model_logs" in call_kwargs["reference_logs"]


class __retain_fixture_file__:
    """Context manager that creates a minimal canonical retain JSON and yields its path."""

    def __init__(self):
        self.tmp = None

    def __enter__(self):
        import tempfile
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        # Canonical form: mia_min_k = {agg_value: number}, forget_truth_ratio = {value_by_index and/or agg_value}
        json.dump({
            "mia_min_k": {"agg_value": 0.1},
            "forget_truth_ratio": {"value_by_index": {"0": {"score": 0.5}}, "agg_value": 0.5},
        }, self.tmp)
        self.tmp.close()
        return self.tmp.name

    def __exit__(self, *args):
        if self.tmp and Path(self.tmp.name).exists():
            Path(self.tmp.name).unlink(missing_ok=True)
