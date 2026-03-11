"""Tests for reference_logs loading in evals.metrics.base (retain trajectory by_step keys)."""

import json
import sys
from pathlib import Path

import pytest

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

    def test_prepare_kwargs_finds_by_step_keys_under_trajectory_all(self):
        """When retain file has by_step keys under trajectory_all (coalesced run), loader still finds them."""
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        mock_logs = {
            "trajectory_all": {
                "agg_value": {},
                "mia_min_k_by_step": {"0": {"agg_value": 0.2}, "1": {"agg_value": 0.3}},
                "forget_truth_ratio_by_step": {
                    "0": {"value_by_index": {"0": {"score": 0.5}}},
                    "1": {"value_by_index": {"0": {"score": 0.6}}},
                },
            },
            "run_info": {},
        }
        with patch.object(metric, "load_logs_from_file", return_value=mock_logs):
            kwargs = metric.prepare_kwargs_evaluate_metric(
                Mock(),
                "test_metric",
                reference_logs={
                    "retain_model_logs": {
                        "path": "/tmp/fake_retain.json",
                        "include": {},
                    }
                },
            )
        ref = kwargs["reference_logs"].get("retain_model_logs") or {}
        assert "retain_mia_by_step" in ref
        assert ref["retain_mia_by_step"] == mock_logs["trajectory_all"]["mia_min_k_by_step"]
        assert "retain_forget_tr_by_step" in ref
        assert ref["retain_forget_tr_by_step"] == mock_logs["trajectory_all"]["forget_truth_ratio_by_step"]


class TestReferenceLogsDoNotOverwriteWithNone:
    """When multiple include keys map to the same access_key (e.g. retain), do not overwrite an existing value with None when a later key is missing. When any requested key is missing, strict policy: do not use ref at all."""

    def test_loader_strict_when_any_key_missing(self):
        """Strict policy: when any requested include key is missing, do not use reference_logs for that ref (pop and set _required_but_missing)."""
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        mock_logs = {
            "mia_min_k": {"forget": {}, "holdout": {}, "auc": 0.5, "agg_value": 0.5},
        }
        with patch.object(metric, "load_logs_from_file", return_value=mock_logs):
            kwargs = metric.prepare_kwargs_evaluate_metric(
                Mock(),
                "test_metric",
                reference_logs={
                    "retain_model_logs": {
                        "path": "/tmp/fake_retain.json",
                        "include": {
                            "mia_min_k": {"access_key": "retain"},
                            "forget_truth_ratio": {"access_key": "retain"},
                        },
                    }
                },
            )
        ref_logs = kwargs.get("reference_logs", {})
        assert ref_logs.get("_required_but_missing") is True
        assert ref_logs.get("retain_model_logs") is None


class TestLoaderRetainScalarAndPrivleakNoTypeError:
    """Regression: loader output has numeric retain['agg_value'] and by_step keys; privleak with that reference_logs does not raise TypeError (T007)."""

    def test_loader_output_scalar_retain_and_by_step_privleak_no_type_error(self):
        """Fixture with scalar aggregate + by_step: loader yields numeric retain.agg_value; privleak(..., reference_logs=...) returns agg_value as number or None (T007)."""
        from evals.metrics.privacy import privleak
        from tests.fixtures.retain_reference import RETAIN_REFERENCE_MODE

        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        with patch.object(metric, "load_logs_from_file", return_value=RETAIN_REFERENCE_MODE):
            kwargs = metric.prepare_kwargs_evaluate_metric(
                Mock(),
                "test_metric",
                reference_logs={
                    "retain_model_logs": {
                        "path": "/tmp/fake_retain.json",
                        "include": {
                            "mia_min_k": {"access_key": "retain"},
                            "forget_truth_ratio": {"access_key": "retain"},
                        },
                    }
                },
            )
        ref_logs = kwargs.get("reference_logs", {}) or {}
        retain_logs = ref_logs.get("retain_model_logs") or {}
        assert "retain" in retain_logs
        retain = retain_logs["retain"]
        assert retain is not None
        agg = retain.get("agg_value")
        assert agg is None or isinstance(agg, (int, float)), "retain.agg_value must be scalar or None"
        assert "retain_mia_by_step" in retain_logs
        assert "retain_forget_tr_by_step" in retain_logs

        # Call privleak's underlying fn with loader output; must not raise TypeError (no int - dict).
        pre_compute = {"forget": {"agg_value": 0.26}}
        result = privleak._metric_fn(
            model=None,
            pre_compute=pre_compute,
            reference_logs=ref_logs,
            ref_value=0.5,
        )
        assert "agg_value" in result
        assert result["agg_value"] is None or isinstance(result["agg_value"], (int, float))


class TestPrivleakRelDiffDictAggValue:
    """T008: When retain['agg_value'] is dict, privleak/rel_diff log ERROR and return agg_value=None; ref_value not used."""

    def test_privleak_dict_agg_value_error_and_none(self, caplog):
        """reference_logs with retain.agg_value = dict → ERROR logged and result agg_value=None (contract test 3)."""
        from evals.metrics.privacy import privleak

        ref_logs = {"retain_model_logs": {"retain": {"agg_value": {"k": 1}}}}
        result = privleak._metric_fn(
            model=None,
            pre_compute={"forget": {"agg_value": 0.3}},
            reference_logs=ref_logs,
            ref_value=0.5,
        )
        assert result.get("agg_value") is None
        assert "not a number" in caplog.text


class TestLoaderNestedAggregate:
    """T009: Loader with only nested aggregate normalizes to scalar or sets _required_but_missing."""

    def test_loader_nested_extractable_normalizes_to_scalar(self):
        """Fixture with nested but extractable agg_value (e.g. full.steps.privleak) → loader outputs scalar retain.agg_value."""
        from tests.fixtures.retain_reference import NESTED_AGGREGATE_EXTRACTABLE

        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        with patch.object(metric, "load_logs_from_file", return_value=NESTED_AGGREGATE_EXTRACTABLE):
            kwargs = metric.prepare_kwargs_evaluate_metric(
                Mock(),
                "test_metric",
                reference_logs={
                    "retain_model_logs": {
                        "path": "/tmp/fake.json",
                        "include": {"mia_min_k": {"access_key": "retain"}},
                    }
                },
            )
        ref_logs = kwargs.get("reference_logs", {}) or {}
        assert ref_logs.get("_required_but_missing") is not True
        retain = (ref_logs.get("retain_model_logs") or {}).get("retain")
        assert retain is not None and isinstance(retain.get("agg_value"), (int, float))

    def test_loader_nested_not_extractable_sets_required_but_missing(self):
        """Fixture with nested agg_value that has no extractable number → _required_but_missing and retain not usable."""
        from tests.fixtures.retain_reference import NESTED_AGGREGATE_NOT_EXTRACTABLE

        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        with patch.object(metric, "load_logs_from_file", return_value=NESTED_AGGREGATE_NOT_EXTRACTABLE):
            kwargs = metric.prepare_kwargs_evaluate_metric(
                Mock(),
                "test_metric",
                reference_logs={
                    "retain_model_logs": {
                        "path": "/tmp/fake.json",
                        "include": {"mia_min_k": {"access_key": "retain"}},
                    }
                },
            )
        ref_logs = kwargs.get("reference_logs", {}) or {}
        assert ref_logs.get("_required_but_missing") is True
        retain = (ref_logs.get("retain_model_logs") or {}).get("retain")
        assert retain is None


class TestLoaderEmptyOrMalformedRetainJson:
    """T016: Empty or malformed retain JSON fails clearly; no silent use of invalid data. Uses tmp_path only."""

    def test_loader_malformed_json_raises(self, tmp_path):
        """When path points to malformed JSON, load fails with clear error (no silent use)."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {")
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        with pytest.raises((ValueError, json.JSONDecodeError)):
            metric.prepare_kwargs_evaluate_metric(
                Mock(),
                "test_metric",
                reference_logs={
                    "retain_model_logs": {"path": str(bad_file), "include": {"mia_min_k": {"access_key": "retain"}}}
                },
            )

    def test_loader_empty_json_missing_keys_sets_required_but_missing(self, tmp_path):
        """When path points to valid but empty/minimal JSON, requested keys missing → _required_but_missing (no silent use)."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("{}")
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(),
            "test_metric",
            reference_logs={
                "retain_model_logs": {"path": str(empty_file), "include": {"mia_min_k": {"access_key": "retain"}}}
            },
        )
        ref_logs = kwargs.get("reference_logs", {}) or {}
        assert ref_logs.get("_required_but_missing") is True
        assert ref_logs.get("retain_model_logs") is None


class TestRealJsonLoaderRoundTrip:
    """T017/T018/T019: Real JSON on tmp_path, no persistent temp files; loader and report structure validation."""

    def test_real_retain_json_on_tmp_path_loader_output_structure(self, tmp_path):
        """T017: Create retain-style JSON (programmatic dict), write to tmp_path, invoke loader with path; assert output has retain.agg_value scalar, retain_mia_by_step, retain_forget_tr_by_step."""
        from tests.fixtures.retain_reference import RETAIN_REFERENCE_MODE

        path = tmp_path / "retain_ref.json"
        path.write_text(json.dumps(RETAIN_REFERENCE_MODE))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(),
            "test_metric",
            reference_logs={
                "retain_model_logs": {
                    "path": str(path),
                    "include": {
                        "mia_min_k": {"access_key": "retain"},
                        "forget_truth_ratio": {"access_key": "retain"},
                    },
                }
            },
        )
        ref_logs = kwargs.get("reference_logs", {}) or {}
        rml = ref_logs.get("retain_model_logs") or {}
        assert "retain" in rml
        assert isinstance(rml["retain"].get("agg_value"), (int, float))
        assert "retain_mia_by_step" in rml
        assert "retain_forget_tr_by_step" in rml

    def test_report_json_structure_against_contract(self, tmp_path):
        """T018: Write results.json to tmp_path with contract keys; assert presence/shape of mia_min_k, forget_truth_ratio, by_step keys; load with loader and assert structure."""
        report = {
            "mia_min_k": {"agg_value": 0.28, "auc": 0.28},
            "forget_truth_ratio": {"agg_value": 0.82},
            "mia_min_k_by_step": {"0": {"agg_value": 0.27}, "1": {"agg_value": 0.29}},
            "forget_truth_ratio_by_step": {"0": {"value_by_index": {"0": {"score": 0.81}}}, "1": {"value_by_index": {"0": {"score": 0.83}}}},
        }
        path = tmp_path / "results.json"
        path.write_text(json.dumps(report))
        assert path.read_text()
        assert "mia_min_k" in report and "forget_truth_ratio" in report
        assert "mia_min_k_by_step" in report and "forget_truth_ratio_by_step" in report
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(),
            "test_metric",
            reference_logs={
                "retain_model_logs": {
                    "path": str(path),
                    "include": {"mia_min_k": {"access_key": "retain"}, "forget_truth_ratio": {"access_key": "retain"}},
                }
            },
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert rml.get("retain") and isinstance(rml["retain"].get("agg_value"), (int, float))
        assert "retain_mia_by_step" in rml and "retain_forget_tr_by_step" in rml

    def test_writer_shape_round_trip_via_loader(self, tmp_path):
        """T019: Dict matching writer (retain_reference_mode) shape → JSON in tmp_path → load via loader → assert full round-trip (retain scalar, by_step present)."""
        writer_shape = {
            "mia_min_k": {"agg_value": 0.27, "auc": 0.27},
            "forget_truth_ratio": {"agg_value": 0.85},
            "mia_min_k_by_step": {"5": {"agg_value": 0.26}, "10": {"agg_value": 0.28}},
            "forget_truth_ratio_by_step": {"5": {"value_by_index": {"0": {"score": 0.84}}}, "10": {"value_by_index": {"0": {"score": 0.86}}}},
        }
        path = tmp_path / "writer_output.json"
        path.write_text(json.dumps(writer_shape))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(),
            "test_metric",
            reference_logs={
                "retain_model_logs": {
                    "path": str(path),
                    "include": {"mia_min_k": {"access_key": "retain"}, "forget_truth_ratio": {"access_key": "retain"}},
                }
            },
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert rml.get("retain") and isinstance(rml["retain"].get("agg_value"), (int, float))
        assert rml.get("retain_mia_by_step") == writer_shape["mia_min_k_by_step"]
        assert rml.get("retain_forget_tr_by_step") == writer_shape["forget_truth_ratio_by_step"]
