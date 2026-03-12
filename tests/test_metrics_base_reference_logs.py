"""Tests for reference_logs loading in evals.metrics.base (retain trajectory by_step keys)."""

import json
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.base import UnlearningMetric, RetainReferenceValidationError
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
    """When any requested key is missing from file, strict policy: loader MUST fail (raise). No _required_but_missing, no ignoring."""

    def test_loader_strict_when_any_key_missing_must_fail(self):
        """Strict policy: when any requested include key is missing from file, loader MUST raise with clear reason."""
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        mock_logs = {
            "mia_min_k": {"forget": {}, "holdout": {}, "auc": 0.5, "agg_value": 0.5},
        }
        with patch.object(metric, "load_logs_from_file", return_value=mock_logs):
            with pytest.raises(Exception) as exc_info:
                metric.prepare_kwargs_evaluate_metric(
                    Mock(),
                    "test_metric",
                    reference_logs={
                        "retain_model_logs": {
                            "path": "/tmp/fake_retain.json",
                            "include": {
                                "mia_min_k": {"access_key": "retain"},
                                "forget_truth_ratio": {"access_key": "retain_ftr"},
                            },
                        }
                    },
                )
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "forget_truth_ratio" in msg or "missing" in msg or "invalid" in msg, (
            "Failure reason must be clear and understandable."
        )


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
                            "forget_truth_ratio": {"access_key": "retain_ftr"},
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
    """T008: When retain['agg_value'] is non-canonical (e.g. dict), privleak must fail (raise); no ref_value."""

    def test_privleak_dict_agg_value_error_and_none(self, caplog):
        """reference_logs with retain.agg_value = dict (non-canonical) → must raise; no fallback to ref_value."""
        from evals.metrics.base import RetainReferenceValidationError
        from evals.metrics.privacy import privleak

        ref_logs = {"retain_model_logs": {"retain": {"agg_value": {"k": 1}}}}
        with pytest.raises((RetainReferenceValidationError, ValueError, TypeError)):
            privleak._metric_fn(
                model=None,
                pre_compute={"forget": {"agg_value": 0.3}},
                reference_logs=ref_logs,
                ref_value=0.5,
            )
        assert "not a number" in caplog.text or "agg_value" in caplog.text.lower()


class TestLoaderNestedAggregate:
    """Strict canonical only: nested or non-canonical aggregate → loader MUST raise. No normalization, no _required_but_missing."""

    def test_loader_nested_aggregate_must_fail(self):
        """Strict canonical: mia_min_k with nested agg_value (e.g. full.steps.privleak) is NON-CANONICAL → loader MUST raise with clear reason."""
        from tests.fixtures.retain_reference import NESTED_AGGREGATE_EXTRACTABLE

        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        with patch.object(metric, "load_logs_from_file", return_value=NESTED_AGGREGATE_EXTRACTABLE):
            with pytest.raises(Exception) as exc_info:
                metric.prepare_kwargs_evaluate_metric(
                    Mock(),
                    "test_metric",
                    reference_logs={
                        "retain_model_logs": {
                            "path": "/tmp/fake.json",
                            "include": {"mia_min_k": {"access_key": "retain"}},
                        }
                    },
                )
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "mia_min_k" in msg or "canonical" in msg or "invalid" in msg or "nested" in msg, (
            "Failure reason must be clear and understandable."
        )

    def test_loader_nested_not_extractable_must_fail(self):
        """Fixture with nested agg_value that has no extractable number → loader MUST raise. No _required_but_missing."""
        from tests.fixtures.retain_reference import NESTED_AGGREGATE_NOT_EXTRACTABLE

        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        with patch.object(metric, "load_logs_from_file", return_value=NESTED_AGGREGATE_NOT_EXTRACTABLE):
            with pytest.raises(Exception) as exc_info:
                metric.prepare_kwargs_evaluate_metric(
                    Mock(),
                    "test_metric",
                    reference_logs={
                        "retain_model_logs": {
                            "path": "/tmp/fake.json",
                            "include": {"mia_min_k": {"access_key": "retain"}},
                        }
                    },
                )
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "mia_min_k" in msg or "canonical" in msg or "invalid" in msg or "missing" in msg, (
            "Failure reason must be clear and understandable."
        )


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

    def test_loader_empty_json_missing_keys_must_fail(self, tmp_path):
        """When path points to valid but empty JSON, requested keys missing → loader MUST raise. No _required_but_missing."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("{}")
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        with pytest.raises(Exception) as exc_info:
            metric.prepare_kwargs_evaluate_metric(
                Mock(),
                "test_metric",
                reference_logs={
                    "retain_model_logs": {"path": str(empty_file), "include": {"mia_min_k": {"access_key": "retain"}}}
                },
            )
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "mia_min_k" in msg or "missing" in msg or "invalid" in msg, (
            "Failure reason must be clear and understandable."
        )


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
                        "forget_truth_ratio": {"access_key": "retain_ftr"},
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
                    "include": {"mia_min_k": {"access_key": "retain"}, "forget_truth_ratio": {"access_key": "retain_ftr"}},
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
                    "include": {"mia_min_k": {"access_key": "retain"}, "forget_truth_ratio": {"access_key": "retain_ftr"}},
                }
            },
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert rml.get("retain") and isinstance(rml["retain"].get("agg_value"), (int, float))
        assert rml.get("retain_mia_by_step") == writer_shape["mia_min_k_by_step"]
        assert rml.get("retain_forget_tr_by_step") == writer_shape["forget_truth_ratio_by_step"]


class TestSlotBugTwoKeysOneSlot:
    """Regression: when config has both mia_min_k and forget_truth_ratio → retain, loader overwrites;
    if last value (forget_truth_ratio) does not normalize to scalar, privleak gets no ref. Load real-shaped JSON."""

    def test_load_json_with_both_keys_retain_must_be_scalar_for_privleak(self, tmp_path):
        """Load canonical JSON with both keys; config: mia_min_k → retain, forget_truth_ratio → retain_ftr.
        Assert: retain has scalar (mia_min_k), retain_ftr has forget_truth_ratio; privleak and ks_test get correct refs."""
        from tests.fixtures.retain_reference import RETAIN_REFERENCE_MODE

        path = tmp_path / "retain_report.json"
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
                        "forget_truth_ratio": {"access_key": "retain_ftr"},
                    },
                }
            },
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        retain = rml.get("retain")
        assert retain is not None, "retain slot should be set"
        agg = retain.get("agg_value") if isinstance(retain, dict) else None
        assert isinstance(agg, (int, float)), "retain.agg_value must be a scalar so privleak can use it"
        assert rml.get("retain_ftr") is not None, "retain_ftr slot should be set for ks_test"


# --- Spec 005: Loader two slots (A), canonical (B), one key (C), strict (D), by_step (E) ---

class TestLoaderTwoSlotsNoOverwrite:
    """A: Two slots (retain, retain_ftr); no overwrite when both keys in config with distinct access_key."""

    def test_A1_both_keys_two_access_keys_both_slots_present(self, tmp_path):
        """A1: JSON with mia_min_k + forget_truth_ratio (canonical); include mia_min_k→retain, forget_truth_ratio→retain_ftr; both retain and retain_ftr in output."""
        canonical = {
            "mia_min_k": {"agg_value": 0.27},
            "forget_truth_ratio": {"value_by_index": {"0": {"score": 0.85}}, "agg_value": 0.85},
        }
        path = tmp_path / "retain.json"
        path.write_text(json.dumps(canonical))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(),
            "test_metric",
            reference_logs={
                "retain_model_logs": {
                    "path": str(path),
                    "include": {
                        "mia_min_k": {"access_key": "retain"},
                        "forget_truth_ratio": {"access_key": "retain_ftr"},
                    },
                }
            },
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert "retain" in rml, "retain slot (mia_min_k) must be present"
        assert "retain_ftr" in rml, "retain_ftr slot (forget_truth_ratio) must be present"
        assert isinstance(rml["retain"].get("agg_value"), (int, float))
        assert rml["retain_ftr"].get("value_by_index") or isinstance(rml["retain_ftr"].get("agg_value"), (int, float))


class TestLoaderCanonicalOnly:
    """B: Canonical form only; non-canonical rejected."""

    def test_B1_mia_min_k_canonical_agg_value_accepted(self, tmp_path):
        """B1: mia_min_k = {agg_value: float} → loader accepts; retain slot has agg_value."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"mia_min_k": {"agg_value": 0.3}}))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "test_metric",
            reference_logs={"retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}}}},
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert rml.get("retain") is not None
        assert isinstance(rml["retain"].get("agg_value"), (int, float))

    def test_B2_mia_min_k_non_canonical_loader_must_fail(self, tmp_path):
        """Strict canonical: mia_min_k = {auc: 0.3} only (no agg_value) is NON-CANONICAL → loader MUST raise with clear reason. No fallback, no _required_but_missing."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"mia_min_k": {"auc": 0.3}}))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        with pytest.raises(Exception) as exc_info:
            metric.prepare_kwargs_evaluate_metric(
                Mock(), "test_metric",
                reference_logs={"retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}}}},
            )
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "mia_min_k" in msg or "canonical" in msg or "agg_value" in msg or "invalid" in msg, (
            "Failure reason must be clear and understandable."
        )

    def test_B3_mia_min_k_missing_agg_value_loader_must_fail(self, tmp_path):
        """Strict canonical: mia_min_k missing agg_value (e.g. empty dict) → loader MUST raise. No fallback, no _required_but_missing."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"mia_min_k": {}}))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        with pytest.raises(Exception) as exc_info:
            metric.prepare_kwargs_evaluate_metric(
                Mock(), "test_metric",
                reference_logs={"retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}}}},
            )
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "mia_min_k" in msg or "canonical" in msg or "agg_value" in msg or "invalid" in msg or "missing" in msg, (
            "Failure reason must be clear and understandable."
        )

    def test_B4_forget_truth_ratio_canonical_retain_ftr_only(self, tmp_path):
        """B4: forget_truth_ratio canonical → loader sets retain_ftr only (no fallback to retain)."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"forget_truth_ratio": {"value_by_index": {"0": {"score": 0.8}}, "agg_value": 0.8}}))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "test_metric",
            reference_logs={"retain_model_logs": {"path": str(path), "include": {"forget_truth_ratio": {"access_key": "retain_ftr"}}}},
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert rml.get("retain_ftr") is not None, "Loader must set retain_ftr from forget_truth_ratio."
        assert rml.get("retain") is None, "retain must not be set when only forget_truth_ratio in include; no fallback."
        slot = rml["retain_ftr"]
        assert slot.get("value_by_index") or isinstance(slot.get("agg_value"), (int, float))

    def test_B5_forget_truth_ratio_non_canonical_loader_must_fail(self, tmp_path):
        """Strict canonical: forget_truth_ratio with non-canonical shape (e.g. agg_value is list) → loader MUST raise. No _required_but_missing."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"forget_truth_ratio": {"agg_value": [0.8, 0.9]}}))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        with pytest.raises(Exception) as exc_info:
            metric.prepare_kwargs_evaluate_metric(
                Mock(), "test_metric",
                reference_logs={"retain_model_logs": {"path": str(path), "include": {"forget_truth_ratio": {"access_key": "retain_ftr"}}}},
            )
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "forget_truth_ratio" in msg or "canonical" in msg or "invalid" in msg, (
            "Failure reason must be clear and understandable."
        )


class TestLoaderOneKeyOnly:
    """C: Include only one key → only that slot set."""

    def test_C1_include_only_mia_min_k_only_retain_slot(self, tmp_path):
        """C1: include only mia_min_k → retain; retain_ftr absent."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"mia_min_k": {"agg_value": 0.3}}))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "test_metric",
            reference_logs={"retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}}}},
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert "retain" in rml
        assert rml.get("retain_ftr") is None

    def test_C2_include_only_forget_truth_ratio_only_retain_ftr_slot(self, tmp_path):
        """C2: include only forget_truth_ratio → only retain_ftr set; retain absent (no fallback)."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"forget_truth_ratio": {"agg_value": 0.8}}))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "test_metric",
            reference_logs={"retain_model_logs": {"path": str(path), "include": {"forget_truth_ratio": {"access_key": "retain_ftr"}}}},
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert rml.get("retain_ftr") is not None
        assert rml.get("retain") is None


class TestLoaderMissingKeysStrict:
    """D: Missing keys, strict, malformed."""

    def test_D1_config_requests_both_file_has_only_mia_min_k_loader_must_fail(self, tmp_path):
        """D1: Config requests both keys; file has only mia_min_k → reference incomplete; loader MUST raise. No _required_but_missing."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"mia_min_k": {"agg_value": 0.3}}))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        with pytest.raises(Exception) as exc_info:
            metric.prepare_kwargs_evaluate_metric(
                Mock(), "test_metric",
                reference_logs={
                    "retain_model_logs": {
                        "path": str(path),
                        "include": {
                            "mia_min_k": {"access_key": "retain"},
                            "forget_truth_ratio": {"access_key": "retain_ftr"},
                        },
                    }
                },
            )
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "forget_truth_ratio" in msg or "missing" in msg or "invalid" in msg, (
            "Failure reason must be clear and understandable."
        )

    def test_D2_config_requests_one_key_file_empty_loader_must_fail(self, tmp_path):
        """D2: Config requests one key; file empty → reference invalid; loader MUST raise. No _required_but_missing."""
        path = tmp_path / "r.json"
        path.write_text("{}")
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        with pytest.raises(Exception) as exc_info:
            metric.prepare_kwargs_evaluate_metric(
                Mock(), "test_metric",
                reference_logs={"retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}}}},
            )
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "mia_min_k" in msg or "missing" in msg or "invalid" in msg, (
            "Failure reason must be clear and understandable."
        )


class TestLoaderByStepCanonical:
    """E: By_step canonical shape."""

    def test_E1_file_has_by_step_canonical_loader_exposes_retain_mia_and_retain_ftr_by_step(self, tmp_path):
        """E1: File has mia_min_k_by_step and forget_truth_ratio_by_step canonical → retain_mia_by_step, retain_forget_tr_by_step in output."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({
            "mia_min_k": {"agg_value": 0.27},
            "forget_truth_ratio": {"agg_value": 0.85},
            "mia_min_k_by_step": {"0": {"agg_value": 0.25}, "1": {"agg_value": 0.28}},
            "forget_truth_ratio_by_step": {"0": {"value_by_index": {"0": {"score": 0.84}}}, "1": {"value_by_index": {"0": {"score": 0.85}}}},
        }))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "test_metric",
            reference_logs={
                "retain_model_logs": {
                    "path": str(path),
                    "include": {"mia_min_k": {"access_key": "retain"}, "forget_truth_ratio": {"access_key": "retain_ftr"}},
                }
            },
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert "retain_mia_by_step" in rml
        assert "retain_forget_tr_by_step" in rml
        assert rml["retain_mia_by_step"]["0"]["agg_value"] == 0.25
        assert "value_by_index" in list(rml["retain_forget_tr_by_step"].values())[0]


# --- Loader edge cases and scenarios ---

class TestLoaderEdgeCases:
    """Loader: edge cases – agg_value types, null, empty value_by_index, by_step shape, extra keys, etc."""

    def test_loader_mia_min_k_agg_value_string_must_fail(self, tmp_path):
        """Strict canonical: mia_min_k.agg_value as string (e.g. "0.3") is non-canonical → loader MUST raise. No coercion."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"mia_min_k": {"agg_value": "0.3"}}))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        with pytest.raises(Exception) as exc_info:
            metric.prepare_kwargs_evaluate_metric(
                Mock(), "test_metric",
                reference_logs={"retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}}}},
            )
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "agg_value" in msg or "canonical" in msg or "invalid" in msg, (
            "Failure reason must be clear and understandable."
        )

    def test_loader_mia_min_k_agg_value_int_accepted(self, tmp_path):
        """Canonical: mia_min_k.agg_value as int (e.g. 0) is valid; loader accepts."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"mia_min_k": {"agg_value": 0}}))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "test_metric",
            reference_logs={"retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}}}},
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert (rml.get("retain") or {}).get("agg_value") == 0

    def test_loader_mia_min_k_agg_value_null_must_fail(self, tmp_path):
        """Strict canonical: mia_min_k.agg_value null → loader MUST raise. No valid retain from null."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"mia_min_k": {"agg_value": None}}))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        with pytest.raises(Exception) as exc_info:
            metric.prepare_kwargs_evaluate_metric(
                Mock(), "test_metric",
                reference_logs={"retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}}}},
            )
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "agg_value" in msg or "canonical" in msg or "invalid" in msg or "null" in msg, (
            "Failure reason must be clear and understandable."
        )

    def test_loader_forget_truth_ratio_empty_value_by_index_retain_ftr_only(self, tmp_path):
        """Canonical: forget_truth_ratio with value_by_index {} is valid shape; loader sets retain_ftr only (no fallback)."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"forget_truth_ratio": {"value_by_index": {}, "agg_value": 0.8}}))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "test_metric",
            reference_logs={"retain_model_logs": {"path": str(path), "include": {"forget_truth_ratio": {"access_key": "retain_ftr"}}}},
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert rml.get("retain_ftr") is not None
        assert rml.get("retain") is None
        slot = rml["retain_ftr"]
        assert "value_by_index" in slot or "agg_value" in slot

    def test_loader_by_step_single_step_canonical(self, tmp_path):
        """Canonical: file with single step in by_step; loader exposes retain_mia_by_step and retain_forget_tr_by_step."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({
            "mia_min_k": {"agg_value": 0.27},
            "forget_truth_ratio": {"agg_value": 0.85},
            "mia_min_k_by_step": {"0": {"agg_value": 0.26}},
            "forget_truth_ratio_by_step": {"0": {"value_by_index": {"0": {"score": 0.85}}}},
        }))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "test_metric",
            reference_logs={"retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}, "forget_truth_ratio": {"access_key": "retain_ftr"}}}},
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert "retain_mia_by_step" in rml and len(rml["retain_mia_by_step"]) == 1
        assert "retain_forget_tr_by_step" in rml and len(rml["retain_forget_tr_by_step"]) == 1

    def test_loader_config_requests_nonexistent_key_must_fail(self, tmp_path):
        """Config requests key not in file (e.g. forget_truth_ratio missing) → loader MUST raise."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"mia_min_k": {"agg_value": 0.3}}))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        with pytest.raises(Exception) as exc_info:
            metric.prepare_kwargs_evaluate_metric(
                Mock(), "test_metric",
                reference_logs={"retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}, "forget_truth_ratio": {"access_key": "retain_ftr"}}}},
            )
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "forget_truth_ratio" in msg or "missing" in msg or "invalid" in msg, (
            "Failure reason must be clear and understandable."
        )

    def test_loader_mia_min_k_key_missing_must_fail(self, tmp_path):
        """File has forget_truth_ratio but not mia_min_k; config requests both → loader MUST raise."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"forget_truth_ratio": {"agg_value": 0.85}}))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        with pytest.raises(Exception) as exc_info:
            metric.prepare_kwargs_evaluate_metric(
                Mock(), "test_metric",
                reference_logs={"retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}, "forget_truth_ratio": {"access_key": "retain_ftr"}}}},
            )
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "mia_min_k" in msg or "missing" in msg or "invalid" in msg, (
            "Failure reason must be clear and understandable."
        )

    def test_loader_by_step_under_trajectory_all_found(self, tmp_path):
        """File has by_step only under trajectory_all; loader finds them (existing behavior)."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({
            "trajectory_all": {
                "mia_min_k": {"agg_value": 0.27},
                "forget_truth_ratio": {"agg_value": 0.85},
                "mia_min_k_by_step": {"0": {"agg_value": 0.26}},
                "forget_truth_ratio_by_step": {"0": {"value_by_index": {"0": {"score": 0.85}}}},
            },
        }))
        metric = UnlearningMetric("test_metric", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "test_metric",
            reference_logs={"retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}, "forget_truth_ratio": {"access_key": "retain_ftr"}}}},
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert rml.get("retain_mia_by_step") is not None
        assert rml.get("retain_forget_tr_by_step") is not None


class TestLoaderSameAccessKeyRejected:
    """A2: Config with both keys mapping to same access_key (e.g. both → retain) must raise; no silent overwrite."""

    def test_A2_both_keys_same_access_key_retain_loader_raises(self, tmp_path):
        """Both mia_min_k and forget_truth_ratio with access_key retain → loader raises distinct access_key required."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({
            "mia_min_k": {"agg_value": 0.27},
            "forget_truth_ratio": {"value_by_index": {"0": {"score": 0.85}}, "agg_value": 0.85},
        }))
        metric = UnlearningMetric("m", _dummy_metric_fn)
        with pytest.raises((RetainReferenceValidationError, ValueError)) as exc_info:
            metric.prepare_kwargs_evaluate_metric(
                Mock(), "m",
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
        msg = str(exc_info.value).lower()
        assert "access_key" in msg or "distinct" in msg


class TestLoaderByStepNonCanonicalRejected:
    """E3: By_step value non-canonical (e.g. step → string or wrong shape) → loader rejects."""

    def test_E3_mia_min_k_by_step_value_non_canonical_loader_raises(self, tmp_path):
        """mia_min_k_by_step has a step value that is not canonical (e.g. string) → loader raises."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({
            "mia_min_k": {"agg_value": 0.27},
            "mia_min_k_by_step": {"0": {"agg_value": 0.25}, "1": "not a dict"},
        }))
        metric = UnlearningMetric("m", _dummy_metric_fn)
        with pytest.raises((RetainReferenceValidationError, ValueError)) as exc_info:
            metric.prepare_kwargs_evaluate_metric(
                Mock(), "m",
                reference_logs={
                    "retain_model_logs": {
                        "path": str(path),
                        "include": {"mia_min_k": {"access_key": "retain"}},
                    }
                },
            )
        msg = str(exc_info.value).lower()
        assert "by_step" in msg or "canonical" in msg or "agg_value" in msg

    def test_E3_forget_truth_ratio_by_step_value_non_canonical_loader_raises(self, tmp_path):
        """forget_truth_ratio_by_step has a step value that is not canonical → loader raises."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({
            "mia_min_k": {"agg_value": 0.27},
            "forget_truth_ratio": {"agg_value": 0.85},
            "forget_truth_ratio_by_step": {"0": {"value_by_index": {"0": {"score": 0.8}}}, "1": []},
        }))
        metric = UnlearningMetric("m", _dummy_metric_fn)
        with pytest.raises((RetainReferenceValidationError, ValueError)) as exc_info:
            metric.prepare_kwargs_evaluate_metric(
                Mock(), "m",
                reference_logs={
                    "retain_model_logs": {
                        "path": str(path),
                        "include": {"mia_min_k": {"access_key": "retain"}, "forget_truth_ratio": {"access_key": "retain_ftr"}},
                    }
                },
            )
        msg = str(exc_info.value).lower()
        assert "by_step" in msg or "canonical" in msg


class TestLoaderPathFileMissing:
    """Path provided but file does not exist → load raises; no silent use."""

    def test_D_path_nonexistent_file_raises(self, tmp_path):
        """reference_logs path points to non-existent file → loader raises (e.g. ValueError from load_logs_from_file)."""
        nonexistent = tmp_path / "does_not_exist.json"
        assert not nonexistent.exists()
        metric = UnlearningMetric("m", _dummy_metric_fn)
        with pytest.raises((RetainReferenceValidationError, ValueError)) as exc_info:
            metric.prepare_kwargs_evaluate_metric(
                Mock(), "m",
                reference_logs={"retain_model_logs": {"path": str(nonexistent), "include": {"mia_min_k": {"access_key": "retain"}}}},
            )
        msg = str(exc_info.value).lower()
        assert "exist" in msg or "don't" in msg or "not found" in msg or "could not" in msg
