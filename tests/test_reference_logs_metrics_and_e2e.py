"""Tests for reference_logs contract: privleak/rel_diff (F), ks_test (G), E2E (I), Config (J), Trajectory (H). Spec 005."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics import METRICS_REGISTRY
from evals.metrics.base import UnlearningMetric, RetainReferenceValidationError


def _dummy_metric_fn(**kwargs):
    return {"agg_value": 0.5}


# --- F: privleak / rel_diff ---

class TestPrivleakRelDiffSlotAndShape:
    """F: reference_logs with retain slot; privleak/rel_diff return numeric or None."""

    def test_F1_retain_slot_scalar_agg_value_privleak_returns_numeric(self):
        """F1: reference_logs with retain.agg_value = scalar → privleak returns numeric."""
        pl = METRICS_REGISTRY.get("privleak")
        assert pl is not None
        ref = {"retain_model_logs": {"retain": {"agg_value": 0.5}}}
        res = pl._metric_fn(
            model=None,
            pre_compute={"forget": {"agg_value": 0.3}},
            reference_logs=ref,
            ref_value=0.5,
        )
        assert res.get("agg_value") is not None
        assert isinstance(res["agg_value"], (int, float))

    def test_F2_only_retain_ftr_no_retain_privleak_must_fail(self):
        """F2: reference_logs with only retain_ftr (no retain) → privleak must fail (raise). No fallback, no None."""
        pl = METRICS_REGISTRY.get("privleak")
        ref = {"retain_model_logs": {"retain_ftr": {"value_by_index": {"0": {"score": 0.8}}}, "retain": None}}
        with pytest.raises(Exception) as exc_info:
            pl._metric_fn(
                model=None,
                pre_compute={"forget": {"agg_value": 0.3}},
                reference_logs=ref,
                ref_value=0.5,
            )
        assert exc_info.value is not None
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "reference" in msg or "missing" in msg or "invalid" in msg, (
            "Failure reason must be clear and understandable."
        )

    def test_F4_invalid_reference_privleak_must_fail(self):
        """F4: Invalid reference (e.g. required slot missing or _required_but_missing) → privleak must fail (raise). No None."""
        pl = METRICS_REGISTRY.get("privleak")
        ref = {"retain_model_logs": {"_required_but_missing": True}, "_required_but_missing": True}
        with pytest.raises(Exception) as exc_info:
            pl._metric_fn(
                model=None,
                pre_compute={"forget": {"agg_value": 0.3}},
                reference_logs=ref,
                ref_value=0.5,
            )
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "reference" in msg or "missing" in msg or "invalid" in msg, (
            "Failure reason must be clear and understandable."
        )

    def test_F5_no_reference_path_privleak_may_use_ref_value(self):
        """F5: No reference path (reference_logs empty) → no reference file to validate; privleak may use ref_value."""
        pl = METRICS_REGISTRY.get("privleak")
        ref = {}
        res = pl._metric_fn(
            model=None,
            pre_compute={"forget": {"agg_value": 0.3}},
            reference_logs=ref,
            ref_value=0.5,
        )
        assert res.get("agg_value") is not None


# --- G: ks_test (forget_quality) ---

class TestKsTestSlotAndShape:
    """G: ks_test reads retain_ftr only (no fallback). Returns p-value when ref valid; fails when ref invalid."""

    def test_G1_retain_ftr_value_by_index_ks_test_returns_pvalue(self):
        """G1: reference_logs with retain_ftr slot, value_by_index with scores → ks_test returns numeric p-value."""
        ks = METRICS_REGISTRY.get("ks_test")
        assert ks is not None
        ref = {
            "retain_model_logs": {
                "retain_ftr": {"value_by_index": {"0": {"score": 0.8}, "1": {"score": 0.85}}},
            }
        }
        pre = {"forget": {"value_by_index": {"0": {"score": 0.7}, "1": {"score": 0.75}}}}
        res = ks._metric_fn(model=None, pre_compute=pre, reference_logs=ref)
        assert "agg_value" in res
        if res["agg_value"] is not None:
            assert isinstance(res["agg_value"], (int, float))

    def test_G2_only_retain_no_retain_ftr_ks_test_must_fail(self):
        """G2: Only retain (MIA slot), no retain_ftr → ks_test must fail (raise). No fallback to retain."""
        ks = METRICS_REGISTRY.get("ks_test")
        ref = {"retain_model_logs": {"retain": {"agg_value": 0.5}}}
        pre = {"forget": {"value_by_index": {"0": {"score": 0.7}}}}
        with pytest.raises(Exception) as exc_info:
            ks._metric_fn(model=None, pre_compute=pre, reference_logs=ref)
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "reference" in msg or "missing" in msg or "invalid" in msg or "ftr" in msg, (
            "Failure reason must be clear and understandable."
        )

    def test_G4_invalid_reference_ks_test_must_fail(self):
        """G4: Invalid reference (_required_but_missing or missing retain_ftr) → ks_test must fail (raise). No None."""
        ks = METRICS_REGISTRY.get("ks_test")
        ref = {"retain_model_logs": {"_required_but_missing": True}}
        pre = {"forget": {"value_by_index": {"0": {"score": 0.7}}}}
        with pytest.raises(Exception) as exc_info:
            ks._metric_fn(model=None, pre_compute=pre, reference_logs=ref)
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "reference" in msg or "missing" in msg or "invalid" in msg, (
            "Failure reason must be clear and understandable."
        )


# --- I: E2E ---

class TestE2EIntegration:
    """I: Load canonical JSON → loader → privleak + ks_test; no TypeError."""

    def test_I1_load_canonical_call_privleak_and_ks_test_no_type_error(self, tmp_path):
        """I1: Load canonical JSON (both keys, two slots); call privleak(loader_output), ks_test(loader_output); no TypeError."""
        path = tmp_path / "retain.json"
        path.write_text(json.dumps({
            "mia_min_k": {"agg_value": 0.27},
            "forget_truth_ratio": {"value_by_index": {"0": {"score": 0.85}}, "agg_value": 0.85},
        }))
        metric = UnlearningMetric("m", _dummy_metric_fn)
        include = {
            "mia_min_k": {"access_key": "retain"},
            "forget_truth_ratio": {"access_key": "retain_ftr"},
        }
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "m",
            reference_logs={"retain_model_logs": {"path": str(path), "include": include}},
        )
        ref = kwargs.get("reference_logs", {}) or {}
        pl = METRICS_REGISTRY.get("privleak")
        ks = METRICS_REGISTRY.get("ks_test")
        try:
            pl_res = pl._metric_fn(model=None, pre_compute={"forget": {"agg_value": 0.3}}, reference_logs=ref, ref_value=0.5)
            ks_res = ks._metric_fn(model=None, pre_compute={"forget": {"value_by_index": {"0": {"score": 0.7}}}}, reference_logs=ref)
        except TypeError as e:
            pytest.fail(f"E2E must not raise TypeError: {e}")
        assert "agg_value" in pl_res and "agg_value" in ks_res


# --- J: Config ---

class TestConfigDrivenLoader:
    """J: trajectory_all include mia_min_k→retain, forget_truth_ratio→retain_ftr; one-key configs."""

    def test_J1_trajectory_include_two_access_keys_fills_both_slots(self, tmp_path):
        """J1: include mia_min_k→retain, forget_truth_ratio→retain_ftr; file has both → both slots filled."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({
            "mia_min_k": {"agg_value": 0.27},
            "forget_truth_ratio": {"agg_value": 0.85},
        }))
        metric = UnlearningMetric("m", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "m",
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
        assert "retain" in rml
        assert "retain_ftr" in rml


class TestE2ERealConfig:
    """E2E: load real config file (trajectory_all or forget_quality) and validate reference_logs flow."""

    def test_real_config_trajectory_all_reference_logs_include_has_retain_ftr(self):
        """Real trajectory_all.yaml must use access_key retain_ftr for forget_truth_ratio (ks_test reads retain_ftr only)."""
        configs_dir = repo_root / "configs" / "eval" / "tofu_metrics"
        yaml_path = configs_dir / "trajectory_all.yaml"
        if not yaml_path.exists():
            pytest.skip("trajectory_all.yaml not found (run from open-unlearning root)")
        text = yaml_path.read_text()
        assert "forget_truth_ratio" in text and "retain_ftr" in text, (
            "trajectory_all.yaml reference_logs.include must map forget_truth_ratio to access_key: retain_ftr"
        )
        assert "mia_min_k" in text and "retain" in text

    def test_real_config_forget_quality_has_retain_ftr(self):
        """Real forget_quality.yaml must use access_key retain_ftr for forget_truth_ratio."""
        configs_dir = repo_root / "configs" / "eval" / "tofu_metrics"
        yaml_path = configs_dir / "forget_quality.yaml"
        if not yaml_path.exists():
            pytest.skip("forget_quality.yaml not found")
        text = yaml_path.read_text()
        assert "retain_ftr" in text, "forget_quality.yaml must use access_key: retain_ftr for ks_test"

    def test_load_and_validate_with_real_config_structure_succeeds(self, tmp_path):
        """Use the same reference_logs.include structure as trajectory_all.yaml; load canonical file → both slots filled."""
        from evals.metrics.base import load_and_validate_reference

        path = tmp_path / "retain.json"
        path.write_text(json.dumps({
            "mia_min_k": {"agg_value": 0.27},
            "forget_truth_ratio": {"value_by_index": {"0": {"score": 0.85}}, "agg_value": 0.85},
        }))
        reference_logs_cfgs = {
            "retain_model_logs": {
                "path": str(path),
                "include": {
                    "mia_min_k": {"access_key": "retain"},
                    "forget_truth_ratio": {"access_key": "retain_ftr"},
                },
            },
        }

        def load_fn(p):
            with open(p) as f:
                return json.load(f)

        result = load_and_validate_reference(reference_logs_cfgs, load_fn)
        assert "retain_model_logs" in result
        rml = result["retain_model_logs"]
        assert rml.get("retain") is not None and isinstance(rml["retain"].get("agg_value"), (int, float))
        assert rml.get("retain_ftr") is not None


# --- Metrics and E2E edge cases ---

class TestPrivleakEdgeCases:
    """privleak: edge cases – agg_value 0, 1, missing key, empty retain_model_logs."""

    def test_privleak_retain_agg_value_zero_returns_numeric(self):
        """retain.agg_value = 0.0 → privleak returns numeric (division guarded)."""
        pl = METRICS_REGISTRY.get("privleak")
        ref = {"retain_model_logs": {"retain": {"agg_value": 0.0}}}
        res = pl._metric_fn(model=None, pre_compute={"forget": {"agg_value": 0.1}}, reference_logs=ref, ref_value=0.5)
        assert "agg_value" in res
        assert res["agg_value"] is not None and isinstance(res["agg_value"], (int, float))

    def test_privleak_retain_agg_value_one_returns_numeric(self):
        """retain.agg_value = 1.0 → privleak returns numeric."""
        pl = METRICS_REGISTRY.get("privleak")
        ref = {"retain_model_logs": {"retain": {"agg_value": 1.0}}}
        res = pl._metric_fn(model=None, pre_compute={"forget": {"agg_value": 0.5}}, reference_logs=ref, ref_value=0.5)
        assert "agg_value" in res
        assert res["agg_value"] is not None

    def test_privleak_retain_model_logs_empty_dict_must_fail(self):
        """reference_logs = {retain_model_logs: {}} → no retain key → must fail (raise). No fallback."""
        pl = METRICS_REGISTRY.get("privleak")
        ref = {"retain_model_logs": {}}
        with pytest.raises(Exception) as exc_info:
            pl._metric_fn(model=None, pre_compute={"forget": {"agg_value": 0.3}}, reference_logs=ref, ref_value=0.5)
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "reference" in msg or "missing" in msg or "invalid" in msg, (
            "Failure reason must be clear and understandable."
        )


class TestKsTestEdgeCases:
    """ks_test: edge cases – single sample, many samples, empty value_by_index, None score."""

    def test_ks_test_single_sample_each_side_returns_value_or_none(self):
        """Single sample in forget and retain value_by_index; ks_test returns agg_value (may be None if test undefined)."""
        ks = METRICS_REGISTRY.get("ks_test")
        ref = {"retain_model_logs": {"retain_ftr": {"value_by_index": {"0": {"score": 0.8}}}}}
        pre = {"forget": {"value_by_index": {"0": {"score": 0.7}}}}
        res = ks._metric_fn(model=None, pre_compute=pre, reference_logs=ref)
        assert "agg_value" in res

    def test_ks_test_many_samples_returns_pvalue(self):
        """Many samples in value_by_index → ks_test returns numeric p-value."""
        ks = METRICS_REGISTRY.get("ks_test")
        ref = {"retain_model_logs": {"retain_ftr": {"value_by_index": {str(i): {"score": 0.8 + i * 0.01} for i in range(20)}}}}
        pre = {"forget": {"value_by_index": {str(i): {"score": 0.7 + i * 0.01} for i in range(20)}}}
        res = ks._metric_fn(model=None, pre_compute=pre, reference_logs=ref)
        assert "agg_value" in res
        if res["agg_value"] is not None:
            assert isinstance(res["agg_value"], (int, float)) and 0 <= res["agg_value"] <= 1

    def test_ks_test_retain_empty_value_by_index_returns_none_or_warning(self):
        """retain_ftr with value_by_index {} or no scores → ks_test returns None or warning."""
        ks = METRICS_REGISTRY.get("ks_test")
        ref = {"retain_model_logs": {"retain_ftr": {"value_by_index": {}, "agg_value": 0.8}}}
        pre = {"forget": {"value_by_index": {"0": {"score": 0.7}}}}
        res = ks._metric_fn(model=None, pre_compute=pre, reference_logs=ref)
        assert "agg_value" in res


class TestE2EEdgeCases:
    """E2E: edge cases – minimal canonical, both metrics with same ref_logs, rel_diff."""

    def test_e2e_minimal_canonical_both_keys_load_and_both_metrics_called(self, tmp_path):
        """Minimal canonical (only agg_value for both); load then call privleak and ks_test; no exception."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"mia_min_k": {"agg_value": 0.27}, "forget_truth_ratio": {"agg_value": 0.85}}))
        metric = UnlearningMetric("m", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "m",
            reference_logs={"retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}, "forget_truth_ratio": {"access_key": "retain_ftr"}}}},
        )
        ref = kwargs.get("reference_logs", {}) or {}
        pl = METRICS_REGISTRY.get("privleak")
        ks = METRICS_REGISTRY.get("ks_test")
        pl_res = pl._metric_fn(model=None, pre_compute={"forget": {"agg_value": 0.3}}, reference_logs=ref, ref_value=0.5)
        ks_res = ks._metric_fn(model=None, pre_compute={"forget": {"value_by_index": {"0": {"score": 0.7}}}}, reference_logs=ref)
        assert "agg_value" in pl_res and "agg_value" in ks_res

    def test_e2e_rel_diff_with_retain_slot(self):
        """rel_diff with retain.agg_value scalar returns numeric (same contract as privleak for ref)."""
        rel = METRICS_REGISTRY.get("rel_diff")
        if rel is None:
            pytest.skip("rel_diff not registered")
        ref = {"retain_model_logs": {"retain": {"agg_value": 0.5}}}
        res = rel._metric_fn(model=None, pre_compute={"forget": {"agg_value": 0.4}}, reference_logs=ref, ref_value=0.5)
        assert "agg_value" in res
        assert res["agg_value"] is not None


# --- F6: rel_diff parity with privleak (same slot, same pass/fail behavior) ---

class TestRelDiffSlotAndShape:
    """F6: rel_diff reads same slot as privleak; same tests F1–F5 style for that slot."""

    def test_rel_diff_F1_retain_slot_scalar_returns_numeric(self):
        """rel_diff with retain.agg_value scalar → returns numeric."""
        rel = METRICS_REGISTRY.get("rel_diff")
        if rel is None:
            pytest.skip("rel_diff not registered")
        ref = {"retain_model_logs": {"retain": {"agg_value": 0.5}}}
        res = rel._metric_fn(model=None, pre_compute={"forget": {"agg_value": 0.3}}, reference_logs=ref, ref_value=0.5)
        assert res.get("agg_value") is not None and isinstance(res["agg_value"], (int, float))

    def test_rel_diff_F2_only_retain_ftr_no_retain_must_fail(self):
        """rel_diff with only retain_ftr (no retain slot) → must fail (raise)."""
        rel = METRICS_REGISTRY.get("rel_diff")
        if rel is None:
            pytest.skip("rel_diff not registered")
        ref = {"retain_model_logs": {"retain_ftr": {"value_by_index": {"0": {"score": 0.8}}}, "retain": None}}
        with pytest.raises(Exception) as exc_info:
            rel._metric_fn(model=None, pre_compute={"forget": {"agg_value": 0.3}}, reference_logs=ref, ref_value=0.5)
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "reference" in msg or "missing" in msg

    def test_rel_diff_F3_retain_agg_value_not_number_must_fail(self):
        """rel_diff with retain.agg_value dict (non-canonical) → must raise; no ref_value."""
        rel = METRICS_REGISTRY.get("rel_diff")
        if rel is None:
            pytest.skip("rel_diff not registered")
        ref = {"retain_model_logs": {"retain": {"agg_value": {"nested": 1}}}}
        with pytest.raises(Exception):
            rel._metric_fn(model=None, pre_compute={"forget": {"agg_value": 0.3}}, reference_logs=ref, ref_value=0.5)

    def test_rel_diff_F4_required_but_missing_must_fail(self):
        """rel_diff with _required_but_missing → must raise."""
        rel = METRICS_REGISTRY.get("rel_diff")
        if rel is None:
            pytest.skip("rel_diff not registered")
        ref = {"retain_model_logs": {"_required_but_missing": True}}
        with pytest.raises(Exception):
            rel._metric_fn(model=None, pre_compute={"forget": {"agg_value": 0.3}}, reference_logs=ref, ref_value=0.5)

    def test_rel_diff_F5_no_reference_path_may_use_ref_value(self):
        """rel_diff with empty reference_logs → may use ref_value; no fail."""
        rel = METRICS_REGISTRY.get("rel_diff")
        if rel is None:
            pytest.skip("rel_diff not registered")
        res = rel._metric_fn(model=None, pre_compute={"forget": {"agg_value": 0.3}}, reference_logs={}, ref_value=0.5)
        assert "agg_value" in res
        assert res["agg_value"] is not None


# --- H: Trajectory per-step refs ---

class TestTrajectoryPerStepRefs:
    """H1–H5: trajectory passes step-specific ref in correct slot to privleak and ks_test."""

    def test_H1_privleak_at_step_receives_retain_from_retain_mia_by_step(self):
        """H1: retain_mia_by_step present; at step 5 privleak receives retain_model_logs['retain'] = retain_mia_by_step['5']."""
        from evals.metrics.trajectory_metrics import _call_metric_at_step

        captured = {}
        def capture_fn(**kwargs):
            captured["reference_logs"] = kwargs.get("reference_logs")
            return {"agg_value": 0.5}

        mock_metric = Mock()
        mock_metric.name = "privleak"
        mock_metric._metric_fn = capture_fn
        ref_logs = {
            "retain_model_logs": {
                "retain_mia_by_step": {"5": {"agg_value": 0.28}, "10": {"agg_value": 0.30}},
            }
        }
        logits = torch.zeros(50, 5)
        batch_template = {"input_ids": torch.zeros(1, 5, dtype=torch.long), "labels": torch.zeros(1, 5, dtype=torch.long)}
        _call_metric_at_step(
            metric=mock_metric,
            logits=logits,
            batch_template=batch_template,
            reference_logs=ref_logs,
            step_index=5,
            step=5,
        )
        assert captured.get("reference_logs") is not None
        rml = captured["reference_logs"].get("retain_model_logs") or {}
        assert rml.get("retain") == {"agg_value": 0.28}

    def test_H2_ks_test_at_step_receives_retain_ftr_from_retain_forget_tr_by_step(self):
        """H2: retain_forget_tr_by_step present; at step 5 ks_test receives retain_model_logs['retain_ftr'] from by_step['5']."""
        from evals.metrics.trajectory_metrics import _call_metric_at_step

        captured = {}
        def capture_fn(**kwargs):
            captured["reference_logs"] = kwargs.get("reference_logs")
            return {"agg_value": 0.75}

        mock_metric = Mock()
        mock_metric.name = "ks_test"
        mock_metric._metric_fn = capture_fn
        step_val = {"value_by_index": {"0": {"score": 0.85}}, "agg_value": 0.85}
        ref_logs = {
            "retain_model_logs": {
                "retain_forget_tr_by_step": {"5": step_val, "10": {"value_by_index": {"0": {"score": 0.86}}}},
            }
        }
        logits = torch.zeros(50, 5)
        batch_template = {"input_ids": torch.zeros(1, 5, dtype=torch.long), "labels": torch.zeros(1, 5, dtype=torch.long)}
        _call_metric_at_step(
            metric=mock_metric,
            logits=logits,
            batch_template=batch_template,
            reference_logs=ref_logs,
            step_index=5,
            step=5,
        )
        assert captured.get("reference_logs") is not None
        rml = captured["reference_logs"].get("retain_model_logs") or {}
        assert rml.get("retain_ftr") == step_val

    def test_H3_step_key_string_lookup_succeeds(self):
        """H3: by_step keys are strings; trajectory looks up by str(step) → no KeyError."""
        from evals.metrics.trajectory_metrics import _call_metric_at_step

        captured = {}
        def capture_fn(**kwargs):
            captured["reference_logs"] = kwargs.get("reference_logs")
            return {"agg_value": 0.5}

        mock_metric = Mock()
        mock_metric.name = "privleak"
        mock_metric._metric_fn = capture_fn
        ref_logs = {"retain_model_logs": {"retain_mia_by_step": {"5": {"agg_value": 0.28}}}}
        logits = torch.zeros(50, 5)
        batch_template = {"input_ids": torch.zeros(1, 5, dtype=torch.long), "labels": torch.zeros(1, 5, dtype=torch.long)}
        _call_metric_at_step(
            metric=mock_metric,
            logits=logits,
            batch_template=batch_template,
            reference_logs=ref_logs,
            step_index=5,
            step=5,
        )
        assert captured.get("reference_logs", {}).get("retain_model_logs", {}).get("retain") == {"agg_value": 0.28}

    def test_H4_per_step_ref_missing_raises(self):
        """H4: reference provided but step 10 not in retain_mia_by_step → RetainReferenceValidationError raised (real error, not just logged)."""
        from evals.metrics.trajectory_metrics import _call_metric_at_step

        mock_metric = Mock()
        mock_metric.name = "privleak"
        mock_metric._metric_fn = lambda **kw: {"agg_value": 0.5}
        ref_logs = {
            "retain_model_logs": {
                "retain_mia_by_step": {"0": {"agg_value": 0.25}, "1": {"agg_value": 0.28}},
            }
        }
        logits = torch.zeros(50, 5)
        batch_template = {"input_ids": torch.zeros(1, 5, dtype=torch.long), "labels": torch.zeros(1, 5, dtype=torch.long)}
        with pytest.raises(RetainReferenceValidationError) as exc_info:
            _call_metric_at_step(
                metric=mock_metric,
                logits=logits,
                batch_template=batch_template,
                reference_logs=ref_logs,
                step_index=10,
                step=10,
            )
        assert "reference_logs was provided" in str(exc_info.value)

    def test_H4_ks_test_per_step_ref_missing_raises(self):
        """Reference provided but step not in retain_forget_tr_by_step → RetainReferenceValidationError raised (real error, not just logged)."""
        from evals.metrics.trajectory_metrics import _call_metric_at_step

        mock_metric = Mock()
        mock_metric.name = "ks_test"
        mock_metric._metric_fn = lambda **kw: {"agg_value": 0.5}
        ref_logs = {
            "retain_model_logs": {
                "retain_forget_tr_by_step": {"0": {"value_by_index": {"0": {"score": 0.5}}, "agg_value": 0.5}},
            }
        }
        logits = torch.zeros(50, 5)
        batch_template = {"input_ids": torch.zeros(1, 5, dtype=torch.long), "labels": torch.zeros(1, 5, dtype=torch.long)}
        with pytest.raises(RetainReferenceValidationError) as exc_info:
            _call_metric_at_step(
                metric=mock_metric,
                logits=logits,
                batch_template=batch_template,
                reference_logs=ref_logs,
                step_index=5,
                step=5,
            )
        assert "reference_logs was provided" in str(exc_info.value)

    def test_H5_both_metrics_at_step_receive_correct_slots(self):
        """H5: Both privleak and ks_test at same step; privleak gets retain from retain_mia_by_step, ks_test gets retain_ftr from retain_forget_tr_by_step."""
        from evals.metrics.trajectory_metrics import _call_metric_at_step

        ref_logs = {
            "retain_model_logs": {
                "retain_mia_by_step": {"3": {"agg_value": 0.27}},
                "retain_forget_tr_by_step": {"3": {"value_by_index": {"0": {"score": 0.84}}, "agg_value": 0.84}},
            }
        }
        logits = torch.zeros(50, 5)
        batch_template = {"input_ids": torch.zeros(1, 5, dtype=torch.long), "labels": torch.zeros(1, 5, dtype=torch.long)}
        kw = {"reference_logs": ref_logs, "step_index": 3, "step": 3}

        cap_priv = {}
        mock_priv = Mock()
        mock_priv.name = "privleak"
        mock_priv._metric_fn = lambda **kwargs: cap_priv.update({"ref": kwargs.get("reference_logs")}) or {"agg_value": 0.5}
        _call_metric_at_step(metric=mock_priv, logits=logits, batch_template=batch_template, **kw)
        assert cap_priv.get("ref", {}).get("retain_model_logs", {}).get("retain") == {"agg_value": 0.27}
        assert "retain_ftr" not in (cap_priv.get("ref", {}).get("retain_model_logs") or {})

        cap_ks = {}
        mock_ks = Mock()
        mock_ks.name = "ks_test"
        mock_ks._metric_fn = lambda **kwargs: cap_ks.update({"ref": kwargs.get("reference_logs")}) or {"agg_value": 0.8}
        _call_metric_at_step(metric=mock_ks, logits=logits, batch_template=batch_template, **kw)
        assert cap_ks.get("ref", {}).get("retain_model_logs", {}).get("retain_ftr") == {"value_by_index": {"0": {"score": 0.84}}, "agg_value": 0.84}


# --- I2 / N4: E2E trajectory one-step with step refs ---

class TestE2ETrajectoryOneStep:
    """I2, N4: Load canonical (with by_step), pass to trajectory one step; both metrics receive correct step refs."""

    def test_I2_N4_load_canonical_with_by_step_trajectory_one_step_both_metrics_receive_step_refs(self, tmp_path):
        """Load canonical file (both keys + both by_step); run trajectory one step for privleak and ks_test; assert both receive correct slot from by_step."""
        from evals.metrics.trajectory_metrics import _call_metric_at_step

        path = tmp_path / "r.json"
        path.write_text(json.dumps({
            "mia_min_k": {"agg_value": 0.27},
            "forget_truth_ratio": {"value_by_index": {"0": {"score": 0.85}}, "agg_value": 0.85},
            "mia_min_k_by_step": {"0": {"agg_value": 0.26}},
            "forget_truth_ratio_by_step": {"0": {"value_by_index": {"0": {"score": 0.84}}, "agg_value": 0.84}},
        }))
        metric = UnlearningMetric("m", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "m",
            reference_logs={
                "retain_model_logs": {
                    "path": str(path),
                    "include": {"mia_min_k": {"access_key": "retain"}, "forget_truth_ratio": {"access_key": "retain_ftr"}},
                }
            },
        )
        ref_logs = kwargs.get("reference_logs") or {}
        assert ref_logs.get("retain_model_logs", {}).get("retain_mia_by_step")
        assert ref_logs.get("retain_model_logs", {}).get("retain_forget_tr_by_step")

        logits = torch.zeros(50, 5)
        batch_template = {"input_ids": torch.zeros(1, 5, dtype=torch.long), "labels": torch.zeros(1, 5, dtype=torch.long)}
        kw = {"reference_logs": ref_logs, "step_index": 0, "step": 0}

        cap_priv = {}
        mock_priv = Mock()
        mock_priv.name = "privleak"
        mock_priv._metric_fn = lambda **k: cap_priv.update({"ref": k.get("reference_logs")}) or {"agg_value": 0.5}
        _call_metric_at_step(metric=mock_priv, logits=logits, batch_template=batch_template, **kw)
        assert cap_priv.get("ref", {}).get("retain_model_logs", {}).get("retain") == {"agg_value": 0.26}

        cap_ks = {}
        mock_ks = Mock()
        mock_ks.name = "ks_test"
        mock_ks._metric_fn = lambda **k: cap_ks.update({"ref": k.get("reference_logs")}) or {"agg_value": 0.8}
        _call_metric_at_step(metric=mock_ks, logits=logits, batch_template=batch_template, **kw)
        assert cap_ks.get("ref", {}).get("retain_model_logs", {}).get("retain_ftr", {}).get("value_by_index")


# --- J2, J3: Config-driven loader (forget_quality / privleak config shape) ---

class TestConfigDrivenLoaderJ2J3:
    """J2: forget_quality config (only forget_truth_ratio → retain_ftr). J3: privleak config (only mia_min_k → retain)."""

    def test_J2_loader_with_forget_quality_config_fills_only_retain_ftr(self, tmp_path):
        """Config shape like forget_quality.yaml: include only forget_truth_ratio → access_key retain_ftr; loader fills only retain_ftr."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"forget_truth_ratio": {"value_by_index": {"0": {"score": 0.85}}, "agg_value": 0.85}}))
        metric = UnlearningMetric("m", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "m",
            reference_logs={
                "retain_model_logs": {"path": str(path), "include": {"forget_truth_ratio": {"access_key": "retain_ftr"}}},
            },
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert rml.get("retain_ftr") is not None
        assert rml.get("retain") is None

    def test_J3_loader_with_privleak_config_fills_only_retain(self, tmp_path):
        """Config shape like privleak.yaml: include only mia_min_k → access_key retain; loader fills only retain."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"mia_min_k": {"agg_value": 0.27}}))
        metric = UnlearningMetric("m", _dummy_metric_fn)
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "m",
            reference_logs={
                "retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}}},
            },
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert rml.get("retain") is not None
        assert rml.get("retain_ftr") is None
