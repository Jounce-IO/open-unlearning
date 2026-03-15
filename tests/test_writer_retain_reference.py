"""Tests for retain reference JSON writer (Evaluator.save_logs, _ensure_retain_reference_keys)
and combined scenarios (writer → file → loader → metrics). Spec: 005 test-list-slots-and-canonical.md M, N."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


def _make_evaluator(retain_reference_mode=True):
    """Minimal Evaluator for writer tests; avoids loading real metrics."""
    from omegaconf import OmegaConf
    from evals.base import Evaluator
    eval_cfg = OmegaConf.create({
        "metrics": {},
        "output_dir": "/tmp",
        "retain_reference_mode": retain_reference_mode,
        "overwrite": False,
    })
    with patch.object(Evaluator, "load_metrics", return_value={}):
        return Evaluator("TOFU", eval_cfg)


# Canonical shapes per contract (single type, no order dependence).
CANONICAL_MIA_MIN_K = {"agg_value": 0.27}
CANONICAL_FORGET_TRUTH_RATIO = {"value_by_index": {"0": {"score": 0.85}}, "agg_value": 0.85}
CANONICAL_MIA_BY_STEP = {"0": {"agg_value": 0.25}, "1": {"agg_value": 0.28}}
CANONICAL_FTR_BY_STEP = {
    "0": {"value_by_index": {"0": {"score": 0.84}}},
    "1": {"value_by_index": {"0": {"score": 0.85}}},
}


class TestWriterSaveLogsCanonicalShape:
    """M1, M2: save_logs writes exact canonical shape; keep_value_by_index controls value_by_index."""

    def test_M1_save_logs_keep_value_by_index_preserves_canonical_shape(self, tmp_path):
        """M1: save_logs(keep_value_by_index=True) preserves mia_min_k, forget_truth_ratio, by_step canonical shape."""
        ev = _make_evaluator(retain_reference_mode=True)
        logs = {
            "mia_min_k": dict(CANONICAL_MIA_MIN_K),
            "forget_truth_ratio": dict(CANONICAL_FORGET_TRUTH_RATIO),
            "mia_min_k_by_step": dict(CANONICAL_MIA_BY_STEP),
            "forget_truth_ratio_by_step": dict(CANONICAL_FTR_BY_STEP),
        }
        path = tmp_path / "retain.json"
        ev.save_logs(logs, str(path), keep_value_by_index=True)
        with open(path) as f:
            loaded = json.load(f)
        assert "mia_min_k" in loaded
        assert loaded["mia_min_k"] == CANONICAL_MIA_MIN_K
        assert "forget_truth_ratio" in loaded
        assert "value_by_index" in loaded["forget_truth_ratio"]
        assert loaded["mia_min_k_by_step"] == CANONICAL_MIA_BY_STEP
        assert loaded["forget_truth_ratio_by_step"] == CANONICAL_FTR_BY_STEP
        for step, val in loaded["mia_min_k_by_step"].items():
            assert isinstance(val, dict) and "agg_value" in val
            assert isinstance(val["agg_value"], (int, float))
        for step, val in loaded["forget_truth_ratio_by_step"].items():
            assert isinstance(val, dict) and "value_by_index" in val

    def test_M2_save_logs_strips_value_by_index_when_false(self, tmp_path):
        """M2: save_logs(keep_value_by_index=False) removes all value_by_index; agg_value preserved."""
        ev = _make_evaluator(retain_reference_mode=False)
        logs = {
            "forget_truth_ratio": {"value_by_index": {"0": {"score": 0.8}}, "agg_value": 0.8},
            "mia_min_k": {"agg_value": 0.3},
        }
        path = tmp_path / "retain.json"
        ev.save_logs(logs, str(path), keep_value_by_index=False)
        with open(path) as f:
            loaded = json.load(f)
        assert "value_by_index" not in loaded.get("forget_truth_ratio", {})
        assert loaded["forget_truth_ratio"].get("agg_value") == 0.8
        assert loaded["mia_min_k"]["agg_value"] == 0.3


class TestWriterEnsureRetainReferenceKeys:
    """M3: _ensure_retain_reference_keys copies forget_Truth_Ratio → forget_truth_ratio when missing."""

    def test_M3_ensure_retain_reference_keys_adds_forget_truth_ratio(self):
        """When forget_truth_ratio missing and forget_Truth_Ratio present, copy added."""
        ev = _make_evaluator(retain_reference_mode=True)
        logs = {"forget_Truth_Ratio": {"agg_value": 0.9}}
        ev._ensure_retain_reference_keys(logs)
        assert "forget_truth_ratio" in logs
        assert logs["forget_truth_ratio"] == logs["forget_Truth_Ratio"]

    def test_M3_ensure_retain_reference_keys_no_change_when_already_present(self):
        """When forget_truth_ratio already present, no change."""
        ev = _make_evaluator(retain_reference_mode=True)
        logs = {"forget_truth_ratio": {"agg_value": 0.85}}
        ev._ensure_retain_reference_keys(logs)
        assert logs["forget_truth_ratio"]["agg_value"] == 0.85

    def test_M3_ensure_retain_reference_keys_skips_when_mode_false(self):
        """When retain_reference_mode False, no copy."""
        ev = _make_evaluator(retain_reference_mode=False)
        logs = {"forget_Truth_Ratio": {"agg_value": 0.9}}
        ev._ensure_retain_reference_keys(logs)
        assert "forget_truth_ratio" not in logs


class TestWriterTrajectoryOutputShape:
    """M4: Canonical by_step structure (step_str -> {agg_value: float} for mia; step_str -> {value_by_index} for ftr)."""

    def test_M4_canonical_by_step_structure_preserved_by_save_logs(self, tmp_path):
        """Dict with canonical by_step shape is preserved by save_logs round-trip."""
        ev = _make_evaluator(retain_reference_mode=True)
        logs = {
            "mia_min_k_by_step": {"5": {"agg_value": 0.26}, "10": {"agg_value": 0.28}},
            "forget_truth_ratio_by_step": {
                "5": {"value_by_index": {"0": {"score": 0.84}}},
                "10": {"value_by_index": {"0": {"score": 0.86}}},
            },
        }
        path = tmp_path / "by_step.json"
        ev.save_logs(logs, str(path), keep_value_by_index=True)
        with open(path) as f:
            loaded = json.load(f)
        assert set(loaded["mia_min_k_by_step"].keys()) == {"5", "10"}
        for k, v in loaded["mia_min_k_by_step"].items():
            assert isinstance(v, dict) and "agg_value" in v and isinstance(v["agg_value"], (int, float))
        for k, v in loaded["forget_truth_ratio_by_step"].items():
            assert isinstance(v, dict) and "value_by_index" in v


class TestWriterFullPath:
    """M5, M6: Full writer path; writer must not emit non-canonical shapes."""

    def test_M5_full_writer_path_canonical_logs_round_trip(self, tmp_path):
        """Logs dict (as from trajectory retain_reference_mode) → _ensure_retain_reference_keys → save_logs → file matches canonical."""
        ev = _make_evaluator(retain_reference_mode=True)
        logs = {
            "forget_Truth_Ratio": {"agg_value": 0.82, "value_by_index": {"0": {"score": 0.82}}},
            "mia_min_k": CANONICAL_MIA_MIN_K,
            "mia_min_k_by_step": CANONICAL_MIA_BY_STEP,
            "forget_truth_ratio_by_step": CANONICAL_FTR_BY_STEP,
        }
        ev._ensure_retain_reference_keys(logs)
        path = tmp_path / "results.json"
        ev.save_logs(logs, str(path), keep_value_by_index=True)
        with open(path) as f:
            loaded = json.load(f)
        assert "mia_min_k" in loaded and "forget_truth_ratio" in loaded
        assert isinstance(loaded["mia_min_k"].get("agg_value"), (int, float))
        assert "value_by_index" in loaded["forget_truth_ratio"] or "agg_value" in loaded["forget_truth_ratio"]

    def test_M6_loader_must_fail_auc_only_no_agg_value(self, tmp_path):
        """Strict canonical: file with mia_min_k = {auc: 0.3} only (no agg_value) → loader MUST raise. No _required_but_missing, no exposing invalid data."""
        from evals.metrics.base import UnlearningMetric
        from unittest.mock import Mock

        def _dummy_fn(**kwargs):
            return {"agg_value": 0.5}

        metric = UnlearningMetric("m", _dummy_fn)
        bad_file = tmp_path / "bad.json"
        bad_file.write_text(json.dumps({"mia_min_k": {"auc": 0.3}}))
        with pytest.raises(Exception) as exc_info:
            metric.prepare_kwargs_evaluate_metric(
                Mock(),
                "m",
                reference_logs={
                    "retain_model_logs": {
                        "path": str(bad_file),
                        "include": {"mia_min_k": {"access_key": "retain"}},
                    }
                },
            )
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "mia_min_k" in msg or "canonical" in msg or "agg_value" in msg or "invalid" in msg, (
            "Failure reason must be clear and understandable."
        )


class TestCombinedBothKeysCanonical:
    """N1: Both keys canonical → load with two slots → privleak and ks_test get correct refs."""

    def test_N1_both_keys_two_slots_privleak_and_ks_test_receive_refs(self, tmp_path):
        """Writer writes canonical mia_min_k + forget_truth_ratio; loader with mia_min_k→retain, forget_truth_ratio→retain_ftr; both metrics get refs."""
        from evals.metrics.base import UnlearningMetric
        from evals.metrics import METRICS_REGISTRY
        from unittest.mock import Mock

        def _dummy_fn(**kwargs):
            return {"agg_value": 0.5}

        path = tmp_path / "retain.json"
        path.write_text(json.dumps({
            "mia_min_k": CANONICAL_MIA_MIN_K,
            "forget_truth_ratio": CANONICAL_FORGET_TRUTH_RATIO,
        }))
        metric = UnlearningMetric("m", _dummy_fn)
        include = {
            "mia_min_k": {"access_key": "retain"},
            "forget_truth_ratio": {"access_key": "retain_ftr"},
        }
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "m",
            reference_logs={"retain_model_logs": {"path": str(path), "include": include}},
        )
        ref = kwargs.get("reference_logs", {}) or {}
        rml = ref.get("retain_model_logs") or {}
        retain = rml.get("retain")
        retain_ftr = rml.get("retain_ftr")
        assert retain is not None, "retain slot (mia_min_k) must be present"
        assert retain_ftr is not None, "retain_ftr slot (forget_truth_ratio) must be present when two access_keys"
        if retain and isinstance(retain.get("agg_value"), (int, float)):
            pl = METRICS_REGISTRY.get("privleak")
            res = pl._metric_fn(model=None, pre_compute={"forget": {"agg_value": 0.3}}, reference_logs=ref, ref_value=0.5)
            assert res.get("agg_value") is not None
        if retain_ftr and (retain_ftr.get("value_by_index") or isinstance(retain_ftr.get("agg_value"), (int, float))):
            ks = METRICS_REGISTRY.get("ks_test")
            pre = {"forget": {"value_by_index": {"0": {"score": 0.7}}}}
            res = ks._metric_fn(model=None, pre_compute=pre, reference_logs=ref)
            assert "agg_value" in res


class TestCombinedOnlyMiaMinK:
    """N2: Only mia_min_k → privleak gets ref, ks_test does not."""

    def test_N2_only_mia_min_k_privleak_gets_ref(self, tmp_path):
        """File has only mia_min_k; config include mia_min_k→retain; privleak gets ref."""
        from evals.metrics.base import UnlearningMetric
        from evals.metrics import METRICS_REGISTRY
        from unittest.mock import Mock

        path = tmp_path / "retain.json"
        path.write_text(json.dumps({"mia_min_k": CANONICAL_MIA_MIN_K}))
        metric = UnlearningMetric("m", lambda **kw: {"agg_value": 0.5})
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "m",
            reference_logs={
                "retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}}},
            },
        )
        ref = kwargs.get("reference_logs", {}) or {}
        rml = ref.get("retain_model_logs") or {}
        assert rml.get("retain") is not None
        pl = METRICS_REGISTRY.get("privleak")
        res = pl._metric_fn(model=None, pre_compute={"forget": {"agg_value": 0.3}}, reference_logs=ref, ref_value=0.5)
        assert res.get("agg_value") is not None


class TestCombinedOnlyForgetTruthRatio:
    """N3: Only forget_truth_ratio → ks_test reads retain_ftr only; privleak has no ref (would fail if called with this ref)."""

    def test_N3_only_forget_truth_ratio_ks_test_gets_retain_ftr_only(self, tmp_path):
        """File has only forget_truth_ratio; config include forget_truth_ratio→retain_ftr; loader sets retain_ftr only (no fallback to retain)."""
        from evals.metrics.base import UnlearningMetric
        from evals.metrics import METRICS_REGISTRY
        from unittest.mock import Mock

        path = tmp_path / "retain.json"
        path.write_text(json.dumps({"forget_truth_ratio": CANONICAL_FORGET_TRUTH_RATIO}))
        metric = UnlearningMetric("m", lambda **kw: {"agg_value": 0.5})
        include = {"forget_truth_ratio": {"access_key": "retain_ftr"}}
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "m",
            reference_logs={"retain_model_logs": {"path": str(path), "include": include}},
        )
        ref = kwargs.get("reference_logs", {}) or {}
        rml = ref.get("retain_model_logs") or {}
        assert rml.get("retain_ftr") is not None, "Loader must set retain_ftr from forget_truth_ratio; no fallback."
        assert rml.get("retain") is None, "retain must not be set when only forget_truth_ratio in file; no fallback."
        retain_ftr = rml["retain_ftr"]
        vbi = (retain_ftr or {}).get("value_by_index")
        if vbi:
            ks = METRICS_REGISTRY.get("ks_test")
            pre = {"forget": {"value_by_index": {"0": {"score": 0.7}}}}
            res = ks._metric_fn(model=None, pre_compute=pre, reference_logs=ref)
            assert "agg_value" in res


class TestCombinedWriterLoaderRoundTrip:
    """N5: Writer output → save → load → prepare_kwargs → reference_logs has both slots and by_step."""

    def test_N5_writer_output_load_preserves_slots_and_by_step(self, tmp_path):
        """Canonical dict → write to file → load via loader with two-slot include → both slots and by_step present."""
        from evals.metrics.base import UnlearningMetric
        from unittest.mock import Mock

        canonical = {
            "mia_min_k": CANONICAL_MIA_MIN_K,
            "forget_truth_ratio": CANONICAL_FORGET_TRUTH_RATIO,
            "mia_min_k_by_step": CANONICAL_MIA_BY_STEP,
            "forget_truth_ratio_by_step": CANONICAL_FTR_BY_STEP,
        }
        path = tmp_path / "retain.json"
        path.write_text(json.dumps(canonical))
        metric = UnlearningMetric("m", lambda **kw: {"agg_value": 0.5})
        include = {
            "mia_min_k": {"access_key": "retain"},
            "forget_truth_ratio": {"access_key": "retain_ftr"},
        }
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "m",
            reference_logs={"retain_model_logs": {"path": str(path), "include": include}},
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert rml.get("retain") is not None and rml.get("retain_ftr") is not None, "Both slots required when file has both keys; no fallback."
        assert rml.get("retain_mia_by_step") == canonical["mia_min_k_by_step"]
        assert rml.get("retain_forget_tr_by_step") == canonical["forget_truth_ratio_by_step"]


class TestCombinedMalformedWriterOutput:
    """N6: Strict canonical – non-canonical file → loader MUST fail (raise) with clear reason. No _required_but_missing, no exposing invalid data."""

    def test_N6_malformed_mia_min_k_loader_must_fail(self, tmp_path):
        """Strict canonical: file with mia_min_k = only auc (no agg_value) → loader MUST raise. No _required_but_missing."""
        from evals.metrics.base import UnlearningMetric
        from unittest.mock import Mock

        path = tmp_path / "retain.json"
        path.write_text(json.dumps({"mia_min_k": {"auc": 0.3}}))
        metric = UnlearningMetric("m", lambda **kw: {"agg_value": 0.5})
        with pytest.raises(Exception) as exc_info:
            metric.prepare_kwargs_evaluate_metric(
                Mock(), "m",
                reference_logs={
                    "retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}}},
                },
            )
        msg = str(exc_info.value).lower()
        assert "retain" in msg or "mia_min_k" in msg or "canonical" in msg or "agg_value" in msg or "invalid" in msg, (
            "Failure reason must be clear and understandable."
        )


class TestCombinedAggregateOnlyNoByStep:
    """N7: File has only aggregate keys (no by_step); loader fills aggregate slots; by_step absent."""

    def test_N7_aggregate_only_load_succeeds_aggregate_refs_available(self, tmp_path):
        """Top-level mia_min_k and forget_truth_ratio only; loader fills aggregate slots; by_step absent."""
        from evals.metrics.base import UnlearningMetric
        from unittest.mock import Mock

        path = tmp_path / "retain.json"
        path.write_text(json.dumps({
            "mia_min_k": CANONICAL_MIA_MIN_K,
            "forget_truth_ratio": CANONICAL_FORGET_TRUTH_RATIO,
        }))
        metric = UnlearningMetric("m", lambda **kw: {"agg_value": 0.5})
        include = {
            "mia_min_k": {"access_key": "retain"},
            "forget_truth_ratio": {"access_key": "retain_ftr"},
        }
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "m",
            reference_logs={"retain_model_logs": {"path": str(path), "include": include}},
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert rml.get("retain") is not None, "Loader must set retain from mia_min_k."
        assert rml.get("retain_ftr") is not None, "Loader must set retain_ftr from forget_truth_ratio. No fallback."


class TestCombinedByStepOnlyNoAggregate:
    """N8: File has only by_step keys; config requests aggregate keys → loader uses first step as aggregate (canonical)."""

    def test_N8_by_step_only_config_requests_aggregate_loader_uses_first_step(self, tmp_path):
        """Only by_step in file; config requests both aggregate keys → loader uses first step as aggregate when canonical."""
        from evals.metrics.base import UnlearningMetric
        from unittest.mock import Mock

        path = tmp_path / "retain.json"
        path.write_text(json.dumps({
            "mia_min_k_by_step": CANONICAL_MIA_BY_STEP,
            "forget_truth_ratio_by_step": CANONICAL_FTR_BY_STEP,
        }))
        metric = UnlearningMetric("m", lambda **kw: {"agg_value": 0.5})
        include = {
            "mia_min_k": {"access_key": "retain"},
            "forget_truth_ratio": {"access_key": "retain_ftr"},
        }
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "m",
            reference_logs={"retain_model_logs": {"path": str(path), "include": include}},
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert rml.get("retain") is not None, "Loader must set retain from first step of mia_min_k_by_step"
        assert rml.get("retain_ftr") is not None, "Loader must set retain_ftr from first step of forget_truth_ratio_by_step"
        assert rml.get("retain_mia_by_step") == CANONICAL_MIA_BY_STEP
        assert rml.get("retain_forget_tr_by_step") == CANONICAL_FTR_BY_STEP


# --- Writer edge cases and scenarios ---

class TestWriterEdgeCases:
    """Writer: edge cases – empty by_step, single step, extra keys, only value_by_index, etc."""

    def test_writer_save_logs_empty_by_step_dict_preserved(self, tmp_path):
        """Canonical: by_step can be empty dict; round-trip preserves it."""
        ev = _make_evaluator(retain_reference_mode=True)
        logs = {
            "mia_min_k": CANONICAL_MIA_MIN_K,
            "mia_min_k_by_step": {},
            "forget_truth_ratio_by_step": {},
        }
        path = tmp_path / "retain.json"
        ev.save_logs(logs, str(path), keep_value_by_index=True)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded.get("mia_min_k_by_step") == {}
        assert loaded.get("forget_truth_ratio_by_step") == {}

    def test_writer_save_logs_single_step_by_step(self, tmp_path):
        """Canonical: single step in by_step; shape preserved."""
        ev = _make_evaluator(retain_reference_mode=True)
        logs = {
            "mia_min_k_by_step": {"0": {"agg_value": 0.26}},
            "forget_truth_ratio_by_step": {"0": {"value_by_index": {"0": {"score": 0.85}}}},
        }
        path = tmp_path / "retain.json"
        ev.save_logs(logs, str(path), keep_value_by_index=True)
        with open(path) as f:
            loaded = json.load(f)
        assert list(loaded["mia_min_k_by_step"].keys()) == ["0"]
        assert loaded["mia_min_k_by_step"]["0"]["agg_value"] == 0.26
        assert "value_by_index" in loaded["forget_truth_ratio_by_step"]["0"]

    def test_writer_save_logs_forget_truth_ratio_only_value_by_index(self, tmp_path):
        """Canonical: forget_truth_ratio with only value_by_index (no agg_value) is valid for ks_test; writer preserves."""
        ev = _make_evaluator(retain_reference_mode=True)
        logs = {"forget_truth_ratio": {"value_by_index": {"0": {"score": 0.8}, "1": {"score": 0.82}}}}
        path = tmp_path / "retain.json"
        ev.save_logs(logs, str(path), keep_value_by_index=True)
        with open(path) as f:
            loaded = json.load(f)
        assert "value_by_index" in loaded["forget_truth_ratio"]
        assert len(loaded["forget_truth_ratio"]["value_by_index"]) == 2

    def test_writer_ensure_retain_reference_keys_both_present_prefers_forget_truth_ratio(self):
        """When both forget_truth_ratio and forget_Truth_Ratio exist, _ensure_retain_reference_keys does not overwrite forget_truth_ratio."""
        ev = _make_evaluator(retain_reference_mode=True)
        logs = {"forget_truth_ratio": {"agg_value": 0.85}, "forget_Truth_Ratio": {"agg_value": 0.9}}
        ev._ensure_retain_reference_keys(logs)
        assert logs["forget_truth_ratio"]["agg_value"] == 0.85

    def test_writer_save_logs_with_run_info_and_config_extra_keys_preserved(self, tmp_path):
        """Writer can save logs that include run_info/config; canonical retain keys preserved."""
        ev = _make_evaluator(retain_reference_mode=True)
        logs = {
            "config": {"evaluator_name": "TOFU"},
            "run_info": {"world_size": 1, "total_samples": 100},
            "mia_min_k": CANONICAL_MIA_MIN_K,
            "forget_truth_ratio": CANONICAL_FORGET_TRUTH_RATIO,
        }
        path = tmp_path / "retain.json"
        ev.save_logs(logs, str(path), keep_value_by_index=True)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded.get("config", {}).get("evaluator_name") == "TOFU"
        assert loaded["mia_min_k"] == CANONICAL_MIA_MIN_K
        assert loaded["forget_truth_ratio"]["agg_value"] == 0.85


# --- Integration edge cases and scenarios ---

class TestCombinedEdgeCasesAndScenarios:
    """Integration: edge cases – round-trip, extra keys in file, numeric boundaries, single sample, etc."""

    def test_integration_round_trip_two_files_same_canonical(self, tmp_path):
        """Canonical file → load → write to second file → load again; structure preserved."""
        from evals.metrics.base import UnlearningMetric
        from unittest.mock import Mock

        path1 = tmp_path / "r1.json"
        path2 = tmp_path / "r2.json"
        canonical = {"mia_min_k": CANONICAL_MIA_MIN_K, "forget_truth_ratio": CANONICAL_FORGET_TRUTH_RATIO}
        path1.write_text(json.dumps(canonical))
        metric = UnlearningMetric("m", lambda **kw: {"agg_value": 0.5})
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "m",
            reference_logs={"retain_model_logs": {"path": str(path1), "include": {"mia_min_k": {"access_key": "retain"}, "forget_truth_ratio": {"access_key": "retain_ftr"}}}},
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs")
        assert rml is not None and rml.get("retain") is not None and rml.get("retain_ftr") is not None
        path2.write_text(json.dumps(canonical))
        kwargs2 = metric.prepare_kwargs_evaluate_metric(
            Mock(), "m",
            reference_logs={"retain_model_logs": {"path": str(path2), "include": {"mia_min_k": {"access_key": "retain"}, "forget_truth_ratio": {"access_key": "retain_ftr"}}}},
        )
        rml2 = (kwargs2.get("reference_logs") or {}).get("retain_model_logs")
        assert rml2 is not None
        assert (rml.get("retain") is not None) == (rml2.get("retain") is not None)
        assert (rml.get("retain_ftr") is not None) == (rml2.get("retain_ftr") is not None)

    def test_integration_file_has_extra_top_level_keys_canonical_subset_loaded(self, tmp_path):
        """File with extra keys (run_info, config, other_metric); loader must still find canonical mia_min_k and forget_truth_ratio."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({
            "config": {},
            "run_info": {"world_size": 1},
            "other_metric": {"agg_value": 99},
            "mia_min_k": {"agg_value": 0.27},
            "forget_truth_ratio": {"agg_value": 0.85},
        }))
        from evals.metrics.base import UnlearningMetric
        from unittest.mock import Mock
        metric = UnlearningMetric("m", lambda **kw: {"agg_value": 0.5})
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "m",
            reference_logs={"retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}, "forget_truth_ratio": {"access_key": "retain_ftr"}}}},
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert rml.get("retain") is not None and isinstance((rml["retain"] or {}).get("agg_value"), (int, float))
        assert rml.get("retain_ftr") is not None

    def test_integration_mia_min_k_agg_value_zero_accepted(self, tmp_path):
        """Canonical: mia_min_k.agg_value = 0.0 is valid; loader accepts; privleak can use."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"mia_min_k": {"agg_value": 0.0}}))
        from evals.metrics.base import UnlearningMetric
        from evals.metrics import METRICS_REGISTRY
        from unittest.mock import Mock
        metric = UnlearningMetric("m", lambda **kw: {"agg_value": 0.5})
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "m",
            reference_logs={"retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}}}},
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert (rml.get("retain") or {}).get("agg_value") == 0.0
        res = METRICS_REGISTRY["privleak"]._metric_fn(model=None, pre_compute={"forget": {"agg_value": 0.1}}, reference_logs=kwargs["reference_logs"], ref_value=0.5)
        assert "agg_value" in res and res["agg_value"] is not None

    def test_integration_mia_min_k_agg_value_one_accepted(self, tmp_path):
        """Canonical: mia_min_k.agg_value = 1.0 is valid; loader accepts."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"mia_min_k": {"agg_value": 1.0}}))
        from evals.metrics.base import UnlearningMetric
        from unittest.mock import Mock
        metric = UnlearningMetric("m", lambda **kw: {"agg_value": 0.5})
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "m",
            reference_logs={"retain_model_logs": {"path": str(path), "include": {"mia_min_k": {"access_key": "retain"}}}},
        )
        rml = (kwargs.get("reference_logs") or {}).get("retain_model_logs") or {}
        assert (rml.get("retain") or {}).get("agg_value") == 1.0

    def test_integration_ks_test_single_sample_value_by_index(self, tmp_path):
        """Canonical: forget_truth_ratio with single entry in value_by_index; ks_test receives it (may return None if insufficient for test)."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"forget_truth_ratio": {"value_by_index": {"0": {"score": 0.8}}, "agg_value": 0.8}}))
        from evals.metrics.base import UnlearningMetric
        from evals.metrics import METRICS_REGISTRY
        from unittest.mock import Mock
        metric = UnlearningMetric("m", lambda **kw: {"agg_value": 0.5})
        kwargs = metric.prepare_kwargs_evaluate_metric(
            Mock(), "m",
            reference_logs={"retain_model_logs": {"path": str(path), "include": {"forget_truth_ratio": {"access_key": "retain_ftr"}}}},
        )
        ref = kwargs.get("reference_logs") or {}
        pre = {"forget": {"value_by_index": {"0": {"score": 0.7}}}}
        res = METRICS_REGISTRY["ks_test"]._metric_fn(model=None, pre_compute=pre, reference_logs=ref)
        assert "agg_value" in res
