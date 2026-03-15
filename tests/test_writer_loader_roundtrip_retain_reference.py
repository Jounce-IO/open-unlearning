"""Round-trip tests: writer (save_logs) → file → loader (load_and_validate_reference).

Uses real Evaluator.save_logs and real load_and_validate_reference; minimal mocks.
Covers many scenarios so that non-canonical or fallback writer output is rejected when
loaded as reference. Catches bugs where trajectory_metrics writes a shape that cannot
be loaded (e.g. only trajectory_all with agg_value/value_by_index/step_distribution
and no mia_min_k, forget_truth_ratio, or by_step keys).
"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.base import (
    RetainReferenceValidationError,
    load_and_validate_reference,
)

# Canonical shapes per contract
CANONICAL_MIA = {"agg_value": 0.27}
CANONICAL_FTR = {"value_by_index": {"0": {"score": 0.85}}, "agg_value": 0.85}
CANONICAL_MIA_BY_STEP = {"0": {"agg_value": 0.25}, "1": {"agg_value": 0.28}}
CANONICAL_FTR_BY_STEP = {
    "0": {"value_by_index": {"0": {"score": 0.84}}},
    "1": {"value_by_index": {"0": {"score": 0.85}}},
}

DEFAULT_INCLUDE = {
    "mia_min_k": {"access_key": "retain"},
    "forget_truth_ratio": {"access_key": "retain_ftr"},
}


def _make_evaluator(retain_reference_mode=True):
    """Minimal Evaluator for writer tests; only load_metrics is mocked."""
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


def _load_fn(path):
    with open(path) as f:
        return json.load(f)


def _reference_cfg(path, include=None):
    return {
        "retain_model_logs": {
            "path": str(path),
            "include": include or DEFAULT_INCLUDE,
        },
    }


# --- Writer → file → loader: canonical shapes must load successfully ---


class TestRoundtripCanonicalSucceeds:
    """Canonical file content → save (or write) → load → success."""

    def test_full_canonical_top_level_save_then_load_succeeds(self, tmp_path):
        """Full canonical at top-level: mia_min_k, forget_truth_ratio, both by_step → load succeeds."""
        ev = _make_evaluator(retain_reference_mode=True)
        logs = {
            "mia_min_k": dict(CANONICAL_MIA),
            "forget_truth_ratio": dict(CANONICAL_FTR),
            "mia_min_k_by_step": dict(CANONICAL_MIA_BY_STEP),
            "forget_truth_ratio_by_step": dict(CANONICAL_FTR_BY_STEP),
        }
        path = tmp_path / "ref.json"
        ev.save_logs(logs, str(path), keep_value_by_index=True)
        ref_cfg = _reference_cfg(path)
        result = load_and_validate_reference(ref_cfg, _load_fn)
        rml = result["retain_model_logs"]
        assert rml.get("retain") is not None and isinstance(rml["retain"].get("agg_value"), (int, float))
        assert rml.get("retain_ftr") is not None
        assert rml.get("retain_mia_by_step") == CANONICAL_MIA_BY_STEP
        assert rml.get("retain_forget_tr_by_step") == CANONICAL_FTR_BY_STEP

    def test_canonical_under_trajectory_all_load_succeeds(self, tmp_path):
        """Canonical keys under trajectory_all (as loader expects _traj.get(key)) → load succeeds."""
        path = tmp_path / "ref.json"
        path.write_text(json.dumps({
            "trajectory_all": {
                "mia_min_k": dict(CANONICAL_MIA),
                "forget_truth_ratio": dict(CANONICAL_FTR),
                "mia_min_k_by_step": dict(CANONICAL_MIA_BY_STEP),
                "forget_truth_ratio_by_step": dict(CANONICAL_FTR_BY_STEP),
            },
        }))
        result = load_and_validate_reference(_reference_cfg(path), _load_fn)
        rml = result["retain_model_logs"]
        assert rml.get("retain") is not None and rml.get("retain_ftr") is not None
        assert rml.get("retain_mia_by_step") == CANONICAL_MIA_BY_STEP
        assert rml.get("retain_forget_tr_by_step") == CANONICAL_FTR_BY_STEP

    def test_aggregate_only_no_by_step_load_succeeds(self, tmp_path):
        """Only mia_min_k and forget_truth_ratio (no by_step) → load succeeds."""
        path = tmp_path / "ref.json"
        path.write_text(json.dumps({
            "mia_min_k": dict(CANONICAL_MIA),
            "forget_truth_ratio": dict(CANONICAL_FTR),
        }))
        result = load_and_validate_reference(_reference_cfg(path), _load_fn)
        rml = result["retain_model_logs"]
        assert rml.get("retain") is not None and rml.get("retain_ftr") is not None
        assert "retain_mia_by_step" not in rml or rml.get("retain_mia_by_step") is None
        assert "retain_forget_tr_by_step" not in rml or rml.get("retain_forget_tr_by_step") is None

    def test_by_step_only_loader_uses_first_step_load_succeeds(self, tmp_path):
        """Only by_step keys; loader uses first step as aggregate → load succeeds."""
        path = tmp_path / "ref.json"
        path.write_text(json.dumps({
            "mia_min_k_by_step": dict(CANONICAL_MIA_BY_STEP),
            "forget_truth_ratio_by_step": dict(CANONICAL_FTR_BY_STEP),
        }))
        result = load_and_validate_reference(_reference_cfg(path), _load_fn)
        rml = result["retain_model_logs"]
        assert rml.get("retain") is not None and rml.get("retain_ftr") is not None
        assert rml.get("retain_mia_by_step") == CANONICAL_MIA_BY_STEP
        assert rml.get("retain_forget_tr_by_step") == CANONICAL_FTR_BY_STEP

    def test_keep_value_by_index_false_ftr_agg_only_still_loads(self, tmp_path):
        """After save with keep_value_by_index=False, forget_truth_ratio has only agg_value; load still succeeds."""
        ev = _make_evaluator(retain_reference_mode=False)
        logs = {
            "mia_min_k": dict(CANONICAL_MIA),
            "forget_truth_ratio": {"value_by_index": {"0": {"score": 0.85}}, "agg_value": 0.85},
        }
        path = tmp_path / "ref.json"
        ev.save_logs(logs, str(path), keep_value_by_index=False)
        with open(path) as f:
            data = json.load(f)
        assert "value_by_index" not in data.get("forget_truth_ratio", {})
        result = load_and_validate_reference(_reference_cfg(path), _load_fn)
        rml = result["retain_model_logs"]
        assert rml.get("retain") is not None and rml.get("retain_ftr") is not None

    def test_include_only_mia_min_k_load_succeeds(self, tmp_path):
        """File has both keys; include only mia_min_k → load succeeds with retain only."""
        path = tmp_path / "ref.json"
        path.write_text(json.dumps({
            "mia_min_k": dict(CANONICAL_MIA),
            "forget_truth_ratio": dict(CANONICAL_FTR),
        }))
        include = {"mia_min_k": {"access_key": "retain"}}
        result = load_and_validate_reference(_reference_cfg(path, include=include), _load_fn)
        rml = result["retain_model_logs"]
        assert rml.get("retain") is not None

    def test_include_only_forget_truth_ratio_load_succeeds(self, tmp_path):
        """File has both keys; include only forget_truth_ratio → load succeeds with retain_ftr only."""
        path = tmp_path / "ref.json"
        path.write_text(json.dumps({
            "mia_min_k": dict(CANONICAL_MIA),
            "forget_truth_ratio": dict(CANONICAL_FTR),
        }))
        include = {"forget_truth_ratio": {"access_key": "retain_ftr"}}
        result = load_and_validate_reference(_reference_cfg(path, include=include), _load_fn)
        rml = result["retain_model_logs"]
        assert rml.get("retain_ftr") is not None


# --- Fallback / invalid shapes must raise ---


class TestRoundtripFallbackOrInvalidRaises:
    """Non-canonical or fallback writer output → load must raise RetainReferenceValidationError."""

    def test_fallback_shape_trajectory_all_only_agg_value_value_by_index_step_distribution_raises(self, tmp_path):
        """File with only trajectory_all = {agg_value, value_by_index, step_distribution} (no mia_min_k, no by_step) → load raises.
        This is the shape produced when trajectory_metrics returns its fallback dict."""
        path = tmp_path / "ref.json"
        path.write_text(json.dumps({
            "trajectory_all": {
                "agg_value": {"full": {"steps": {"privleak": [0.3]}}},
                "value_by_index": {},
                "step_distribution": {"full": {"steps": {}}},
            },
        }))
        ref_cfg = _reference_cfg(path)
        with pytest.raises(RetainReferenceValidationError) as exc_info:
            load_and_validate_reference(ref_cfg, _load_fn)
        msg = str(exc_info.value).lower()
        assert "mia_min_k" in msg or "forget_truth_ratio" in msg or "required" in msg or "not found" in msg

    def test_missing_mia_min_k_include_both_raises(self, tmp_path):
        """File has only forget_truth_ratio; include requests both → raise."""
        path = tmp_path / "ref.json"
        path.write_text(json.dumps({"forget_truth_ratio": dict(CANONICAL_FTR)}))
        with pytest.raises(RetainReferenceValidationError) as exc_info:
            load_and_validate_reference(_reference_cfg(path), _load_fn)
        msg = str(exc_info.value).lower()
        assert "mia_min_k" in msg or "required" in msg or "not found" in msg

    def test_missing_forget_truth_ratio_include_both_raises(self, tmp_path):
        """File has only mia_min_k; include requests both → raise."""
        path = tmp_path / "ref.json"
        path.write_text(json.dumps({"mia_min_k": dict(CANONICAL_MIA)}))
        with pytest.raises(RetainReferenceValidationError) as exc_info:
            load_and_validate_reference(_reference_cfg(path), _load_fn)
        msg = str(exc_info.value).lower()
        assert "forget_truth_ratio" in msg or "required" in msg or "not found" in msg

    def test_malformed_mia_auc_only_no_agg_value_raises(self, tmp_path):
        """mia_min_k = {auc: 0.3} only (no agg_value) → raise."""
        path = tmp_path / "ref.json"
        path.write_text(json.dumps({"mia_min_k": {"auc": 0.3}, "forget_truth_ratio": dict(CANONICAL_FTR)}))
        with pytest.raises(RetainReferenceValidationError):
            load_and_validate_reference(_reference_cfg(path), _load_fn)

    def test_malformed_ftr_empty_dict_raises(self, tmp_path):
        """forget_truth_ratio = {} (no value_by_index or agg_value) → raise."""
        path = tmp_path / "ref.json"
        path.write_text(json.dumps({"mia_min_k": dict(CANONICAL_MIA), "forget_truth_ratio": {}}))
        with pytest.raises(RetainReferenceValidationError):
            load_and_validate_reference(_reference_cfg(path), _load_fn)

    def test_malformed_mia_agg_value_not_number_raises(self, tmp_path):
        """mia_min_k.agg_value is dict instead of number → raise."""
        path = tmp_path / "ref.json"
        path.write_text(json.dumps({
            "mia_min_k": {"agg_value": {"nested": 0.3}},
            "forget_truth_ratio": dict(CANONICAL_FTR),
        }))
        with pytest.raises(RetainReferenceValidationError):
            load_and_validate_reference(_reference_cfg(path), _load_fn)

    def test_by_step_only_but_malformed_step_value_raises(self, tmp_path):
        """by_step present but one step value is non-canonical → raise."""
        path = tmp_path / "ref.json"
        path.write_text(json.dumps({
            "mia_min_k_by_step": {"0": {"agg_value": 0.25}, "1": {"auc": 0.28}},
            "forget_truth_ratio_by_step": dict(CANONICAL_FTR_BY_STEP),
        }))
        with pytest.raises(RetainReferenceValidationError):
            load_and_validate_reference(_reference_cfg(path), _load_fn)

    def test_file_missing_raises(self, tmp_path):
        """Path points to non-existent file → raise."""
        path = tmp_path / "nonexistent.json"
        assert not path.exists()
        ref_cfg = _reference_cfg(path)
        with pytest.raises(RetainReferenceValidationError):
            load_and_validate_reference(ref_cfg, _load_fn)

    def test_ftr_value_by_index_score_not_number_raises(self, tmp_path):
        """forget_truth_ratio.value_by_index["0"].score must be number; string → raise."""
        path = tmp_path / "ref.json"
        path.write_text(json.dumps({
            "mia_min_k": dict(CANONICAL_MIA),
            "forget_truth_ratio": {"value_by_index": {"0": {"score": "not-a-number"}}, "agg_value": 0.85},
        }))
        with pytest.raises(RetainReferenceValidationError):
            load_and_validate_reference(_reference_cfg(path), _load_fn)

    def test_ftr_agg_value_only_no_value_by_index_load_succeeds(self, tmp_path):
        """forget_truth_ratio with only agg_value (no value_by_index) is valid → load succeeds."""
        path = tmp_path / "ref.json"
        path.write_text(json.dumps({
            "mia_min_k": dict(CANONICAL_MIA),
            "forget_truth_ratio": {"agg_value": 0.9},
        }))
        result = load_and_validate_reference(_reference_cfg(path), _load_fn)
        assert result["retain_model_logs"].get("retain_ftr") is not None
        assert result["retain_model_logs"]["retain_ftr"].get("agg_value") == 0.9


# --- Combinations: with/without retain_reference_mode, with/without value_by_index ---


class TestRoundtripCombinations:
    """Combinations: retain_reference_mode, keep_value_by_index, extra keys."""

    @pytest.mark.parametrize("keep_value_by_index", [True, False], ids=["keep_vbi", "strip_vbi"])
    def test_canonical_save_with_keep_value_by_index_then_load(self, tmp_path, keep_value_by_index):
        """Canonical logs → save with keep_value_by_index True/False → load succeeds."""
        ev = _make_evaluator(retain_reference_mode=keep_value_by_index)
        logs = {
            "mia_min_k": dict(CANONICAL_MIA),
            "forget_truth_ratio": dict(CANONICAL_FTR),
        }
        path = tmp_path / "ref.json"
        ev.save_logs(logs, str(path), keep_value_by_index=keep_value_by_index)
        result = load_and_validate_reference(_reference_cfg(path), _load_fn)
        assert result["retain_model_logs"].get("retain") is not None
        assert result["retain_model_logs"].get("retain_ftr") is not None

    def test_extra_keys_run_info_config_loader_ignores_canonical_loaded(self, tmp_path):
        """File has config, run_info; loader still finds mia_min_k and forget_truth_ratio."""
        ev = _make_evaluator(retain_reference_mode=True)
        logs = {
            "config": {"evaluator_name": "TOFU"},
            "run_info": {"world_size": 1, "total_samples": 10},
            "mia_min_k": dict(CANONICAL_MIA),
            "forget_truth_ratio": dict(CANONICAL_FTR),
        }
        path = tmp_path / "ref.json"
        ev.save_logs(logs, str(path), keep_value_by_index=True)
        result = load_and_validate_reference(_reference_cfg(path), _load_fn)
        rml = result["retain_model_logs"]
        assert rml.get("retain") is not None and rml.get("retain_ftr") is not None


# --- Evaluator merge path: metric return shape → save → load ---


class TestEvaluatorMergePathRoundtrip:
    """Simulate evaluator merge output: logs[first_name] = metric result → save → load. Minimal mock: only load_metrics."""

    def test_metric_returns_canonical_flat_then_saved_file_loads(self, tmp_path):
        """Merge puts trajectory_all = {mia_min_k, forget_truth_ratio, by_step}. Save via real save_logs → load succeeds."""
        ev = _make_evaluator(retain_reference_mode=True)
        # Simulate coalesced merge: result = metric return; logs[first_name] = result
        logs = {
            "trajectory_all": {
                "mia_min_k": dict(CANONICAL_MIA),
                "forget_truth_ratio": dict(CANONICAL_FTR),
                "mia_min_k_by_step": dict(CANONICAL_MIA_BY_STEP),
                "forget_truth_ratio_by_step": dict(CANONICAL_FTR_BY_STEP),
            },
        }
        logs_file = tmp_path / "results.json"
        ev.save_logs(logs, str(logs_file), keep_value_by_index=True)
        result = load_and_validate_reference(_reference_cfg(logs_file), _load_fn)
        rml = result["retain_model_logs"]
        assert rml.get("retain") is not None and rml.get("retain_ftr") is not None
        assert rml.get("retain_mia_by_step") == CANONICAL_MIA_BY_STEP
        assert rml.get("retain_forget_tr_by_step") == CANONICAL_FTR_BY_STEP

    def test_metric_returns_fallback_shape_then_saved_file_load_raises(self, tmp_path):
        """File with only trajectory_all = {agg_value, value_by_index, step_distribution} (no canonical keys) → load raises."""
        path = tmp_path / "results.json"
        path.write_text(json.dumps({
            "trajectory_all": {
                "agg_value": {"full": {"steps": {}}},
                "value_by_index": {},
                "step_distribution": {"full": {"steps": {}}},
            },
        }))
        with pytest.raises(RetainReferenceValidationError) as exc_info:
            load_and_validate_reference(_reference_cfg(path), _load_fn)
        msg = str(exc_info.value).lower()
        assert "mia_min_k" in msg or "forget_truth_ratio" in msg or "required" in msg or "not found" in msg

    def test_evaluate_with_serializable_model_then_load_succeeds(self, tmp_path):
        """Full path: evaluate() with model that has JSON-serializable config → save_logs → load succeeds."""
        from omegaconf import OmegaConf
        from evals.base import Evaluator

        logs_file = tmp_path / "TOFU_EVAL.json"
        eval_cfg = OmegaConf.create({
            "metrics": {"trajectory_all": {"datasets": [], "metrics": []}},
            "output_dir": str(tmp_path),
            "retain_reference_mode": True,
            "overwrite": False,
        })
        model = Mock()
        model.config = Mock()
        model.config._name_or_path = "test-model"

        def stub_metric(model, metric_name, cache, **kwargs):
            return {
                "mia_min_k": dict(CANONICAL_MIA),
                "forget_truth_ratio": dict(CANONICAL_FTR),
                "mia_min_k_by_step": dict(CANONICAL_MIA_BY_STEP),
                "forget_truth_ratio_by_step": dict(CANONICAL_FTR_BY_STEP),
            }

        mock_metric = Mock(side_effect=stub_metric)
        with patch.object(Evaluator, "load_metrics", return_value={"trajectory_all": mock_metric}):
            ev = Evaluator("TOFU", eval_cfg)
        with patch.object(ev, "load_logs_from_file", return_value={}):
            ev.evaluate(model, logs_file_path=str(logs_file), tokenizer=None, template_args=None)
        assert logs_file.exists()
        result = load_and_validate_reference(_reference_cfg(logs_file), _load_fn)
        rml = result["retain_model_logs"]
        assert rml.get("retain") is not None and rml.get("retain_ftr") is not None


# --- Multiple reference_log names and duplicate access_key ---


class TestRoundtripLoaderEdgeCases:
    """Loader behavior: duplicate access_key, missing path."""

    def test_duplicate_access_key_raises(self, tmp_path):
        """Include has two keys with same access_key → raise."""
        path = tmp_path / "ref.json"
        path.write_text(json.dumps({
            "mia_min_k": dict(CANONICAL_MIA),
            "forget_truth_ratio": dict(CANONICAL_FTR),
        }))
        include = {
            "mia_min_k": {"access_key": "retain"},
            "forget_truth_ratio": {"access_key": "retain"},
        }
        ref_cfg = _reference_cfg(path, include=include)
        with pytest.raises(RetainReferenceValidationError):
            load_and_validate_reference(ref_cfg, _load_fn)

    def test_no_path_skipped_no_error(self):
        """reference_logs_cfg with no path → loader skips, no error."""
        ref_cfg = {"retain_model_logs": {"path": None, "include": DEFAULT_INCLUDE}}
        result = load_and_validate_reference(ref_cfg, _load_fn)
        assert result == {}

    def test_empty_include_no_required_keys_load_returns_empty_ref(self, tmp_path):
        """Path valid but include empty → no keys requested → loader does not add retain_model_logs (ref is empty)."""
        path = tmp_path / "ref.json"
        path.write_text(json.dumps({"mia_min_k": dict(CANONICAL_MIA)}))
        ref_cfg = {"retain_model_logs": {"path": str(path), "include": {}}}
        result = load_and_validate_reference(ref_cfg, _load_fn)
        # Loader only adds to result when ref is non-empty; with no include keys, ref stays {} and is not added
        assert result == {}
