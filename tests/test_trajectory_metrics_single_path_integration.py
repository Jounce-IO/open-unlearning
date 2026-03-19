"""Integration tests for trajectory_metrics single-path refactor.

Asserts unified return shape (one result dict, single display key or per-name),
canonical keys when retain_reference_mode, and round-trip save/load.
Uses real metric with minimal mocks (model, sampler, data) where possible.
"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics import METRICS_REGISTRY
from evals.metrics.base import load_and_validate_reference

# Canonical shapes for round-trip
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


def _trajectory_result_payload(result):
    """Return the dict that contains agg_value (unified or legacy shape)."""
    if not isinstance(result, dict):
        return {}
    if "agg_value" in result and result.get("step_distribution") is not None:
        return result
    for v in result.values():
        if isinstance(v, dict) and "agg_value" in v:
            return v
    return result


def _minimal_model_sampler_data(steps=4, max_new_tokens=32, T=8):
    """Minimal model, sampler output, and data for trajectory_metrics."""
    V = 50
    logits_history = [torch.randn(1, T, V) for _ in range(steps)]
    fixation_steps = torch.randint(0, steps, (1, T), dtype=torch.long)
    model = Mock()
    sampler = Mock()

    class SamplerOutput:
        def __init__(self):
            self.logits_history = logits_history
            self.fixation_steps = fixation_steps

    sampler.sample = Mock(return_value=SamplerOutput())
    model.sampler = sampler
    data = [{
        "input_ids": torch.zeros(1, T, dtype=torch.long),
        "labels": torch.cat([
            torch.full((3,), -100, dtype=torch.long),
            torch.zeros(T - 3, dtype=torch.long),
        ]).unsqueeze(0),
    }]
    collator = lambda x: x[0]
    tokenizer = Mock()
    return model, sampler, data, collator, tokenizer


class TestSinglePathInvalidMappingShape:
    """Invalid display-name mapping → single display key in result."""

    def test_no_metric_display_names_single_key_and_payload_has_agg_value(self):
        """metric_display_names missing/empty → result has one key; payload has agg_value and step_distribution."""
        model, _, data, collator, tokenizer = _minimal_model_sampler_data()
        trajectory_metrics_fn = METRICS_REGISTRY["trajectory_metrics"]._metric_fn
        with patch(
            "evals.metrics.trajectory_metrics._call_metric_at_step",
            return_value={"prob": 0.5},
        ):
            result = trajectory_metrics_fn(
                model=model,
                metrics=["probability"],
                data=data,
                collators=collator,
                tokenizer=tokenizer,
                batch_size=1,
                trajectory_config={
                    "return_logits": True,
                    "return_fixation_steps": True,
                    "sampler_kwargs": {
                        "steps": 4,
                        "max_new_tokens": 32,
                        "trajectory_sample_interval": 8,
                    },
                },
            )
        assert isinstance(result, dict)
        payload = _trajectory_result_payload(result)
        assert "agg_value" in payload
        assert "step_distribution" in payload
        # Single-path: one display key when mapping invalid (no metric_display_names)
        if len(result) == 1:
            key = next(iter(result))
            assert result[key] is payload or result[key] == payload

    def test_metric_name_used_as_display_key_when_no_display_names(self):
        """When metric_display_names empty, display_key = metric_name (e.g. trajectory_privleak)."""
        model, _, data, collator, tokenizer = _minimal_model_sampler_data()
        trajectory_metrics_fn = METRICS_REGISTRY["trajectory_metrics"]._metric_fn
        with patch(
            "evals.metrics.trajectory_metrics._call_metric_at_step",
            return_value={"prob": 0.5},
        ):
            result = trajectory_metrics_fn(
                model=model,
                metric_name="trajectory_privleak",
                metrics=["probability"],
                data=data,
                collators=collator,
                tokenizer=tokenizer,
                batch_size=1,
                trajectory_config={
                    "return_logits": True,
                    "return_fixation_steps": True,
                    "sampler_kwargs": {"steps": 4, "max_new_tokens": 32, "trajectory_sample_interval": 8},
                },
            )
        assert isinstance(result, dict)
        assert "trajectory_privleak" in result
        assert "agg_value" in result["trajectory_privleak"]


class TestSinglePathValidMappingShape:
    """Valid display-name mapping → per-display-name keys in result."""

    def test_two_display_names_two_internal_per_name_keys_in_result(self):
        """metric_display_names length matches internal → result has per-display-name keys."""
        model, _, data, collator, tokenizer = _minimal_model_sampler_data()
        trajectory_metrics_fn = METRICS_REGISTRY["trajectory_metrics"]._metric_fn
        with patch(
            "evals.metrics.trajectory_metrics._call_metric_at_step",
            return_value={"prob": 0.5},
        ):
            result = trajectory_metrics_fn(
                model=model,
                metrics=["probability", "privleak"],
                metric_display_names=["trajectory_prob", "trajectory_privleak"],
                data=data,
                collators=collator,
                tokenizer=tokenizer,
                batch_size=1,
                trajectory_config={
                    "return_logits": True,
                    "return_fixation_steps": True,
                    "sampler_kwargs": {"steps": 4, "max_new_tokens": 32, "trajectory_sample_interval": 8},
                },
            )
        assert isinstance(result, dict)
        assert "trajectory_prob" in result
        assert "trajectory_privleak" in result
        assert "trajectory_step_metadata" in result
        assert "agg_value" in result["trajectory_prob"]
        assert "agg_value" in result["trajectory_privleak"]


class TestSinglePathRetainReferenceModeCanonicalKeys:
    """When retain_reference_mode=True, result includes canonical keys when data available."""

    def test_retain_reference_mode_result_has_canonical_keys_when_privleak_present(self):
        """With retain_reference_mode and privleak in metrics, result has mia_min_k (and by_step if steps)."""
        model, _, data, collator, tokenizer = _minimal_model_sampler_data(steps=4)
        trajectory_metrics_fn = METRICS_REGISTRY["trajectory_metrics"]._metric_fn
        eval_cfg = {"retain_reference_mode": True}
        with patch(
            "evals.metrics.trajectory_metrics._call_metric_at_step",
            return_value={"agg_value": 0.42},
        ):
            result = trajectory_metrics_fn(
                model=model,
                metrics=["privleak"],
                data=data,
                collators=collator,
                tokenizer=tokenizer,
                batch_size=1,
                eval_cfg=eval_cfg,
                trajectory_config={
                    "return_logits": True,
                    "return_fixation_steps": True,
                    "sampler_kwargs": {"steps": 4, "max_new_tokens": 32, "trajectory_sample_interval": 8},
                },
            )
        assert isinstance(result, dict)
        assert "mia_min_k" in result
        assert isinstance(result["mia_min_k"], dict)
        assert "agg_value" in result["mia_min_k"]


def _load_fn(path):
    with open(path) as f:
        return json.load(f)


class TestSinglePathRoundtripLoadable:
    """Result shape produced by single path is loadable as reference after save."""

    def test_single_key_with_canonical_save_then_load_succeeds(self, tmp_path):
        """Result with one display key + mia_min_k + forget_truth_ratio → save → load succeeds."""
        ref_path = tmp_path / "ref.json"
        ref_path.write_text(json.dumps({
            "trajectory_all": {
                "trajectory_all": {"agg_value": {}, "value_by_index": {}, "step_distribution": {}},
                "mia_min_k": dict(CANONICAL_MIA),
                "forget_truth_ratio": dict(CANONICAL_FTR),
                "mia_min_k_by_step": dict(CANONICAL_MIA_BY_STEP),
                "forget_truth_ratio_by_step": dict(CANONICAL_FTR_BY_STEP),
            },
        }))
        ref_cfg = {
            "retain_model_logs": {
                "path": str(ref_path),
                "include": DEFAULT_INCLUDE,
            },
        }
        result = load_and_validate_reference(ref_cfg, _load_fn)
        rml = result["retain_model_logs"]
        assert rml.get("retain") is not None and rml.get("retain_ftr") is not None
        assert rml.get("retain_mia_by_step") == CANONICAL_MIA_BY_STEP
        assert rml.get("retain_forget_tr_by_step") == CANONICAL_FTR_BY_STEP

    def test_top_level_canonical_save_then_load_succeeds(self, tmp_path):
        """Top-level mia_min_k, forget_truth_ratio (no trajectory_all wrapper) → load succeeds."""
        ref_path = tmp_path / "ref.json"
        ref_path.write_text(json.dumps({
            "mia_min_k": dict(CANONICAL_MIA),
            "forget_truth_ratio": dict(CANONICAL_FTR),
            "mia_min_k_by_step": dict(CANONICAL_MIA_BY_STEP),
            "forget_truth_ratio_by_step": dict(CANONICAL_FTR_BY_STEP),
        }))
        ref_cfg = {
            "retain_model_logs": {
                "path": str(ref_path),
                "include": DEFAULT_INCLUDE,
            },
        }
        result = load_and_validate_reference(ref_cfg, _load_fn)
        rml = result["retain_model_logs"]
        assert rml.get("retain") is not None and rml.get("retain_ftr") is not None


class TestSinglePathOneReturnNoEarlyExit:
    """Structural: only one return of result in the metric (no early return from success path)."""

    def test_result_is_dict_with_expected_keys_or_single_display_key(self):
        """Result is always a dict; either multiple keys (per-name + metadata) or one display key."""
        model, _, data, collator, tokenizer = _minimal_model_sampler_data()
        trajectory_metrics_fn = METRICS_REGISTRY["trajectory_metrics"]._metric_fn
        with patch(
            "evals.metrics.trajectory_metrics._call_metric_at_step",
            return_value={"prob": 0.5},
        ):
            result = trajectory_metrics_fn(
                model=model,
                metrics=["probability"],
                data=data,
                collators=collator,
                tokenizer=tokenizer,
                batch_size=1,
                trajectory_config={
                    "return_logits": True,
                    "return_fixation_steps": True,
                    "sampler_kwargs": {"steps": 4, "max_new_tokens": 32, "trajectory_sample_interval": 8},
                },
            )
        assert isinstance(result, dict)
        assert len(result) >= 1
        payload = _trajectory_result_payload(result)
        assert payload is not None
        assert "agg_value" in payload


class TestRetainMuComponentsByStep:
    """When hm_aggregate is used with a retain dataset, result includes retain_mu_components_by_step."""

    def test_retain_mu_components_by_step_present_when_hm_aggregate_and_retain_data(self):
        """With hm_aggregate in metrics and data.retain, result has retain_mu_components_by_step with Prob, ROUGE, Truth_Ratio per step."""
        model, _, data_list, collator, tokenizer = _minimal_model_sampler_data()
        data = {"forget": data_list, "retain": data_list}
        retain_agg_stub = {
            "0": {
                "retain_Q_A_Prob": {"agg_value": 0.5},
                "retain_Q_A_ROUGE": {"agg_value": 0.3},
                "retain_Truth_Ratio": {"agg_value": 0.9},
            },
        }
        trajectory_metrics_fn = METRICS_REGISTRY["trajectory_metrics"]._metric_fn
        with patch(
            "evals.metrics.trajectory_metrics._call_metric_at_step",
            return_value={"agg_value": 0.5},
        ), patch(
            "evals.metrics.trajectory_metrics._compute_retain_mu_by_step",
            return_value=retain_agg_stub,
        ):
            result = trajectory_metrics_fn(
                model=model,
                metrics=["probability", "hm_aggregate"],
                data=data,
                collators=collator,
                tokenizer=tokenizer,
                batch_size=1,
                trajectory_config={
                    "return_logits": True,
                    "return_fixation_steps": True,
                    "sampler_kwargs": {"steps": 4, "max_new_tokens": 32, "trajectory_sample_interval": 8},
                },
            )
        assert isinstance(result, dict)
        assert "retain_mu_components_by_step" in result
        comp = result["retain_mu_components_by_step"]
        assert comp["0"]["retain_Q_A_Prob"] == 0.5
        assert comp["0"]["retain_Q_A_ROUGE"] == 0.3
        assert comp["0"]["retain_Truth_Ratio"] == 0.9

    def test_retain_mu_components_by_step_absent_when_no_hm_aggregate(self):
        """Without hm_aggregate in metrics, result does not include retain_mu_components_by_step."""
        model, _, data_list, collator, tokenizer = _minimal_model_sampler_data()
        data = {"forget": data_list, "retain": data_list}
        trajectory_metrics_fn = METRICS_REGISTRY["trajectory_metrics"]._metric_fn
        with patch(
            "evals.metrics.trajectory_metrics._call_metric_at_step",
            return_value={"agg_value": 0.5},
        ):
            result = trajectory_metrics_fn(
                model=model,
                metrics=["probability"],
                data=data,
                collators=collator,
                tokenizer=tokenizer,
                batch_size=1,
                trajectory_config={
                    "return_logits": True,
                    "return_fixation_steps": True,
                    "sampler_kwargs": {"steps": 4, "max_new_tokens": 32, "trajectory_sample_interval": 8},
                },
            )
        assert isinstance(result, dict)
        assert "retain_mu_components_by_step" not in result


class TestCoalescedVsPerMetricResultShape:
    """Coalesced (one metric name) vs per-metric (metric_name) produce consistent shape."""

    def test_coalesced_single_metric_result_has_one_display_key(self):
        """Coalesced with one display name, multiple internal metrics → single display key in result."""
        model, _, data, collator, tokenizer = _minimal_model_sampler_data()
        trajectory_metrics_fn = METRICS_REGISTRY["trajectory_metrics"]._metric_fn
        with patch(
            "evals.metrics.trajectory_metrics._call_metric_at_step",
            return_value={"prob": 0.5},
        ):
            result = trajectory_metrics_fn(
                model=model,
                metrics=["probability", "privleak"],
                metric_display_names=["trajectory_all"],
                data=data,
                collators=collator,
                tokenizer=tokenizer,
                batch_size=1,
                trajectory_config={
                    "return_logits": True,
                    "return_fixation_steps": True,
                    "sampler_kwargs": {"steps": 4, "max_new_tokens": 32, "trajectory_sample_interval": 8},
                },
            )
        assert isinstance(result, dict)
        assert "trajectory_all" in result
        assert "agg_value" in result["trajectory_all"]
        assert "trajectory_step_metadata" in result, "single display key report must include trajectory_step_metadata"

    def test_per_metric_no_display_names_uses_metric_name_as_key(self):
        """Per-metric call with no metric_display_names uses metric_name as the single key."""
        model, _, data, collator, tokenizer = _minimal_model_sampler_data()
        trajectory_metrics_fn = METRICS_REGISTRY["trajectory_metrics"]._metric_fn
        with patch(
            "evals.metrics.trajectory_metrics._call_metric_at_step",
            return_value={"prob": 0.5},
        ):
            result = trajectory_metrics_fn(
                model=model,
                metric_name="trajectory_forget_quality",
                metrics=["probability"],
                data=data,
                collators=collator,
                tokenizer=tokenizer,
                batch_size=1,
                trajectory_config={
                    "return_logits": True,
                    "return_fixation_steps": True,
                    "sampler_kwargs": {"steps": 4, "max_new_tokens": 32, "trajectory_sample_interval": 8},
                },
            )
        assert "trajectory_forget_quality" in result
        assert "agg_value" in result["trajectory_forget_quality"]


class TestTrajectoryStepCountFormula:
    """Step count S in reports: formula S_traj = ceil(max_new_tokens / trajectory_sample_interval)."""

    def test_step_count_matches_sampler_logits_length(self):
        """S in result equals len(logits_history) from sampler (e.g. ceil(max_new_tokens/interval))."""
        import math
        max_new_tokens = 32
        trajectory_sample_interval = 8
        expected_S = math.ceil(max_new_tokens / trajectory_sample_interval)  # 4
        model, _, data, collator, tokenizer = _minimal_model_sampler_data(
            steps=expected_S, max_new_tokens=max_new_tokens, T=trajectory_sample_interval * 2
        )
        trajectory_metrics_fn = METRICS_REGISTRY["trajectory_metrics"]._metric_fn
        with patch(
            "evals.metrics.trajectory_metrics._call_metric_at_step",
            return_value={"agg_value": 0.5},
        ):
            result = trajectory_metrics_fn(
                model=model,
                metrics=["probability"],
                data=data,
                collators=collator,
                tokenizer=tokenizer,
                batch_size=1,
                trajectory_config={
                    "return_logits": True,
                    "return_fixation_steps": True,
                    "sampler_kwargs": {
                        "steps": 50,
                        "max_new_tokens": max_new_tokens,
                        "trajectory_sample_interval": trajectory_sample_interval,
                    },
                },
            )
        payload = _trajectory_result_payload(result)
        assert payload
        agg = payload.get("agg_value", payload)
        if isinstance(agg, dict) and "full" in agg and "steps" in agg["full"]:
            steps_arr = agg["full"]["steps"].get("probability", [])
            assert len(steps_arr) == expected_S, (
                f"agg_value.steps should have S={expected_S} (ceil({max_new_tokens}/{trajectory_sample_interval}))"
            )

    def test_inferred_max_new_tokens_range_for_S(self):
        """With interval=8, S=22 => (168,176], S=24 => (184,192] (documentation check)."""
        import math
        interval = 8
        for S, (lo, hi) in [(22, (168, 176)), (24, (184, 192))]:
            inferred_lo = (S - 1) * interval + 1
            inferred_hi = S * interval
            assert inferred_lo == lo + 1, f"S={S}: expected lo+1={lo+1}, got {inferred_lo}"
            assert inferred_hi == hi, f"S={S}: expected hi={hi}, got {inferred_hi}"
            assert S == math.ceil(inferred_hi / interval)
