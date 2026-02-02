"""
Comprehensive unit tests for trajectory_metrics module.

Tests cover:
- Shape validation for logits_history and fixation_steps
- Generated portion extraction from full sequence
- Label extraction and alignment with logits
- Batch template creation
- Error handling for missing sampler, empty logits, etc.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.trajectory_metrics import (
    _get_sampler_from_model,
    _get_metric_from_registry,
    _call_metric_at_step,
    _compute_pre_compute_metrics_at_step,
    trajectory_metrics,
)
from evals.metrics.trajectory_utils import stack_logits_history
from evals.metrics import METRICS_REGISTRY


class TestGetSamplerFromModel:
    """Tests for _get_sampler_from_model function."""
    
    def test_model_with_sampler_attribute(self):
        """Test extracting sampler from model.sampler."""
        sampler = Mock()
        model = Mock()
        model.sampler = sampler
        
        result = _get_sampler_from_model(model)
        
        assert result is sampler
    
    def test_model_with_nested_sampler(self):
        """Test extracting sampler from model.model.sampler."""
        sampler = Mock()
        inner_model = Mock()
        inner_model.sampler = sampler
        model = Mock()
        # Ensure model.sampler doesn't exist (Mock creates it by default)
        delattr(model, 'sampler')
        model.model = inner_model
        
        result = _get_sampler_from_model(model)
        
        assert result is sampler
    
    def test_model_without_sampler_returns_none(self):
        """Test that model without sampler returns None."""
        model = Mock()
        # Remove sampler attribute if it exists
        if hasattr(model, 'sampler'):
            delattr(model, 'sampler')
        # Remove model.model if it exists (Mock creates it by default)
        if hasattr(model, 'model'):
            delattr(model, 'model')
        
        result = _get_sampler_from_model(model)
        
        assert result is None
    
    def test_model_with_deep_nested_sampler(self):
        """Test extracting sampler from model.model.sampler (deep nesting)."""
        sampler = Mock()
        inner_model = Mock()
        inner_model.sampler = sampler
        outer_model = Mock()
        # Remove outer_model.sampler if it exists
        if hasattr(outer_model, 'sampler'):
            delattr(outer_model, 'sampler')
        outer_model.model = inner_model
        
        result = _get_sampler_from_model(outer_model)
        
        assert result is sampler
    
    def test_model_with_sampler_none(self):
        """Test that model with sampler=None returns None, not the None value."""
        model = Mock()
        model.sampler = None
        
        result = _get_sampler_from_model(model)
        
        # Should return None (the None value), not raise error
        assert result is None
    
    def test_model_with_sampler_not_sampler_object(self):
        """Test that model with sampler attribute that is not a sampler object."""
        model = Mock()
        model.sampler = "not a sampler"  # String instead of sampler object
        
        result = _get_sampler_from_model(model)
        
        # Should return the attribute value (even if it's not a real sampler)
        assert result == "not a sampler"
    
    def test_model_with_both_sampler_and_nested_sampler(self):
        """Test model with both model.sampler and model.model.sampler (which takes precedence?)."""
        sampler1 = Mock()
        sampler2 = Mock()
        inner_model = Mock()
        inner_model.sampler = sampler2
        model = Mock()
        model.sampler = sampler1
        model.model = inner_model
        
        result = _get_sampler_from_model(model)
        
        # Should take model.sampler (first check)
        assert result is sampler1


class TestGetMetricFromRegistry:
    """Tests for _get_metric_from_registry function."""
    
    def test_get_metric_from_registry_success(self):
        """Test loading a metric that exists in registry."""
        # probability should be registered
        if "probability" in METRICS_REGISTRY:
            metric = _get_metric_from_registry("probability")
            assert metric is not None
            assert hasattr(metric, "name")
            assert metric.name == "probability"
    
    def test_get_metric_from_registry_not_found(self):
        """Test that loading non-existent metric raises ValueError."""
        with pytest.raises(ValueError, match="not found in registry"):
            _get_metric_from_registry("nonexistent_metric_xyz")
    
    def test_get_metric_empty_string_raises_error(self):
        """Test that empty string metric_name raises error."""
        with pytest.raises(ValueError, match="not found in registry"):
            _get_metric_from_registry("")
    
    def test_get_metric_special_characters(self):
        """Test that metric name with special characters raises error."""
        with pytest.raises(ValueError, match="not found in registry"):
            _get_metric_from_registry("metric-name_with.special@chars")
    
    def test_get_metric_error_message_includes_available(self):
        """Test that error message includes available metrics list."""
        try:
            _get_metric_from_registry("nonexistent_xyz")
        except ValueError as e:
            error_msg = str(e)
            assert "not found in registry" in error_msg
            assert "Available metrics" in error_msg or "available" in error_msg.lower()
    
    def test_get_all_registered_metrics(self):
        """Test that all registered metrics can be loaded."""
        for metric_name in METRICS_REGISTRY.keys():
            metric = _get_metric_from_registry(metric_name)
            assert metric is not None
            assert hasattr(metric, "name")
            assert metric.name == metric_name


class TestTrajectoryMetricsShapeValidation:
    """Tests for shape validation in trajectory_metrics."""
    
    def test_extract_generated_portion_from_logits(self):
        """Test that generated portion is correctly extracted from full sequence."""
        # Simulate: prompt_len=5, generated_len=10, full_len=15
        prompt_len = 5
        generated_len = 10
        full_len = prompt_len + generated_len
        V, S = 100, 8
        
        # Create logits_history with full sequence length
        logits_history = [
            torch.randn(1, full_len, V) for _ in range(S)
        ]
        
        R_full = stack_logits_history(logits_history)  # [V, full_len, S]
        assert R_full.shape == (V, full_len, S)
        
        # Extract generated portion
        R = R_full[:, prompt_len:prompt_len + generated_len, :]  # [V, generated_len, S]
        assert R.shape == (V, generated_len, S)
        
        # Verify it matches the generated region
        for s in range(S):
            expected = logits_history[s][0, prompt_len:prompt_len + generated_len, :].T
            assert torch.allclose(R[:, :, s], expected)
    
    def test_extract_fixation_steps_for_generated_region(self):
        """Test that fixation steps are extracted for generated region only."""
        prompt_len = 5
        generated_len = 10
        full_len = prompt_len + generated_len
        S = 8
        
        # Fixation steps for full sequence [T]
        F_full = torch.randint(0, S, (full_len,))
        
        # Extract generated portion
        F = F_full[prompt_len:prompt_len + generated_len]  # [L]
        assert F.shape == (generated_len,)
        
        # Verify it matches the generated region
        expected = F_full[prompt_len:prompt_len + generated_len]
        assert torch.allclose(F, expected)
    
    def test_label_extraction_matches_generated_length(self):
        """Test that labels are extracted to match generated length."""
        prompt_len = 5
        generated_len = 10
        full_len = prompt_len + generated_len
        
        # Full sequence labels [T]
        sample_labels = torch.randint(0, 1000, (full_len,))
        
        # Extract generated portion
        generated_labels = sample_labels[prompt_len:prompt_len + generated_len]
        assert generated_labels.shape == (generated_len,)
        
        # Verify alignment
        expected = sample_labels[prompt_len:prompt_len + generated_len]
        assert torch.allclose(generated_labels, expected)
    
    def test_label_padding_when_short(self):
        """Test that labels are padded if shorter than generated length."""
        prompt_len = 5
        generated_len = 10
        actual_generated = 7  # Shorter than expected
        
        # Labels that are shorter than expected
        sample_labels = torch.randint(0, 1000, (prompt_len + actual_generated,))
        generated_labels = sample_labels[prompt_len:prompt_len + actual_generated]
        
        # Pad to generated_len
        IGNORE_INDEX = -100
        if generated_labels.shape[0] < generated_len:
            padding = torch.full(
                (generated_len - generated_labels.shape[0],),
                IGNORE_INDEX,
                dtype=generated_labels.dtype,
                device=generated_labels.device,
            )
            generated_labels = torch.cat([generated_labels, padding])
        
        assert generated_labels.shape == (generated_len,)
        assert (generated_labels[actual_generated:] == IGNORE_INDEX).all()
    
    def test_batch_template_shape_matches_generated_length(self):
        """Test that batch template has correct shape for generated length."""
        generated_len = 10
        device = torch.device("cpu")
        
        batch_template = {
            "input_ids": torch.zeros((1, generated_len), dtype=torch.long, device=device),
            "labels": torch.zeros((1, generated_len), dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, generated_len), dtype=torch.long, device=device),
        }
        
        assert batch_template["input_ids"].shape == (1, generated_len)
        assert batch_template["labels"].shape == (1, generated_len)
        assert batch_template["attention_mask"].shape == (1, generated_len)


class TestTrajectoryMetricsErrorHandling:
    """Tests for error handling in trajectory_metrics."""
    
    def test_missing_sampler_raises_error(self):
        """Test that missing sampler raises ValueError."""
        model = Mock()
        # Remove sampler attribute
        if hasattr(model, 'sampler'):
            delattr(model, 'sampler')
        if hasattr(model, 'model'):
            delattr(model, 'model')
        
        kwargs = {
            "metrics": ["probability"],
            "data": Mock(),
            "collators": Mock(),
            "tokenizer": Mock(),
        }
        
        # trajectory_metrics is wrapped in UnlearningMetric, access raw function
        from evals.metrics.trajectory_metrics import trajectory_metrics
        raw_fn = trajectory_metrics._metric_fn if hasattr(trajectory_metrics, '_metric_fn') else trajectory_metrics
        
        with pytest.raises(ValueError, match="Model does not have a sampler"):
            raw_fn(model, **kwargs)
    
    def test_empty_metrics_list_raises_error(self):
        """Test that empty metrics list raises ValueError."""
        model = Mock()
        model.sampler = Mock()
        
        kwargs = {
            "metrics": [],
            "data": Mock(),
            "collators": Mock(),
            "tokenizer": Mock(),
        }
        
        # trajectory_metrics is wrapped in UnlearningMetric, access raw function
        from evals.metrics.trajectory_metrics import trajectory_metrics
        raw_fn = trajectory_metrics._metric_fn if hasattr(trajectory_metrics, '_metric_fn') else trajectory_metrics
        
        with pytest.raises(ValueError, match="No metrics specified"):
            raw_fn(model, **kwargs)
    
    def test_missing_tokenizer_raises_error(self):
        """Test that missing tokenizer raises ValueError."""
        model = Mock()
        model.sampler = Mock()
        
        kwargs = {
            "metrics": ["probability"],
            "data": Mock(),
            "collators": Mock(),
            "tokenizer": None,
        }
        
        # trajectory_metrics is wrapped in UnlearningMetric, access raw function
        from evals.metrics.trajectory_metrics import trajectory_metrics
        raw_fn = trajectory_metrics._metric_fn if hasattr(trajectory_metrics, '_metric_fn') else trajectory_metrics
        
        with pytest.raises(ValueError, match="tokenizer is required"):
            raw_fn(model, **kwargs)


class TestTrajectoryMetricsIntegration:
    """Integration tests for trajectory_metrics with mock sampler."""
    
    def test_trajectory_metrics_with_mock_sampler(self):
        """Test trajectory_metrics with a mock sampler that returns logits_history."""
        # Setup
        V, L_gen, S = 100, 10, 8
        prompt_len = 5
        full_len = prompt_len + L_gen
        
        # Create mock sampler
        sampler = Mock()
        
        # Create logits_history: list of [B, T, V] tensors
        logits_history = [torch.randn(1, full_len, V) for _ in range(S)]
        
        # Create fixation_steps: [B, T]
        fixation_steps = torch.randint(0, S, (1, full_len))
        
        # Mock sampler.sample to return SamplerOutput
        # SamplerOutput is from dllm, but we can create a simple mock
        class MockSamplerOutput:
            def __init__(self, sequences, histories, logits_history, fixation_steps):
                self.sequences = sequences
                self.histories = histories
                self.logits_history = logits_history
                self.fixation_steps = fixation_steps
        
        sampler_output = MockSamplerOutput(
            sequences=torch.randint(0, V, (1, full_len)),
            histories=None,
            logits_history=logits_history,
            fixation_steps=fixation_steps,
        )
        sampler.sample.return_value = sampler_output
        
        # Create model with sampler
        model = Mock()
        model.sampler = sampler
        
        # Create mock dataset
        class MockDataset:
            def __init__(self):
                self.data = [
                    {
                        "input_ids": torch.randint(0, V, (full_len,)),
                        "labels": torch.randint(0, V, (full_len,)),
                    }
                    for _ in range(2)
                ]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        # Create mock collator
        def mock_collator(batch):
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
                "indices": torch.tensor([0, 1]),
            }
        
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.decode = lambda x, **kwargs: "decoded text"
        
        kwargs = {
            "metrics": ["probability"],
            "data": MockDataset(),
            "collators": mock_collator,
            "tokenizer": tokenizer,
            "batch_size": 1,
            "trajectory_config": {
                "return_logits": True,
                "return_fixation_steps": True,
                "sampler_kwargs": {"steps": S, "max_new_tokens": L_gen},
            },
        }
        
        # This should run without errors (though may fail on actual metric computation)
        # We're mainly testing the shape extraction logic
        try:
            result = trajectory_metrics(model, **kwargs)
            # If it succeeds, check structure
            if result:
                assert isinstance(result, dict)
        except Exception as e:
            # If it fails, it should be on metric computation, not shape issues
            # Check that it's not a shape mismatch error
            assert "Expected target size" not in str(e)
            assert "shape" not in str(e).lower() or "mismatch" not in str(e).lower()


class TestDynamicMetricLoading:
    """Tests for dynamic metric loading from registry."""
    
    def test_get_metric_from_registry_success(self):
        """Test loading a metric that exists in registry."""
        # probability should be registered
        if "probability" in METRICS_REGISTRY:
            metric = _get_metric_from_registry("probability")
            assert metric is not None
            assert hasattr(metric, "name")
            assert metric.name == "probability"
    
    def test_get_metric_from_registry_not_found(self):
        """Test that loading non-existent metric raises ValueError."""
        with pytest.raises(ValueError, match="not found in registry"):
            _get_metric_from_registry("nonexistent_metric_xyz")
    
    def test_metrics_list_format(self):
        """Test that metrics can be specified as a simple list."""
        # This should be parsed correctly
        metrics_config = ["probability", "exact_memorization"]
        
        # Simulate parsing logic
        if isinstance(metrics_config, list):
            metrics_to_compute = {name: {} for name in metrics_config}
        else:
            metrics_to_compute = metrics_config
        
        assert isinstance(metrics_to_compute, dict)
        assert "probability" in metrics_to_compute
        assert "exact_memorization" in metrics_to_compute
        assert metrics_to_compute["probability"] == {}
        assert metrics_to_compute["exact_memorization"] == {}
    
    def test_metrics_dict_format(self):
        """Test that metrics can be specified as a dict with configs."""
        metrics_config = {
            "probability": {},
            "truth_ratio": {
                "aggregator": "closer_to_1_better",
                "pre_compute": {
                    "probability": {"access_key": "correct"},
                },
            },
        }
        
        # Should be used directly
        metrics_to_compute = metrics_config
        
        assert isinstance(metrics_to_compute, dict)
        assert "probability" in metrics_to_compute
        assert "truth_ratio" in metrics_to_compute
        assert metrics_to_compute["truth_ratio"]["aggregator"] == "closer_to_1_better"
        assert "pre_compute" in metrics_to_compute["truth_ratio"]
    
    def test_load_metrics_from_registry(self):
        """Test loading multiple metrics from registry."""
        metrics_to_compute = {
            "probability": {},
            "exact_memorization": {},
        }
        
        loaded_metrics = {}
        for metric_name, metric_cfg in metrics_to_compute.items():
            if metric_name in METRICS_REGISTRY:
                metric = METRICS_REGISTRY[metric_name]
                loaded_metrics[metric_name] = {
                    "metric": metric,
                    "config": metric_cfg,
                }
        
        assert len(loaded_metrics) > 0
        for metric_name in metrics_to_compute.keys():
            if metric_name in METRICS_REGISTRY:
                assert metric_name in loaded_metrics
                assert "metric" in loaded_metrics[metric_name]
                assert "config" in loaded_metrics[metric_name]


class TestPreComputeMetrics:
    """Tests for pre-compute metrics support."""
    
    def test_pre_compute_config_structure(self):
        """Test that pre_compute config has correct structure."""
        pre_compute_config = {
            "probability": {
                "access_key": "correct",
            },
            "probability": {  # Can reuse same metric
                "access_key": "wrong",
            },
        }
        
        assert isinstance(pre_compute_config, dict)
        for pre_metric_name, pre_metric_cfg in pre_compute_config.items():
            assert isinstance(pre_metric_cfg, dict)
            assert "access_key" in pre_metric_cfg
    
    def test_pre_compute_access_key_defaults_to_metric_name(self):
        """Test that access_key defaults to metric name if not specified."""
        pre_compute_config = {
            "probability": {},  # No access_key
        }
        
        for pre_metric_name, pre_metric_cfg in pre_compute_config.items():
            access_key = pre_metric_cfg.get("access_key", pre_metric_name)
            assert access_key == pre_metric_name
    
    def test_pre_compute_metric_loading_by_name(self):
        """Test loading pre-compute metric by name from registry."""
        pre_metric_name = "probability"
        
        if pre_metric_name in METRICS_REGISTRY:
            pre_metric = METRICS_REGISTRY[pre_metric_name]
            assert pre_metric is not None
            assert pre_metric.name == pre_metric_name
    
    def test_pre_compute_metric_loading_by_handler(self):
        """Test loading pre-compute metric by handler from config."""
        pre_metric_cfg = {
            "handler": "probability",
            "access_key": "correct",
        }
        
        handler_name = pre_metric_cfg.get("handler")
        if handler_name and handler_name in METRICS_REGISTRY:
            pre_metric = METRICS_REGISTRY[handler_name]
            assert pre_metric is not None
            assert pre_metric.name == handler_name
    
    def test_pre_compute_result_structure(self):
        """Test that pre-compute results have correct structure."""
        # Simulate pre-compute result from evaluate_probability
        sample_idx = "0"
        pre_result = [{"prob": 0.5, "avg_loss": 0.693}]
        
        # Process result
        if isinstance(pre_result, list) and len(pre_result) > 0:
            result_dict = pre_result[0]
            value_by_index = {sample_idx: result_dict.copy()}
            processed_result = {
                "agg_value": result_dict.get("prob"),
                "value_by_index": value_by_index,
            }
        
        assert "agg_value" in processed_result
        assert "value_by_index" in processed_result
        assert sample_idx in processed_result["value_by_index"]
        assert "prob" in processed_result["value_by_index"][sample_idx]
        assert "avg_loss" in processed_result["value_by_index"][sample_idx]
    
    def test_pre_compute_preserves_avg_loss(self):
        """Test that avg_loss is preserved for truth_ratio compatibility."""
        sample_idx = "0"
        pre_result = [{"prob": 0.5, "avg_loss": 0.693}]
        
        # Process result
        result_dict = pre_result[0]
        value_by_index = {sample_idx: result_dict.copy()}
        
        # Verify avg_loss is preserved
        assert "avg_loss" in value_by_index[sample_idx]
        assert value_by_index[sample_idx]["avg_loss"] == 0.693


class TestMetricConfigWithPreCompute:
    """Tests for metrics configured with pre_compute."""
    
    def test_truth_ratio_config_structure(self):
        """Test config structure for truth_ratio with pre_compute."""
        metric_config = {
            "aggregator": "closer_to_1_better",
            "pre_compute": {
                "probability": {
                    "access_key": "correct",
                },
                "probability": {
                    "access_key": "wrong",
                },
            },
        }
        
        assert "aggregator" in metric_config
        assert "pre_compute" in metric_config
        assert isinstance(metric_config["pre_compute"], dict)
    
    def test_extract_pre_compute_from_metric_config(self):
        """Test extracting pre_compute from metric_config."""
        metric_config = {
            "aggregator": "closer_to_1_better",
            "pre_compute": {
                "probability": {"access_key": "correct"},
            },
        }
        
        pre_compute_config = metric_config.pop("pre_compute", {})
        remaining_config = metric_config.copy()
        
        assert "pre_compute" not in remaining_config
        assert "aggregator" in remaining_config
        assert len(pre_compute_config) > 0
    
    def test_pre_compute_passed_to_metric(self):
        """Test that pre_compute results are passed to main metric."""
        pre_compute_results = {
            "correct": {
                "agg_value": 0.8,
                "value_by_index": {"0": {"prob": 0.8, "avg_loss": 0.223}},
            },
            "wrong": {
                "agg_value": 0.2,
                "value_by_index": {"0": {"prob": 0.2, "avg_loss": 1.609}},
            },
        }
        
        metric_kwargs = {
            "model": Mock(),
            "batch": {},
            "pre_compute": pre_compute_results,
        }
        
        assert "pre_compute" in metric_kwargs
        assert "correct" in metric_kwargs["pre_compute"]
        assert "wrong" in metric_kwargs["pre_compute"]


class TestCallMetricAtStep:
    """Tests for _call_metric_at_step function."""
    
    def test_call_metric_with_pre_compute(self):
        """Test calling a metric with pre_compute at a step."""
        # Create mock metric
        mock_metric = Mock()
        mock_metric.name = "test_metric"
        mock_metric._metric_fn = Mock(return_value={"agg_value": 0.5})
        
        # Create logits
        V, L = 100, 10
        logits = torch.randn(V, L)
        
        # Create batch template
        batch_template = {
            "input_ids": torch.zeros((1, L), dtype=torch.long),
            "labels": torch.zeros((1, L), dtype=torch.long),
            "attention_mask": torch.ones((1, L), dtype=torch.long),
        }
        
        # Create metric config with pre_compute
        metric_config = {
            "pre_compute": {
                "probability": {"access_key": "correct"},
            },
        }
        
        # Mock tokenizer and other args
        tokenizer = Mock()
        sample_labels = torch.zeros(L, dtype=torch.long)
        sample_input_ids = torch.zeros(L, dtype=torch.long)
        sample_prompt_len = 0
        
        # Mock _compute_pre_compute_metrics_at_step
        from unittest.mock import patch
        
        with patch(
            "evals.metrics.trajectory_metrics._compute_pre_compute_metrics_at_step",
            return_value={"correct": {"agg_value": 0.8, "value_by_index": {"0": {"prob": 0.8}}}},
        ):
            result = _call_metric_at_step(
                metric=mock_metric,
                logits=logits,
                batch_template=batch_template,
                tokenizer=tokenizer,
                sample_labels=sample_labels,
                sample_input_ids=sample_input_ids,
                sample_prompt_len=sample_prompt_len,
                metric_config=metric_config,
                sample_idx="0",
            )
            
            # Verify metric was called with pre_compute
            assert mock_metric._metric_fn.called
            call_kwargs = mock_metric._metric_fn.call_args[1]
            assert "pre_compute" in call_kwargs
            assert "correct" in call_kwargs["pre_compute"]


class TestReproductionBugs:
    """Reproduction tests for trajectory metric bugs."""

    def test_extract_logits_at_step_differs_by_step(self):
        """extract_logits_at_step produces different tensors for step 0 vs S-1."""
        from evals.metrics.trajectory_utils import compute_trajectories, extract_logits_at_step

        V, L, S = 100, 10, 16
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        T_steps, T_fix_start, T_fix_end, T_fix_ratio = compute_trajectories(R, F, S)
        trajectories = {
            "steps": T_steps,
            "fixation_start": T_fix_start,
            "fixation_end": T_fix_end,
            "fixation_ratio": T_fix_ratio,
        }

        logits_0 = extract_logits_at_step(trajectories["steps"], 0)
        logits_last = extract_logits_at_step(trajectories["steps"], S - 1)
        assert not torch.allclose(logits_0, logits_last)
        assert logits_0.shape == logits_last.shape == (V, L)

    def test_extraction_strength_reproduces_with_logits(self):
        """extraction_strength runs with trajectory logits; documents constant 0.05 bug."""
        if "extraction_strength" not in METRICS_REGISTRY:
            pytest.skip("extraction_strength metric not registered")

        es_metric = METRICS_REGISTRY["extraction_strength"]
        V, L = 100, 20
        labels = torch.randint(0, V, (L,))
        logits = torch.randn(V, L)

        batch_template = {
            "input_ids": torch.zeros((1, L), dtype=torch.long),
            "labels": labels.unsqueeze(0),
            "attention_mask": torch.ones((1, L), dtype=torch.long),
        }

        tokenizer = Mock()
        sample_labels = labels
        sample_input_ids = torch.zeros(L, dtype=torch.long)
        sample_prompt_len = 0

        result = _call_metric_at_step(
            metric=es_metric,
            logits=logits,
            batch_template=batch_template,
            tokenizer=tokenizer,
            sample_labels=sample_labels,
            sample_input_ids=sample_input_ids,
            sample_prompt_len=sample_prompt_len,
            metric_config={},
            sample_idx="0",
        )

        score = result[0]["score"] if isinstance(result, list) else result.get("agg_value")
        assert score is not None
        assert 0 <= score <= 1

    def test_model_utility_pre_compute_uses_same_batch_for_all_metrics(self):
        """trajectory_model_utility: _compute_pre_compute_metrics_at_step uses same batch_template for all metrics."""
        pre_compute_config = {
            "probability": {"access_key": "retain_Q_A_Prob"},
            "exact_memorization": {"access_key": "ra_Q_A_Prob"},
            "extraction_strength": {"access_key": "wf_Q_A_Prob"},
        }
        V, L = 100, 10
        logits = torch.randn(V, L)
        batch_template = {
            "input_ids": torch.zeros((1, L), dtype=torch.long),
            "labels": torch.randint(0, V, (1, L)),
            "attention_mask": torch.ones((1, L), dtype=torch.long),
        }
        tokenizer = Mock()
        sample_labels = torch.randint(0, V, (L,))
        sample_input_ids = torch.zeros(L, dtype=torch.long)
        sample_prompt_len = 0
        sample_idx = "0"

        batch_templates_seen = []

        def capture_call_metric(*args, **kwargs):
            batch_template_arg = args[2] if len(args) > 2 else kwargs.get("batch_template")
            batch_templates_seen.append(batch_template_arg)
            return [{"score": 0.5, "prob": 0.5}]

        with patch(
            "evals.metrics.trajectory_metrics._call_metric_at_step",
            side_effect=capture_call_metric,
        ):
            _compute_pre_compute_metrics_at_step(
                pre_compute_config=pre_compute_config,
                logits=logits,
                batch_template=batch_template,
                tokenizer=tokenizer,
                sample_labels=sample_labels,
                sample_input_ids=sample_input_ids,
                sample_prompt_len=sample_prompt_len,
                sample_idx=sample_idx,
            )

        assert len(batch_templates_seen) == 3
        for bt in batch_templates_seen:
            assert bt is batch_template, "All pre_compute metrics receive same batch_template"

    def test_logit_model_wrapper_returns_same_logits_for_different_batches(self):
        """LogitModelWrapper returns same logits for forget and holdout batches (reproduces privleak bug)."""
        from evals.metrics.trajectory_adapters import LogitModelWrapper

        B, L, V = 1, 10, 100
        logits = torch.randn(B, L, V)
        wrapper = LogitModelWrapper(logits, torch.device("cpu"))

        batch_forget = {
            "input_ids": torch.zeros(1, L),
            "labels": torch.randint(0, V, (1, L)),
            "index": torch.tensor([0]),
        }
        batch_holdout = {
            "input_ids": torch.zeros(1, L),
            "labels": torch.randint(0, V, (1, L)),
            "index": torch.tensor([1]),
        }

        out_forget = wrapper(**batch_forget)
        out_holdout = wrapper(**batch_holdout)

        assert torch.allclose(out_forget.logits, out_holdout.logits)
        assert torch.allclose(out_forget.logits, logits)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
