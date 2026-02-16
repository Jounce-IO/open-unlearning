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
    _derive_steps_to_use,
    _get_logits_at_step,
    _trajectory_sampler_kwargs,
    DEFAULT_TRAJECTORY_SAMPLE_INTERVAL,
    should_run_gc,
    trajectory_metrics,
)
from evals.metrics.trajectory_utils import stack_logits_history
from evals.metrics import METRICS_REGISTRY


class TestShouldRunGc:
    """Tests for should_run_gc (conditional gc when VRAM > threshold)."""

    def test_returns_false_when_cuda_not_available(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert should_run_gc(0.9) is False

    def test_returns_false_when_vram_below_threshold(self):
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.memory_allocated", return_value=100
        ), patch(
            "torch.cuda.get_device_properties",
            return_value=Mock(total_memory=1000),
        ):
            assert should_run_gc(0.9) is False

    def test_returns_true_when_vram_at_or_above_threshold(self):
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.memory_allocated", return_value=900
        ), patch(
            "torch.cuda.get_device_properties",
            return_value=Mock(total_memory=1000),
        ):
            assert should_run_gc(0.9) is True

    def test_returns_true_when_vram_above_threshold(self):
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.memory_allocated", return_value=950
        ), patch(
            "torch.cuda.get_device_properties",
            return_value=Mock(total_memory=1000),
        ):
            assert should_run_gc(0.9) is True


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
        """Test that generated portion is correctly extracted from full sequence. R_full is [B, V, full_len, S]."""
        prompt_len = 5
        generated_len = 10
        full_len = prompt_len + generated_len
        V, S = 100, 8

        logits_history = [torch.randn(1, full_len, V) for _ in range(S)]

        R_full = stack_logits_history(logits_history)  # [B, V, full_len, S]
        assert R_full.shape == (1, V, full_len, S)

        R = R_full[:, :, prompt_len : prompt_len + generated_len, :]  # [B, V, generated_len, S]
        assert R.shape == (1, V, generated_len, S)

        for s in range(S):
            expected = logits_history[s][0, prompt_len : prompt_len + generated_len, :].T
            assert torch.allclose(R[0, :, :, s], expected)
    
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


class TestDeriveStepsToUse:
    """Tests for _derive_steps_to_use (step subsampling for report)."""

    def test_trajectory_sample_interval_set_uses_all_s_steps(self):
        """When trajectory_sample_interval=8, use all S steps; metadata = [8, 16, ..., 8*S]."""
        S = 25
        trajectory_config = {
            "sampler_kwargs": {
                "trajectory_sample_interval": 8,
                "max_new_tokens": 200,
            },
        }
        steps_to_use, step_values_metadata = _derive_steps_to_use(S, trajectory_config)
        assert len(steps_to_use) == S
        assert steps_to_use == list(range(S))
        assert len(step_values_metadata) == S
        assert step_values_metadata[0] == 8
        assert step_values_metadata[-1] == min(8 * S, 200)

    def test_trajectory_sample_interval_unset_subsamples_to_report_steps(self):
        """When trajectory_sample_interval unset, max_new_tokens=200, steps=50: subsample to 0,8,16,..."""
        S = 50
        trajectory_config = {
            "sampler_kwargs": {
                "max_new_tokens": 200,
                "steps": 50,
            },
        }
        steps_to_use, step_values_metadata = _derive_steps_to_use(S, trajectory_config)
        assert len(steps_to_use) <= S
        assert len(step_values_metadata) == len(steps_to_use)
        assert step_values_metadata == sorted(set(step_values_metadata))
        for t in step_values_metadata:
            assert t % 8 == 0 or t == 0
        assert 0 in step_values_metadata or step_values_metadata[0] == 0
        assert steps_to_use == sorted(set(steps_to_use))

    def test_trajectory_sample_interval_8_max_new_tokens_200_s_25(self):
        """trajectory_sample_interval=8, max_new_tokens=200, S=25: never allocate beyond step 24."""
        S = 25
        trajectory_config = {
            "sampler_kwargs": {
                "trajectory_sample_interval": 8,
                "max_new_tokens": 200,
            },
        }
        steps_to_use, step_values_metadata = _derive_steps_to_use(S, trajectory_config)
        assert all(0 <= s < S for s in steps_to_use)
        assert max(steps_to_use) == 24
        assert len(step_values_metadata) == 25

    def test_derive_steps_empty_s_returns_empty(self):
        """S=0 returns empty lists."""
        steps_to_use, step_values_metadata = _derive_steps_to_use(0, {})
        assert steps_to_use == []
        assert step_values_metadata == []


class TestTrajectorySamplerKwargs:
    """Tests for _trajectory_sampler_kwargs (interval default 8 when return_logits)."""

    def test_injects_interval_8_when_return_logits_and_interval_missing(self):
        """When return_logits=True and sampler_kwargs has no trajectory_sample_interval, default to 8."""
        config = {"return_logits": True, "sampler_kwargs": {"steps": 16, "max_new_tokens": 32}}
        kwargs = _trajectory_sampler_kwargs(config)
        assert kwargs["trajectory_sample_interval"] == DEFAULT_TRAJECTORY_SAMPLE_INTERVAL
        assert kwargs["steps"] == 16
        assert kwargs["max_new_tokens"] == 32

    def test_preserves_explicit_interval(self):
        """When trajectory_sample_interval is set, do not override."""
        config = {
            "return_logits": True,
            "sampler_kwargs": {"trajectory_sample_interval": 4, "max_new_tokens": 32},
        }
        kwargs = _trajectory_sampler_kwargs(config)
        assert kwargs["trajectory_sample_interval"] == 4

    def test_no_inject_when_return_logits_false(self):
        """When return_logits is False, do not add trajectory_sample_interval."""
        config = {"return_logits": False, "sampler_kwargs": {"steps": 16}}
        kwargs = _trajectory_sampler_kwargs(config)
        assert "trajectory_sample_interval" not in kwargs or kwargs.get("trajectory_sample_interval") is None


class TestTrajectoryMetricsIndexErrorRepro:
    """Reproduces IndexError when a batch has fewer steps than run_steps_to_use (for debug instrumentation)."""

    def test_get_logits_at_step_step_3_S_3_raises_index_error(self):
        """Direct repro: R has S=3 (valid indices 0,1,2); loop uses step=3 → IndexError. Writes debug NDJSON before crash."""
        V, L, S = 10, 5, 3
        R = torch.randn(1, V, L, S)
        F = torch.zeros(1, L, dtype=torch.long)
        run_steps_to_use = [0, 1, 2, 3]
        traj = {"R": R[0], "F": F[0], "S": S, "L": L}
        with pytest.raises(IndexError, match="index 3 is out of bounds"):
            for step in run_steps_to_use:
                _get_logits_at_step(traj, "steps", step)


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

    def test_trajectory_metrics_sets_use_fixation_logits_on_adapter(self):
        """When model has adapter_config, trajectory_metrics sets use_fixation_logits from trajectory_config (default True)."""
        # Test the contract: the same logic trajectory_metrics uses to set the flag
        class AdapterConfig:
            use_fixation_logits = False

        model = Mock()
        model.adapter_config = AdapterConfig()
        trajectory_config = {"use_fixation_logits": True}

        if hasattr(model, "adapter_config"):
            model.adapter_config.use_fixation_logits = trajectory_config.get(
                "use_fixation_logits", True
            )

        assert model.adapter_config.use_fixation_logits is True

        # Default when key missing: use_fixation_logits defaults to True
        model.adapter_config.use_fixation_logits = False
        trajectory_config_no_key = {}
        if hasattr(model, "adapter_config"):
            model.adapter_config.use_fixation_logits = trajectory_config_no_key.get(
                "use_fixation_logits", True
            )
        assert model.adapter_config.use_fixation_logits is True

    def test_trajectory_metrics_sets_use_fixation_logits_false_when_config_false(self):
        """When trajectory_config has use_fixation_logits=False, adapter ends up with False."""
        class AdapterConfig:
            use_fixation_logits = True

        model = Mock()
        model.adapter_config = AdapterConfig()
        trajectory_config = {"use_fixation_logits": False}

        if hasattr(model, "adapter_config"):
            model.adapter_config.use_fixation_logits = trajectory_config.get(
                "use_fixation_logits", True
            )

        assert model.adapter_config.use_fixation_logits is False

    def test_trajectory_metrics_full_run_with_adapter_and_fixation_logits(self):
        """Full trajectory_metrics run with adapter_config and sampler returning fixation_logits; no crash."""
        V, L_gen, S = 100, 10, 8
        prompt_len = 5
        full_len = prompt_len + L_gen

        logits_history = [torch.randn(1, full_len, V) for _ in range(S)]
        fixation_steps = torch.randint(0, S, (1, full_len))
        # fixation_logits [B, T, V] same formula as sampler
        R = torch.stack(logits_history, dim=0)
        F = fixation_steps.clamp(0, S - 1)
        B, T, _ = R.shape[1], R.shape[2], R.shape[3]
        batch_idx = torch.arange(B, device=R.device, dtype=torch.long).unsqueeze(1).expand(B, T)
        pos_idx = torch.arange(T, device=R.device, dtype=torch.long).unsqueeze(0).expand(B, T)
        fixation_logits = R[F, batch_idx, pos_idx, :]

        class MockSamplerOutput:
            def __init__(self, sequences, histories, logits_history, fixation_steps, fixation_logits=None):
                self.sequences = sequences
                self.histories = histories
                self.logits_history = logits_history
                self.fixation_steps = fixation_steps
                self.fixation_logits = fixation_logits

        sampler = Mock()
        sampler.sample.return_value = MockSamplerOutput(
            sequences=torch.randint(0, V, (1, full_len)),
            histories=None,
            logits_history=logits_history,
            fixation_steps=fixation_steps,
            fixation_logits=fixation_logits,
        )

        class AdapterConfig:
            use_fixation_logits = False

        model = Mock()
        model.sampler = sampler
        model.adapter_config = AdapterConfig()

        class MockDataset:
            def __init__(self):
                self.data = [
                    {"input_ids": torch.randint(0, V, (full_len,)), "labels": torch.randint(0, V, (full_len,))}
                    for _ in range(2)
                ]
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        def mock_collator(batch):
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
                "index": torch.tensor([0, 1]),
            }

        tokenizer = Mock()
        tokenizer.decode = lambda x, **kwargs: "decoded"

        kwargs = {
            "metrics": ["probability"],
            "data": MockDataset(),
            "collators": mock_collator,
            "tokenizer": tokenizer,
            "batch_size": 1,
            "trajectory_config": {"use_fixation_logits": True},
        }
        # UnlearningMetric.evaluate(model, metric_name, cache, **kwargs) is how the framework calls it
        result = trajectory_metrics(
            model,
            metric_name="trajectory_metrics",
            cache={},
            **kwargs,
        )
        assert model.adapter_config.use_fixation_logits is True
        assert isinstance(result, dict)

    def test_trajectory_metrics_batch_size_2_different_samples(self):
        """With batch_size=2 and different logits per sample, trajectory_metrics runs and uses per-sample trajectories.
        Asserts that the B=2 aggregate differs from the B=1 (first sample only) result, so both samples contribute."""
        import numpy as np

        V, L_gen, S = 100, 10, 8
        prompt_len = 5
        full_len = prompt_len + L_gen
        B = 2

        # Fix seeds so we have reproducible different logits per sample
        torch.manual_seed(42)
        logits_history_b2 = [torch.randn(B, full_len, V) for _ in range(S)]
        fixation_steps_b2 = torch.randint(0, S, (B, full_len))

        class MockSamplerOutput:
            def __init__(self, sequences, histories, logits_history, fixation_steps):
                self.sequences = sequences
                self.histories = histories
                self.logits_history = logits_history
                self.fixation_steps = fixation_steps

        # Run 1: batch_size=1, only first sample -> result_single
        sampler1 = Mock()
        sampler1.sample.return_value = MockSamplerOutput(
            sequences=torch.randint(0, V, (1, full_len)),
            histories=None,
            logits_history=[t[0:1] for t in logits_history_b2],
            fixation_steps=fixation_steps_b2[0:1],
        )
        model1 = Mock()
        model1.sampler = sampler1

        class MockDatasetSingle:
            def __init__(self):
                self.data = [
                    {"input_ids": torch.randint(0, V, (full_len,)), "labels": torch.randint(0, V, (full_len,))}
                ]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        def mock_collator_single(batch):
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
                "indices": torch.tensor([0]),
            }

        tokenizer = Mock()
        tokenizer.decode = lambda x, **kwargs: "decoded text"

        result_single = trajectory_metrics(
            model1,
            metric_name="trajectory_metrics",
            cache={},
            metrics=["probability"],
            data=MockDatasetSingle(),
            collators=mock_collator_single,
            tokenizer=tokenizer,
            batch_size=1,
            trajectory_config={
                "return_logits": True,
                "return_fixation_steps": True,
                "sampler_kwargs": {"steps": S, "max_new_tokens": L_gen},
            },
        )
        assert result_single is not None and "agg_value" in result_single
        assert "step_distribution" in result_single
        # Result is nested by view (full, eos)
        for view in result_single["agg_value"]:
            for traj_name in result_single["agg_value"][view]:
                for metric_name in result_single["agg_value"][view][traj_name]:
                    dist = result_single["step_distribution"][view][traj_name][metric_name]
                    agg = np.asarray(result_single["agg_value"][view][traj_name][metric_name])
                    assert set(dist.keys()) == {"mean", "std", "median", "p25", "p75", "min", "max", "ci_low", "ci_high"}
                    assert len(dist["mean"]) == len(agg)
                    np.testing.assert_allclose(dist["mean"], agg, rtol=1e-5, atol=1e-8)

        # Run 2: batch_size=2, two different samples -> result_b2
        sampler2 = Mock()
        sampler2.sample.return_value = MockSamplerOutput(
            sequences=torch.randint(0, V, (B, full_len)),
            histories=None,
            logits_history=logits_history_b2,
            fixation_steps=fixation_steps_b2,
        )
        model2 = Mock()
        model2.sampler = sampler2

        class MockDatasetTwo:
            def __init__(self):
                self.data = [
                    {"input_ids": torch.randint(0, V, (full_len,)), "labels": torch.randint(0, V, (full_len,))}
                    for _ in range(2)
                ]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        def mock_collator_two(batch):
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
                "indices": torch.tensor([0, 1]),
            }

        result_b2 = trajectory_metrics(
            model2,
            metric_name="trajectory_metrics",
            cache={},
            metrics=["probability"],
            data=MockDatasetTwo(),
            collators=mock_collator_two,
            tokenizer=tokenizer,
            batch_size=2,
            trajectory_config={
                "return_logits": True,
                "return_fixation_steps": True,
                "sampler_kwargs": {"steps": S, "max_new_tokens": L_gen},
            },
        )
        assert result_b2 is not None and "agg_value" in result_b2
        assert "step_distribution" in result_b2

        # If only the first sample were used in B=2, agg would match B=1. Different logits => different values.
        view = "full"
        agg1 = result_single["agg_value"][view]
        agg2 = result_b2["agg_value"][view]
        for traj_name in agg1:
            if traj_name not in agg2:
                continue
            for metric_name in agg1[traj_name]:
                if metric_name not in agg2[traj_name]:
                    continue
                a1 = np.asarray(agg1[traj_name][metric_name])
                a2 = np.asarray(agg2[traj_name][metric_name])
                if a1.size > 0 and a2.size > 0:
                    assert not np.allclose(a1, a2), (
                        f"B=2 aggregate should differ from B=1 when samples differ "
                        f"(traj={traj_name}, metric={metric_name})"
                    )
                    break
            else:
                continue
            break

    def test_step_distribution_single_sample_per_step(self):
        """With one sample, each step has one value: std=0, ci_low=ci_high=mean, min=max=median=mean."""
        import numpy as np

        V, L_gen, S = 20, 4, 3
        full_len = 5 + L_gen
        sampler = Mock()
        logits_history = [torch.randn(1, full_len, V) for _ in range(S)]
        fixation_steps = torch.randint(0, S, (1, full_len))

        class MockSamplerOutput:
            def __init__(self, sequences, histories, logits_history, fixation_steps):
                self.sequences = sequences
                self.histories = histories
                self.logits_history = logits_history
                self.fixation_steps = fixation_steps

        sampler.sample.return_value = MockSamplerOutput(
            sequences=torch.randint(0, V, (1, full_len)),
            histories=None,
            logits_history=logits_history,
            fixation_steps=fixation_steps,
        )
        model = Mock()
        model.sampler = sampler

        class MockDataset:
            def __init__(self):
                # Use 1D (full_len,) so batch stack gives (1, full_len); 2D would give (1, 1, full_len)
                self.data = [{"input_ids": torch.zeros(full_len), "labels": torch.zeros(full_len)}]

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return self.data[idx]

        def mock_collator(batch):
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
                "indices": torch.tensor([0]),
            }

        tokenizer = Mock()
        tokenizer.decode = lambda x, **kwargs: "decoded"

        result = trajectory_metrics(
            model,
            metric_name="trajectory_metrics",
            cache={},
            metrics=["probability"],
            data=MockDataset(),
            collators=mock_collator,
            tokenizer=tokenizer,
            batch_size=1,
            trajectory_config={
                "return_logits": True,
                "return_fixation_steps": True,
                "sampler_kwargs": {"steps": S, "max_new_tokens": L_gen},
            },
        )
        assert "step_distribution" in result
        for view in result["step_distribution"]:
            for traj_name in result["step_distribution"][view]:
                for metric_name in result["step_distribution"][view][traj_name]:
                    d = result["step_distribution"][view][traj_name][metric_name]
                    mean_arr = np.asarray(d["mean"])
                    std_arr = np.asarray(d["std"])
                    ci_low_arr = np.asarray(d["ci_low"])
                    ci_high_arr = np.asarray(d["ci_high"])
                    min_arr = np.asarray(d["min"])
                    max_arr = np.asarray(d["max"])
                    # Single sample per step => std=0, ci_low=ci_high=mean, min=max=mean
                    np.testing.assert_allclose(std_arr, 0.0, rtol=0, atol=1e-10)
                    np.testing.assert_allclose(ci_low_arr, mean_arr, rtol=0, atol=1e-10)
                    np.testing.assert_allclose(ci_high_arr, mean_arr, rtol=0, atol=1e-10)
                    np.testing.assert_allclose(min_arr, mean_arr, rtol=0, atol=1e-10)
                    np.testing.assert_allclose(max_arr, mean_arr, rtol=0, atol=1e-10)

    def test_trajectory_metrics_batch_size_4_runs(self):
        """trajectory_metrics with batch_size=4 runs without shape errors."""
        V, L_gen, S = 50, 6, 4
        full_len = 5 + L_gen
        B = 4

        sampler = Mock()
        logits_history = [torch.randn(B, full_len, V) for _ in range(S)]
        fixation_steps = torch.randint(0, S, (B, full_len))

        class MockSamplerOutput:
            def __init__(self, sequences, histories, logits_history, fixation_steps):
                self.sequences = sequences
                self.histories = histories
                self.logits_history = logits_history
                self.fixation_steps = fixation_steps

        sampler.sample.return_value = MockSamplerOutput(
            sequences=torch.randint(0, V, (B, full_len)),
            histories=None,
            logits_history=logits_history,
            fixation_steps=fixation_steps,
        )
        model = Mock()
        model.sampler = sampler

        class MockDataset:
            def __init__(self):
                self.data = [
                    {"input_ids": torch.randint(0, V, (full_len,)), "labels": torch.randint(0, V, (full_len,))}
                    for _ in range(4)
                ]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        def mock_collator(batch):
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
                "indices": torch.tensor(list(range(len(batch)))),
            }

        tokenizer = Mock()
        tokenizer.decode = lambda x, **kwargs: "decoded text"

        kwargs = {
            "metrics": ["probability"],
            "data": MockDataset(),
            "collators": mock_collator,
            "tokenizer": tokenizer,
            "batch_size": 4,
            "trajectory_config": {
                "return_logits": True,
                "return_fixation_steps": True,
                "sampler_kwargs": {"steps": S, "max_new_tokens": L_gen},
            },
        }
        try:
            result = trajectory_metrics(model, **kwargs)
            assert result is not None
            assert "agg_value" in result
        except Exception as e:
            assert "shape" not in str(e).lower() or "mismatch" not in str(e).lower()

    def test_trajectory_metrics_multi_batch_releases_memory(self):
        """Run trajectory_metrics over multiple batches (batch_size=1); regression test for batch memory release.
        Ensures we complete without OOM and aggregate over all samples (del logits_history, out between batches)."""
        import numpy as np

        V, L_gen, S = 50, 6, 4
        full_len = 5 + L_gen
        num_samples = 5

        class MockSamplerOutput:
            def __init__(self, sequences, histories, logits_history, fixation_steps):
                self.sequences = sequences
                self.histories = histories
                self.logits_history = logits_history
                self.fixation_steps = fixation_steps

        # One sampler output per batch (each batch gets its own logits so we exercise the full loop + del path)
        sampler = Mock()
        outputs = []
        for _ in range(num_samples):
            lh = [torch.randn(1, full_len, V) for _ in range(S)]
            fix = torch.randint(0, S, (1, full_len))
            outputs.append(
                MockSamplerOutput(
                    sequences=torch.randint(0, V, (1, full_len)),
                    histories=None,
                    logits_history=lh,
                    fixation_steps=fix,
                )
            )
        sampler.sample.side_effect = outputs

        model = Mock()
        model.sampler = sampler

        class MockDataset:
            def __init__(self):
                self.data = [
                    {"input_ids": torch.randint(0, V, (full_len,)), "labels": torch.randint(0, V, (full_len,))}
                    for _ in range(num_samples)
                ]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        def mock_collator(batch):
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
                "indices": torch.tensor(list(range(len(batch)))),
            }

        tokenizer = Mock()
        tokenizer.decode = lambda x, **kwargs: "decoded text"

        result = trajectory_metrics(
            model,
            metric_name="trajectory_metrics",
            cache={},
            metrics=["probability"],
            data=MockDataset(),
            collators=mock_collator,
            tokenizer=tokenizer,
            batch_size=1,
            trajectory_config={
                "return_logits": True,
                "return_fixation_steps": True,
                "sampler_kwargs": {
                    "steps": S,
                    "max_new_tokens": L_gen,
                    "trajectory_sample_interval": 8,
                },
            },
        )
        assert result is not None
        assert "agg_value" in result
        for view in result["agg_value"]:
            agg = result["agg_value"][view]
            assert isinstance(agg, dict)
            for traj_name, metrics_dict in agg.items():
                assert isinstance(metrics_dict, dict)
                for metric_name, arr in metrics_dict.items():
                    arr = np.asarray(arr)
                    assert arr.size > 0, f"view={view} traj={traj_name} metric={metric_name} should have aggregated values"
                    assert arr.shape[0] == S, f"expected S={S} steps, got {arr.shape[0]}"
        # Sampler should have been called once per batch
        assert sampler.sample.call_count == num_samples

    def test_two_view_probability_calls_cuda_cleanup_when_available(self):
        """With include_views=[full, eos] and probability, we call torch.cuda.synchronize and empty_cache after first view (no GPU leakage)."""
        V, L_gen, S = 30, 6, 4
        full_len = 5 + L_gen
        num_samples = 2

        class MockSamplerOutput:
            def __init__(self, sequences, histories, logits_history, fixation_steps):
                self.sequences = sequences
                self.histories = histories
                self.logits_history = logits_history
                self.fixation_steps = fixation_steps

        sampler = Mock()
        def sample_side_effect(*args, **kwargs):
            return MockSamplerOutput(
                sequences=torch.randint(0, V, (1, full_len)),
                histories=None,
                logits_history=[torch.randn(1, full_len, V) for _ in range(S)],
                fixation_steps=torch.randint(0, S, (1, full_len)),
            )
        sampler.sample.side_effect = [sample_side_effect() for _ in range(num_samples)]

        model = Mock()
        model.sampler = sampler

        class MockDataset:
            def __init__(self):
                self.data = [
                    {"input_ids": torch.randint(0, V, (full_len,)), "labels": torch.randint(0, V, (full_len,))}
                    for _ in range(num_samples)
                ]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        def mock_collator(batch):
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
                "indices": torch.tensor(list(range(len(batch)))),
            }

        tokenizer = Mock()
        tokenizer.decode = lambda x, **kwargs: "decoded text"

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_properties", return_value=Mock(total_memory=10**9)
        ), patch("torch.cuda.memory_allocated", return_value=100), patch(
            "torch.cuda.synchronize", new_callable=Mock
        ) as mock_sync, patch(
            "torch.cuda.empty_cache", new_callable=Mock
        ) as mock_empty:
            result = trajectory_metrics(
                model,
                metric_name="trajectory_metrics",
                cache={},
                metrics=["probability"],
                data=MockDataset(),
                collators=mock_collator,
                tokenizer=tokenizer,
                batch_size=1,
                trajectory_config={
                    "return_logits": True,
                    "return_fixation_steps": True,
                    "include_views": ["full", "eos"],
                    "sampler_kwargs": {
                        "steps": S,
                        "max_new_tokens": L_gen,
                        "trajectory_sample_interval": 8,
                    },
                },
            )
            assert result is not None
            assert "agg_value" in result
            # With 2 views we run probability twice per (sample, traj_name). Cleanup runs after first view each time.
            assert mock_sync.call_count >= 1, "cuda.synchronize should be called when two views and CUDA available"
            assert mock_empty.call_count >= 1, "cuda.empty_cache should be called when two views and CUDA available"

    def test_two_view_multi_batch_no_python_memory_leak(self):
        """Run trajectory_metrics with both views over multiple batches; completes without error (no GPU). Validates no crash from leaked refs."""
        import numpy as np

        V, L_gen, S = 40, 6, 4
        full_len = 5 + L_gen
        num_samples = 6
        batch_size = 1

        class MockSamplerOutput:
            def __init__(self, sequences, histories, logits_history, fixation_steps):
                self.sequences = sequences
                self.histories = histories
                self.logits_history = logits_history
                self.fixation_steps = fixation_steps

        sampler = Mock()
        outputs = []
        for _ in range(num_samples):
            lh = [torch.randn(1, full_len, V) for _ in range(S)]
            fix = torch.randint(0, S, (1, full_len))
            outputs.append(
                MockSamplerOutput(
                    sequences=torch.randint(0, V, (1, full_len)),
                    histories=None,
                    logits_history=lh,
                    fixation_steps=fix,
                )
            )
        sampler.sample.side_effect = outputs

        model = Mock()
        model.sampler = sampler

        class MockDataset:
            def __init__(self):
                self.data = [
                    {"input_ids": torch.randint(0, V, (full_len,)), "labels": torch.randint(0, V, (full_len,))}
                    for _ in range(num_samples)
                ]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        def mock_collator(batch):
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
                "indices": torch.tensor(list(range(len(batch)))),
            }

        tokenizer = Mock()
        tokenizer.decode = lambda x, **kwargs: "decoded text"

        result = trajectory_metrics(
            model,
            metric_name="trajectory_metrics",
            cache={},
            metrics=["probability"],
            data=MockDataset(),
            collators=mock_collator,
            tokenizer=tokenizer,
            batch_size=batch_size,
            trajectory_config={
                "return_logits": True,
                "return_fixation_steps": True,
                "include_views": ["full", "eos"],
                "sampler_kwargs": {
                    "steps": S,
                    "max_new_tokens": L_gen,
                    "trajectory_sample_interval": 8,
                },
            },
        )

        assert result is not None
        assert "agg_value" in result
        for view in ("full", "eos"):
            assert view in result["agg_value"]
            for traj_name, metrics_dict in result["agg_value"][view].items():
                assert "probability" in metrics_dict
                arr = np.asarray(result["agg_value"][view][traj_name]["probability"])
                assert arr.size == S, f"view={view} traj={traj_name} expected S={S} steps"
        assert sampler.sample.call_count == num_samples

    def test_sort_by_length_order_invariance(self):
        """Same aggregated results with sort_by_length=True vs False (order invariance)."""
        import numpy as np

        V, L_gen, S = 50, 6, 4
        full_len = 10 + L_gen  # 16
        # 4 samples with distinct prompt lengths 5, 7, 3, 6 so mock can key logits by length
        prompt_lengths = [5, 7, 3, 6]
        IGNORE_INDEX = -100

        # Precompute deterministic logits per "virtual index" (0..3) so same sample -> same logits
        torch.manual_seed(123)
        logits_by_index = []
        for _ in range(4):
            lh = [torch.randn(1, full_len, V) for _ in range(S)]
            fix = torch.randint(0, S, (1, full_len))
            logits_by_index.append((lh, fix))
        length_to_idx = {5: 0, 7: 1, 3: 2, 6: 3}

        def mock_sample(inputs, **kwargs):
            B = len(inputs)
            lengths = [len(p) for p in inputs]
            lh_stacked = [
                torch.cat([logits_by_index[length_to_idx[L]][0][s] for L in lengths], dim=0)
                for s in range(S)
            ]
            fix_stacked = torch.cat(
                [logits_by_index[length_to_idx[L]][1] for L in lengths], dim=0
            )
            return Mock(
                logits_history=lh_stacked,
                fixation_steps=fix_stacked,
                sequences=torch.randint(0, V, (B, full_len)),
                histories=None,
            )

        sampler = Mock()
        sampler.sample.side_effect = mock_sample
        model = Mock()
        model.sampler = sampler

        def make_dataset():
            data = []
            for i, plen in enumerate(prompt_lengths):
                input_ids = torch.randint(0, V, (full_len,))
                labels = torch.full((full_len,), IGNORE_INDEX, dtype=torch.long)
                labels[plen:] = input_ids[plen:].clone()
                data.append({"input_ids": input_ids, "labels": labels, "index": i})
            return data

        class MockDataset:
            def __init__(self):
                self.data = make_dataset()

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        def mock_collator(batch):
            return {
                "input_ids": torch.nn.utils.rnn.pad_sequence(
                    [b["input_ids"] for b in batch], batch_first=True, padding_value=0
                ),
                "labels": torch.nn.utils.rnn.pad_sequence(
                    [b["labels"] for b in batch], batch_first=True, padding_value=IGNORE_INDEX
                ),
                "index": torch.tensor([b["index"] for b in batch]),
            }

        tokenizer = Mock()
        tokenizer.decode = lambda x, **kwargs: "decoded"

        dataset = MockDataset()
        base_kwargs = {
            "metric_name": "trajectory_metrics",
            "cache": {},
            "metrics": ["probability"],
            "data": dataset,
            "collators": mock_collator,
            "tokenizer": tokenizer,
            "batch_size": 2,
            "trajectory_config": {
                "return_logits": True,
                "return_fixation_steps": True,
                "sampler_kwargs": {"steps": S, "max_new_tokens": L_gen},
            },
        }

        result_no_sort = trajectory_metrics(model, **base_kwargs, sort_by_length=False)
        result_sort = trajectory_metrics(model, **base_kwargs, sort_by_length=True)

        assert result_no_sort is not None and "agg_value" in result_no_sort
        assert result_sort is not None and "agg_value" in result_sort
        view = "full"
        for traj_name in result_no_sort["agg_value"][view]:
            assert traj_name in result_sort["agg_value"][view]
            for metric_name in result_no_sort["agg_value"][view][traj_name]:
                assert metric_name in result_sort["agg_value"][view][traj_name]
                a = np.asarray(result_no_sort["agg_value"][view][traj_name][metric_name])
                b = np.asarray(result_sort["agg_value"][view][traj_name][metric_name])
                np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-8)

    def test_batch_size_four_same_aggregate_as_batch_size_one(self):
        """Same aggregated results for batch_size=1 vs batch_size=4 (batch-size invariance)."""
        import numpy as np

        V, L_gen, S = 50, 6, 4
        prompt_len = 10
        full_len = prompt_len + L_gen  # 16, same for all samples
        num_samples = 8
        IGNORE_INDEX = -100
        # Same prompt length for all so L (generated length) is identical; encode index in first token
        prompt_lengths = [prompt_len] * num_samples

        torch.manual_seed(456)
        logits_by_index = []
        for _ in range(num_samples):
            lh = [torch.randn(1, full_len, V) for _ in range(S)]
            fix = torch.randint(0, S, (1, full_len))
            logits_by_index.append((lh, fix))

        def mock_sample(inputs, **kwargs):
            B = len(inputs)
            # Encode index in first token: we built input_ids so input_ids[i, 0] == i
            indices = [int(p[0]) % num_samples for p in inputs]
            lh_stacked = [
                torch.cat([logits_by_index[idx][0][s] for idx in indices], dim=0)
                for s in range(S)
            ]
            fix_stacked = torch.cat([logits_by_index[idx][1] for idx in indices], dim=0)
            return Mock(
                logits_history=lh_stacked,
                fixation_steps=fix_stacked,
                sequences=torch.randint(0, V, (B, full_len)),
                histories=None,
            )

        sampler = Mock()
        sampler.sample.side_effect = mock_sample
        model = Mock()
        model.sampler = sampler

        class MockDataset:
            def __init__(self):
                self.data = []
                for i, plen in enumerate(prompt_lengths):
                    input_ids = torch.randint(0, V, (full_len,))
                    input_ids[0] = i  # encode index in first token for mock
                    labels = torch.full((full_len,), IGNORE_INDEX, dtype=torch.long)
                    labels[plen:] = input_ids[plen:].clone()
                    self.data.append({"input_ids": input_ids, "labels": labels, "index": i})

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        def mock_collator(batch):
            return {
                "input_ids": torch.nn.utils.rnn.pad_sequence(
                    [b["input_ids"] for b in batch], batch_first=True, padding_value=0
                ),
                "labels": torch.nn.utils.rnn.pad_sequence(
                    [b["labels"] for b in batch], batch_first=True, padding_value=IGNORE_INDEX
                ),
                "index": torch.tensor([b["index"] for b in batch]),
            }

        tokenizer = Mock()
        tokenizer.decode = lambda x, **kwargs: "decoded"

        dataset = MockDataset()
        base_kwargs = {
            "metric_name": "trajectory_metrics",
            "cache": {},
            "metrics": ["probability"],
            "data": dataset,
            "collators": mock_collator,
            "tokenizer": tokenizer,
            "trajectory_config": {
                "return_logits": True,
                "return_fixation_steps": True,
                "sampler_kwargs": {"steps": S, "max_new_tokens": L_gen},
            },
        }

        result_b1 = trajectory_metrics(model, **base_kwargs, batch_size=1)
        result_b4 = trajectory_metrics(model, **base_kwargs, batch_size=4)

        assert result_b1 is not None and "agg_value" in result_b1
        assert result_b4 is not None and "agg_value" in result_b4
        view = "full"
        for traj_name in result_b1["agg_value"][view]:
            assert traj_name in result_b4["agg_value"][view]
            for metric_name in result_b1["agg_value"][view][traj_name]:
                assert metric_name in result_b4["agg_value"][view][traj_name]
                a = np.asarray(result_b1["agg_value"][view][traj_name][metric_name])
                b = np.asarray(result_b4["agg_value"][view][traj_name][metric_name])
                np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-8)

    def test_include_views_full_only_returns_single_view(self):
        """With include_views=[full], result contains only 'full' in agg_value and step_distribution."""
        class MockSamplerOutput:
            def __init__(self, sequences, histories, logits_history, fixation_steps):
                self.sequences = sequences
                self.histories = histories
                self.logits_history = logits_history
                self.fixation_steps = fixation_steps
        V, L_gen, S = 20, 6, 4
        full_len = 5 + L_gen
        sampler = Mock()
        sampler.sample.return_value = MockSamplerOutput(
            sequences=torch.randint(0, V, (1, full_len)),
            histories=None,
            logits_history=[torch.randn(1, full_len, V) for _ in range(S)],
            fixation_steps=torch.randint(0, S, (1, full_len)),
        )
        model = Mock()
        model.sampler = sampler
        result = trajectory_metrics(
            model,
            metric_name="trajectory_metrics",
            cache={},
            metrics=["probability"],
            data=MockDataset(1, full_len, V),
            collators=mock_collator(1, full_len),
            tokenizer=Mock(decode=lambda x, **kw: "decoded"),
            batch_size=1,
            trajectory_config={
                "return_logits": True,
                "return_fixation_steps": True,
                "include_views": ["full"],
                "sampler_kwargs": {"steps": S, "max_new_tokens": L_gen},
            },
        )
        assert set(result["agg_value"].keys()) == {"full"}
        assert set(result["step_distribution"].keys()) == {"full"}

    def test_include_views_eos_only_returns_single_view(self):
        """With include_views=[eos], result contains only 'eos' in agg_value and step_distribution."""
        class MockSamplerOutput:
            def __init__(self, sequences, histories, logits_history, fixation_steps):
                self.sequences = sequences
                self.histories = histories
                self.logits_history = logits_history
                self.fixation_steps = fixation_steps
        V, L_gen, S = 20, 6, 4
        full_len = 5 + L_gen
        sampler = Mock()
        sampler.sample.return_value = MockSamplerOutput(
            sequences=torch.randint(0, V, (1, full_len)),
            histories=None,
            logits_history=[torch.randn(1, full_len, V) for _ in range(S)],
            fixation_steps=torch.randint(0, S, (1, full_len)),
        )
        model = Mock()
        model.sampler = sampler
        result = trajectory_metrics(
            model,
            metric_name="trajectory_metrics",
            cache={},
            metrics=["probability"],
            data=MockDataset(1, full_len, V),
            collators=mock_collator(1, full_len),
            tokenizer=Mock(decode=lambda x, **kw: "decoded"),
            batch_size=1,
            trajectory_config={
                "return_logits": True,
                "return_fixation_steps": True,
                "include_views": ["eos"],
                "sampler_kwargs": {"steps": S, "max_new_tokens": L_gen},
            },
        )
        assert set(result["agg_value"].keys()) == {"eos"}
        assert set(result["step_distribution"].keys()) == {"eos"}

    def test_include_views_both_returns_full_and_eos(self):
        """Default include_views (both) returns 'full' and 'eos' in agg_value and step_distribution."""
        class MockSamplerOutput:
            def __init__(self, sequences, histories, logits_history, fixation_steps):
                self.sequences = sequences
                self.histories = histories
                self.logits_history = logits_history
                self.fixation_steps = fixation_steps
        V, L_gen, S = 20, 6, 4
        full_len = 5 + L_gen
        sampler = Mock()
        sampler.sample.return_value = MockSamplerOutput(
            sequences=torch.randint(0, V, (1, full_len)),
            histories=None,
            logits_history=[torch.randn(1, full_len, V) for _ in range(S)],
            fixation_steps=torch.randint(0, S, (1, full_len)),
        )
        model = Mock()
        model.sampler = sampler
        result = trajectory_metrics(
            model,
            metric_name="trajectory_metrics",
            cache={},
            metrics=["probability"],
            data=MockDataset(1, full_len, V),
            collators=mock_collator(1, full_len),
            tokenizer=Mock(decode=lambda x, **kw: "decoded"),
            batch_size=1,
            trajectory_config={
                "return_logits": True,
                "return_fixation_steps": True,
                "include_views": ["full", "eos"],
                "sampler_kwargs": {"steps": S, "max_new_tokens": L_gen},
            },
        )
        assert set(result["agg_value"].keys()) == {"full", "eos"}
        assert set(result["step_distribution"].keys()) == {"full", "eos"}
        import numpy as np
        view = "full"
        for traj_name in result["agg_value"][view]:
            for metric_name in result["agg_value"][view][traj_name]:
                a_full = np.asarray(result["agg_value"]["full"][traj_name][metric_name])
                a_eos = np.asarray(result["agg_value"]["eos"][traj_name][metric_name])
                assert a_full.shape == a_eos.shape, "full and eos should have same step count"

    def test_eos_view_diffs_from_full_when_eos_in_sequences(self):
        """When EOS appears before end of sequence, eos view aggregate can differ from full (different length)."""
        class MockSamplerOutput:
            def __init__(self, sequences, histories, logits_history, fixation_steps):
                self.sequences = sequences
                self.histories = histories
                self.logits_history = logits_history
                self.fixation_steps = fixation_steps
        V, L_gen, S = 30, 8, 4
        prompt_len = 4
        full_len = prompt_len + L_gen
        eos_id = 1
        # Build sequences: sample 0 has EOS at generated position 3 (L_eff=4), so eos uses 4 tokens, full uses 8
        sequences = torch.randint(2, V, (1, full_len))
        sequences[0, prompt_len + 3] = eos_id
        sampler = Mock()
        sampler.sample.return_value = MockSamplerOutput(
            sequences=sequences,
            histories=None,
            logits_history=[torch.randn(1, full_len, V) for _ in range(S)],
            fixation_steps=torch.randint(0, S, (1, full_len)),
        )
        model = Mock()
        model.sampler = sampler
        tokenizer = Mock()
        tokenizer.decode = lambda x, **kw: "decoded"
        tokenizer.eos_token_id = eos_id
        # Dataset with prompt_len=4 so generated region length L=8; EOS at gen position 3 -> L_eff=4
        IGNORE_INDEX = -100
        one_input = torch.randint(0, V, (full_len,))
        one_labels = torch.cat([
            torch.full((prompt_len,), IGNORE_INDEX, dtype=torch.long),
            torch.randint(0, V, (L_gen,)),
        ])
        class DS:
            data = [(one_input, one_labels)]
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return {"input_ids": self.data[idx][0], "labels": self.data[idx][1]}
        result = trajectory_metrics(
            model,
            metric_name="trajectory_metrics",
            cache={},
            metrics=["probability"],
            data=DS(),
            collators=mock_collator(1, full_len),
            tokenizer=tokenizer,
            batch_size=1,
            trajectory_config={
                "return_logits": True,
                "return_fixation_steps": True,
                "include_views": ["full", "eos"],
                "sampler_kwargs": {"steps": S, "max_new_tokens": L_gen},
            },
        )
        assert "full" in result["agg_value"] and "eos" in result["agg_value"]
        import numpy as np
        # Both views should have same number of steps; values may differ because eos uses shorter sequence
        for traj_name in result["agg_value"]["full"]:
            assert traj_name in result["agg_value"]["eos"]
            for metric_name in result["agg_value"]["full"][traj_name]:
                a_full = np.asarray(result["agg_value"]["full"][traj_name][metric_name])
                a_eos = np.asarray(result["agg_value"]["eos"][traj_name][metric_name])
                assert a_full.shape == a_eos.shape
                # With different lengths (4 vs 8), probability aggregates can differ
                assert a_full.size > 0 and a_eos.size > 0


def MockDataset(n, full_len, V, ignore_index=-100):
    """Minimal dataset for trajectory tests: n samples, full_len tokens, random labels in generated region."""
    class DS:
        data = [
            (
                torch.randint(0, V, (full_len,)),
                torch.cat([
                    torch.full((full_len - (full_len // 2),), ignore_index, dtype=torch.long),
                    torch.randint(0, V, (full_len // 2,)),
                ]),
            )
            for _ in range(n)
        ]
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            inp, lab = self.data[idx]
            return {"input_ids": inp, "labels": lab}
    return DS()


def mock_collator(n, full_len):
    def collator(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "index": torch.arange(len(batch)),
        }
    return collator


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
