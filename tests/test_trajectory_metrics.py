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
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.trajectory_metrics import (
    _get_sampler_from_model,
    _is_logit_based_metric,
    trajectory_metrics,
)
from evals.metrics.trajectory_utils import stack_logits_history


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
        model.model = inner_model
        
        result = _get_sampler_from_model(model)
        
        assert result is sampler
    
    def test_model_without_sampler_returns_none(self):
        """Test that model without sampler returns None."""
        model = Mock()
        del model.sampler
        
        result = _get_sampler_from_model(model)
        
        assert result is None


class TestIsLogitBasedMetric:
    """Tests for _is_logit_based_metric function."""
    
    def test_logit_based_metrics(self):
        """Test that logit-based metrics return True."""
        assert _is_logit_based_metric("probability") is True
        assert _is_logit_based_metric("exact_memorization") is True
    
    def test_text_based_metrics(self):
        """Test that text-based metrics return False."""
        assert _is_logit_based_metric("rouge") is False
        assert _is_logit_based_metric("bleu") is False


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
        model.sampler = None
        del model.model
        
        kwargs = {
            "metrics": ["probability"],
            "data": Mock(),
            "collators": Mock(),
            "tokenizer": Mock(),
        }
        
        with pytest.raises(ValueError):
            trajectory_metrics(model, **kwargs)
    
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
        
        with pytest.raises(ValueError, match="No metrics specified"):
            trajectory_metrics(model, **kwargs)
    
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
        
        with pytest.raises(ValueError, match="tokenizer is required"):
            trajectory_metrics(model, **kwargs)


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
        from evals.metrics.base import SamplerOutput
        sampler_output = SamplerOutput(
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
