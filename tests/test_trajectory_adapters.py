"""
Comprehensive unit tests for trajectory_adapters module.

Tests cover:
- LogitModelWrapper: Wrapping logits to be callable as model
- compute_logit_metric_at_step: Shape conversion and metric computation
- compute_text_metric_at_step: Text metric computation
"""

import pytest
import torch
from typing import Dict, List

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.trajectory_adapters import (
    LogitModelWrapper,
    compute_logit_metric_at_step,
    compute_text_metric_at_step,
)


class TestLogitModelWrapper:
    """Tests for LogitModelWrapper class."""
    
    def test_wrapper_stores_logits(self):
        """Test that wrapper stores logits correctly."""
        B, L, V = 1, 10, 100
        logits = torch.randn(B, L, V)
        device = torch.device("cpu")
        
        wrapper = LogitModelWrapper(logits, device)
        
        assert torch.allclose(wrapper.logits, logits)
        assert wrapper.device == device
    
    def test_wrapper_callable_returns_output(self):
        """Test that wrapper is callable and returns output with logits."""
        B, L, V = 1, 10, 100
        logits = torch.randn(B, L, V)
        device = torch.device("cpu")
        
        wrapper = LogitModelWrapper(logits, device)
        output = wrapper(input_ids=torch.zeros(1, 10))
        
        assert hasattr(output, "logits")
        assert torch.allclose(output.logits, logits)
    
    def test_wrapper_ignores_batch_kwargs(self):
        """Test that wrapper ignores batch kwargs and always returns same logits."""
        B, L, V = 1, 10, 100
        logits = torch.randn(B, L, V)
        device = torch.device("cpu")
        
        wrapper = LogitModelWrapper(logits, device)
        
        # Different batch kwargs should still return same logits
        output1 = wrapper(input_ids=torch.zeros(1, 5))
        output2 = wrapper(input_ids=torch.zeros(1, 20), attention_mask=torch.ones(1, 20))
        
        assert torch.allclose(output1.logits, output2.logits)
        assert torch.allclose(output1.logits, logits)


class TestComputeLogitMetricAtStep:
    """Tests for compute_logit_metric_at_step function."""
    
    def test_2d_logits_converted_to_3d(self):
        """Test that [V, L] logits are converted to [1, L, V]."""
        V, L = 100, 10
        logits_2d = torch.randn(V, L)
        
        # Mock metric function that checks shape
        def mock_metric_fn(model, batch, **kwargs):
            assert model.logits.shape == (1, L, V)
            return [{"test": 1.0}]
        
        batch_template = {
            "input_ids": torch.zeros(1, L),
            "labels": torch.zeros(1, L),
            "attention_mask": torch.ones(1, L),
        }
        
        result = compute_logit_metric_at_step(
            mock_metric_fn, logits_2d, batch_template
        )
        
        assert result == [{"test": 1.0}]
    
    def test_3d_logits_preserved(self):
        """Test that [1, L, V] logits are preserved."""
        V, L = 100, 10
        logits_3d = torch.randn(1, L, V)
        
        def mock_metric_fn(model, batch, **kwargs):
            assert model.logits.shape == (1, L, V)
            return [{"test": 1.0}]
        
        batch_template = {
            "input_ids": torch.zeros(1, L),
            "labels": torch.zeros(1, L),
            "attention_mask": torch.ones(1, L),
        }
        
        result = compute_logit_metric_at_step(
            mock_metric_fn, logits_3d, batch_template
        )
        
        assert result == [{"test": 1.0}]
    
    def test_unexpected_shape_raises_error(self):
        """Test that unexpected logits shapes raise ValueError."""
        logits_4d = torch.randn(1, 1, 10, 100)  # 4D - invalid
        
        def mock_metric_fn(model, batch, **kwargs):
            return [{"test": 1.0}]
        
        batch_template = {
            "input_ids": torch.zeros(1, 10),
            "labels": torch.zeros(1, 10),
            "attention_mask": torch.ones(1, 10),
        }
        
        with pytest.raises(ValueError, match="Unexpected logits shape"):
            compute_logit_metric_at_step(
                mock_metric_fn, logits_4d, batch_template
            )
    
    def test_batch_template_tensors_moved_to_device(self):
        """Test that batch template tensors are moved to logits device."""
        V, L = 100, 10
        logits = torch.randn(V, L)
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logits = logits.to(device)
        else:
            device = torch.device("cpu")
        
        def mock_metric_fn(model, batch, **kwargs):
            # Check all tensors are on same device
            assert batch["input_ids"].device == model.device
            assert batch["labels"].device == model.device
            assert batch["attention_mask"].device == model.device
            return [{"test": 1.0}]
        
        batch_template = {
            "input_ids": torch.zeros(1, L),
            "labels": torch.zeros(1, L),
            "attention_mask": torch.ones(1, L),
        }
        
        compute_logit_metric_at_step(
            mock_metric_fn, logits, batch_template
        )
    
    def test_metric_function_receives_correct_args(self):
        """Test that metric function receives model and batch."""
        V, L = 100, 10
        logits = torch.randn(V, L)
        
        received_args = {}
        
        def mock_metric_fn(model, batch, **kwargs):
            received_args["model"] = model
            received_args["batch"] = batch
            received_args["kwargs"] = kwargs
            return [{"test": 1.0}]
        
        batch_template = {
            "input_ids": torch.zeros(1, L),
            "labels": torch.zeros(1, L),
        }
        
        extra_kwargs = {"custom_arg": 42}
        
        compute_logit_metric_at_step(
            mock_metric_fn, logits, batch_template, **extra_kwargs
        )
        
        assert isinstance(received_args["model"], LogitModelWrapper)
        assert "input_ids" in received_args["batch"]
        assert "labels" in received_args["batch"]
        assert received_args["kwargs"]["custom_arg"] == 42


class TestComputeTextMetricAtStep:
    """Tests for compute_text_metric_at_step function."""
    
    def test_text_metric_receives_texts_and_ground_truths(self):
        """Test that text metric receives texts and ground truths."""
        texts = ["Generated text 1", "Generated text 2"]
        ground_truths = ["Ground truth 1", "Ground truth 2"]
        
        received_batch = None
        
        def mock_metric_fn(model, tokenizer, batch, **kwargs):
            nonlocal received_batch
            received_batch = batch
            return [{"rouge": 0.5}]
        
        # Mock tokenizer
        class MockTokenizer:
            pass
        
        tokenizer = MockTokenizer()
        
        result = compute_text_metric_at_step(
            mock_metric_fn, texts, ground_truths, tokenizer
        )
        
        assert received_batch is not None
        assert received_batch["generation"] == texts
        assert received_batch["ground_truth"] == ground_truths
        assert result == [{"rouge": 0.5}]
    
    def test_text_metric_receives_tokenizer(self):
        """Test that text metric receives tokenizer."""
        texts = ["Text"]
        ground_truths = ["Truth"]
        
        received_tokenizer = None
        
        def mock_metric_fn(model, tokenizer, batch, **kwargs):
            nonlocal received_tokenizer
            received_tokenizer = tokenizer
            return [{"rouge": 0.5}]
        
        tokenizer = object()  # Any object
        
        compute_text_metric_at_step(
            mock_metric_fn, texts, ground_truths, tokenizer
        )
        
        assert received_tokenizer is tokenizer
    
    def test_text_metric_passes_extra_kwargs(self):
        """Test that extra kwargs are passed to metric function."""
        texts = ["Text"]
        ground_truths = ["Truth"]
        
        received_kwargs = {}
        
        def mock_metric_fn(model, tokenizer, batch, **kwargs):
            received_kwargs.update(kwargs)
            return [{"rouge": 0.5}]
        
        tokenizer = object()
        
        compute_text_metric_at_step(
            mock_metric_fn,
            texts,
            ground_truths,
            tokenizer,
            custom_arg=42,
            another_arg="test",
        )
        
        assert received_kwargs["custom_arg"] == 42
        assert received_kwargs["another_arg"] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
