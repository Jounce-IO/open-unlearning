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
    DualLogitModelWrapper,
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
    
    @pytest.mark.parametrize("B", [1, 2, 4])
    def test_wrapper_different_batch_sizes(self, B):
        """Test wrapper with different batch sizes."""
        L, V = 10, 100
        logits = torch.randn(B, L, V)
        device = torch.device("cpu")
        
        wrapper = LogitModelWrapper(logits, device)
        assert wrapper.logits.shape == (B, L, V)
        assert torch.allclose(wrapper.logits, logits)
    
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_wrapper_different_dtypes(self, dtype):
        """Test wrapper with different dtypes."""
        B, L, V = 1, 10, 100
        logits = torch.randn(B, L, V, dtype=dtype)
        device = torch.device("cpu")
        
        wrapper = LogitModelWrapper(logits, device)
        assert wrapper.logits.dtype == dtype
        assert torch.allclose(wrapper.logits, logits)
    
    def test_wrapper_different_devices(self):
        """Test wrapper with different devices."""
        B, L, V = 1, 10, 100
        logits = torch.randn(B, L, V)
        
        # CPU
        device_cpu = torch.device("cpu")
        wrapper_cpu = LogitModelWrapper(logits, device_cpu)
        assert wrapper_cpu.device.type == "cpu"
        
        # CUDA if available
        if torch.cuda.is_available():
            device_cuda = torch.device("cuda")
            logits_cuda = logits.cuda()
            wrapper_cuda = LogitModelWrapper(logits_cuda, device_cuda)
            assert wrapper_cuda.device.type == "cuda"
    
    def test_wrapper_output_same_tensor(self):
        """Test that output.logits is same tensor (not copy)."""
        B, L, V = 1, 10, 100
        logits = torch.randn(B, L, V)
        device = torch.device("cpu")
        
        wrapper = LogitModelWrapper(logits, device)
        output = wrapper(input_ids=torch.zeros(1, 10))
        
        # output.logits should be the same tensor object
        assert output.logits is wrapper.logits
        assert output.logits is logits
    
    def test_wrapper_output_has_correct_attributes(self):
        """Test that output object has correct attributes."""
        B, L, V = 1, 10, 100
        logits = torch.randn(B, L, V)
        device = torch.device("cpu")
        
        wrapper = LogitModelWrapper(logits, device)
        output = wrapper(input_ids=torch.zeros(1, 10))
        
        assert hasattr(output, "logits")
        assert output.logits.shape == (B, L, V)
    
    def test_wrapper_with_empty_batch_kwargs(self):
        """Test wrapper with empty batch kwargs."""
        B, L, V = 1, 10, 100
        logits = torch.randn(B, L, V)
        device = torch.device("cpu")
        
        wrapper = LogitModelWrapper(logits, device)
        output = wrapper()
        
        assert hasattr(output, "logits")
        assert torch.allclose(output.logits, logits)
    
    def test_wrapper_with_many_batch_kwargs(self):
        """Test wrapper with many batch kwargs (stress test)."""
        B, L, V = 1, 10, 100
        logits = torch.randn(B, L, V)
        device = torch.device("cpu")
        
        wrapper = LogitModelWrapper(logits, device)
        output = wrapper(
            input_ids=torch.zeros(1, 10),
            attention_mask=torch.ones(1, 10),
            token_type_ids=torch.zeros(1, 10),
            position_ids=torch.arange(10).unsqueeze(0),
            labels=torch.zeros(1, 10),
            custom_arg="value",
        )
        
        assert hasattr(output, "logits")
        assert torch.allclose(output.logits, logits)


class TestDualLogitModelWrapper:
    """Tests for DualLogitModelWrapper (per-sample logits for MIA)."""

    def test_returns_different_logits_for_forget_vs_holdout(self):
        """DualLogitModelWrapper returns different logits for forget vs holdout batch."""
        B, L, V = 1, 10, 100
        logits_forget = torch.randn(V, L)
        logits_holdout = torch.randn(V, L)
        logits_by_key = {
            "forget": {"0": logits_forget},
            "holdout": {"0": logits_holdout},
        }
        wrapper = DualLogitModelWrapper(logits_by_key, torch.device("cpu"))

        batch = {
            "input_ids": torch.zeros(1, L),
            "labels": torch.randint(0, V, (1, L)),
            "index": torch.tensor([0]),
        }

        wrapper.set_dataset_key("forget")
        out_forget = wrapper(**batch)
        wrapper.set_dataset_key("holdout")
        out_holdout = wrapper(**batch)

        assert not torch.allclose(out_forget.logits, out_holdout.logits)

    def test_set_dataset_key_required(self):
        """DualLogitModelWrapper raises if set_dataset_key not called."""
        logits_by_key = {"forget": {"0": torch.randn(100, 10)}, "holdout": {"0": torch.randn(100, 10)}}
        wrapper = DualLogitModelWrapper(logits_by_key, torch.device("cpu"))
        batch = {"input_ids": torch.zeros(1, 10), "index": torch.tensor([0])}
        with pytest.raises(RuntimeError, match="set_dataset_key"):
            wrapper(**batch)


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
    
    def test_logits_2d_conversion_exact_values(self):
        """Test that [V, L] logits are correctly converted to [1, L, V] with exact values."""
        V, L = 100, 10
        logits_2d = torch.arange(V * L, dtype=torch.float32).reshape(V, L)
        
        def mock_metric_fn(model, batch, **kwargs):
            # Verify shape and values
            assert model.logits.shape == (1, L, V)
            # Check that values are transposed correctly
            # logits_2d[i, j] should become model.logits[0, j, i]
            for j in range(L):
                for i in range(V):
                    assert torch.allclose(model.logits[0, j, i], logits_2d[i, j])
            return [{"test": 1.0}]
        
        batch_template = {
            "input_ids": torch.zeros(1, L),
            "labels": torch.zeros(1, L),
        }
        
        result = compute_logit_metric_at_step(mock_metric_fn, logits_2d, batch_template)
        assert result == [{"test": 1.0}]
    
    def test_batch_template_with_non_tensor_values(self):
        """Test that batch template with non-tensor values is preserved."""
        V, L = 100, 10
        logits = torch.randn(V, L)
        
        received_batch = {}
        
        def mock_metric_fn(model, batch, **kwargs):
            received_batch.update(batch)
            return [{"test": 1.0}]
        
        batch_template = {
            "input_ids": torch.zeros(1, L),
            "labels": torch.zeros(1, L),
            "index": [0],  # Non-tensor value
            "text": "sample text",  # Non-tensor value
        }
        
        compute_logit_metric_at_step(mock_metric_fn, logits, batch_template)
        
        assert "input_ids" in received_batch
        assert "labels" in received_batch
        assert "index" in received_batch
        assert received_batch["index"] == [0]
        assert "text" in received_batch
        assert received_batch["text"] == "sample text"
    
    def test_batch_template_missing_keys(self):
        """Test that batch template with missing keys still works."""
        V, L = 100, 10
        logits = torch.randn(V, L)
        
        def mock_metric_fn(model, batch, **kwargs):
            # Should work even if batch is minimal
            return [{"test": 1.0}]
        
        batch_template = {
            "input_ids": torch.zeros(1, L),
            # Missing labels, attention_mask, etc.
        }
        
        result = compute_logit_metric_at_step(mock_metric_fn, logits, batch_template)
        assert result == [{"test": 1.0}]
    
    def test_batch_template_with_extra_keys(self):
        """Test that batch template with extra keys is passed through."""
        V, L = 100, 10
        logits = torch.randn(V, L)
        
        received_batch = {}
        
        def mock_metric_fn(model, batch, **kwargs):
            received_batch.update(batch)
            return [{"test": 1.0}]
        
        batch_template = {
            "input_ids": torch.zeros(1, L),
            "labels": torch.zeros(1, L),
            "extra_key": "extra_value",
            "another_key": torch.ones(1, L),
        }
        
        compute_logit_metric_at_step(mock_metric_fn, logits, batch_template)
        
        assert "extra_key" in received_batch
        assert received_batch["extra_key"] == "extra_value"
        assert "another_key" in received_batch
    
    def test_metric_function_raises_exception(self):
        """Test that metric function exceptions are propagated."""
        V, L = 100, 10
        logits = torch.randn(V, L)
        
        def mock_metric_fn(model, batch, **kwargs):
            raise ValueError("Test error")
        
        batch_template = {
            "input_ids": torch.zeros(1, L),
            "labels": torch.zeros(1, L),
        }
        
        with pytest.raises(ValueError, match="Test error"):
            compute_logit_metric_at_step(mock_metric_fn, logits, batch_template)
    
    def test_metric_function_returns_none(self):
        """Test that metric function returning None is handled."""
        V, L = 100, 10
        logits = torch.randn(V, L)
        
        def mock_metric_fn(model, batch, **kwargs):
            return None
        
        batch_template = {
            "input_ids": torch.zeros(1, L),
            "labels": torch.zeros(1, L),
        }
        
        result = compute_logit_metric_at_step(mock_metric_fn, logits, batch_template)
        assert result is None
    
    def test_metric_function_returns_empty_list(self):
        """Test that metric function returning empty list is handled."""
        V, L = 100, 10
        logits = torch.randn(V, L)
        
        def mock_metric_fn(model, batch, **kwargs):
            return []
        
        batch_template = {
            "input_ids": torch.zeros(1, L),
            "labels": torch.zeros(1, L),
        }
        
        result = compute_logit_metric_at_step(mock_metric_fn, logits, batch_template)
        assert result == []
    
    def test_kwargs_with_tensors(self):
        """Test that kwargs containing tensors are passed through."""
        V, L = 100, 10
        logits = torch.randn(V, L)
        
        received_kwargs = {}
        
        def mock_metric_fn(model, batch, **kwargs):
            received_kwargs.update(kwargs)
            return [{"test": 1.0}]
        
        batch_template = {
            "input_ids": torch.zeros(1, L),
        }
        
        tensor_kwarg = torch.ones(1, L)
        compute_logit_metric_at_step(
            mock_metric_fn, logits, batch_template, tensor_arg=tensor_kwarg
        )
        
        assert "tensor_arg" in received_kwargs
        assert torch.allclose(received_kwargs["tensor_arg"], tensor_kwarg)
    
    def test_kwargs_with_non_tensor_values(self):
        """Test that kwargs with non-tensor values are passed through."""
        V, L = 100, 10
        logits = torch.randn(V, L)
        
        received_kwargs = {}
        
        def mock_metric_fn(model, batch, **kwargs):
            received_kwargs.update(kwargs)
            return [{"test": 1.0}]
        
        batch_template = {
            "input_ids": torch.zeros(1, L),
        }
        
        compute_logit_metric_at_step(
            mock_metric_fn,
            logits,
            batch_template,
            string_arg="value",
            int_arg=42,
            list_arg=[1, 2, 3],
            dict_arg={"key": "value"},
        )
        
        assert received_kwargs["string_arg"] == "value"
        assert received_kwargs["int_arg"] == 42
        assert received_kwargs["list_arg"] == [1, 2, 3]
        assert received_kwargs["dict_arg"] == {"key": "value"}


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
    
    def test_text_metric_empty_texts_list(self):
        """Test text metric with empty texts list."""
        texts = []
        ground_truths = []
        
        def mock_metric_fn(model, tokenizer, batch, **kwargs):
            return [{"rouge": 0.0}]
        
        tokenizer = object()
        
        result = compute_text_metric_at_step(
            mock_metric_fn, texts, ground_truths, tokenizer
        )
        
        assert result == [{"rouge": 0.0}]
    
    def test_text_metric_empty_ground_truths_list(self):
        """Test text metric with empty ground_truths list."""
        texts = ["Text 1", "Text 2"]
        ground_truths = []
        
        def mock_metric_fn(model, tokenizer, batch, **kwargs):
            assert batch["generation"] == texts
            assert batch["ground_truth"] == ground_truths
            return [{"rouge": 0.0}]
        
        tokenizer = object()
        
        result = compute_text_metric_at_step(
            mock_metric_fn, texts, ground_truths, tokenizer
        )
        
        assert result == [{"rouge": 0.0}]
    
    def test_text_metric_mismatched_lengths(self):
        """Test text metric with mismatched lengths (texts vs ground_truths)."""
        texts = ["Text 1", "Text 2"]
        ground_truths = ["Truth 1"]  # Different length
        
        def mock_metric_fn(model, tokenizer, batch, **kwargs):
            # Function should receive what we pass, even if mismatched
            return [{"rouge": 0.0}]
        
        tokenizer = object()
        
        result = compute_text_metric_at_step(
            mock_metric_fn, texts, ground_truths, tokenizer
        )
        
        assert result == [{"rouge": 0.0}]
    
    def test_text_metric_very_long_texts(self):
        """Test text metric with very long texts (stress test)."""
        long_text = "word " * 1000  # Very long text
        texts = [long_text]
        ground_truths = ["Ground truth"]
        
        def mock_metric_fn(model, tokenizer, batch, **kwargs):
            assert len(batch["generation"][0]) > 1000
            return [{"rouge": 0.5}]
        
        tokenizer = object()
        
        result = compute_text_metric_at_step(
            mock_metric_fn, texts, ground_truths, tokenizer
        )
        
        assert result == [{"rouge": 0.5}]
    
    def test_text_metric_empty_strings(self):
        """Test text metric with empty strings."""
        texts = ["", ""]
        ground_truths = ["", ""]
        
        def mock_metric_fn(model, tokenizer, batch, **kwargs):
            assert batch["generation"] == texts
            assert batch["ground_truth"] == ground_truths
            return [{"rouge": 0.0}]
        
        tokenizer = object()
        
        result = compute_text_metric_at_step(
            mock_metric_fn, texts, ground_truths, tokenizer
        )
        
        assert result == [{"rouge": 0.0}]
    
    def test_text_metric_special_characters(self):
        """Test text metric with special characters."""
        texts = ["Text with\nnewlines\tand\ttabs", "Text with \"quotes\" and 'apostrophes'"]
        ground_truths = ["Truth with\nnewlines", "Truth with special chars: !@#$%"]
        
        def mock_metric_fn(model, tokenizer, batch, **kwargs):
            assert batch["generation"] == texts
            assert batch["ground_truth"] == ground_truths
            return [{"rouge": 0.5}]
        
        tokenizer = object()
        
        result = compute_text_metric_at_step(
            mock_metric_fn, texts, ground_truths, tokenizer
        )
        
        assert result == [{"rouge": 0.5}]
    
    def test_text_metric_unicode_characters(self):
        """Test text metric with unicode characters."""
        texts = ["Text with unicode: 你好世界 🌍", "Text with émojis: 🎉🎊"]
        ground_truths = ["Truth with unicode: مرحبا", "Truth with émojis: 🚀"]
        
        def mock_metric_fn(model, tokenizer, batch, **kwargs):
            assert batch["generation"] == texts
            assert batch["ground_truth"] == ground_truths
            return [{"rouge": 0.5}]
        
        tokenizer = object()
        
        result = compute_text_metric_at_step(
            mock_metric_fn, texts, ground_truths, tokenizer
        )
        
        assert result == [{"rouge": 0.5}]
    
    def test_text_metric_function_raises_exception(self):
        """Test that text metric function exceptions are propagated."""
        texts = ["Text"]
        ground_truths = ["Truth"]
        
        def mock_metric_fn(model, tokenizer, batch, **kwargs):
            raise ValueError("Test error")
        
        tokenizer = object()
        
        with pytest.raises(ValueError, match="Test error"):
            compute_text_metric_at_step(mock_metric_fn, texts, ground_truths, tokenizer)
    
    def test_text_metric_function_returns_none(self):
        """Test that text metric function returning None is handled."""
        texts = ["Text"]
        ground_truths = ["Truth"]
        
        def mock_metric_fn(model, tokenizer, batch, **kwargs):
            return None
        
        tokenizer = object()
        
        result = compute_text_metric_at_step(
            mock_metric_fn, texts, ground_truths, tokenizer
        )
        
        assert result is None
    
    def test_text_metric_kwargs_with_complex_objects(self):
        """Test text metric with kwargs containing complex objects."""
        texts = ["Text"]
        ground_truths = ["Truth"]
        
        received_kwargs = {}
        
        def mock_metric_fn(model, tokenizer, batch, **kwargs):
            received_kwargs.update(kwargs)
            return [{"rouge": 0.5}]
        
        tokenizer = object()
        
        complex_obj = {"nested": {"deep": {"value": 42}}}
        list_obj = [1, 2, {"key": "value"}]
        
        compute_text_metric_at_step(
            mock_metric_fn,
            texts,
            ground_truths,
            tokenizer,
            complex_arg=complex_obj,
            list_arg=list_obj,
        )
        
        assert received_kwargs["complex_arg"] == complex_obj
        assert received_kwargs["list_arg"] == list_obj


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
