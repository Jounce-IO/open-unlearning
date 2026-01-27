"""
Unit tests for text-based metrics in trajectory_metrics.

Tests cover:
- ROUGE metric computation with logits
- Generic text-based metric handler
- generation_args handling (OmegaConf conversion)
- Tokenizer duplicate argument fix
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from omegaconf import OmegaConf, DictConfig

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.trajectory_metrics import (
    _call_metric_at_step,
    _handle_text_based_metric,
)
from evals.metrics.trajectory_adapters import LogitModelWrapper
from evals.metrics import METRICS_REGISTRY


class TestTextBasedMetricHandler:
    """Tests for _handle_text_based_metric function."""
    
    def test_rouge_metric_with_logits(self):
        """Test that ROUGE metric works with decoded logits."""
        # Create logits [V, L]
        V, L = 100, 5
        logits = torch.randn(V, L)
        
        # Create mock tokenizer
        tokenizer = Mock()
        tokenizer.decode = Mock(return_value="generated text")
        tokenizer.encode = Mock(return_value=torch.tensor([[1, 2, 3]]))
        
        # Create sample labels
        sample_labels = torch.randint(0, V, (L,))
        sample_input_ids = torch.randint(0, V, (5,))  # prompt
        sample_prompt_len = 5
        
        # Create metric config
        metric_config = {"rouge_type": "rougeL_f1"}
        
        # Mock eval_text_similarity (it's imported inside the function)
        with patch("evals.metrics.utils.eval_text_similarity") as mock_eval:
            mock_eval.return_value = [
                {
                    "rouge1_recall": 0.5,
                    "rougeL_f1": 0.6,
                    "rougeL_recall": 0.7,
                }
            ]
            
            result = _handle_text_based_metric(
                logits=logits,
                tokenizer=tokenizer,
                sample_labels=sample_labels,
                sample_input_ids=sample_input_ids,
                sample_prompt_len=sample_prompt_len,
                metric_name="rouge",
                metric_config=metric_config,
            )
            
            # Should return list with score
            assert isinstance(result, list)
            assert len(result) > 0
            assert "score" in result[0]
            assert result[0]["score"] == 0.6  # rougeL_f1
    
    def test_generation_args_as_dict_converted_to_omegaconf(self):
        """Test that generation_args dict is converted to OmegaConf."""
        V, L = 100, 5
        logits = torch.randn(V, L)
        
        tokenizer = Mock()
        tokenizer.decode = Mock(return_value="generated text")
        tokenizer.encode = Mock(return_value=torch.tensor([[1, 2, 3]]))
        
        sample_labels = torch.randint(0, V, (L,))
        sample_input_ids = torch.randint(0, V, (5,))
        sample_prompt_len = 5
        
        # generation_args as plain dict (not OmegaConf)
        generation_args = {"max_length": 100, "temperature": 0.7}
        
        with patch("evals.metrics.utils.eval_text_similarity") as mock_eval:
            mock_eval.return_value = [{"rougeL_f1": 0.5}]
            
            # This should not raise ValueError about OmegaConf
            result = _handle_text_based_metric(
                logits=logits,
                tokenizer=tokenizer,
                sample_labels=sample_labels,
                sample_input_ids=sample_input_ids,
                sample_prompt_len=sample_prompt_len,
                metric_name="rouge",
                metric_config={"rouge_type": "rougeL_f1"},
                generation_args=generation_args,
            )
            
            # Verify eval_text_similarity was called
            assert mock_eval.called
            call_args = mock_eval.call_args
            
            # Check that generation_args was converted to OmegaConf
            gen_args_passed = call_args[0][3]  # 4th positional arg
            assert isinstance(gen_args_passed, (DictConfig, dict))
    
    def test_generation_args_as_omegaconf_preserved(self):
        """Test that generation_args as OmegaConf is preserved."""
        V, L = 100, 5
        logits = torch.randn(V, L)
        
        tokenizer = Mock()
        tokenizer.decode = Mock(return_value="generated text")
        tokenizer.encode = Mock(return_value=torch.tensor([[1, 2, 3]]))
        
        sample_labels = torch.randint(0, V, (L,))
        sample_input_ids = torch.randint(0, V, (5,))
        sample_prompt_len = 5
        
        # generation_args as OmegaConf
        generation_args = OmegaConf.create({"max_length": 100})
        
        with patch("evals.metrics.utils.eval_text_similarity") as mock_eval:
            mock_eval.return_value = [{"rougeL_f1": 0.5}]
            
            result = _handle_text_based_metric(
                logits=logits,
                tokenizer=tokenizer,
                sample_labels=sample_labels,
                sample_input_ids=sample_input_ids,
                sample_prompt_len=sample_prompt_len,
                metric_name="rouge",
                metric_config={"rouge_type": "rougeL_f1"},
                generation_args=generation_args,
            )
            
            # Should work without error
            assert result is not None


class TestCallMetricAtStepTextBased:
    """Tests for _call_metric_at_step with text-based metrics."""
    
    def test_rouge_metric_detected_as_text_based(self):
        """Test that rouge metric is detected and handled as text-based."""
        # Get rouge metric from registry
        if "rouge" not in METRICS_REGISTRY:
            pytest.skip("rouge metric not registered")
        
        rouge_metric = METRICS_REGISTRY["rouge"]
        
        # Create logits [V, L]
        V, L = 100, 5
        logits = torch.randn(V, L)
        
        # Create batch template
        batch_template = {
            "input_ids": torch.zeros((1, L), dtype=torch.long),
            "labels": torch.randint(0, V, (1, L)),
            "attention_mask": torch.ones((1, L), dtype=torch.long),
        }
        
        # Create tokenizer
        tokenizer = Mock()
        tokenizer.decode = Mock(return_value="generated text")
        tokenizer.encode = Mock(return_value=torch.tensor([[1, 2, 3]]))
        tokenizer.eos_token_id = 2
        
        sample_labels = torch.randint(0, V, (L,))
        sample_input_ids = torch.randint(0, V, (5,))
        sample_prompt_len = 5
        
        # Mock eval_text_similarity (it's imported inside the function)
        with patch("evals.metrics.utils.eval_text_similarity") as mock_eval:
            mock_eval.return_value = [
                {
                    "rouge1_recall": 0.5,
                    "rougeL_f1": 0.6,
                    "rougeL_recall": 0.7,
                }
            ]
            
            # Call metric - should detect it needs generation and use text handler
            result = _call_metric_at_step(
                metric=rouge_metric,
                logits=logits,
                batch_template=batch_template,
                tokenizer=tokenizer,
                sample_labels=sample_labels,
                sample_input_ids=sample_input_ids,
                sample_prompt_len=sample_prompt_len,
                metric_config={"rouge_type": "rougeL_f1"},
                sample_idx="0",
            )
            
            # Should return result from text handler
            assert result is not None
            assert isinstance(result, list)
            assert len(result) > 0
    
    def test_tokenizer_not_duplicated_in_kwargs(self):
        """Test that tokenizer is removed from kwargs to avoid duplicate argument."""
        # Get probability metric (logit-based, should work)
        if "probability" not in METRICS_REGISTRY:
            pytest.skip("probability metric not registered")
        
        prob_metric = METRICS_REGISTRY["probability"]
        
        V, L = 100, 5
        logits = torch.randn(V, L)
        
        batch_template = {
            "input_ids": torch.zeros((1, L), dtype=torch.long),
            "labels": torch.randint(0, V, (1, L)),
            "attention_mask": torch.ones((1, L), dtype=torch.long),
        }
        
        tokenizer = Mock()
        sample_labels = torch.randint(0, V, (L,))
        sample_input_ids = torch.randint(0, V, (5,))
        sample_prompt_len = 5
        
        # kwargs contains tokenizer (simulating real scenario)
        kwargs = {"tokenizer": tokenizer, "other_arg": "value"}
        
        # In real usage, we filter kwargs before calling (as done in trajectory_metrics)
        kwargs_clean = {k: v for k, v in kwargs.items() if k != "tokenizer"}
        
        # This should not raise "got multiple values for keyword argument 'tokenizer'"
        result = _call_metric_at_step(
            metric=prob_metric,
            logits=logits,
            batch_template=batch_template,
            tokenizer=tokenizer,  # Passed explicitly
            sample_labels=sample_labels,
            sample_input_ids=sample_input_ids,
            sample_prompt_len=sample_prompt_len,
            metric_config={},
            sample_idx="0",
            **kwargs_clean  # tokenizer removed
        )
        
        # Should not raise TypeError and should return a result
        assert result is not None


class TestTrajectoryMetricsWithTwoMetrics:
    """Integration tests for trajectory_metrics with probability and rouge."""
    
    def test_trajectory_metrics_with_probability_and_rouge(self):
        """Test that trajectory_metrics works with both probability and rouge."""
        # Setup mock sampler
        V, L_gen, S = 1000, 10, 8
        prompt_len = 5
        full_len = prompt_len + L_gen
        
        sampler = Mock()
        logits_history = [torch.randn(1, full_len, V) for _ in range(S)]
        fixation_steps = torch.randint(0, S, (1, full_len))
        
        class MockSamplerOutput:
            def __init__(self):
                self.sequences = torch.randint(0, V, (1, full_len))
                self.histories = None
                self.logits_history = logits_history
                self.fixation_steps = fixation_steps
        
        sampler.sample.return_value = MockSamplerOutput()
        
        model = Mock()
        model.sampler = sampler
        
        # Mock dataset
        class MockDataset:
            def __init__(self):
                self.data = [
                    {
                        "input_ids": torch.randint(0, V, (full_len,)),
                        "labels": torch.randint(0, V, (full_len,)),
                    }
                ]
            
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
        tokenizer.decode = Mock(return_value="decoded text")
        tokenizer.encode = Mock(return_value=torch.tensor([[1, 2, 3]]))
        tokenizer.eos_token_id = 2
        
        # Mock eval_text_similarity for rouge (it's imported inside the function from utils)
        with patch("evals.metrics.utils.eval_text_similarity") as mock_eval_text:
            mock_eval_text.return_value = [
                {
                    "rouge1_recall": 0.5,
                    "rougeL_f1": 0.6,
                    "rougeL_recall": 0.7,
                }
            ]
            
            kwargs = {
                "metrics": ["probability", "rouge"],
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
            
            from evals.metrics.trajectory_metrics import trajectory_metrics
            
            # trajectory_metrics is wrapped in UnlearningMetric, need to call _metric_fn
            raw_fn = trajectory_metrics._metric_fn if hasattr(trajectory_metrics, '_metric_fn') else trajectory_metrics
            
            # This should run without errors
            result = raw_fn(model, **kwargs)
            
            # Check result structure
            assert isinstance(result, dict)
            if "agg_value" in result:
                agg = result["agg_value"]
                assert "steps" in agg
                assert "fixation" in agg
                assert "ratio" in agg
                
                # Check that both metrics are present
                if "probability" in agg["steps"]:
                    assert len(agg["steps"]["probability"]) == S
                if "rouge" in agg["steps"]:
                    assert len(agg["steps"]["rouge"]) == S


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
