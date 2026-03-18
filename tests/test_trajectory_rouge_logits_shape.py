"""
Regression and shape tests for trajectory ROUGE logits path.

- _handle_text_based_metric: [1, L, V] yields gen_text length ~L; [V, L] yields ~V (wrong).
- _call_metric_at_step fallback: passes [1, L, V] to text handler (no transpose).
- One test fails before the fallback fix (transpose to [V, L]) and passes after.
"""

import pytest
import torch
from unittest.mock import Mock, patch

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.trajectory_metrics import (
    _call_metric_at_step,
    _handle_text_based_metric,
)


class TestHandleTextBasedMetricLogitsShape:
    """_handle_text_based_metric: correct [1, L, V] vs wrong [V, L] yields different decode length."""

    def test_logits_1_L_V_yields_gen_text_length_on_order_of_L(self):
        """With [1, L, V], argmax(dim=-1) gives L tokens; decoded text length is on order of L."""
        L, V = 20, 1000
        logits = torch.randn(1, L, V)
        tokenizer = Mock()
        decoded_lengths = []

        def capture_decode(tok_ids, **kwargs):
            ids = tok_ids if isinstance(tok_ids, list) else tok_ids.tolist()
            decoded_lengths.append(len(ids))
            return "x" * len(ids)

        tokenizer.decode = capture_decode

        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        with patch("evals.metrics.utils.eval_rouge_recall_batch") as mock_rouge:
            mock_rouge.return_value = [{"rougeL_f1": 0.5, "rougeL_recall": 0.5}]
            _ = _handle_text_based_metric(
                logits=logits,
                tokenizer=tokenizer,
                sample_labels=torch.zeros(L),
                sample_input_ids=torch.zeros(L),
                sample_prompt_len=0,
                metric_name="rouge",
                metric_config={"rouge_type": "rougeL_f1"},
                ground_truth="reference",
                rouge_scorer=scorer,
            )
        assert len(decoded_lengths) == 1
        assert decoded_lengths[0] == L

    def test_logits_V_L_wrong_shape_yields_gen_text_length_on_order_of_V(self):
        """With [V, L] (wrong), argmax(dim=-1) gives V tokens; decoded length is on order of V."""
        L, V = 20, 1000
        logits = torch.randn(V, L)
        tokenizer = Mock()
        decoded_lengths = []

        def capture_decode(tok_ids, **kwargs):
            ids = tok_ids if isinstance(tok_ids, list) else tok_ids.tolist()
            decoded_lengths.append(len(ids))
            return "x" * len(ids)

        tokenizer.decode = capture_decode

        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        with patch("evals.metrics.utils.eval_rouge_recall_batch") as mock_rouge:
            mock_rouge.return_value = [{"rougeL_f1": 0.0, "rougeL_recall": 0.0}]
            _ = _handle_text_based_metric(
                logits=logits,
                tokenizer=tokenizer,
                sample_labels=torch.zeros(L),
                sample_input_ids=torch.zeros(L),
                sample_prompt_len=0,
                metric_name="rouge",
                metric_config={"rouge_type": "rougeL_f1"},
                ground_truth="reference",
                rouge_scorer=scorer,
            )
        assert len(decoded_lengths) == 1
        assert decoded_lengths[0] == V

    def test_logits_2d_L_V_yields_gen_text_length_L(self):
        """With [L, V] (2D), argmax(dim=-1) gives L tokens."""
        L, V = 15, 500
        logits = torch.randn(L, V)
        tokenizer = Mock()
        decoded_lengths = []

        def capture_decode(tok_ids, **kwargs):
            ids = tok_ids if isinstance(tok_ids, list) else tok_ids.tolist()
            decoded_lengths.append(len(ids))
            return "y" * len(ids)

        tokenizer.decode = capture_decode

        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        with patch("evals.metrics.utils.eval_rouge_recall_batch") as mock_rouge:
            mock_rouge.return_value = [{"rougeL_f1": 0.5, "rougeL_recall": 0.5}]
            _ = _handle_text_based_metric(
                logits=logits,
                tokenizer=tokenizer,
                sample_labels=torch.zeros(L),
                sample_input_ids=torch.zeros(L),
                sample_prompt_len=0,
                metric_name="rouge",
                metric_config={"rouge_type": "rougeL_f1"},
                ground_truth="ref",
                rouge_scorer=scorer,
            )
        assert len(decoded_lengths) == 1
        assert decoded_lengths[0] == L


class TestCallMetricAtStepFallbackPassesCorrectLogitsShape:
    """
    When the direct metric call fails and we use the generic text-based handler,
    logits must be passed as [1, L, V]; not transposed to [V, L].
    This test would fail before the fix (fallback passed [V, L] -> decode received V tokens)
    and passes after the fix (fallback passes [1, L, V] -> decode receives L tokens).
    """

    def test_fallback_receives_logits_1_L_V_not_V_L(self):
        """_call_metric_at_step fallback must pass [1, L, V] to _handle_text_based_metric so decode gets L tokens."""
        L, V = 200, 32000
        logits = torch.randn(1, L, V)
        batch_template = {
            "input_ids": torch.zeros(1, L, dtype=torch.long),
            "labels": torch.zeros(1, L, dtype=torch.long),
        }
        tokenizer = Mock()
        decoded_token_counts = []

        def capture_decode(tok_ids, **kwargs):
            ids = tok_ids if isinstance(tok_ids, list) else tok_ids.tolist()
            decoded_token_counts.append(len(ids))
            return "a" * min(len(ids), 1000)

        tokenizer.decode = capture_decode

        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        rouge_metric = Mock()
        rouge_metric.name = "rouge"
        rouge_metric._metric_fn = Mock(side_effect=KeyError("generate"))

        with patch("evals.metrics.utils.eval_rouge_recall_batch") as mock_rouge:
            mock_rouge.return_value = [{"rougeL_f1": 0.3, "rougeL_recall": 0.3}]
            result = _call_metric_at_step(
                metric=rouge_metric,
                logits=logits,
                batch_template=batch_template,
                tokenizer=tokenizer,
                sample_labels=torch.zeros(1, L),
                sample_input_ids=torch.zeros(1, L),
                sample_prompt_len=0,
                metric_config={"rouge_type": "rougeL_f1"},
                sample_idx="0",
                ground_truth="reference text",
                rouge_scorer=scorer,
            )
        assert result is not None
        assert len(decoded_token_counts) == 1
        assert decoded_token_counts[0] == L, (
            f"Fallback must pass [1, L, V] so decode gets L={L} tokens; got {decoded_token_counts[0]} (wrong shape [V, L] would give V)"
        )

    def test_fallback_re_raises_when_text_handler_fails(self):
        """When the generic text-based handler raises, we re-raise instead of returning None."""
        L, V = 10, 100
        logits = torch.randn(1, L, V)
        batch_template = {
            "input_ids": torch.zeros(1, L, dtype=torch.long),
            "labels": torch.zeros(1, L, dtype=torch.long),
        }
        tokenizer = Mock()
        tokenizer.decode = Mock(return_value="ok")

        rouge_metric = Mock()
        rouge_metric.name = "rouge"
        rouge_metric._metric_fn = Mock(side_effect=KeyError("generate"))

        with patch("evals.metrics.utils.eval_rouge_recall_batch") as mock_rouge:
            mock_rouge.side_effect = RuntimeError("rouge failed")
            with pytest.raises(RuntimeError, match="rouge failed"):
                _call_metric_at_step(
                    metric=rouge_metric,
                    logits=logits,
                    batch_template=batch_template,
                    tokenizer=tokenizer,
                    sample_labels=torch.zeros(1, L),
                    sample_input_ids=torch.zeros(1, L),
                    sample_prompt_len=0,
                    metric_config={"rouge_type": "rougeL_f1"},
                    sample_idx="0",
                    ground_truth="ref",
                    rouge_scorer=Mock(),
                )


class TestPipelineLogitsShape:
    """Shapes through the trajectory MU path: _get_logits_at_step -> [V, L]; after transpose -> [1, L, V]."""

    def test_get_logits_at_step_returns_V_L(self):
        """_get_logits_at_step returns [V, L] for single-sample trajectory (contract)."""
        from evals.metrics.trajectory_metrics import _get_logits_at_step

        V, L, S = 100, 50, 5
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        traj = {"R": R, "F": F, "S": S, "L": L}
        logits = _get_logits_at_step(traj, "steps", step=0)
        assert logits.shape == (V, L)

    def test_normalize_2d_V_L_to_1_L_V(self):
        """_call_metric_at_step normalizes 2D [V, L] to [1, L, V] before passing to metric/fallback."""
        from evals.metrics.trajectory_metrics import _call_metric_at_step

        V, L = 100, 20
        logits_2d = torch.randn(V, L)
        batch_template = {
            "input_ids": torch.zeros(1, L, dtype=torch.long),
            "labels": torch.zeros(1, L, dtype=torch.long),
        }
        tokenizer = Mock()
        decoded_token_counts = []

        def capture_decode(tok_ids, **kwargs):
            ids = tok_ids if isinstance(tok_ids, list) else tok_ids.tolist()
            decoded_token_counts.append(len(ids))
            return "x" * len(ids)

        tokenizer.decode = capture_decode

        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        rouge_metric = Mock()
        rouge_metric.name = "rouge"
        rouge_metric._metric_fn = Mock(side_effect=TypeError("generate"))

        with patch("evals.metrics.utils.eval_rouge_recall_batch") as mock_rouge:
            mock_rouge.return_value = [{"rougeL_f1": 0.5, "rougeL_recall": 0.5}]
            _ = _call_metric_at_step(
                metric=rouge_metric,
                logits=logits_2d,
                batch_template=batch_template,
                tokenizer=tokenizer,
                sample_labels=torch.zeros(1, L),
                sample_input_ids=torch.zeros(1, L),
                sample_prompt_len=0,
                metric_config={"rouge_type": "rougeL_f1"},
                sample_idx="0",
                ground_truth="gt",
                rouge_scorer=scorer,
            )
        assert len(decoded_token_counts) == 1
        assert decoded_token_counts[0] == L
