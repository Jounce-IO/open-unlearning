"""
Tests for step-wise score provider (generalized unlearning metrics).

- AR provider: same geometric mean as evaluate_probability for a batch.
- sequence_probability_from_scores and evaluate_probability_via_provider format.
"""

import pytest
import torch
from unittest.mock import Mock

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from data.utils import IGNORE_INDEX
from evals.metrics.utils import evaluate_probability
from evals.metrics.step_wise_score import (
    ARStepWiseScoreProvider,
    FixationStepWiseScoreProvider,
    build_effective_step_fixation_logits,
    build_fixation_logits_from_R_F,
    sequence_probability_from_scores,
    evaluate_probability_via_provider,
    extraction_strength_from_fixation,
)


class TestSequenceProbabilityFromScores:
    def test_empty_returns_zero(self):
        assert sequence_probability_from_scores([]) == 0.0

    def test_single_score(self):
        assert abs(sequence_probability_from_scores([0.5]) - 0.5) < 1e-9

    def test_geometric_mean(self):
        import math
        scores = [0.25, 0.5, 0.5]
        expected = (0.25 * 0.5 * 0.5) ** (1 / 3)
        assert abs(sequence_probability_from_scores(scores) - expected) < 1e-9


class TestARStepWiseScoreProvider:
    """AR provider yields same geometric mean as evaluate_probability."""

    def test_get_per_position_scores_same_geom_mean_as_evaluate_probability(self):
        batch_size = 2
        seq_len = 6
        vocab_size = 8
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[:, 0] = IGNORE_INDEX
        input_ids = labels.clone()
        input_ids[labels == IGNORE_INDEX] = 0
        model = Mock()
        model.device = torch.device("cpu")
        model.return_value = Mock(logits=torch.randn(batch_size, seq_len, vocab_size))
        model.side_effect = lambda **kw: model.return_value
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones(batch_size, seq_len),
        }
        direct = evaluate_probability(model, batch)
        provider = ARStepWiseScoreProvider()
        via_provider = evaluate_probability_via_provider(provider, model, batch)
        assert len(direct) == len(via_provider)
        for d, v in zip(direct, via_provider):
            if d["prob"] is None:
                assert v["prob"] is None
                continue
            assert abs(d["prob"] - v["prob"]) < 1e-5
            assert abs(d["avg_loss"] - v["avg_loss"]) < 1e-5

    def test_get_per_position_scores_returns_scores_and_fixation_steps(self):
        batch_size = 1
        seq_len = 5
        vocab_size = 10
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[0, 0] = IGNORE_INDEX
        input_ids = labels.clone()
        input_ids[labels == IGNORE_INDEX] = 0
        model = Mock()
        model.device = torch.device("cpu")
        model.return_value = Mock(logits=torch.randn(batch_size, seq_len, vocab_size))
        model.side_effect = lambda **kw: model.return_value
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones(batch_size, seq_len),
        }
        provider = ARStepWiseScoreProvider()
        results = provider.get_per_position_scores(model, batch)
        assert len(results) == 1
        scores, fixation_steps = results[0]
        assert isinstance(scores, list)
        assert isinstance(fixation_steps, list)
        assert len(scores) == len(fixation_steps)
        assert all(0 <= p <= 1 for p in scores)
        assert fixation_steps == list(range(len(scores)))
        geom = sequence_probability_from_scores(scores)
        direct = evaluate_probability(model, batch)[0]["prob"]
        assert abs(geom - direct) < 1e-5


class TestBuildFixationLogitsFromRF:
    def test_single_sample(self):
        V, L, S = 3, 2, 4
        R = torch.randn(V, L, S)
        F = torch.tensor([0, 2])
        out = build_fixation_logits_from_R_F(R, F)
        assert out.shape == (1, L, V)
        assert torch.allclose(out[0, 0], R[:, 0, 0])
        assert torch.allclose(out[0, 1], R[:, 1, 2])

    def test_batch(self):
        B, V, L, S = 2, 4, 3, 5
        R = torch.randn(B, V, L, S)
        F = torch.randint(0, S, (B, L))
        out = build_fixation_logits_from_R_F(R, F)
        assert out.shape == (B, L, V)


class TestBuildEffectiveStepFixationLogits:
    """Effective step s_eff(ell, s) = min(s, F[ell]); at report_step S-1 equals full fixation."""

    def test_at_last_step_equals_fixation_logits(self):
        V, L, S = 4, 3, 5
        R = torch.randn(V, L, S)
        F = torch.tensor([0, 2, 4])
        full = build_fixation_logits_from_R_F(R, F).squeeze(0)
        at_s = build_effective_step_fixation_logits(R, F, S - 1).squeeze(0)
        assert full.shape == at_s.shape
        assert torch.allclose(full, at_s)

    def test_at_step_zero_uses_step_zero_for_all_positions(self):
        V, L, S = 4, 3, 5
        R = torch.randn(V, L, S)
        F = torch.tensor([1, 2, 4])
        at_0 = build_effective_step_fixation_logits(R, F, 0).squeeze(0)
        for ell in range(L):
            assert torch.allclose(at_0[ell], R[:, ell, 0])

    def test_provider_with_report_step_returns_scores_at_that_step(self):
        V, L, S = 10, 4, 6
        R = torch.randn(V, L, S)
        F = torch.tensor([0, 1, 3, 5])
        lab = torch.randint(0, V, (L,))
        provider = FixationStepWiseScoreProvider(logit_alignment="same_position")
        batch = {"labels": lab.unsqueeze(0)}
        no_report = provider.get_per_position_scores(
            {"R": R.unsqueeze(0), "F": F.unsqueeze(0)}, batch
        )
        at_step_0 = provider.get_per_position_scores(
            {"R": R.unsqueeze(0), "F": F.unsqueeze(0), "report_step": 0}, batch
        )
        at_step_last = provider.get_per_position_scores(
            {"R": R.unsqueeze(0), "F": F.unsqueeze(0), "report_step": S - 1}, batch
        )
        assert len(no_report) == 1 and len(at_step_0[0][0]) == L and len(at_step_last[0][0]) == L
        assert torch.allclose(
            torch.tensor(sequence_probability_from_scores(at_step_last[0][0])),
            torch.tensor(sequence_probability_from_scores(no_report[0][0])),
        )
        assert sequence_probability_from_scores(at_step_0[0][0]) != sequence_probability_from_scores(no_report[0][0]) or L == 0


class TestFixationStepWiseScoreProvider:
    """Hand-crafted R, F for causal and same_position alignment."""

    def test_causal_uses_prev_position_logits(self):
        # R: [1, V, L, S]. Position 0 at step 0: logits [1,0,0]; position 1 at step 1: [0.5,0.5,0]
        # Labels: token 0 at pos 0, token 1 at pos 1.
        # Causal: score at ell=0 uses logit_idx 0 -> [1,0,0] -> P(0)=1; score at ell=1 uses logit_idx 0 -> [1,0,0] -> P(1)=0
        V, L, S = 3, 2, 2
        R = torch.zeros(1, V, L, S)
        R[0, 0, 0, 0] = 10.0
        R[0, 1, 0, 0] = -10.0
        R[0, 2, 0, 0] = -10.0
        R[0, :, 1, 1] = 0.0
        F = torch.tensor([[0, 1]])
        labels = torch.tensor([[0, 1]])
        batch = {"labels": labels}
        provider = FixationStepWiseScoreProvider(logit_alignment="causal")
        results = provider.get_per_position_scores({"R": R, "F": F}, batch)
        assert len(results) == 1
        scores, _ = results[0]
        assert len(scores) == 2
        assert abs(scores[0] - 1.0) < 1e-5
        assert abs(scores[1] - 0.0) < 1e-5

    def test_same_position_uses_same_index_logits(self):
        # fixation_logits[0,0,:] = [10,-10,-10] -> P(0)=1; fixation_logits[0,1,:] = [0,0,0] -> P(1)=1/3
        V, L, S = 3, 2, 2
        R = torch.zeros(1, V, L, S)
        R[0, 0, 0, 0] = 10.0
        # R[0, :, 1, 1] left zero -> softmax = [1/3, 1/3, 1/3]
        F = torch.tensor([[0, 1]])
        labels = torch.tensor([[0, 1]])
        batch = {"labels": labels}
        provider = FixationStepWiseScoreProvider(logit_alignment="same_position")
        results = provider.get_per_position_scores({"R": R, "F": F}, batch)
        assert len(results) == 1
        scores, _ = results[0]
        assert len(scores) == 2
        assert abs(scores[0] - 1.0) < 1e-3
        assert abs(scores[1] - (1.0 / 3.0)) < 1e-3

    def test_invalid_alignment_raises(self):
        with pytest.raises(ValueError, match="logit_alignment"):
            FixationStepWiseScoreProvider(logit_alignment="invalid")


class TestExtractionStrengthFromFixation:
    """dLLM ES: smallest fraction of fixation steps to drop so remaining match target; result in [0,1]."""

    def test_all_match_es_one(self):
        L, V, S = 3, 10, 4
        fixation_logits = torch.zeros(L, V)
        fixation_logits[0, 1] = 10.0
        fixation_logits[1, 2] = 10.0
        fixation_logits[2, 3] = 10.0
        labels = torch.tensor([1, 2, 3])
        F = torch.tensor([0, 1, 2])
        es = extraction_strength_from_fixation(
            fixation_logits, labels, F, S, logit_alignment="same_position"
        )
        assert 0 <= es <= 1
        assert abs(es - 1.0) < 1e-5

    def test_one_wrong_at_late_step_es_half(self):
        L, V, S = 3, 10, 4
        fixation_logits = torch.zeros(L, V)
        fixation_logits[0, 1] = 10.0
        fixation_logits[1, 0] = 10.0
        fixation_logits[2, 3] = 10.0
        labels = torch.tensor([1, 2, 3])
        F = torch.tensor([0, 1, 2])
        es = extraction_strength_from_fixation(
            fixation_logits, labels, F, S, logit_alignment="same_position"
        )
        assert 0 <= es <= 1
        assert abs(es - 0.5) < 1e-5

    def test_empty_or_zero_steps_returns_zero(self):
        es = extraction_strength_from_fixation(
            torch.zeros(0, 5), torch.tensor([]), torch.tensor([]), 0
        )
        assert es == 0.0
