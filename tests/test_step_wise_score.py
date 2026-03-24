"""
Tests for step-wise score provider (generalized unlearning metrics).

- AR provider: same geometric mean as evaluate_probability for a batch.
- sequence_probability_from_scores and evaluate_probability_via_provider format.
"""

import logging

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
    evaluate_probability_confidence_ordered_from_fixation_logits,
    sequence_probability_from_scores,
    evaluate_probability_via_provider,
    extraction_strength_from_fixation,
    log_es_trajectory_diagnostics,
    trajectory_step_logits_to_prob_batch,
)


class TestSequenceProbabilityFromScores:
    def test_empty_returns_zero(self):
        assert sequence_probability_from_scores([]) == 0.0

    def test_single_score(self):
        assert abs(sequence_probability_from_scores([0.5]) - 0.5) < 1e-9

    def test_geometric_mean(self):
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

    def test_fixation_logits_longer_than_labels_no_index_error(self):
        """Regression: trajectory R/F can have one more position than labels; cap to min(L, lab.shape[1])."""
        V, L_lab, L_fix, S = 4, 86, 87, 10
        # R/F have L_fix positions (87); labels have L_lab (86)
        R = torch.randn(1, V, L_fix, S)
        F = torch.randint(0, S, (1, L_fix))
        labels = torch.randint(0, V, (1, L_lab))
        labels[0, 0] = IGNORE_INDEX
        batch = {"labels": labels}
        provider = FixationStepWiseScoreProvider(logit_alignment="causal")
        results = provider.get_per_position_scores({"R": R, "F": F}, batch)
        assert len(results) == 1
        scores, fixation_steps = results[0]
        # Should cap to L_lab positions; one position is ignore_index so L_lab - 1 scores
        assert len(scores) == L_lab - 1
        assert len(fixation_steps) == L_lab - 1
        assert all(0 <= p <= 1 for p in scores)


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

    def test_fixation_logits_longer_than_labels_no_index_error(self):
        """Regression: fixation logits can have more positions than labels; cap to min(L, labels.shape[0])."""
        L_fix, L_lab, V, S = 12, 11, 4, 10
        fixation_logits = torch.randn(L_fix, V)
        labels = torch.randint(0, V, (L_lab,))
        F = torch.randint(0, S, (L_fix,))
        es = extraction_strength_from_fixation(
            fixation_logits, labels, F, S, logit_alignment="causal"
        )
        assert 0 <= es <= 1

    def test_audit_text_decode_skips_ignore_index(self, caplog):
        """Labels often include IGNORE_INDEX (-100); audit decode must not error."""
        caplog.set_level(logging.DEBUG, logger="evaluator")

        class _Tok:
            def decode(self, ids, skip_special_tokens=True):
                for i in ids:
                    if i < 0:
                        raise ValueError("negative id")
                return "x"

        L, V, S = 2, 10, 4
        fixation_logits = torch.zeros(L, V)
        fixation_logits[0, 1] = 10.0
        fixation_logits[1, 2] = 10.0
        labels = torch.tensor([IGNORE_INDEX, 2])
        F = torch.tensor([0, 1])
        extraction_strength_from_fixation(
            fixation_logits,
            labels,
            F,
            S,
            logit_alignment="same_position",
            audit=True,
            audit_ctx={
                "sample_idx": "0",
                "step": 0,
                "traj_name": "steps",
                "view": "full",
                "tokenizer": _Tok(),
            },
        )
        joined = "\n".join(r.message for r in caplog.records)
        assert "text_decode failed" not in joined
        assert "text_decode pred_argmax_seq=" in joined

    def test_audit_logs_branches_and_best_t(self, caplog):
        """DEBUG audit: EXTRACTION_STRENGTH_AUDIT traces branches, loop_t, result."""
        caplog.set_level(logging.DEBUG, logger="evaluator")
        L, V, S = 3, 10, 4
        fixation_logits = torch.zeros(L, V)
        fixation_logits[0, 1] = 10.0
        fixation_logits[1, 0] = 10.0
        fixation_logits[2, 3] = 10.0
        labels = torch.tensor([1, 2, 3])
        F = torch.tensor([0, 1, 2])
        es = extraction_strength_from_fixation(
            fixation_logits,
            labels,
            F,
            S,
            logit_alignment="same_position",
            audit=True,
            audit_compact=False,
            audit_ctx={"sample_idx": "0", "step": 7, "traj_name": "steps", "view": "full"},
        )
        assert abs(es - 0.5) < 1e-5
        joined = "\n".join(r.message for r in caplog.records)
        assert "EXTRACTION_STRENGTH_AUDIT" in joined
        assert "enter L=" in joined
        assert "loop_t t=0 FAIL" in joined
        assert "loop_t t=2 PASS" in joined
        assert "result best_t=2" in joined

    def test_audit_compact_omits_loop_and_enter(self, caplog):
        """Compact audit: preds_summary + result only (no per-t loop spam)."""
        caplog.set_level(logging.DEBUG, logger="evaluator")
        L, V, S = 3, 10, 4
        fixation_logits = torch.zeros(L, V)
        fixation_logits[0, 1] = 10.0
        fixation_logits[1, 0] = 10.0
        fixation_logits[2, 3] = 10.0
        labels = torch.tensor([1, 2, 3])
        F = torch.tensor([0, 1, 2])
        extraction_strength_from_fixation(
            fixation_logits,
            labels,
            F,
            S,
            logit_alignment="same_position",
            audit=True,
            audit_compact=True,
            audit_ctx={"sample_idx": "0", "step": 7, "traj_name": "steps", "view": "full"},
        )
        joined = "\n".join(r.message for r in caplog.records)
        assert "loop_t t=" not in joined
        assert "enter L=" not in joined
        assert "preds_summary" in joined
        assert "result best_t=" in joined

    def test_diag_out_populated(self):
        L, V, S = 3, 10, 4
        fixation_logits = torch.zeros(L, V)
        fixation_logits[0, 1] = 10.0
        fixation_logits[1, 0] = 10.0
        fixation_logits[2, 3] = 10.0
        labels = torch.tensor([1, 2, 3])
        F = torch.tensor([0, 1, 2])
        d: dict = {}
        es = extraction_strength_from_fixation(
            fixation_logits,
            labels,
            F,
            S,
            logit_alignment="same_position",
            diag_out=d,
        )
        assert abs(es - 0.5) < 1e-5
        assert d["best_t"] == 2
        assert d["n_valid"] == 3
        assert d["argmax_mismatch_on_valid"] == 1

    def test_traj_diag_warns_missing_report_step(self, caplog):
        caplog.set_level(logging.WARNING, logger="evaluator")
        V, L, S = 4, 3, 5
        R = torch.randn(V, L, S)
        F = torch.tensor([0, 1, 2])
        fl = build_fixation_logits_from_R_F(R, F).squeeze(0)
        log_es_trajectory_diagnostics(
            R,
            F,
            S,
            None,
            fl,
            batch_idx=0,
            sample_idx="0",
            traj_name="steps",
            view="full",
            step_index=0,
            last_step_index=4,
            audit_runtime=False,
        )
        joined = "\n".join(r.message for r in caplog.records)
        assert "EXTRACTION_STRENGTH_TRAJ_DIAG missing report_step" in joined

    def test_traj_diag_warns_flat_r_on_step_zero(self, caplog):
        caplog.set_level(logging.WARNING, logger="evaluator")
        V, L, S = 4, 3, 8
        R = torch.ones(V, L, S)
        F = torch.tensor([1, 2, 3])
        fl = build_effective_step_fixation_logits(R, F, 0).squeeze(0)
        log_es_trajectory_diagnostics(
            R,
            F,
            S,
            0,
            fl,
            batch_idx=0,
            sample_idx="0",
            traj_name="steps",
            view="full",
            step_index=0,
            last_step_index=7,
            audit_runtime=False,
        )
        joined = "\n".join(r.message for r in caplog.records)
        assert "near) constant along trajectory axis" in joined

    def test_traj_diag_per_position_vocab_change_counts(self, caplog):
        """DEBUG detail: per-position counts of vocab logits that changed vs prev."""
        caplog.set_level(logging.DEBUG, logger="evaluator")
        V, L, S = 5, 4, 8
        torch.manual_seed(0)
        R = torch.randn(V, L, S)
        F = torch.tensor([1, 2, 3, 1])
        fl_prev = build_effective_step_fixation_logits(R, F, 1).squeeze(0).float()
        fl_now = fl_prev.clone()
        fl_now[0] = fl_now[0] + 1.0
        fl_now[2, 3] = fl_now[2, 3] + 5.0
        log_es_trajectory_diagnostics(
            R,
            F,
            S,
            1,
            fl_now,
            batch_idx=0,
            sample_idx="0",
            traj_name="steps",
            view="full",
            step_index=1,
            last_step_index=7,
            audit_runtime=True,
            prev_fixation_logits=fl_prev,
            es_diag_every_step=True,
            fix_vs_prev_count_atol=0.5,
            es_diag_per_position=True,
            fix_vs_prev_max_positions_list=256,
            es_score=0.25,
            best_t=6,
        )
        joined = "\n".join(r.message for r in caplog.records)
        assert " es=" in joined and "best_t=6" in joined
        assert "fix_vs_prev_n_pos_vocab_changed=2" in joined
        assert "fix_vs_prev_max_vocab_dims_one_pos=5" in joined
        assert "EXTRACTION_STRENGTH_TRAJ_DIAG per_position" in joined
        assert "fix_vs_prev_pos_n_changed_vocab=0:5,2:1" in joined


def test_trajectory_step_logits_to_prob_batch_shape() -> None:
    V, L = 5, 7
    logits_vl = torch.randn(V, L)
    bat = trajectory_step_logits_to_prob_batch(logits_vl)
    assert bat.shape == (1, L, V)


def test_confidence_ordered_from_fixation_geometric_mean_ordering() -> None:
    """Descending sort of per-position probs then geom mean (two valid positions)."""
    B, L, V = 1, 4, 8
    labels = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    labels[0, 1] = 2
    labels[0, 2] = 3
    fl = torch.zeros(B, L, V)
    # causal: ell=1 uses logit_idx 0; ell=2 uses logit_idx 1
    fl[0, 0, 2] = 10.0
    fl[0, 1, 3] = 5.0
    out = evaluate_probability_confidence_ordered_from_fixation_logits(
        fl, labels, logit_alignment="causal", ignore_index=IGNORE_INDEX
    )
    assert out[0]["prob"] is not None
    p0 = torch.softmax(fl[0, 0].float(), dim=-1)[2].item()
    p1 = torch.softmax(fl[0, 1].float(), dim=-1)[3].item()
    high, low = max(p0, p1), min(p0, p1)
    expected = float((high * low) ** 0.5)
    assert abs(out[0]["prob"] - expected) < 1e-5
