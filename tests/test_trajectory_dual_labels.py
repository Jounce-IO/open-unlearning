"""
Tests for trajectory metrics when metrics require dual-answer labels (labels_correct, labels_wrong).

Reproduces the "labels missing" bug when the batch does not provide labels_correct/labels_wrong,
and validates that when the batch (and thus batch_template) has them, pre_compute returns
valid results. No model or GPU required; uses mocks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

import sys
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.trajectory_metrics import (
    _compute_pre_compute_metrics_at_step,
    IGNORE_INDEX,
)


def _make_batch_template_without_dual_labels(L: int = 10, vocab_size: int = 100):
    """Batch template as built when dataset does NOT provide labels_correct/labels_wrong (e.g. QADataset)."""
    return {
        "input_ids": torch.randint(0, vocab_size, (1, L)),
        "labels": torch.randint(0, vocab_size, (1, L)),
        "attention_mask": torch.ones((1, L), dtype=torch.long),
        "index": torch.tensor([0], dtype=torch.long),
    }


def _make_batch_template_with_dual_labels(L: int = 10, vocab_size: int = 100):
    """Batch template as built when dataset provides labels_correct and labels_wrong (QAwithDualAnswersDataset)."""
    base = _make_batch_template_without_dual_labels(L, vocab_size)
    base["labels_correct"] = torch.randint(0, vocab_size, (1, L))
    base["labels_wrong"] = torch.randint(0, vocab_size, (1, L))
    return base


def _truth_ratio_pre_compute_config():
    """Pre-compute config for truth_ratio: correct and wrong with labels_field."""
    return {
        "correct": {
            "handler": "probability",
            "access_key": "correct",
            "labels_field": "labels_correct",
        },
        "wrong": {
            "handler": "probability",
            "access_key": "wrong",
            "labels_field": "labels_wrong",
        },
    }


class TestTrajectoryDualLabelsReproduceBug:
    """Reproduce: when batch_template lacks labels_correct/labels_wrong, pre_compute returns None."""

    def test_pre_compute_returns_none_when_labels_correct_missing(self):
        """With batch_template without labels_correct/labels_wrong, generalized path returns agg_value None."""
        L, V = 12, 100
        batch_template = _make_batch_template_without_dual_labels(L, V)
        pre_compute_config = _truth_ratio_pre_compute_config()
        trajectory_config = {
            "use_generalized_sequence_probability": True,
            "logit_alignment": "causal",
        }
        sample_traj = {
            "R": torch.randn(1, L, 8),
            "F": torch.randint(0, 8, (1, L)),
        }
        tokenizer = Mock()
        sample_labels = torch.randint(0, V, (L,))
        sample_input_ids = torch.randint(0, V, (L,))
        logits = torch.randn(V, L)

        with patch(
            "evals.metrics.trajectory_metrics.FixationStepWiseScoreProvider",
            spec=True,
        ) as mock_provider_cls:
            # Provider should not be used for the None-lab branch; we only need lab = None
            mock_provider_cls.return_value.get_per_position_scores.return_value = []

            result = _compute_pre_compute_metrics_at_step(
                pre_compute_config=pre_compute_config,
                logits=logits,
                batch_template=batch_template,
                tokenizer=tokenizer,
                sample_labels=sample_labels,
                sample_input_ids=sample_input_ids,
                sample_prompt_len=0,
                sample_idx="0",
                trajectory_config=trajectory_config,
                sample_traj=sample_traj,
                step=0,
            )

        assert "correct" in result
        assert "wrong" in result
        assert result["correct"]["agg_value"] is None
        assert result["wrong"]["agg_value"] is None
        assert result["correct"]["value_by_index"]["0"]["avg_loss"] is None
        assert result["wrong"]["value_by_index"]["0"]["avg_loss"] is None

    def test_pre_compute_logs_labels_missing_when_dual_labels_absent(self, caplog):
        """When labels_field is set but key missing in batch_template, we log 'labels missing'."""
        L, V = 8, 50
        batch_template = _make_batch_template_without_dual_labels(L, V)
        pre_compute_config = {
            "correct": {
                "handler": "probability",
                "access_key": "correct",
                "labels_field": "labels_correct",
            },
        }
        trajectory_config = {"use_generalized_sequence_probability": True, "logit_alignment": "causal"}
        sample_traj = {"R": torch.randn(1, L, 4), "F": torch.randint(0, 4, (1, L))}
        tokenizer = Mock()
        logits = torch.randn(V, L)

        with caplog.at_level(logging.INFO):
            _compute_pre_compute_metrics_at_step(
                pre_compute_config=pre_compute_config,
                logits=logits,
                batch_template=batch_template,
                tokenizer=tokenizer,
                sample_labels=torch.zeros(L, dtype=torch.long),
                sample_input_ids=torch.zeros(L, dtype=torch.long),
                sample_prompt_len=0,
                sample_idx="0",
                trajectory_config=trajectory_config,
                sample_traj=sample_traj,
                step=1,
            )

        assert "labels missing" in caplog.text
        assert "labels_correct" in caplog.text or "labels_field" in caplog.text


class TestTrajectoryDualLabelsFixed:
    """Validate: when batch_template has labels_correct/labels_wrong, pre_compute returns non-None (with mocked provider)."""

    def test_pre_compute_returns_value_when_dual_labels_present(self):
        """With batch_template containing labels_correct (and mocked provider), generalized path returns non-None agg_value."""
        L, V = 10, 100
        batch_template = _make_batch_template_with_dual_labels(L, V)
        pre_compute_config = {
            "correct": {
                "handler": "probability",
                "access_key": "correct",
                "labels_field": "labels_correct",
            },
        }
        trajectory_config = {"use_generalized_sequence_probability": True, "logit_alignment": "causal"}
        sample_traj = {
            "R": torch.randn(1, L, 6),
            "F": torch.randint(0, 6, (1, L)),
        }
        tokenizer = Mock()
        logits = torch.randn(V, L)
        # Per-position scores that yield a valid probability (geometric mean)
        mock_scores = [0.5, 0.4, 0.6]

        with patch(
            "evals.metrics.trajectory_metrics.FixationStepWiseScoreProvider",
            spec=True,
        ) as mock_provider_cls:
            mock_instance = Mock()
            mock_instance.get_per_position_scores.return_value = [(mock_scores, None)]
            mock_provider_cls.return_value = mock_instance

            result = _compute_pre_compute_metrics_at_step(
                pre_compute_config=pre_compute_config,
                logits=logits,
                batch_template=batch_template,
                tokenizer=tokenizer,
                sample_labels=batch_template["labels_correct"].squeeze(0),
                sample_input_ids=batch_template["input_ids"].squeeze(0),
                sample_prompt_len=0,
                sample_idx="0",
                trajectory_config=trajectory_config,
                sample_traj=sample_traj,
                step=0,
            )

        assert "correct" in result
        assert result["correct"]["agg_value"] is not None
        assert result["correct"]["value_by_index"]["0"]["avg_loss"] is not None
        assert result["correct"]["value_by_index"]["0"]["prob"] is not None

    def test_pre_compute_both_correct_and_wrong_when_batch_has_both(self):
        """When batch_template has both labels_correct and labels_wrong, both pre_compute entries get values."""
        L, V = 8, 80
        batch_template = _make_batch_template_with_dual_labels(L, V)
        pre_compute_config = _truth_ratio_pre_compute_config()
        trajectory_config = {"use_generalized_sequence_probability": True, "logit_alignment": "causal"}
        sample_traj = {"R": torch.randn(1, L, 4), "F": torch.randint(0, 4, (1, L))}
        tokenizer = Mock()
        logits = torch.randn(V, L)
        mock_scores = [0.3, 0.5]

        with patch(
            "evals.metrics.trajectory_metrics.FixationStepWiseScoreProvider",
            spec=True,
        ) as mock_provider_cls:
            mock_instance = Mock()
            mock_instance.get_per_position_scores.return_value = [(mock_scores, None)]
            mock_provider_cls.return_value = mock_instance

            result = _compute_pre_compute_metrics_at_step(
                pre_compute_config=pre_compute_config,
                logits=logits,
                batch_template=batch_template,
                tokenizer=tokenizer,
                sample_labels=batch_template["labels_correct"].squeeze(0),
                sample_input_ids=batch_template["input_ids"].squeeze(0),
                sample_prompt_len=0,
                sample_idx="0",
                trajectory_config=trajectory_config,
                sample_traj=sample_traj,
                step=2,
            )

        assert result["correct"]["agg_value"] is not None
        assert result["wrong"]["agg_value"] is not None


class TestTrajectoryConfigDualAnswerDataset:
    """Config-level: eval configs that include truth_ratio must use a dataset that provides labels_correct/labels_wrong."""

    def test_tofu_trajectory_all_forget_dataset_has_dual_answer_handler(self):
        """TOFU trajectory_all uses a forget dataset that provides dual labels (QAwithDualAnswersDataset or equivalent)."""
        configs_dir = repo_root / "configs" / "eval" / "tofu_metrics"
        trajectory_all_yaml = configs_dir / "trajectory_all.yaml"
        if not trajectory_all_yaml.exists():
            pytest.skip("trajectory_all.yaml not found")
        text = trajectory_all_yaml.read_text()
        # Must reference dual-answer dataset for forget when truth_ratio is in metrics:
        # either handler QAwithDualAnswersDataset for TOFU_QA_forget, or correct_answer_key/wrong_answer_key
        has_dual = (
            "QAwithDualAnswersDataset" in text
            or "correct_answer_key" in text
            or "wrong_answer_key" in text
        )
        assert has_dual, (
            "trajectory_all.yaml must use a dataset that provides labels_correct/labels_wrong "
            "(e.g. QAwithDualAnswersDataset with correct_answer_key/wrong_answer_key) for the forget set when truth_ratio is in metrics"
        )

    def test_tofu_trajectory_forget_quality_uses_dual_answer_dataset(self):
        """trajectory_forget_quality uses TOFU_QA_forget_para_pert (dual-answer) for forget_truth_ratio."""
        configs_dir = repo_root / "configs" / "eval" / "tofu_metrics"
        yaml_path = configs_dir / "trajectory_forget_quality.yaml"
        if not yaml_path.exists():
            pytest.skip("trajectory_forget_quality.yaml not found")
        text = yaml_path.read_text()
        assert "TOFU_QA_forget_para_pert" in text or "correct_answer_key" in text
