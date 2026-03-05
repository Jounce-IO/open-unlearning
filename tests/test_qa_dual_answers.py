"""
Tests for QAwithDualAnswersDataset when correct/wrong answer keys return a list or string.

When wrong_answer is a list of N options (e.g. TOFU perturbed_answer), we build N wrong
label tensors per sample and return labels_wrong as a list of N tensors. When it is a
string, we return a single tensor (backward compatible). No model or GPU required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

import sys
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from data.qa import _ensure_single_answer, QAwithDualAnswersDataset


class TestEnsureSingleAnswer:
    """Unit tests for _ensure_single_answer (list -> first element, string -> passthrough)."""

    def test_returns_first_element_when_list(self):
        out = _ensure_single_answer(["a", "b", "c"], "wrong_key", 0, 0)
        assert out == "a"

    def test_returns_same_when_string(self):
        out = _ensure_single_answer("single answer", "wrong_key", 0, 0)
        assert out == "single answer"

    def test_raises_when_empty_list(self):
        with pytest.raises(ValueError, match="Empty list for wrong_key"):
            _ensure_single_answer([], "wrong_key", 0, 0)


class TestQAwithDualAnswersDatasetListAnswer:
    """When wrong_answer is a list we build N wrong label tensors; when string we return one."""

    def test_labels_wrong_is_list_of_n_tensors_when_wrong_answer_is_list(self):
        """When wrong_answer is a list of N strings, labels_wrong is a list of N tensors."""
        hf_args = {"path": "locuslab/TOFU", "name": "forget01_perturbed", "split": "train"}
        template_args = {}
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = [1, 2, 3]
        tokenizer.__len__ = MagicMock(return_value=1000)

        with patch("data.qa.load_hf_dataset") as load_mock:
            load_mock.return_value = MagicMock(
                __len__=lambda _: 1,
                __getitem__=lambda self, idx: {
                    "question": "What is the author's name?",
                    "paraphrased_answer": "Basil Mahfouz Al-Kuwaiti",
                    "perturbed_answer": ["Wrong1", "Wrong2", "Wrong3"],
                    "index": 0,
                },
            )
            with patch("data.qa.add_dataset_index", side_effect=lambda x: x):
                def fake_process(*_args, **_kwargs):
                    return {
                        "input_ids": torch.tensor([[1, 2, 3]]),
                        "labels": torch.tensor([[1, 2, 3]]),
                        "attention_mask": torch.tensor([[1, 1, 1]]),
                    }

                with patch.object(
                    QAwithDualAnswersDataset,
                    "_process_sample",
                    side_effect=fake_process,
                ) as mock_process:
                    ds = QAwithDualAnswersDataset(
                        correct_answer_key="paraphrased_answer",
                        wrong_answer_key="perturbed_answer",
                        hf_args=hf_args,
                        template_args=template_args,
                        tokenizer=tokenizer,
                    )
                    item = ds[0]

        assert mock_process.call_count == 1 + 3
        assert mock_process.call_args_list[0].kwargs["answer"] == "Basil Mahfouz Al-Kuwaiti"
        for k in range(3):
            assert mock_process.call_args_list[1 + k].kwargs["answer"] == f"Wrong{k+1}"
        assert isinstance(item["labels_wrong"], list)
        assert len(item["labels_wrong"]) == 3
        for t in item["labels_wrong"]:
            assert isinstance(t, torch.Tensor)

    def test_labels_wrong_is_single_tensor_when_wrong_answer_is_string(self):
        """When wrong_answer is a string, labels_wrong is a single tensor (backward compat)."""
        with patch("data.qa.load_hf_dataset") as load_mock:
            load_mock.return_value = MagicMock(
                __len__=lambda _: 1,
                __getitem__=lambda self, idx: {
                    "question": "Q?",
                    "paraphrased_answer": "Correct",
                    "perturbed_answer": "Wrong",
                    "index": 0,
                },
            )
            def fake_process(*_args, **_kwargs):
                return {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "labels": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }

            with patch("data.qa.add_dataset_index", side_effect=lambda x: x):
                with patch.object(
                    QAwithDualAnswersDataset,
                    "_process_sample",
                    side_effect=fake_process,
                ):
                    ds = QAwithDualAnswersDataset(
                        correct_answer_key="paraphrased_answer",
                        wrong_answer_key="perturbed_answer",
                        hf_args={"path": "x", "name": "y", "split": "train"},
                        template_args={},
                        tokenizer=MagicMock(),
                    )
                    item = ds[0]

        assert isinstance(item["labels_wrong"], torch.Tensor)

    def test_process_sample_called_with_string_when_correct_answer_is_list(self):
        """If correct_answer key returns a list, we use first element (single correct per sample)."""
        with patch("data.qa.load_hf_dataset") as load_mock:
            load_mock.return_value = MagicMock(
                __len__=lambda _: 1,
                __getitem__=lambda self, idx: {
                    "question": "Q?",
                    "paraphrased_answer": ["First correct", "Second correct"],
                    "perturbed_answer": "Wrong answer",
                    "index": 0,
                },
            )
            def fake_process(*_args, **_kwargs):
                return {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "labels": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }

            with patch("data.qa.add_dataset_index", side_effect=lambda x: x):
                with patch.object(
                    QAwithDualAnswersDataset,
                    "_process_sample",
                    side_effect=fake_process,
                ) as mock_process:
                    ds = QAwithDualAnswersDataset(
                        correct_answer_key="paraphrased_answer",
                        wrong_answer_key="perturbed_answer",
                        hf_args={"path": "x", "name": "y", "split": "train"},
                        template_args={},
                        tokenizer=MagicMock(),
                    )
                    ds[0]

        assert mock_process.call_count == 2
        correct_passed = mock_process.call_args_list[0].kwargs["answer"]
        assert isinstance(correct_passed, str)
        assert correct_passed == "First correct"
