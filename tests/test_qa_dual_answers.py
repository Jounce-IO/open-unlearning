"""
Regression tests for QAwithDualAnswersDataset when correct/wrong answer keys return a list.

TOFU forget*_perturbed has perturbed_answer as a list of 5 strings; passing the list
to the tokenizer produced invalid labels_wrong and "no scores from fixation provider".
We normalize to the first element. No model or GPU required.
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
    """When the dataset returns a list for wrong_answer (e.g. TOFU perturbed_answer), we use first element."""

    def test_process_sample_called_with_string_when_wrong_answer_is_list(self):
        """Regression: _process_sample must receive a string for wrong answer, not a list."""
        hf_args = {"path": "locuslab/TOFU", "name": "forget01_perturbed", "split": "train"}
        template_args = {}
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = [1, 2, 3]
        # Minimal tokenizer behavior so preprocess_chat_instance can run
        tokenizer.__len__ = MagicMock(return_value=1000)

        with patch("data.qa.load_hf_dataset") as load_mock:
            # In-memory data: one row with list for perturbed_answer (TOFU-style)
            load_mock.return_value = MagicMock(
                __len__=lambda _: 1,
                __getitem__=lambda self, idx: {
                    "question": "What is the author's name?",
                    "paraphrased_answer": "Basil Mahfouz Al-Kuwaiti",
                    "perturbed_answer": [
                        "Gregor Mendel Al-Kuwaiti",
                        "Other wrong 1",
                        "Other wrong 2",
                        "Other wrong 3",
                        "Other wrong 4",
                    ],
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

        # _process_sample should have been called twice: correct, then wrong
        assert mock_process.call_count == 2
        # First call: correct answer (string)
        assert mock_process.call_args_list[0].kwargs["answer"] == "Basil Mahfouz Al-Kuwaiti"
        # Second call: wrong answer must be a string (first element of list), not the list
        wrong_answer_passed = mock_process.call_args_list[1].kwargs["answer"]
        assert isinstance(wrong_answer_passed, str), "wrong answer must be string, not list"
        assert wrong_answer_passed == "Gregor Mendel Al-Kuwaiti"

    def test_process_sample_called_with_string_when_correct_answer_is_list(self):
        """If correct_answer key ever returns a list, we use first element too."""
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
