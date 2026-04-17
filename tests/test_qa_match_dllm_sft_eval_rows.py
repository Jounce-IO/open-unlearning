"""QADataset ``match_dllm_sft_eval_rows`` matches dllm ``load_dataset_tofu`` 95/5 slice."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from data.qa import QADataset  # noqa: E402


@pytest.fixture
def tofu_like_rows():
    n = 200
    return Dataset.from_dict(
        {
            "question": [f"q{i}" for i in range(n)],
            "answer": [f"a{i}" for i in range(n)],
        }
    )


def test_match_dllm_sft_eval_rows_truncates_train_split(tofu_like_rows):
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = [1, 2, 3]
    tokenizer.__len__ = MagicMock(return_value=1000)
    hf_args = {"path": "locuslab/TOFU", "name": "forget01", "split": "train"}

    with patch("data.qa.load_hf_dataset", return_value=tofu_like_rows):
        with patch("data.qa.add_dataset_index", side_effect=lambda x: x):
            ds = QADataset(
                hf_args=hf_args,
                template_args={},
                tokenizer=tokenizer,
                match_dllm_sft_eval_rows=True,
            )
    # 200 * (1 - 0.05) = 190
    assert len(ds) == 190


def test_match_false_uses_full_split(tofu_like_rows):
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = [1, 2, 3]
    tokenizer.__len__ = MagicMock(return_value=1000)
    hf_args = {"path": "locuslab/TOFU", "name": "forget01", "split": "train"}

    with patch("data.qa.load_hf_dataset", return_value=tofu_like_rows):
        with patch("data.qa.add_dataset_index", side_effect=lambda x: x):
            ds = QADataset(
                hf_args=hf_args,
                template_args={},
                tokenizer=tokenizer,
                match_dllm_sft_eval_rows=False,
            )
    assert len(ds) == 200


def test_match_true_with_omegaconf_hf_args(tofu_like_rows):
    omegaconf = pytest.importorskip("omegaconf")
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = [1, 2, 3]
    tokenizer.__len__ = MagicMock(return_value=1000)
    hf_args = omegaconf.OmegaConf.create(
        {"path": "locuslab/TOFU", "name": "forget01", "split": "train"}
    )

    with patch("data.qa.load_hf_dataset", return_value=tofu_like_rows):
        with patch("data.qa.add_dataset_index", side_effect=lambda x: x):
            ds = QADataset(
                hf_args=hf_args,
                template_args={},
                tokenizer=tokenizer,
                match_dllm_sft_eval_rows=True,
            )
    assert len(ds) == 190
