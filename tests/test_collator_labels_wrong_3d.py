"""
Test collator when labels_wrong is a list of N tensors per sample (multi-option wrong answers).
Output should be [B, max_N, L] with IGNORE_INDEX padding for missing options.
"""

from pathlib import Path
import sys
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from data.collators import DataCollatorForSupervisedDataset
from data.utils import IGNORE_INDEX


def test_collator_labels_wrong_list_produces_3d():
    tokenizer = type("T", (), {"pad_token_id": 0})()
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, padding_side="right")
    # Sample 0: 3 wrong options (3 tensors); Sample 1: 2 wrong options
    instances = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([1, 2, 3]),
            "labels_correct": torch.tensor([1, 2, 3]),
            "labels_wrong": [
                torch.tensor([10, 11]),
                torch.tensor([20, 21, 22]),
                torch.tensor([30]),
            ],
            "attention_mask": torch.tensor([1, 1, 1]),
        },
        {
            "input_ids": torch.tensor([1, 2]),
            "labels": torch.tensor([1, 2]),
            "labels_correct": torch.tensor([1, 2]),
            "labels_wrong": [
                torch.tensor([40, 41]),
                torch.tensor([50]),
            ],
            "attention_mask": torch.tensor([1, 1]),
        },
    ]
    out = collator(instances)
    assert "labels_wrong" in out
    lw = out["labels_wrong"]
    assert lw.dim() == 3
    assert lw.shape[0] == 2
    assert lw.shape[1] == 3
    assert lw.shape[2] >= 3
    assert (lw[1, 2, :] == IGNORE_INDEX).all()


def test_collator_labels_wrong_single_tensor_unchanged():
    tokenizer = type("T", (), {"pad_token_id": 0})()
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, padding_side="right")
    instances = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([1, 2, 3]),
            "labels_correct": torch.tensor([1, 2, 3]),
            "labels_wrong": torch.tensor([10, 11, 12]),
            "attention_mask": torch.tensor([1, 1, 1]),
        },
    ]
    out = collator(instances)
    assert out["labels_wrong"].dim() == 2
