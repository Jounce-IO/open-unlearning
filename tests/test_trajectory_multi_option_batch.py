"""Multi-option QADataset batches (perturbed_answer list) for trajectory eval."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from data.qa import QADataset
from evals.metrics.trajectory_metrics import (
    _clone_step_values_prob_layers,
    _is_multi_option_collated_batch,
    _iter_trajectory_mini_batches,
    _merge_pert_option_probability_layers,
)


class _ListAnswerDataset(QADataset):
    """Minimal dataset returning list-valued perturbed_answer (TOFU-style)."""

    def __init__(self) -> None:
        self.data = [
            {
                "question": "Q0?",
                "answer": "correct0",
                "perturbed_answer": ["w0a", "w0b", "w0c"],
            },
            {
                "question": "Q1?",
                "answer": "correct1",
                "perturbed_answer": ["w1a", "w1b"],
            },
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data[int(idx)]
        if isinstance(row["perturbed_answer"], list):
            return {
                i: {
                    **row,
                    "answer": row["perturbed_answer"][i],
                }
                for i in range(len(row["perturbed_answer"]))
            }
        return super().__getitem__(idx)


def test_iter_trajectory_mini_batches_single():
    batch = {
        "input_ids": torch.zeros(2, 4, dtype=torch.long),
        "labels": torch.full((2, 4), -100),
    }
    out = list(_iter_trajectory_mini_batches(batch))
    assert len(out) == 1
    assert "input_ids" in out[0]


def test_iter_trajectory_mini_batches_nested():
    batch = {
        "0": {"input_ids": torch.zeros(2, 4), "index": torch.tensor([0, 1])},
        "1": {"input_ids": torch.ones(2, 4), "index": torch.tensor([0, 1])},
    }
    assert _is_multi_option_collated_batch(batch)
    out = list(_iter_trajectory_mini_batches(batch))
    assert len(out) == 2
    assert out[0]["input_ids"].sum() == 0
    assert out[1]["input_ids"].sum() == 8


def test_collator_style_nested_batch_no_top_level_input_ids():
    """Nested batch shape produced when QADataset rows have list ``answer_key`` values."""
    batch = {
        "0": {"input_ids": torch.zeros(2, 4, dtype=torch.long), "index": torch.tensor([0, 1])},
        "1": {"input_ids": torch.ones(2, 4, dtype=torch.long), "index": torch.tensor([0, 1])},
        "2": {"input_ids": torch.full((2, 4), 2, dtype=torch.long), "index": torch.tensor([0, 1])},
    }
    assert _is_multi_option_collated_batch(batch)
    assert "input_ids" not in batch
    mini = list(_iter_trajectory_mini_batches(batch))
    assert len(mini) == 3
    for mb in mini:
        assert mb["input_ids"].shape[0] == 2


def test_merge_pert_option_mean_and_sum():
    layer0 = {
        "full": {
            "steps": {
                0: {"probability": [0.2, 0.4]},
            }
        }
    }
    layer1 = {
        "full": {
            "steps": {
                0: {"probability": [0.4, 0.6]},
            }
        }
    }
    layer2 = {
        "full": {
            "steps": {
                0: {"probability": [0.6, 0.8]},
            }
        }
    }
    main: dict = {"full": {"steps": {}}}
    _merge_pert_option_probability_layers(
        main, [layer0, layer1, layer2], emit_wrong_sum=True
    )
    assert main["full"]["steps"][0]["probability"] == pytest.approx([0.4, 0.6])
    assert main["full"]["steps"][0]["probability_wrong_sum"] == pytest.approx([1.2, 1.8])


def test_clone_step_values_prob_layers():
    layer = {
        "full": {
            "steps": {
                1: {"probability": [0.1], "probability_wrong_sum": [0.3]},
            }
        }
    }
    cloned = _clone_step_values_prob_layers(layer)
    cloned["full"]["steps"][1]["probability"].append(99.0)
    assert layer["full"]["steps"][1]["probability"] == [0.1]
