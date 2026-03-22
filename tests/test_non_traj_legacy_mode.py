"""US2: explicit use_generalized_sequence_probability=false selects legacy probability path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics import memorization
from evals.metrics.utils import evaluate_probability


class _TinyDs(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, i: int) -> dict:
        return {
            "input_ids": torch.ones(8, dtype=torch.long),
            "labels": torch.tensor([-100, -100, 1, 2, 3, 4, 5, 6], dtype=torch.long),
        }


def _collate(batch: list) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def test_explicit_false_uses_evaluate_probability(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list = []

    def _capture_run_batchwise(model, dataloader, batch_eval_fn, fun_args, desc):
        captured.append(batch_eval_fn)
        return {"0": {"prob": 0.5, "avg_loss": 0.693}}

    monkeypatch.setattr(memorization, "run_batchwise_evals", _capture_run_batchwise)
    model = object()
    memorization.probability._metric_fn(
        model,
        data=_TinyDs(),
        collators=_collate,
        batch_size=1,
        use_generalized_sequence_probability=False,
    )
    assert len(captured) == 1
    assert captured[0] is evaluate_probability


def test_explicit_true_non_diffusion_uses_ar_provider_fn(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list = []

    def _capture_run_batchwise(model, dataloader, batch_eval_fn, fun_args, desc):
        captured.append(batch_eval_fn)
        return {"0": {"prob": 0.5, "avg_loss": 0.693}}

    monkeypatch.setattr(memorization, "run_batchwise_evals", _capture_run_batchwise)
    model = object()
    memorization.probability._metric_fn(
        model,
        data=_TinyDs(),
        collators=_collate,
        batch_size=1,
        use_generalized_sequence_probability=True,
    )
    assert len(captured) == 1
    assert captured[0] is memorization._probability_batch_fn_ar_provider
