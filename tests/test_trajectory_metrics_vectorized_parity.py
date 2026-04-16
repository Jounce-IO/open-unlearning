"""Parity tests for trajectory metric prefetch / batched exact_memorization paths."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.trajectory_metrics import (
    _call_metric_at_step,
    _get_logits_at_step,
    _prefetch_logits_by_step,
    _stack_step_logits_for_prob_batch,
)
from evals.metrics import METRICS_REGISTRY


def test_prefetch_steps_logits_matches_get_logits_at_step() -> None:
    torch.manual_seed(0)
    V, L, S = 11, 7, 13
    R = torch.randn(V, L, S)
    F = torch.randint(0, S, (L,), dtype=torch.long)
    traj = {"R": R, "F": F, "S": S, "L": L}
    steps = [0, 4, 12]
    by_step = _prefetch_logits_by_step(traj, "steps", steps)
    assert set(by_step.keys()) == set(steps)
    for s in steps:
        assert torch.equal(by_step[s], _get_logits_at_step(traj, "steps", s))


def test_stack_step_logits_shape() -> None:
    V, L = 5, 6
    logits_by_step = {0: torch.randn(V, L), 2: torch.randn(V, L)}
    steps = [0, 2]
    stacked = _stack_step_logits_for_prob_batch(logits_by_step, steps, None)
    assert stacked.shape == (2, L, V)


def test_batched_exact_memorization_matches_per_step() -> None:
    metric = METRICS_REGISTRY["exact_memorization"]
    torch.manual_seed(1)
    T, L, V = 3, 8, 17
    logits_b = torch.randn(T, L, V)
    labels = torch.randint(0, V, (1, L), dtype=torch.long)
    labels[0, :2] = -100
    batch_template = {
        "input_ids": torch.zeros((1, L), dtype=torch.long),
        "labels": labels,
        "attention_mask": torch.ones((1, L), dtype=torch.long),
        "index": torch.tensor([0], dtype=torch.long),
    }
    per_scores: list[float] = []
    for t in range(T):
        r = _call_metric_at_step(
            metric=metric,
            logits=logits_b[t : t + 1],
            batch_template=batch_template,
            tokenizer=None,
            sample_labels=None,
            sample_input_ids=torch.zeros(L, dtype=torch.long),
            sample_prompt_len=0,
            metric_config={},
            sample_idx="0",
        )
        assert isinstance(r, list) and r and isinstance(r[0], dict)
        assert r[0]["score"] is not None
        per_scores.append(float(r[0]["score"]))
    batched = _call_metric_at_step(
        metric=metric,
        logits=logits_b,
        batch_template=batch_template,
        tokenizer=None,
        sample_labels=None,
        sample_input_ids=torch.zeros(L, dtype=torch.long),
        sample_prompt_len=0,
        metric_config={},
        sample_idx="0",
    )
    assert isinstance(batched, list) and len(batched) == T
    for t in range(T):
        assert batched[t]["score"] is not None
        assert abs(float(batched[t]["score"]) - per_scores[t]) < 1e-5


def test_worker_exact_memorization_cpu_matches_call() -> None:
    from evals.metrics.trajectory_metrics import _worker_exact_memorization_cpu
    from evals.metrics.step_wise_score import trajectory_step_logits_to_prob_batch

    torch.manual_seed(2)
    V, L = 23, 9
    logits_vl = torch.randn(V, L)
    labels = torch.randint(0, V, (L,), dtype=torch.long)
    labels[:1] = -100
    sync = _call_metric_at_step(
        metric=METRICS_REGISTRY["exact_memorization"],
        logits=logits_vl,
        batch_template={
            "input_ids": torch.zeros((1, L), dtype=torch.long),
            "labels": labels.unsqueeze(0),
            "attention_mask": torch.ones((1, L), dtype=torch.long),
            "index": torch.tensor([0], dtype=torch.long),
        },
        tokenizer=None,
        sample_labels=None,
        sample_input_ids=torch.zeros(L, dtype=torch.long),
        sample_prompt_len=0,
        metric_config={},
        sample_idx="0",
    )
    assert isinstance(sync, list) and sync[0]["score"] is not None
    logits_lv = trajectory_step_logits_to_prob_batch(logits_vl).squeeze(0)
    cpu_score = _worker_exact_memorization_cpu(
        logits_lv.detach().cpu().numpy().astype("float32"),
        labels.detach().cpu().numpy().astype("int64"),
        -100,
    )
    assert cpu_score is not None
    assert abs(cpu_score - float(sync[0]["score"])) < 1e-5
