"""Parity tests for trajectory metric prefetch / batched exact_memorization paths."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
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


def test_trajectory_logits_vl_at_step_steps_matches_index() -> None:
    from evals.metrics.trajectory_metrics import _trajectory_logits_vl_at_step

    torch.manual_seed(0)
    V, L, S = 5, 4, 6
    R = torch.randn(V, L, S)
    F = torch.randint(0, S, (L,), dtype=torch.long)
    step = 3
    got = _trajectory_logits_vl_at_step(R, F, S, L, "steps", step)
    assert torch.equal(got, R[:, :, step])


def test_batch_template_has_list_tensors() -> None:
    from evals.metrics.trajectory_metrics import _batch_template_has_list_tensors

    assert not _batch_template_has_list_tensors({"labels": torch.zeros(1, 3)})
    assert _batch_template_has_list_tensors(
        {"labels": [torch.zeros(1, 3)]}
    )


def test_exact_mem_post_loop_metric_names_excludes_list_template() -> None:
    """When any row has list-of-tensor batch_template (e.g. multi wrong), skip post-loop set."""
    from evals.metrics.trajectory_metrics import _trajectory_exact_mem_post_loop_metric_names
    from evals.metrics import METRICS_REGISTRY

    Vocab = 20
    L = 5
    B = 2
    seq_len = 12
    labels = torch.full((B, seq_len), -100)
    labels[:, 6:] = torch.randint(0, Vocab, (B, seq_len - 6))
    input_ids = torch.zeros((B, seq_len), dtype=torch.long)
    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "labels_wrong": torch.randint(0, Vocab, (B, 2, seq_len)),
    }
    prompt_starts = [0, 0]
    prompt_lens = [6, 6]
    indices = torch.arange(B)
    loaded_metrics = {
        "exact_memorization": {"metric": METRICS_REGISTRY["exact_memorization"], "config": {}},
    }
    metrics_to_run = ["exact_memorization"]
    allow = frozenset({"exact_memorization"})
    tok = None
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        pytest.skip("gpt2 tokenizer unavailable")
    names = _trajectory_exact_mem_post_loop_metric_names(
        metrics_to_run=metrics_to_run,
        loaded_metrics=loaded_metrics,
        step_metric_batch_allowlist=allow,
        B=B,
        batch=batch,
        labels=labels,
        input_ids=input_ids,
        indices=indices,
        prompt_starts=prompt_starts,
        prompt_lens=prompt_lens,
        L=L,
        prompt_only_input_ids=False,
        tokenizer=tok,
    )
    assert names == frozenset()


def test_post_loop_rouge_append_order_b2_executor_none() -> None:
    """Two samples: post-loop ROUGE appends two scores per (step, view) in batch order."""
    from evals.metrics.trajectory_metrics import _trajectory_append_post_loop_rouge
    from evals.metrics import METRICS_REGISTRY

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        pytest.skip("gpt2 tokenizer unavailable")

    B, V, L, S = 2, tok.vocab_size, 4, 5
    R = torch.zeros(B, V, L, S)
    F = torch.zeros(B, L, dtype=torch.long)
    for b in range(B):
        for pos in range(L):
            R[b, (b + 1) % V, pos, 1] = 1.0
    steps_to_use = [1]
    loaded_metrics = {"rouge": {"metric": METRICS_REGISTRY["rouge"], "config": {"rouge_type": "rougeL_recall"}}}
    metrics_to_run = ["rouge"]
    step_values_by_view = {
        "full": {"steps": {}},
        "eos": {"steps": {}},
    }
    labels = torch.randint(0, min(100, V), (B, 20))
    labels[:, :10] = -100
    input_ids = torch.zeros((B, 20), dtype=torch.long)
    batch = {"input_ids": input_ids, "labels": labels}
    indices = torch.arange(B)
    prompt_starts = [10, 10]
    prompt_lens = [0, 0]
    _trajectory_append_post_loop_rouge(
        trajectory_names=["steps"],
        steps_to_use=steps_to_use,
        include_views=["full"],
        R=R,
        F=F,
        S=S,
        L=L,
        B=B,
        effective_lengths=[L, L],
        labels=labels,
        input_ids=input_ids,
        batch=batch,
        indices=indices,
        prompt_starts=prompt_starts,
        prompt_lens=prompt_lens,
        prompt_only_input_ids=False,
        tokenizer=tok,
        metrics_to_run=metrics_to_run,
        loaded_metrics=loaded_metrics,
        step_values_by_view=step_values_by_view,
        executor=None,
        all_rouge_futures=[],
        kwargs={"rouge_type": "rougeL_recall"},
    )
    vals = step_values_by_view["full"]["steps"][1]["rouge"]
    assert len(vals) == B
    assert all(isinstance(x, (int, float, np.number)) for x in vals)
