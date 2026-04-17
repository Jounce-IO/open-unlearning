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
from evals.metrics.utils import IGNORE_INDEX
from evals.metrics.step_wise_score import compute_prob_from_fixation_logits


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
        view_step_batch_allowlist=frozenset(),
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


def test_batch_rouge_decode_matches_per_row_decode() -> None:
    from evals.metrics.trajectory_metrics import (
        _trajectory_batch_decode_predictions_for_rouge,
        _trajectory_decode_prediction_for_rouge,
    )

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        pytest.skip("gpt2 tokenizer unavailable")

    torch.manual_seed(11)
    V, L = tok.vocab_size, 9
    rows = [torch.randn(V, L), torch.randn(V, L), torch.randn(V, L)]
    leff = [L, 2, L]
    for view in ("full", "eos"):
        bat = _trajectory_batch_decode_predictions_for_rouge(tok, rows, view, leff)
        assert len(bat) == 3
        for i, lv in enumerate(rows):
            want = _trajectory_decode_prediction_for_rouge(tok, lv, view, leff[i])
            assert bat[i] == want


def test_fused_truth_ratio_precompute_matches_sequential() -> None:
    from evals.metrics.trajectory_metrics import _compute_pre_compute_metrics_at_step
    from evals.metrics.step_wise_score import FixationStepWiseScoreProvider, sequence_probability_from_scores

    torch.manual_seed(13)
    V, L, S = 11, 7, 8
    R = torch.randn(V, L, S)
    F = torch.randint(0, S, (L,), dtype=torch.long)
    step = 4
    lab_c = torch.randint(0, V, (L,), dtype=torch.long)
    lab_w = torch.randint(0, V, (L,), dtype=torch.long)
    lab_c[0] = -100
    lab_w[1] = -100
    batch_template = {
        "input_ids": torch.zeros(1, L, dtype=torch.long),
        "labels_correct": lab_c.unsqueeze(0),
        "labels_wrong": lab_w.unsqueeze(0),
        "attention_mask": torch.ones(1, L, dtype=torch.long),
        "index": torch.tensor([0], dtype=torch.long),
    }
    sample_traj = {"R": R, "F": F, "S": S, "L": L}
    pre_compute_config = {
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
    logits = torch.zeros(V, L)
    out = _compute_pre_compute_metrics_at_step(
        pre_compute_config=pre_compute_config,
        logits=logits,
        batch_template=batch_template,
        tokenizer=None,
        sample_labels=None,
        sample_input_ids=torch.zeros(L, dtype=torch.long),
        sample_prompt_len=0,
        sample_idx="0",
        trajectory_config={"use_generalized_sequence_probability": True, "logit_alignment": "causal"},
        sample_traj=sample_traj,
        step=step,
    )
    prov = FixationStepWiseScoreProvider(logit_alignment="causal")
    mo = {"R": R.unsqueeze(0), "F": F.unsqueeze(0), "report_step": step}

    def _expected(lab_1d: torch.Tensor) -> float | None:
        r = prov.get_per_position_scores(mo, {"labels": lab_1d.unsqueeze(0)})
        if not r or not r[0][0]:
            return None
        return float(sequence_probability_from_scores(r[0][0]))

    ec = _expected(lab_c)
    ew = _expected(lab_w)
    assert (out["correct"]["agg_value"] is None) == (ec is None)
    assert (out["wrong"]["agg_value"] is None) == (ew is None)
    if ec is not None:
        assert abs(float(out["correct"]["agg_value"]) - ec) < 1e-5
    if ew is not None:
        assert abs(float(out["wrong"]["agg_value"]) - ew) < 1e-5


def test_fused_truth_ratio_traj_step_precompute_matches_sequential() -> None:
    """traj_name set: dual probability pre_compute must match two shifted-CE calls."""
    from evals.metrics.trajectory_metrics import _compute_pre_compute_metrics_at_step

    torch.manual_seed(21)
    L, V = 9, 17
    logits_1lv = torch.randn(1, L, V)
    lab_c = torch.randint(0, V, (L,), dtype=torch.long)
    lab_w = torch.randint(0, V, (L,), dtype=torch.long)
    lab_c[0] = IGNORE_INDEX
    lab_w[2] = IGNORE_INDEX
    batch_template = {
        "input_ids": torch.zeros(1, L, dtype=torch.long),
        "labels_correct": lab_c.unsqueeze(0),
        "labels_wrong": lab_w.unsqueeze(0),
        "attention_mask": torch.ones(1, L, dtype=torch.long),
    }
    pre_compute_config = {
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
    device = logits_1lv.device
    labels_c_b = lab_c.unsqueeze(0)
    labels_w_b = lab_w.unsqueeze(0)
    ref_c = compute_prob_from_fixation_logits(logits_1lv, labels_c_b, device)[0]
    ref_w = compute_prob_from_fixation_logits(logits_1lv, labels_w_b, device)[0]

    def _assert_leg(out_leg: dict, ref: dict) -> None:
        vbi = out_leg["value_by_index"]["0"]
        assert abs(float(out_leg["agg_value"]) - float(ref["prob"])) < 1e-5
        assert abs(float(vbi["prob"]) - float(ref["prob"])) < 1e-5
        assert abs(float(vbi["avg_loss"]) - float(ref["avg_loss"])) < 1e-5

    out_1 = _compute_pre_compute_metrics_at_step(
        pre_compute_config=pre_compute_config,
        logits=logits_1lv,
        batch_template=batch_template,
        tokenizer=None,
        sample_labels=None,
        sample_input_ids=torch.zeros(L, dtype=torch.long),
        sample_prompt_len=0,
        sample_idx="0",
        traj_name="steps",
        step=3,
    )
    _assert_leg(out_1["correct"], ref_c)
    _assert_leg(out_1["wrong"], ref_w)

    logits_vl = logits_1lv.squeeze(0).transpose(0, 1).contiguous()
    assert logits_vl.shape == (V, L)
    out_vl = _compute_pre_compute_metrics_at_step(
        pre_compute_config=pre_compute_config,
        logits=logits_vl,
        batch_template=batch_template,
        tokenizer=None,
        sample_labels=None,
        sample_input_ids=torch.zeros(L, dtype=torch.long),
        sample_prompt_len=0,
        sample_idx="0",
        traj_name="steps",
        step=3,
    )
    _assert_leg(out_vl["correct"], ref_c)
    _assert_leg(out_vl["wrong"], ref_w)


def test_trajectory_should_empty_cuda_cache_respects_flag() -> None:
    from evals.metrics.trajectory_metrics import _trajectory_should_empty_cuda_cache

    if not torch.cuda.is_available():
        assert _trajectory_should_empty_cuda_cache(None) is False
    else:
        assert _trajectory_should_empty_cuda_cache(None) is True
        assert _trajectory_should_empty_cuda_cache({"aggressive_cuda_empty_cache": False}) is False
        assert _trajectory_should_empty_cuda_cache({"aggressive_cuda_empty_cache": True}) is True


def test_packed_shifted_probs_chunked_matches_single_call() -> None:
    from evals.metrics.step_wise_score import compute_prob_packed_shifted_segments
    from evals.metrics.trajectory_metrics import _packed_shifted_probs_chunked

    torch.manual_seed(11)
    device = torch.device("cpu")
    seg_logits: list[torch.Tensor] = []
    seg_labels: list[torch.Tensor] = []
    for Li in (3, 5, 2, 4):
        seg_logits.append(torch.randn(1, Li, 9))
        lab = torch.randint(0, 9, (1, Li), dtype=torch.long)
        lab[0, -1] = IGNORE_INDEX
        seg_labels.append(lab)
    ref = compute_prob_packed_shifted_segments(seg_logits, seg_labels, device, IGNORE_INDEX)
    chunked = _packed_shifted_probs_chunked(
        seg_logits, seg_labels, device, IGNORE_INDEX, chunk_max=2
    )
    assert len(chunked) == len(ref)
    for a, b in zip(ref, chunked, strict=True):
        assert abs(float(a["prob"]) - float(b["prob"])) < 1e-6
        assert abs(float(a["avg_loss"]) - float(b["avg_loss"])) < 1e-5


def test_shifted_ce_segments_lex_matches_individual_probs() -> None:
    from evals.metrics.trajectory_metrics import _build_shifted_ce_segments_step_view_lex
    from evals.metrics.trajectory_metrics import _prefetch_logits_by_step
    from evals.metrics.step_wise_score import (
        compute_prob_from_fixation_logits,
        compute_prob_packed_shifted_segments,
        trajectory_step_logits_to_prob_batch,
    )

    torch.manual_seed(12)
    V, L, S = 13, 10, 5
    R = torch.randn(V, L, S)
    F = torch.randint(0, S, (L,), dtype=torch.long)
    traj = {"R": R, "F": F, "S": S, "L": L}
    steps = [1, 3]
    include_views = ["full", "eos"]
    gl = torch.randint(0, V, (L,), dtype=torch.long)
    gl[0] = IGNORE_INDEX
    L_eff = 6
    gl_eos = gl[:L_eff].clone()

    by_step = _prefetch_logits_by_step(traj, "steps", steps)
    device = R.device
    seg_log, seg_lab = _build_shifted_ce_segments_step_view_lex(
        by_step, steps, include_views, gl, gl_eos, L_eff, device
    )
    packed = compute_prob_packed_shifted_segments(seg_log, seg_lab, device, IGNORE_INDEX)
    assert len(packed) == len(steps) * len(include_views)
    k = 0
    for step in steps:
        sl = by_step[int(step)]
        log_b = trajectory_step_logits_to_prob_batch(sl)
        for view in include_views:
            if view == "full":
                pr = compute_prob_from_fixation_logits(log_b, gl.unsqueeze(0), device, IGNORE_INDEX)[0]
            else:
                Ls = min(L_eff, log_b.shape[1])
                pr = compute_prob_from_fixation_logits(
                    log_b[:, :Ls, :], gl_eos[:Ls].unsqueeze(0), device, IGNORE_INDEX
                )[0]
            assert abs(float(packed[k]["prob"]) - float(pr["prob"])) < 1e-5
            k += 1


@pytest.mark.parametrize("traj_name", ["steps", "fixation_start", "fixation_end", "fixation_ratio"])
def test_exact_mem_tv_stack_parity_all_traj_types(traj_name: str) -> None:
    from evals.metrics.trajectory_metrics import (
        _call_metric_at_step,
        _exact_mem_tv_stack_logits_and_batch,
        _prefetch_logits_by_step,
    )

    torch.manual_seed(13)
    V, L, S = 11, 8, 9
    R = torch.randn(V, L, S)
    F = torch.randint(0, S, (L,), dtype=torch.long)
    traj = {"R": R, "F": F, "S": S, "L": L}
    steps = [2, 5]
    include_views = ["full", "eos"]
    labels = torch.randint(0, V, (1, L), dtype=torch.long)
    labels[0, :1] = IGNORE_INDEX
    L_eff = 5
    batch_template = {
        "input_ids": torch.zeros((1, L), dtype=torch.long),
        "labels": labels,
        "attention_mask": torch.ones((1, L), dtype=torch.long),
        "index": torch.tensor([0], dtype=torch.long),
    }
    labels_eos = labels[:, :L_eff].clone()
    batch_eos = {
        "input_ids": batch_template["input_ids"][:, :L_eff],
        "labels": labels_eos,
        "attention_mask": torch.ones((1, L_eff), dtype=torch.long),
        "index": batch_template["index"],
    }
    logits_by_step = _prefetch_logits_by_step(traj, traj_name, steps)
    logits_tv, bt_exp = _exact_mem_tv_stack_logits_and_batch(
        logits_by_step,
        steps,
        include_views,
        batch_template,
        batch_eos,
        L_eff,
        IGNORE_INDEX,
    )
    metric = METRICS_REGISTRY["exact_memorization"]
    batched = _call_metric_at_step(
        metric=metric,
        logits=logits_tv,
        batch_template=bt_exp,
        tokenizer=None,
        sample_labels=None,
        sample_input_ids=torch.zeros(L, dtype=torch.long),
        sample_prompt_len=0,
        metric_config={},
        sample_idx="0",
        traj_name=traj_name,
        sample_traj=traj,
    )
    assert isinstance(batched, list) and len(batched) == len(steps) * len(include_views)
    j = 0
    for step in steps:
        lv = logits_by_step[int(step)]
        for view in include_views:
            bt = batch_template if view == "full" else batch_eos
            logits_view = lv[:, :L_eff] if view == "eos" else lv
            one = _call_metric_at_step(
                metric=metric,
                logits=logits_view,
                batch_template=bt,
                tokenizer=None,
                sample_labels=None,
                sample_input_ids=torch.zeros(L, dtype=torch.long),
                sample_prompt_len=0,
                metric_config={},
                sample_idx="0",
                traj_name=traj_name,
                sample_traj=traj,
                step=step,
            )
            assert isinstance(one, list) and one[0]["score"] is not None
            assert batched[j]["score"] is not None
            assert abs(float(batched[j]["score"]) - float(one[0]["score"])) < 1e-5
            j += 1
