"""Tests for trajectory ROUGE canvas + proposal merge decode."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from evals.metrics.trajectory_metrics import (
    _trajectory_assert_snapshots_align_logits_history,
    _trajectory_decode_gen_span_canvas_merge,
    _trajectory_merge_snapshot_row_with_diffusion_reindex,
)
from evals.metrics.trajectory_utils import diffusion_source_steps_for_trajectory


def test_decode_canvas_plus_step_x0_fills_masks() -> None:
    canvas = torch.tensor([10, 99, 30, 99, 50], dtype=torch.long)
    proposal = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    tok = MagicMock()
    tok.decode = lambda ids, **kw: ",".join(str(i) for i in ids)

    text = _trajectory_decode_gen_span_canvas_merge(
        canvas_row=canvas,
        proposal_row=proposal,
        pl=0,
        L=5,
        mask_id=99,
        view="full",
        leff=5,
        tokenizer=tok,
        mode="canvas_plus_step_x0",
    )
    assert text == "10,2,30,4,50"


def test_decode_committed_only_skips_masks() -> None:
    canvas = torch.tensor([10, 99, 30, 99, 50], dtype=torch.long)
    tok = MagicMock()
    tok.decode = lambda ids, **kw: ",".join(str(i) for i in ids)

    text = _trajectory_decode_gen_span_canvas_merge(
        canvas_row=canvas,
        proposal_row=None,
        pl=0,
        L=5,
        mask_id=99,
        view="full",
        leff=5,
        tokenizer=tok,
        mode="committed_only",
    )
    assert text == "10,30,50"


def test_assert_snapshots_canvas_plus_requires_both_lists() -> None:
    lh_len = 3
    seq = [torch.zeros(1, 8, dtype=torch.long) for _ in range(lh_len)]
    prop = [torch.zeros(1, 8, dtype=torch.long) for _ in range(lh_len)]
    _trajectory_assert_snapshots_align_logits_history(
        sequence_snapshots=seq,
        proposal_snapshots=prop,
        lh_len=lh_len,
        rouge_prediction_source="canvas_plus_step_x0",
    )


def test_assert_snapshots_raises_on_mismatch() -> None:
    with pytest.raises(ValueError):
        _trajectory_assert_snapshots_align_logits_history(
            sequence_snapshots=[torch.zeros(1, 4)],
            proposal_snapshots=None,
            lh_len=2,
            rouge_prediction_source="canvas_plus_step_x0",
        )


def test_canvas_reindex_merge_differs_steps_vs_fixation_start() -> None:
    """Canvas ROUGE row: per-position diffusion gather (same semantics as logits trajectories)."""
    S = 2
    snap = [
        torch.tensor([[5, 6]], dtype=torch.long),
        torch.tensor([[7, 8]], dtype=torch.long),
    ]
    F_b = torch.tensor([0, 1], dtype=torch.long)
    step = 1
    L = 2
    pl = 0
    tok = MagicMock()
    tok.decode = lambda ids, **kw: ",".join(str(i) for i in ids)

    src_steps = diffusion_source_steps_for_trajectory("steps", step, F_b, S)
    row_steps = _trajectory_merge_snapshot_row_with_diffusion_reindex(
        snap, 0, step, pl, L, 2, src_steps
    )
    text_steps = _trajectory_decode_gen_span_canvas_merge(
        canvas_row=row_steps,
        proposal_row=None,
        pl=pl,
        L=L,
        mask_id=99,
        view="full",
        leff=2,
        tokenizer=tok,
        mode="committed_only",
    )
    assert text_steps == "7,8"

    src_fs = diffusion_source_steps_for_trajectory("fixation_start", step, F_b, S)
    row_fs = _trajectory_merge_snapshot_row_with_diffusion_reindex(
        snap, 0, step, pl, L, 2, src_fs
    )
    text_fs = _trajectory_decode_gen_span_canvas_merge(
        canvas_row=row_fs,
        proposal_row=None,
        pl=pl,
        L=L,
        mask_id=99,
        view="full",
        leff=2,
        tokenizer=tok,
        mode="committed_only",
    )
    assert text_fs == "5,8"
