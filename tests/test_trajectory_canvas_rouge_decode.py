"""Tests for trajectory ROUGE canvas + proposal merge decode."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from evals.metrics.trajectory_metrics import (
    _trajectory_assert_snapshots_align_logits_history,
    _trajectory_decode_gen_span_canvas_merge,
)


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
