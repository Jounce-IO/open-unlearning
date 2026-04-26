"""Regression: lh row [L,V] must be transposed to [V,L] before trajectory_step_logits_to_prob_batch.

Dense R[b,:,:,step] is [V,L]. List-backed lh_batch[s][b] is [L,V]. The post-loop packed
probability path passed lh rows without transposing, so CE saw vocab dim = L (~200) and
device asserts on labels (token ids up to ~128k). Guided-traj runs with dense_r never hit this.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.step_wise_score import (
    compute_prob_packed_shifted_segments,
    trajectory_step_logits_to_prob_batch,
)


def test_trajectory_step_logits_to_prob_batch_expects_vl_not_lv() -> None:
    """Simulate dense slice [V,L] vs lh row [L,V]; only [V,L] yields [1,L,V] for CE."""
    V, Lm = 64, 11
    torch.manual_seed(0)
    logits_vl = torch.randn(V, Lm)
    lh_row_lv = logits_vl.transpose(0, 1).contiguous()
    assert lh_row_lv.shape == (Lm, V)

    good = trajectory_step_logits_to_prob_batch(logits_vl)
    wrong = trajectory_step_logits_to_prob_batch(lh_row_lv)

    assert good.shape == (1, Lm, V), good.shape
    assert wrong.shape == (1, V, Lm), wrong.shape
    assert good.shape[-1] == V
    assert wrong.shape[-1] == Lm
    assert torch.allclose(good, trajectory_step_logits_to_prob_batch(lh_row_lv.transpose(0, 1)))


def test_packed_shifted_ce_fails_when_logits_last_dim_not_vocab() -> None:
    """If last dim is L instead of V, label ids in [0,V) trigger CUDA/CPU CE out-of-range."""
    V, Lm = 32, 7
    torch.manual_seed(1)
    logits_vl = torch.randn(V, Lm)
    lh_row = logits_vl.transpose(0, 1).contiguous()
    labels_1d = torch.randint(0, V, (Lm,), dtype=torch.long)
    labels_1d[0] = -100

    bad_logits_b = trajectory_step_logits_to_prob_batch(lh_row)
    good_logits_b = trajectory_step_logits_to_prob_batch(lh_row.transpose(0, 1).contiguous())

    device = torch.device("cpu")
    good_out = compute_prob_packed_shifted_segments(
        [good_logits_b], [labels_1d.unsqueeze(0)], device, ignore_index=-100
    )
    assert len(good_out) == 1 and good_out[0].get("prob") is not None

    try:
        compute_prob_packed_shifted_segments(
            [bad_logits_b], [labels_1d.unsqueeze(0)], device, ignore_index=-100
        )
    except (RuntimeError, IndexError, ValueError) as e:
        # CPU: IndexError / invalid target; CUDA: device-side assert in nll_loss
        msg = str(e).lower()
        assert "target" in msg or "bounds" in msg or "class" in msg or "assert" in msg
    else:
        raise AssertionError("expected CE failure for wrong [1,V,L] layout vs vocab-sized labels")
