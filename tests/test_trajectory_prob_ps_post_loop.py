"""Phase D: post-loop trajectory probability uses effective-step fixation (P_s), not lh[s] packed CE."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.step_wise_score import (  # noqa: E402
    compute_prob_packed_shifted_segment_details,
    trajectory_step_logits_to_prob_batch,
)
from evals.metrics.trajectory_metrics import (  # noqa: E402
    _build_effective_step_fixation_logits_for_traj,
    _trajectory_report_step_Ps_prob_details,
)


def _gold_friendly_vl(vocab: int, length: int, gold: list[int]) -> torch.Tensor:
    logits = torch.full((vocab, length), -20.0)
    for ell in range(1, len(gold)):
        if ell < length:
            logits[gold[ell], ell - 1] = 20.0
    return logits


def test_ps_path_differs_from_packed_lh_at_last_step() -> None:
    """Production bug path: packed CE on lh[S-1]; Phase D: effective-step fixation at S-1."""
    device = torch.device("cpu")
    vocab, length, steps = 32, 6, 6
    gold = [3, 4, 5, 6, 7]
    lab = torch.tensor(gold, dtype=torch.long)
    lh: list[torch.Tensor] = []
    for s in range(steps):
        if s == 0:
            lh.append(_gold_friendly_vl(vocab, length, gold).transpose(0, 1).unsqueeze(0))
        elif s == steps - 1:
            lh.append(torch.randn(1, length, vocab))
        else:
            lh.append(lh[-1].clone())
    F = torch.arange(length)
    st_b = {"lh": lh, "b": 0, "F": F, "S": steps, "L": length}
    fl = _build_effective_step_fixation_logits_for_traj(st_b, "steps", steps - 1)
    ps_det = _trajectory_report_step_Ps_prob_details(
        fl, lab, view="full", L_eff=length, device=device
    )
    packed_det = compute_prob_packed_shifted_segment_details(
        trajectory_step_logits_to_prob_batch(lh[-1][0].transpose(0, 1).contiguous()),
        lab.unsqueeze(0),
        device,
    )
    assert ps_det["prob"] is not None and packed_det["prob"] is not None
    assert ps_det["prob"] != pytest.approx(packed_det["prob"], rel=0.05)


def test_ps_steps_matches_effective_step_helper() -> None:
    device = torch.device("cpu")
    vocab, length, steps = 24, 5, 4
    gold = [2, 3, 4, 5]
    lab = torch.tensor(gold, dtype=torch.long)
    R_slices = [_gold_friendly_vl(vocab, length, gold) for _ in range(steps)]
    R = torch.stack(R_slices, dim=2)
    F = torch.arange(length)
    st_b = {"R": R, "F": F, "S": steps, "L": length}
    for report_step in (0, steps - 1):
        fl = _build_effective_step_fixation_logits_for_traj(st_b, "steps", report_step)
        det = _trajectory_report_step_Ps_prob_details(
            fl, lab, view="full", L_eff=length, device=device
        )
        assert det["n_valid_tokens"] == len(gold) - 1
        assert det["prob"] is not None and det["prob"] > 0.9
