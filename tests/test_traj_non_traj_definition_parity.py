"""US1: trajectory fixation logits path vs provider geometric mean (aligned positions)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from data.utils import IGNORE_INDEX
from evals.metrics.step_wise_score import (
    FixationStepWiseScoreProvider,
    compute_prob_from_fixation_logits,
    evaluate_probability_via_provider,
)


def test_fixation_logits_prob_matches_provider_when_first_label_ignored() -> None:
    """First position ignored so shifted-CE path and causal provider score the same set."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    b, l, v = 1, 12, 64
    fixation_logits = torch.randn(b, l, v, device=device)
    labels = torch.full((b, l), IGNORE_INDEX, dtype=torch.long, device=device)
    labels[0, 3:10] = torch.randint(1, v, (7,), device=device)

    traj = compute_prob_from_fixation_logits(
        fixation_logits, labels, device, ignore_index=IGNORE_INDEX
    )
    provider = FixationStepWiseScoreProvider(logit_alignment="causal")
    via = evaluate_probability_via_provider(
        provider,
        fixation_logits,
        {"labels": labels},
        ignore_index=IGNORE_INDEX,
    )
    assert traj[0]["prob"] is not None and via[0]["prob"] is not None
    assert abs(traj[0]["prob"] - via[0]["prob"]) < 1e-4
