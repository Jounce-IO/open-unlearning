"""Regression: extraction_strength via _call_metric_at_step must support list_history sample_traj."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics import METRICS_REGISTRY
from evals.metrics.step_wise_score import build_fixation_logits_from_R_F, build_fixation_logits_from_history
from evals.metrics.trajectory_metrics import _call_metric_at_step


def _lh_from_dense_R(R_vls: torch.Tensor, B: int = 2) -> list[torch.Tensor]:
    """Build batch logits history lh[s][b,l,:] == R_vls[:,l,s] for b==0 (other rows arbitrary)."""
    V, L, S = R_vls.shape
    lh: list[torch.Tensor] = []
    for s in range(S):
        t = torch.randn(B, L, V)
        for ell in range(L):
            t[0, ell, :] = R_vls[:, ell, s]
        lh.append(t)
    return lh


def test_build_fixation_logits_from_history_matches_R_F() -> None:
    torch.manual_seed(0)
    V, L, S = 11, 9, 7
    R = torch.randn(V, L, S)
    F = torch.randint(0, S, (L,), dtype=torch.long)
    b = 0
    lh = _lh_from_dense_R(R, B=3)
    ref = build_fixation_logits_from_R_F(R, F).squeeze(0)
    got = build_fixation_logits_from_history(lh, F, b)
    assert ref.shape == got.shape
    assert torch.allclose(ref, got, atol=1e-6, rtol=1e-5)


def test_call_metric_at_step_extraction_strength_list_history_no_r_key() -> None:
    """list_history sample_traj has no 'R'; must not KeyError (cluster regression)."""
    assert "extraction_strength" in METRICS_REGISTRY
    m = METRICS_REGISTRY["extraction_strength"]
    torch.manual_seed(1)
    V, L, S = 13, 8, 6
    R = torch.randn(V, L, S)
    F = torch.randint(0, S, (L,), dtype=torch.long)
    lh = _lh_from_dense_R(R, B=2)
    b = 0
    sample_traj = {"lh": lh, "b": b, "F": F, "S": S, "L": L}
    logits_vl = R[:, :, 3].contiguous()
    batch_template = {
        "input_ids": torch.zeros((1, L), dtype=torch.long),
        "labels": torch.randint(0, V, (1, L)),
        "attention_mask": torch.ones((1, L), dtype=torch.long),
    }
    tokenizer = None
    sample_labels = batch_template["labels"].squeeze(0)
    sample_input_ids = batch_template["input_ids"].squeeze(0)
    sample_prompt_len = 2
    traj_cfg = {"use_generalized_sequence_probability": True, "logit_alignment": "causal"}
    ref = _call_metric_at_step(
        metric=m,
        logits=logits_vl,
        batch_template=batch_template,
        tokenizer=tokenizer,
        sample_labels=sample_labels,
        sample_input_ids=sample_input_ids,
        sample_prompt_len=sample_prompt_len,
        metric_config={},
        sample_idx="0",
        trajectory_config=traj_cfg,
        sample_traj={"R": R, "F": F, "S": S, "L": L},
        step=3,
    )
    out = _call_metric_at_step(
        metric=m,
        logits=logits_vl,
        batch_template=batch_template,
        tokenizer=tokenizer,
        sample_labels=sample_labels,
        sample_input_ids=sample_input_ids,
        sample_prompt_len=sample_prompt_len,
        metric_config={},
        sample_idx="0",
        trajectory_config=traj_cfg,
        sample_traj=sample_traj,
        step=3,
    )
    assert isinstance(ref, list) and isinstance(out, list)
    assert len(ref) == len(out) == 1
    assert abs(float(ref[0]["score"]) - float(out[0]["score"])) < 1e-5
