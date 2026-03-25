"""Trajectory vs non-traj generalized probability definitions (fixation_start, CPU, tiny tensors)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from data.utils import IGNORE_INDEX
from evals.metrics.step_wise_score import (
    FixationStepWiseScoreProvider,
    build_fixation_logits_from_R_F,
    compute_prob_from_fixation_logits,
    evaluate_probability_via_provider,
    trajectory_step_logits_to_prob_batch,
)
from evals.metrics.trajectory_metrics import _get_logits_at_step


def _labels_prompt_eos(
    L: int,
    pl: int,
    V: int,
    eos_pos: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Labels with ignore prefix, random tail, eos at eos_pos; L_eff = eos_pos + 1."""
    labels = torch.full((L,), IGNORE_INDEX, dtype=torch.long, device=device)
    labels[pl:eos_pos] = torch.randint(0, V, (eos_pos - pl,), device=device)
    labels[eos_pos] = 7
    return labels, eos_pos + 1


def _prob_traj_style(
    R: torch.Tensor,
    F: torch.Tensor,
    labels: torch.Tensor,
    *,
    traj_name: str,
    step: int,
    view: str,
    L_eff: int,
    device: torch.device,
    apply_legacy_cat: bool = False,
) -> float:
    """Trajectory loop: optional legacy cat (regression), else ``trajectory_step_logits_to_prob_batch``."""
    traj = {"R": R, "F": F}
    logits_lv = _get_logits_at_step(traj, traj_name, step)
    logits_step = logits_lv.t()
    if apply_legacy_cat:
        logits_step = torch.cat(
            [logits_step[:1, :], logits_step[:-1, :]], dim=0
        )
    logits_batch = logits_step.unsqueeze(0)
    labels_full = labels.unsqueeze(0)
    if view == "full":
        logits_v, lab = logits_batch, labels_full
    else:
        sl = min(L_eff, logits_batch.shape[1])
        logits_v = logits_batch[:, :sl, :].contiguous()
        lab = labels_full[:, :sl].contiguous()
    out = compute_prob_from_fixation_logits(logits_v, lab, device, IGNORE_INDEX)
    return float(out[0]["prob"])


def _prob_non_traj_provider(
    R: torch.Tensor,
    F: torch.Tensor,
    labels: torch.Tensor,
    *,
    view: str,
    L_eff: int,
    device: torch.device,
) -> float:
    fixation_logits = build_fixation_logits_from_R_F(R, F).to(device=device)
    lab = labels.unsqueeze(0)
    if view == "eos":
        sl = min(L_eff, fixation_logits.shape[1])
        fixation_logits = fixation_logits[:, :sl, :].contiguous()
        lab = lab[:, :sl].contiguous()
    provider = FixationStepWiseScoreProvider(logit_alignment="causal")
    out = evaluate_probability_via_provider(
        provider, fixation_logits, {"labels": lab}, ignore_index=IGNORE_INDEX
    )
    prob = out[0]["prob"]
    return float(prob) if prob is not None else float("nan")


@pytest.mark.parametrize("vocab", [8, 32])
@pytest.mark.parametrize("seq_len", [4, 16])
@pytest.mark.parametrize("num_steps", [3, 8])
@pytest.mark.parametrize("view", ["full", "eos"])
def test_fixation_start_final_step_matches_non_traj_provider(
    vocab: int,
    seq_len: int,
    num_steps: int,
    view: str,
) -> None:
    device = torch.device("cpu")
    torch.manual_seed(vocab + seq_len * 17 + num_steps)
    R = torch.randn(vocab, seq_len, num_steps, device=device)
    F = torch.randint(0, num_steps, (seq_len,), device=device, dtype=torch.long)
    pl = min(2, seq_len - 2)
    eos_pos = min(seq_len - 1, pl + max(1, seq_len - pl - 1))
    labels, L_eff = _labels_prompt_eos(seq_len, pl, vocab, eos_pos, device)
    last = num_steps - 1

    fl = build_fixation_logits_from_R_F(R, F)
    fs_batch = trajectory_step_logits_to_prob_batch(
        _get_logits_at_step({"R": R, "F": F}, "fixation_start", last)
    )
    assert torch.allclose(fl, fs_batch)

    p_traj = _prob_traj_style(
        R, F, labels, traj_name="fixation_start", step=last, view=view, L_eff=L_eff, device=device
    )
    p_nt = _prob_non_traj_provider(R, F, labels, view=view, L_eff=L_eff, device=device)
    assert abs(p_traj - p_nt) < 1e-5


@pytest.mark.parametrize("case", ("all_zero", "all_last", "mixed"))
def test_fixation_F_edge_cases_match_provider(case: str) -> None:
    device = torch.device("cpu")
    torch.manual_seed(42)
    V, L, S = 16, 12, 6
    R = torch.randn(V, L, S, device=device)
    if case == "all_zero":
        F = torch.zeros(L, dtype=torch.long, device=device)
    elif case == "all_last":
        F = torch.full((L,), S - 1, dtype=torch.long, device=device)
    else:
        F = torch.randint(0, S, (L,), device=device, dtype=torch.long)
    labels, L_eff = _labels_prompt_eos(L, 3, V, 9, device)
    last = S - 1
    p_traj = _prob_traj_style(
        R, F, labels, traj_name="fixation_start", step=last, view="eos", L_eff=L_eff, device=device
    )
    p_nt = _prob_non_traj_provider(R, F, labels, view="eos", L_eff=L_eff, device=device)
    assert abs(p_traj - p_nt) < 1e-5


def test_mid_step_fixation_start_can_differ_from_non_traj() -> None:
    """At step < S-1, fixation_start uses min(s,F) — not full fixation gather."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    V, L, S = 24, 10, 8
    R = torch.randn(V, L, S, device=device)
    F = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 7, 7], device=device)
    labels = torch.full((L,), IGNORE_INDEX, dtype=torch.long, device=device)
    labels[2:] = torch.randint(0, V, (L - 2,), device=device)
    mid = S // 2
    p_traj_mid = _prob_traj_style(
        R, F, labels, traj_name="fixation_start", step=mid, view="full", L_eff=L, device=device
    )
    p_nt = _prob_non_traj_provider(R, F, labels, view="full", L_eff=L, device=device)
    assert abs(p_traj_mid - p_nt) > 1e-6


def test_steps_slice_at_last_can_differ_from_non_traj() -> None:
    device = torch.device("cpu")
    torch.manual_seed(1)
    V, L, S = 24, 10, 8
    R = torch.randn(V, L, S, device=device)
    F = torch.randint(0, S, (L,), device=device, dtype=torch.long)
    labels = torch.full((L,), IGNORE_INDEX, dtype=torch.long, device=device)
    labels[2:] = torch.randint(0, V, (L - 2,), device=device)
    last = S - 1
    p_steps = _prob_traj_style(
        R, F, labels, traj_name="steps", step=last, view="full", L_eff=L, device=device
    )
    p_nt = _prob_non_traj_provider(R, F, labels, view="full", L_eff=L, device=device)
    assert abs(p_steps - p_nt) > 1e-8


def test_legacy_cat_diverges_from_provider_on_random_fixture() -> None:
    """If legacy torch.cat is reintroduced, this should fail vs provider path."""
    device = torch.device("cpu")
    torch.manual_seed(2)
    V, L, S = 32, 14, 7
    R = torch.randn(V, L, S, device=device)
    F = torch.randint(0, S, (L,), device=device, dtype=torch.long)
    labels, L_eff = _labels_prompt_eos(L, 4, V, 11, device)
    last = S - 1
    p_cat = _prob_traj_style(
        R,
        F,
        labels,
        traj_name="fixation_start",
        step=last,
        view="full",
        L_eff=L_eff,
        device=device,
        apply_legacy_cat=True,
    )
    p_nt = _prob_non_traj_provider(R, F, labels, view="full", L_eff=L_eff, device=device)
    assert abs(p_cat - p_nt) > 1e-6


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
