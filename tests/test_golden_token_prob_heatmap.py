"""Unit tests for golden-token probability heatmap helper (CPU, no model)."""

from __future__ import annotations

import math

import numpy as np
import torch

from evals.metrics.golden_token_prob_heatmap import (
    GoldenTokenHeatmapAccumulator,
    compute_golden_token_prob_heatmap_row,
    log_golden_token_heatmap_sample_diagnostics,
)
from evals.metrics.utils import IGNORE_INDEX


def test_causal_uses_previous_column_for_position_ell():
    """Causal alignment: token ell uses logits column max(0, ell-1)."""
    V, L = 6, 5
    logits = torch.zeros(V, L, dtype=torch.float32)
    logits[4, 2] = 8.0
    labels = torch.tensor([0, 1, 2, 4, 3], dtype=torch.long)
    row = compute_golden_token_prob_heatmap_row(
        logits,
        labels,
        logit_alignment="causal",
        ignore_index=IGNORE_INDEX,
    )
    assert not math.isnan(row[3])
    assert row[3] > 0.95
    row_same = compute_golden_token_prob_heatmap_row(
        logits,
        labels,
        logit_alignment="same_position",
        ignore_index=IGNORE_INDEX,
    )
    assert row_same[3] < 0.25


def test_same_position_uses_current_column():
    V, L = 4, 3
    logits = torch.zeros(V, L, dtype=torch.float32)
    logits[2, 1] = 9.0
    labels = torch.tensor([0, 2, 1], dtype=torch.long)
    row = compute_golden_token_prob_heatmap_row(
        logits,
        labels,
        logit_alignment="same_position",
        ignore_index=IGNORE_INDEX,
    )
    assert row[1] > 0.95


def test_verbose_diag_runs_without_error():
    V, L = 8, 6
    logits_by_step = {
        0: torch.randn(V, L),
        1: torch.randn(V, L),
        2: torch.randn(V, L),
    }
    labels = torch.tensor([1, 2, 3, 4, 5, 0], dtype=torch.long)
    log_golden_token_heatmap_sample_diagnostics(
        sample_idx=0,
        idx_str="0",
        traj_name="steps",
        logits_by_step=logits_by_step,
        gen_labels=labels,
        steps_to_use=[0, 1, 2],
        logit_alignment="causal",
        ignore_index=IGNORE_INDEX,
        L_gen=L,
        L_eff=4,
    )


def test_ignore_index_skips_positions():
    logits = torch.zeros(3, 4, dtype=torch.float32)
    logits[1, :] = 3.0
    labels = torch.tensor([1, IGNORE_INDEX, 1, 0], dtype=torch.long)
    row = compute_golden_token_prob_heatmap_row(
        logits,
        labels,
        logit_alignment="causal",
        ignore_index=IGNORE_INDEX,
    )
    assert math.isnan(row[1])


def test_accumulator_mean_excludes_ignore_only_positions():
    steps = [0, 1]
    L = 3
    acc = GoldenTokenHeatmapAccumulator(
        step_indices=steps,
        L_gen=L,
        trajectory_names=["steps"],
        views=["full", "eos"],
    )
    logits_a = {0: torch.zeros(2, L), 1: torch.zeros(2, L)}
    logits_a[0][0, 0] = 5.0
    logits_a[1][0, 0] = 5.0
    lab_a = torch.tensor([0, IGNORE_INDEX, 1], dtype=torch.long)
    acc.add_traj(
        traj_name="steps",
        logits_by_step=logits_a,
        gen_labels=lab_a,
        logit_alignment="same_position",
        ignore_index=IGNORE_INDEX,
        L_eff=2,
    )
    logits_b = {0: torch.zeros(2, L), 1: torch.zeros(2, L)}
    logits_b[0][1, 1] = 5.0
    logits_b[1][1, 1] = 5.0
    lab_b = torch.tensor([IGNORE_INDEX, 1, 0], dtype=torch.long)
    acc.add_traj(
        traj_name="steps",
        logits_by_step=logits_b,
        gen_labels=lab_b,
        logit_alignment="same_position",
        ignore_index=IGNORE_INDEX,
        L_eff=2,
    )
    out = acc.finalize_agg_value()
    full_steps = out["full"]["steps"]
    mean = np.array(full_steps["matrix_mean"], dtype=np.float64)
    cnt = np.array(full_steps["matrix_count"], dtype=np.int64)
    assert cnt[0, 1] == 1
    assert cnt[0, 0] == 1
    assert mean[0, 1] > 0.9
    assert cnt[0, 2] == 2
    eos_cnt = np.array(out["eos"]["steps"]["matrix_count"], dtype=np.int64)
    assert int(eos_cnt[0, 2]) == 0
