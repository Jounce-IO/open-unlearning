"""Tests for prob_trajectory_dump derived arrays and metric parity."""

from __future__ import annotations

import numpy as np
import torch

from evals.metrics.prob_trajectory_dump import (
    derive_position_arrays,
    geom_mean_from_golden_probs,
    parse_prob_trajectory_dump_config,
    prob_fixation_provider_from_sample_traj,
    prob_packed_shifted_from_step_logits,
)
from evals.metrics.step_wise_score import (
    compute_prob_packed_shifted_segments,
    sequence_probability_from_scores,
    trajectory_step_logits_to_prob_batch,
)


def test_parse_prob_trajectory_dump_config() -> None:
    enabled, n = parse_prob_trajectory_dump_config({"prob_trajectory_dump": {"enabled": True, "max_samples": 5}})
    assert enabled is True
    assert n == 5
    enabled2, n2 = parse_prob_trajectory_dump_config({})
    assert enabled2 is False
    assert n2 == 0


def test_derive_position_arrays_causal_column_sharing() -> None:
    V, L = 8, 4
    logits_vl = torch.zeros(V, L)
    # Column 0: token 3 is strongest; column 1: token 5
    logits_vl[3, 0] = 10.0
    logits_vl[5, 1] = 10.0
    gen_labels = torch.tensor([3, 3, 5, -100], dtype=torch.long)
    golden, argmax, gold = derive_position_arrays(
        logits_vl, gen_labels, logit_alignment="causal"
    )
    assert np.isclose(golden[0], 1.0, rtol=1e-4)
    assert np.isclose(golden[1], 1.0, rtol=1e-4)
    assert argmax[0] == 3
    assert argmax[1] == 3
    assert gold[0] == 3
    assert gold[1] == 3
    assert np.isnan(golden[3])


def test_prob_packed_shifted_matches_compute_prob_packed_shifted_segments() -> None:
    V, L = 6, 5
    logits_vl = torch.randn(V, L)
    labels = torch.tensor([1, 2, 3, 4, -100], dtype=torch.long)
    device = torch.device("cpu")
    p_dump = prob_packed_shifted_from_step_logits(logits_vl, labels, device)
    logits_b = trajectory_step_logits_to_prob_batch(logits_vl)
    lab_b = labels.reshape(1, -1)
    out = compute_prob_packed_shifted_segments([logits_b], [lab_b], device)
    assert np.isclose(p_dump, out[0]["prob"], rtol=1e-5, atol=1e-7)


def test_geom_mean_from_golden_probs() -> None:
    scores = [0.5, 0.25, 0.125]
    expected = sequence_probability_from_scores(scores)
    g = np.array([0.5, 0.25, 0.125, np.nan])
    assert np.isclose(geom_mean_from_golden_probs(g), expected)


def test_prob_fixation_provider_list_history() -> None:
    B, L, V, S = 1, 3, 5, 2
    lh = [
        torch.randn(B, L, V),
        torch.randn(B, L, V),
    ]
    F = torch.tensor([0, 1, 1], dtype=torch.long)
    sample_traj = {"lh": lh, "b": 0, "F": F, "S": S, "L": L}
    gen_labels = torch.tensor([1, 2, 3], dtype=torch.long)
    p = prob_fixation_provider_from_sample_traj(
        sample_traj, gen_labels, report_step=1, logit_alignment="causal"
    )
    assert 0.0 <= p <= 1.0
