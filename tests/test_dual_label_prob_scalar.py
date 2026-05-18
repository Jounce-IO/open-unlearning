"""Unit tests for legacy ra/wf dual-label helper (superseded by merge synthesis)."""

import numpy as np
import pytest

from evals.metrics.trajectory_metrics import _dual_label_prob_scalar


def test_dual_label_prob_scalar_raw_returns_pc():
    assert _dual_label_prob_scalar(0.4, [0.6, 0.8], normalised=False) == pytest.approx(0.4)


def test_dual_label_prob_scalar_normalised_default_sum_matches_ou():
    pc = 0.3
    pw = [0.2, 0.4]
    expected = pc / (pc + float(np.sum(pw)) + 1e-10)
    assert _dual_label_prob_scalar(pc, pw, normalised=True) == pytest.approx(expected)


def test_dual_label_prob_scalar_normalised_mean_deprecated():
    pc = 0.3
    pw = [0.2, 0.4]
    expected = pc / (pc + float(np.mean(pw)) + 1e-10)
    assert _dual_label_prob_scalar(
        pc, pw, normalised=True, wrong_aggregate="mean"
    ) == pytest.approx(expected)


def test_dual_label_prob_scalar_none_pc():
    assert _dual_label_prob_scalar(None, [0.5], normalised=True) is None
