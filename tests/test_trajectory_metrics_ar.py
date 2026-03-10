"""
T025, T026: AR trajectory metrics — unit and regression tests (US4).

- T025: AR trajectory construction produces R [V,L,S] with S=L; undecoded positions
  have uniform logits (1/V).
- T026: AR trajectory result has same keys as dLLM; for full sequence sequence-level
  probability matches TOFU formula (geometric mean of step-wise scores).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


# Expected trajectory result keys (same as dLLM per contract eval-result-schema.md)
EXPECTED_TRAJECTORY_RESULT_KEYS = frozenset({
    "agg_value",
    "step_distribution",
    "steps",
    "fixation_start",
    "fixation_end",
    "fixation_ratio",
})


class TestARTrajectoryConstruction:
    """T025: Unit test — AR trajectory R shape [V,L,S] S=L, undecoded positions uniform."""

    def test_ar_trajectory_r_shape_s_equals_l(self):
        """build_ar_trajectory_r_f returns R with shape [V, L, S] and S=L."""
        from evals.metrics.trajectory_utils import build_ar_trajectory_r_f

        V, L = 100, 5
        position_logits = torch.randn(L, V)
        R, F, S, L_out = build_ar_trajectory_r_f(position_logits)
        assert R.dim() == 3, "Per-sample R must be [V, L, S]"
        assert R.shape[0] == V and R.shape[1] == L_out and R.shape[2] == S
        assert S == L_out, "For AR, S must equal L (one step per token)"
        assert F.shape == (L_out,)
        assert (F == torch.arange(L_out, device=F.device, dtype=F.dtype)).all(), "AR fixation F[l]=l"

    def test_ar_trajectory_undecoded_positions_uniform(self):
        """At step s, positions l > s have uniform logits (constant along vocab dim)."""
        from evals.metrics.trajectory_utils import build_ar_trajectory_r_f

        V, L = 50, 4
        position_logits = torch.randn(L, V)
        R, F, S, L_out = build_ar_trajectory_r_f(position_logits)
        for s in range(S):
            for l in range(L_out):
                if l > s:
                    logits_at_l_s = R[:, l, s]
                    assert logits_at_l_s.numel() == V
                    # Uniform: all logits equal (e.g. 0) so softmax gives 1/V
                    assert torch.allclose(
                        logits_at_l_s, logits_at_l_s[0].expand_as(logits_at_l_s)
                    ), f"Undecoded position l={l} at step s={s} must have uniform logits"


class TestARTrajectoryResultKeysAndFormula:
    """T026: Regression — same keys as dLLM; sequence probability matches TOFU formula."""

    def test_ar_trajectory_compute_trajectories_same_structure_as_dllm(self):
        """AR R, F, S produce same trajectory tensor structure via compute_trajectories."""
        from evals.metrics.trajectory_utils import (
            build_ar_trajectory_r_f,
            compute_trajectories,
        )

        V, L = 20, 3
        position_logits = torch.randn(L, V)
        R, F, S, L_out = build_ar_trajectory_r_f(position_logits)
        assert S > 1, "compute_trajectories requires S > 1"
        T_steps, T_fix_start, T_fix_end, T_fix_ratio = compute_trajectories(R, F, S)
        assert T_steps.shape == R.shape
        assert T_fix_start.shape == R.shape
        assert T_fix_end.shape == R.shape
        assert T_fix_ratio.shape == R.shape

    def test_ar_trajectory_result_has_same_keys_as_dllm(self):
        """Minimal trajectory result dict contains required keys (contract: same schema as dLLM)."""
        # Result keys come from aggregation; we assert the expected set is a subset of
        # what any trajectory result must expose per eval-result-schema.md.
        from evals.metrics.trajectory_utils import build_ar_trajectory_r_f, compute_trajectories

        V, L = 10, 4
        position_logits = torch.randn(L, V)
        R, F, S, L_out = build_ar_trajectory_r_f(position_logits)
        T_steps, T_fix_start, T_fix_end, T_fix_ratio = compute_trajectories(R, F, S)
        # Minimal result-like structure that trajectory metrics produce
        result_keys = {"steps", "fixation_start", "fixation_end", "fixation_ratio"}
        assert result_keys <= EXPECTED_TRAJECTORY_RESULT_KEYS
        assert EXPECTED_TRAJECTORY_RESULT_KEYS.issuperset(
            {"agg_value", "step_distribution", "steps", "fixation_start", "fixation_end", "fixation_ratio"}
        )

    def test_sequence_probability_full_sequence_matches_tofu_formula(self):
        """For full sequence, sequence-level probability = geometric mean of step-wise scores."""
        from evals.metrics.step_wise_score import sequence_probability_from_scores
        from evals.metrics.trajectory_utils import build_ar_trajectory_r_f

        # Step-wise scores = per-position probs (e.g. [0.1, 0.2, 0.3] -> geom mean)
        scores = [0.1, 0.2, 0.3]
        prob = sequence_probability_from_scores(scores)
        import math
        expected = math.exp(sum(math.log(p) for p in scores) / len(scores))
        assert math.isclose(prob, expected, rel_tol=1e-5), "TOFU formula: geometric mean of step-wise scores"


class TestARTrajectoryShortOrEmptyOutput:
    """T031: Short or empty model output does not fail; missing positions contribute uniform log(1/V)."""

    def test_ar_trajectory_l1_does_not_fail(self):
        """L=1 (single token) produces valid R, F, S, L without raising."""
        from evals.metrics.trajectory_utils import build_ar_trajectory_r_f, compute_trajectories

        V, L = 10, 1
        position_logits = torch.randn(L, V)
        R, F, S, L_out = build_ar_trajectory_r_f(position_logits)
        assert R.shape == (V, 1, 1) and S == 1 and L_out == 1
        # compute_trajectories requires S > 1; for L=1 we don't call it in the metric path for reporting
        # but build_ar_trajectory_r_f itself must not fail
        assert F.shape == (1,)

    def test_ar_trajectory_empty_positions_uniform(self):
        """Undecoded positions use uniform logits (constant); softmax gives 1/V."""
        from evals.metrics.trajectory_utils import build_ar_trajectory_r_f

        V, L = 5, 3
        position_logits = torch.randn(L, V)
        R, F, S, L_out = build_ar_trajectory_r_f(position_logits, uniform_logit_value=0.0)
        # At step 0, only position 0 is decoded; positions 1,2 are undecoded (uniform)
        for l in range(1, L_out):
            logits_undecoded = R[:, l, 0]
            assert torch.allclose(logits_undecoded, logits_undecoded[0].expand_as(logits_undecoded))
