"""
Validate that every trajectory-related tensor has the expected shape.

Covers:
- Sampler output (logits_history, fixation_steps) → trajectories_from_logits
- R, F, S, L from trajectories_from_logits (generated-only and full-sequence)
- Per-sample storage in _generate_trajectories_for_dataloader (R[i], F[i])
- _get_logits_at_step for steps, fixation_start, fixation_end, fixation_ratio → [V, L]
- Privleak dual path: trajectories_by_key, logits_by_key, DualLogitModelWrapper input

Parameters match the OOM run: max_new_tokens=200, steps=50, trajectory_sample_interval=8
→ S_traj=25, L_gen=200. Uses small V for speed.
"""

import pytest
import torch
import math
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.trajectory_utils import (
    stack_logits_history,
    trajectories_from_logits,
    compute_fixation_start_trajectory,
    compute_fixation_end_trajectory,
    compute_fixation_ratio_trajectory,
)
from evals.metrics.trajectory_metrics import _get_logits_at_step
from evals.metrics.trajectory_adapters import DualLogitModelWrapper


# --- Constants matching the OOM run (small V for test speed) ---
B = 1
max_new_tokens = 200
steps_diffusion = 50
trajectory_sample_interval = 8
S_traj = math.ceil(max_new_tokens / trajectory_sample_interval)  # 25
L_gen = max_new_tokens  # 200
V = 1000  # small for test; real is ~128256
max_prompt_len = 64  # arbitrary
T_full = max_prompt_len + max_new_tokens  # 264


class TestSamplerOutputShapes:
    """Shapes that the sampler produces (we simulate them)."""

    def test_logits_history_generated_only_interval(self):
        """With trajectory_sample_interval=8: S_traj entries, each [B, L_gen, V]."""
        logits_history = [torch.randn(B, L_gen, V) for _ in range(S_traj)]
        assert len(logits_history) == S_traj
        for t in logits_history:
            assert t.shape == (B, L_gen, V), f"expected (B={B}, L_gen={L_gen}, V={V})"

    def test_fixation_steps_full_sequence_length(self):
        """fixation_steps from sampler is [B, T_full] with T_full = prompt + generated."""
        fixation_steps = torch.zeros(B, T_full, dtype=torch.long)
        assert fixation_steps.shape == (B, T_full)


class TestStackLogitsHistory:
    """stack_logits_history: list of [B, L, V] → R [B, V, L, S]."""

    def test_generated_only_S_traj_steps(self):
        logits_history = [torch.randn(B, L_gen, V) for _ in range(S_traj)]
        R = stack_logits_history(logits_history)
        assert R.shape == (B, V, L_gen, S_traj), (
            f"R should be (B={B}, V={V}, L={L_gen}, S={S_traj}), got {R.shape}"
        )


class TestTrajectoriesFromLogitsGeneratedOnly:
    """trajectories_from_logits in generated-only branch (L_logits < T_fixation)."""

    @pytest.fixture
    def generated_only_inputs(self):
        logits_history = [torch.randn(B, L_gen, V) for _ in range(S_traj)]
        fixation_steps = torch.randint(0, S_traj, (B, T_full), dtype=torch.long)
        prompt_lens = [max_prompt_len] * B
        return logits_history, fixation_steps, prompt_lens

    def test_R_F_S_L_shapes(self, generated_only_inputs):
        logits_history, fixation_steps, prompt_lens = generated_only_inputs
        out = trajectories_from_logits(
            logits_history, fixation_steps, prompt_lens, return_trajectory_tensors=False
        )
        R, F, S, L = out["R"], out["F"], out["S"], out["L"]
        assert R.shape == (B, V, L_gen, S_traj), f"R: expected (B, V, L_gen, S_traj), got {R.shape}"
        assert F.shape == (B, L_gen), f"F: expected (B, L_gen), got {F.shape}"
        assert S == S_traj, f"S: expected {S_traj}, got {S}"
        assert L == L_gen, f"L: expected {L_gen}, got {L}"


class TestPerSampleStorage:
    """What we store per sample in trajectories_by_idx (e.g. in _generate_trajectories_for_dataloader)."""

    @pytest.fixture
    def one_sample_traj(self):
        """One sample: R [V, L, S], F [L], S, L."""
        R_batch = torch.randn(B, V, L_gen, S_traj)
        F_batch = torch.randint(0, S_traj, (B, L_gen), dtype=torch.long)
        R_sample = R_batch[0]
        F_sample = F_batch[0]
        return {
            "R": R_sample,
            "F": F_sample,
            "S": S_traj,
            "L": L_gen,
        }

    def test_per_sample_R_shape(self, one_sample_traj):
        """Each sample's R must be [V, L, S] (generated region only)."""
        R = one_sample_traj["R"]
        assert R.shape == (V, L_gen, S_traj), f"per-sample R: expected (V, L, S), got {R.shape}"

    def test_per_sample_F_shape(self, one_sample_traj):
        """Each sample's F must be [L]."""
        F = one_sample_traj["F"]
        assert F.shape == (L_gen,), f"per-sample F: expected (L,), got {F.shape}"


class TestGetLogitsAtStep:
    """_get_logits_at_step(traj, traj_name, step) → [V, L]."""

    @pytest.fixture
    def one_sample_traj(self):
        R = torch.randn(V, L_gen, S_traj)
        F = torch.randint(0, S_traj, (L_gen,), dtype=torch.long)
        return {"R": R, "F": F, "S": S_traj, "L": L_gen}

    def test_steps_returns_V_L(self, one_sample_traj):
        for step in [0, S_traj - 1]:
            logits = _get_logits_at_step(one_sample_traj, "steps", step)
            assert logits.shape == (V, L_gen), f"steps step {step}: expected (V, L), got {logits.shape}"

    def test_fixation_start_returns_V_L(self, one_sample_traj):
        logits = compute_fixation_start_trajectory(
            one_sample_traj["R"], 0, one_sample_traj["F"]
        )
        assert logits.shape == (V, L_gen)

    def test_fixation_end_returns_V_L(self, one_sample_traj):
        logits = compute_fixation_end_trajectory(
            one_sample_traj["R"], S_traj - 1, one_sample_traj["F"]
        )
        assert logits.shape == (V, L_gen)

    def test_fixation_ratio_returns_V_L(self, one_sample_traj):
        logits = compute_fixation_ratio_trajectory(
            one_sample_traj["R"], S_traj // 2, one_sample_traj["F"]
        )
        assert logits.shape == (V, L_gen)


class TestPrivleakDualPathShapes:
    """Privleak dual path: trajectories_by_key → logits_by_key → DualLogitModelWrapper."""

    @pytest.fixture
    def forget_and_holdout_trajectories(self):
        """Simulate forget_traj and holdout_traj as returned by _generate_trajectories_for_dataloader."""
        def make_traj():
            return {
                "R": torch.randn(V, L_gen, S_traj),
                "F": torch.randint(0, S_traj, (L_gen,), dtype=torch.long),
                "S": S_traj,
                "L": L_gen,
            }
        return {"forget": {"0": make_traj(), "1": make_traj()}, "holdout": {"0": make_traj(), "1": make_traj()}}

    def test_logits_by_key_per_step_shape(self, forget_and_holdout_trajectories):
        """For privleak we build logits_by_key[key][idx] = _get_logits_at_step(..., "steps", step) → [V, L]."""
        trajectories_by_key = forget_and_holdout_trajectories
        for step in range(S_traj):
            logits_by_key = {}
            for key, traj_by_idx in trajectories_by_key.items():
                logits_by_key[key] = {
                    idx: _get_logits_at_step(traj, "steps", step)
                    for idx, traj in traj_by_idx.items()
                }
            for key in ("forget", "holdout"):
                for idx in ("0", "1"):
                    lg = logits_by_key[key][idx]
                    assert lg.shape == (V, L_gen), (
                        f"logits_by_key[{key}][{idx}] at step {step}: expected (V, L), got {lg.shape}"
                    )

    def test_dual_logit_model_wrapper_accepts_V_L(self, forget_and_holdout_trajectories):
        """DualLogitModelWrapper expects logits_by_key[key][idx] as [V, L]; it may transpose to [1, L, V]."""
        trajectories_by_key = forget_and_holdout_trajectories
        step = 0
        logits_by_key = {}
        for key, traj_by_idx in trajectories_by_key.items():
            logits_by_key[key] = {
                idx: _get_logits_at_step(traj, "steps", step)
                for idx, traj in traj_by_idx.items()
            }
        device = torch.device("cpu")
        wrapper = DualLogitModelWrapper(logits_by_key, device)
        wrapper.set_dataset_key("forget")
        # __call__ with batch that has index
        out = wrapper(index=0)
        assert hasattr(out, "logits")
        assert out.logits.dim() == 3, "Output logits should be [B, L, V]"
        assert out.logits.shape[0] == 1 and out.logits.shape[1] == L_gen and out.logits.shape[2] == V


class TestFullSequenceConsistency:
    """Sanity: S_traj and L_gen match the config."""

    def test_S_traj_formula(self):
        assert S_traj == math.ceil(max_new_tokens / trajectory_sample_interval)
        assert S_traj == 25

    def test_L_gen_equals_max_new_tokens(self):
        assert L_gen == max_new_tokens
        assert L_gen == 200


class TestOOMRunExpectedSizes:
    """Document expected tensor sizes for the real OOM run (V=128256, L=200, S=25).
    No large allocations; only shape and element-count checks."""

    V_REAL = 128256
    L_REAL = 200
    S_REAL = 25

    def test_per_sample_R_element_count(self):
        """Per-sample R [V, L, S]: 128256 * 200 * 25 = 641_280_000 elements."""
        elements = self.V_REAL * self.L_REAL * self.S_REAL
        assert elements == 641_280_000
        bytes_fp32 = elements * 4
        assert bytes_fp32 == 2_565_120_000  # ~2.56 GB

    def test_ten_samples_forget_traj_R_total(self):
        """10 samples × R per sample: 10 * 2.56 GB ≈ 25.6 GB."""
        per_sample = self.V_REAL * self.L_REAL * self.S_REAL * 4
        ten_samples = 10 * per_sample
        assert ten_samples == 25_651_200_000  # 25.65 GB

    def test_logits_history_entry_generated_only(self):
        """Each logits_history entry: [B, L_gen, V] = [1, 200, 128256]."""
        B, L, V = 1, self.L_REAL, self.V_REAL
        elements_per_entry = B * L * V
        assert elements_per_entry == 25_651_200  # ~98 MB per entry
        assert self.S_REAL * elements_per_entry * 4 == 2_565_120_000  # same as one R
