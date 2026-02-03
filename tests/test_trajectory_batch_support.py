"""
Tests for trajectory metrics arbitrary batch size (B) support.

Verifies that stack_logits_history, compute_trajectories, and _generate_trajectories_for_dataloader
support any batch size B with per-sample trajectories and no discarded samples.
"""

import pytest
import torch
from unittest.mock import Mock

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.trajectory_utils import stack_logits_history, compute_trajectories
from evals.metrics.trajectory_metrics import _generate_trajectories_for_dataloader


class TestStackLogitsHistoryBatchSupport:
    """stack_logits_history returns [B, V, L, S]; no sample discarded."""

    @pytest.mark.parametrize("B", [1, 2, 4])
    def test_output_shape_B_V_L_S(self, B):
        V, L, S = 50, 10, 5
        logits_history = [torch.randn(B, L, V) for _ in range(S)]
        R = stack_logits_history(logits_history)
        assert R.shape == (B, V, L, S), f"Expected ({B}, {V}, {L}, {S}), got {R.shape}"

    def test_two_samples_different_logits_produce_different_R(self):
        """When sample 0 and sample 1 have different logits, R[0] != R[1]."""
        V, L, S = 20, 5, 3
        torch.manual_seed(1)
        logits_history = [torch.randn(2, L, V) for _ in range(S)]
        R = stack_logits_history(logits_history)
        assert R.shape == (2, V, L, S)
        assert not torch.allclose(R[0], R[1]), "R[0] and R[1] should differ when inputs differ"

    def test_per_sample_content_matches_input(self):
        """For each b, R[b] equals permute of stacked[:, b, :, :]."""
        B, V, L, S = 2, 10, 8, 4
        logits_history = [torch.randn(B, L, V) for _ in range(S)]
        stacked = torch.stack(logits_history, dim=0)  # [S, B, L, V]
        R = stack_logits_history(logits_history)
        for b in range(B):
            expected_b = stacked[:, b, :, :].permute(2, 1, 0)  # [V, L, S]
            assert torch.allclose(R[b], expected_b), f"Sample {b} content mismatch"


class TestComputeTrajectoriesBatchSupport:
    """compute_trajectories accepts R [B, V, L, S], F [B, L]; returns 4 x [B, V, L, S]."""

    def test_batched_output_shapes(self):
        B, V, L, S = 2, 10, 8, 6
        R = torch.randn(B, V, L, S)
        F = torch.randint(0, S, (B, L))
        T_steps, T_fix_start, T_fix_end, T_fix_ratio = compute_trajectories(R, F, S)
        assert T_steps.shape == (B, V, L, S)
        assert T_fix_start.shape == (B, V, L, S)
        assert T_fix_end.shape == (B, V, L, S)
        assert T_fix_ratio.shape == (B, V, L, S)

    def test_batched_per_sample_matches_single_sample(self):
        """For each b, output[b] matches compute_trajectories(R[b], F[b], S)."""
        B, V, L, S = 2, 10, 5, 4
        R = torch.randn(B, V, L, S)
        F = torch.randint(0, S, (B, L))
        T_steps_b, T_fs_b, T_fe_b, T_fr_b = compute_trajectories(R, F, S)
        for b in range(B):
            T_s, T_fs, T_fe, T_fr = compute_trajectories(R[b], F[b], S)
            assert torch.allclose(T_steps_b[b], T_s)
            assert torch.allclose(T_fs_b[b], T_fs)
            assert torch.allclose(T_fe_b[b], T_fe)
            assert torch.allclose(T_fr_b[b], T_fr)

    def test_two_samples_different_F_produce_different_trajectories(self):
        """When F[0] != F[1], trajectory tensors differ for the two samples."""
        B, V, L, S = 2, 10, 5, 8
        R = torch.randn(B, V, L, S)
        F = torch.zeros(B, L, dtype=torch.long)
        F[0] = 2
        F[1] = 6
        T_steps, T_fix_start, T_fix_end, T_fix_ratio = compute_trajectories(R, F, S)
        assert not torch.allclose(T_fix_start[0], T_fix_start[1])
        assert not torch.allclose(T_fix_end[0], T_fix_end[1])
        assert not torch.allclose(T_fix_ratio[0], T_fix_ratio[1])


class TestGenerateTrajectoriesForDataloaderBatchSupport:
    """_generate_trajectories_for_dataloader returns per-sample trajectories."""

    def test_batch_size_2_returns_two_different_trajectories(self):
        """With batch_size=2 and different sampler output per sample, trajectories_by_idx has two different trajectories."""
        V, full_len, S = 50, 15, 6
        B = 2
        max_prompt_len = 5
        generated_len = full_len - max_prompt_len

        sampler = Mock()
        torch.manual_seed(123)
        logits_history = [torch.randn(B, full_len, V) for _ in range(S)]
        fixation_steps = torch.randint(0, S, (B, full_len))

        class MockSamplerOutput:
            def __init__(self, logits_history, fixation_steps):
                self.logits_history = logits_history
                self.fixation_steps = fixation_steps

        sampler.sample.return_value = MockSamplerOutput(
            logits_history=logits_history,
            fixation_steps=fixation_steps,
        )

        IGNORE_INDEX = -100

        class MockDataset:
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                input_ids = torch.randint(0, V, (full_len,))
                labels = torch.full((full_len,), IGNORE_INDEX, dtype=torch.long)
                labels[max_prompt_len:] = torch.randint(0, V, (full_len - max_prompt_len,))
                return {"input_ids": input_ids, "labels": labels}

        def collator(batch):
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
                "index": torch.tensor([0, 1]),
            }

        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            MockDataset(),
            batch_size=2,
            collate_fn=collator,
        )
        trajectories_by_idx = _generate_trajectories_for_dataloader(
            sampler, dataloader, {"sampler_kwargs": {}}
        )
        assert len(trajectories_by_idx) == 2
        assert "0" in trajectories_by_idx
        assert "1" in trajectories_by_idx
        traj0 = trajectories_by_idx["0"]["steps"]
        traj1 = trajectories_by_idx["1"]["steps"]
        assert traj0.shape == (V, generated_len, S)
        assert traj1.shape == (V, generated_len, S)
        assert not torch.allclose(traj0, traj1), "Two samples should have different trajectories when logits differ"
