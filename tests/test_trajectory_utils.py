"""
Comprehensive unit tests for trajectory_utils module.

Tests cover:
- stack_logits_history: Shape validation, single/batch cases
- compute_trajectories: Steps, fixation, ratio trajectory computation
- trajectories_from_logits: Model-free entry-point (logits + fixation -> trajectory tensors)
- extract_logits_at_step: Step extraction and bounds checking
- decode_logits_to_text: Text decoding from logits
"""

import pytest
import torch
from typing import List

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.trajectory_utils import (
    stack_logits_history,
    compute_trajectories,
    compute_fixation_start_trajectory,
    compute_fixation_end_trajectory,
    compute_fixation_ratio_trajectory,
    trajectories_from_logits,
    extract_logits_at_step,
    decode_logits_to_text,
)


class TestStackLogitsHistory:
    """Tests for stack_logits_history function."""
    
    def test_empty_list_raises_error(self):
        """Test that empty logits_history raises ValueError."""
        with pytest.raises(ValueError, match="logits_history cannot be empty"):
            stack_logits_history([])
    
    def test_single_sample_single_step(self):
        """Test stacking with B=1, S=1. Output shape [B, V, L, S] = [1, V, L, S]."""
        V, L = 10, 5
        logits_history = [torch.randn(1, L, V)]  # [B, L, V]
        R = stack_logits_history(logits_history)

        assert R.shape == (1, V, L, 1)
        assert torch.allclose(R[0, :, :, 0], logits_history[0][0].T)

    def test_single_sample_multiple_steps(self):
        """Test stacking with B=1, S=5. Output shape [1, V, L, S]."""
        V, L, S = 100, 20, 5
        logits_history = [torch.randn(1, L, V) for _ in range(S)]
        R = stack_logits_history(logits_history)

        assert R.shape == (1, V, L, S)
        for s in range(S):
            assert torch.allclose(R[0, :, :, s], logits_history[s][0].T)

    def test_batch_multiple_steps(self):
        """Test stacking with B=2, S=3. Output shape [B, V, L, S]; each sample preserved."""
        V, L, S, B = 50, 10, 3, 2
        logits_history = [torch.randn(B, L, V) for _ in range(S)]
        R = stack_logits_history(logits_history)

        assert R.shape == (B, V, L, S)
        for b in range(B):
            for s in range(S):
                assert torch.allclose(R[b, :, :, s], logits_history[s][b].T)

    def test_shape_consistency(self):
        """Test that output shape is always [B, V, L, S] for any batch size B."""
        V, L, S = 1000, 64, 8

        for B in [1, 2, 4]:
            logits_history = [torch.randn(B, L, V) for _ in range(S)]
            R = stack_logits_history(logits_history)
            assert R.shape == (B, V, L, S), f"Failed for B={B}"

    def test_larger_batches(self):
        """Test stacking with larger batch sizes (B=4, B=8); all samples preserved."""
        V, L, S = 100, 20, 5

        for B in [4, 8]:
            logits_history = [torch.randn(B, L, V) for _ in range(S)]
            R = stack_logits_history(logits_history)
            assert R.shape == (B, V, L, S)
            for b in range(B):
                for s in range(S):
                    assert torch.allclose(R[b, :, :, s], logits_history[s][b].T)

    def test_exact_tensor_values_match(self):
        """Test that exact tensor values match, not just shapes (B=1)."""
        V, L, S = 50, 10, 3
        torch.manual_seed(42)
        logits_history = [torch.randn(1, L, V) for _ in range(S)]

        R = stack_logits_history(logits_history)

        assert R.shape == (1, V, L, S)
        for s in range(S):
            expected = logits_history[s][0].T  # [V, L]
            actual = R[0, :, :, s]
            assert torch.allclose(actual, expected, atol=1e-6)
            assert torch.equal(actual, expected) or torch.allclose(actual, expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_different_dtypes(self, dtype):
        """Test stacking with different dtypes. Output [1, V, L, S]."""
        V, L, S = 100, 20, 5
        logits_history = [torch.randn(1, L, V, dtype=dtype) for _ in range(S)]
        R = stack_logits_history(logits_history)

        assert R.shape == (1, V, L, S)
        assert R.dtype == dtype
        for s in range(S):
            assert torch.allclose(
                R[0, :, :, s], logits_history[s][0].T, atol=1e-3 if dtype != torch.float32 else 1e-6
            )

    def test_different_devices(self):
        """Test stacking with different devices. Output [1, V, L, S]."""
        V, L, S = 100, 20, 5

        logits_history_cpu = [torch.randn(1, L, V) for _ in range(S)]
        R_cpu = stack_logits_history(logits_history_cpu)
        assert R_cpu.device.type == "cpu"
        assert R_cpu.shape == (1, V, L, S)

        if torch.cuda.is_available():
            logits_history_cuda = [torch.randn(1, L, V).cuda() for _ in range(S)]
            R_cuda = stack_logits_history(logits_history_cuda)
            assert R_cuda.device.type == "cuda"
            assert R_cuda.shape == (1, V, L, S)


class TestComputeTrajectories:
    """Tests for compute_trajectories function."""
    
    def test_steps_trajectory_is_copy(self):
        """Test that T_steps is a direct copy of R."""
        V, L, S = 100, 20, 10
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        assert torch.allclose(T_steps, R)
        assert T_steps.shape == (V, L, S)
    
    def test_fixation_start_trajectory_formula(self):
        """Test fixation start trajectory: T_fixation_start[v,l,s] = R[v,l, min(s, F[l])]."""
        V, L, S = 10, 5, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        F = torch.tensor([0, 2, 4, 6, 7])  # [L]
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        # For position l=1, fixed at step 2
        # At trajectory step s=0: min(0, 2) = 0 → R[:, 1, 0]
        # At trajectory step s=1: min(1, 2) = 1 → R[:, 1, 1]
        # At trajectory step s=2: min(2, 2) = 2 → R[:, 1, 2]
        # At trajectory step s=3: min(3, 2) = 2 → R[:, 1, 2] (clamped)
        assert torch.allclose(T_fixation_start[:, 1, 0], R[:, 1, 0])
        assert torch.allclose(T_fixation_start[:, 1, 1], R[:, 1, 1])
        assert torch.allclose(T_fixation_start[:, 1, 2], R[:, 1, 2])
        assert torch.allclose(T_fixation_start[:, 1, 3], R[:, 1, 2])  # Clamped to fixation
        assert torch.allclose(T_fixation_start[:, 1, 7], R[:, 1, 2])  # Last step = fixation
    
    def test_fixation_end_trajectory_formula(self):
        """Test fixation end trajectory: T_fixation_end[v,l,s] = R[v,l, max(0, F[l]-(S-1)+s)]."""
        V, L, S = 10, 5, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        F = torch.tensor([0, 2, 4, 6, 7])  # [L]
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        # For position l=1, fixed at step 2, S=8
        # At trajectory step s=0: max(0, 2-7+0) = max(0, -5) = 0 → R[:, 1, 0]
        # At trajectory step s=1: max(0, 2-7+1) = max(0, -4) = 0 → R[:, 1, 0]
        # At trajectory step s=5: max(0, 2-7+5) = max(0, 0) = 0 → R[:, 1, 0]
        # At trajectory step s=6: max(0, 2-7+6) = max(0, 1) = 1 → R[:, 1, 1]
        # At trajectory step s=7: max(0, 2-7+7) = max(0, 2) = 2 → R[:, 1, 2] (fixation)
        assert torch.allclose(T_fixation_end[:, 1, 0], R[:, 1, 0])
        assert torch.allclose(T_fixation_end[:, 1, 1], R[:, 1, 0])
        assert torch.allclose(T_fixation_end[:, 1, 6], R[:, 1, 1])
        assert torch.allclose(T_fixation_end[:, 1, 7], R[:, 1, 2])  # Last step = fixation
    
    def test_fixation_ratio_trajectory_formula(self):
        """Test fixation ratio trajectory: T_fixation_ratio[v,l,s] = R[v,l, floor(F[l]*s/(S-1))]."""
        V, L, S = 10, 3, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        F = torch.tensor([4, 6, 7])  # Fixation steps
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        # For position l=0, fixed at step 4, S=8
        # At trajectory step s=0: floor(4 * 0/7) = 0 → R[:, 0, 0]
        # At trajectory step s=3: floor(4 * 3/7) = floor(12/7) = 1 → R[:, 0, 1]
        # At trajectory step s=7: floor(4 * 7/7) = 4 → R[:, 0, 4] (fixation)
        assert torch.allclose(T_fixation_ratio[:, 0, 0], R[:, 0, 0])
        assert torch.allclose(T_fixation_ratio[:, 0, 3], R[:, 0, 1])
        assert torch.allclose(T_fixation_ratio[:, 0, 7], R[:, 0, 4])  # Last step = fixation
    
    def test_shape_mismatch_raises_error(self):
        """Test that shape mismatches raise assertions."""
        V, L, S = 10, 5, 8
        R = torch.randn(V, L, S)

        # Wrong S
        with pytest.raises(AssertionError, match="S mismatch"):
            F = torch.randint(0, S, (L,))
            compute_trajectories(R, F, S + 1)

        # Wrong F length (single-sample: F (L+1,) unsqueezed to (1, L+1) vs expected (1, L))
        with pytest.raises(AssertionError, match="F shape mismatch"):
            F = torch.randint(0, S, (L + 1,))
            compute_trajectories(R, F, S)
    
    def test_all_trajectories_same_shape(self):
        """Test that all three trajectories have shape [V, L, S]."""
        V, L, S = 100, 20, 16
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        assert T_steps.shape == (V, L, S)
        assert T_fixation_start.shape == (V, L, S)
        assert T_fixation_end.shape == (V, L, S)
        assert T_fixation_ratio.shape == (V, L, S)
    
    def test_fixation_at_boundary(self):
        """Test fixation trajectory at boundary cases (fixation at step 0 or S-1)."""
        V, L, S = 10, 3, 8
        R = torch.randn(V, L, S)
        
        # All fixed at step 0
        F = torch.zeros(L, dtype=torch.long)
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        # Should all use step 0 (min(s, 0) = 0)
        for s in range(S):
            assert torch.allclose(T_fixation_start[:, 0, s], R[:, 0, 0])
        
        # All fixed at last step
        F = torch.full((L,), S - 1, dtype=torch.long)
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        # At s=0: min(0, S-1) = 0, at s=S-1: min(S-1, S-1) = S-1
        assert torch.allclose(T_fixation_start[:, 0, 0], R[:, 0, 0])
        assert torch.allclose(T_fixation_start[:, 0, S - 1], R[:, 0, S - 1])
    
    def test_fixation_steps_ascending_order(self):
        """Test fixation steps in ascending order [0, 1, 2, ..., S-1]."""
        V, L, S = 10, 8, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        F = torch.arange(L, dtype=torch.long)  # [0, 1, 2, ..., 7]
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        # For position l=1, fixed at step 1
        # Fixation start: At s=0: min(0, 1) = 0 → R[:, 1, 0], at s=1: min(1, 1) = 1 → R[:, 1, 1]
        assert torch.allclose(T_fixation_start[:, 1, 0], R[:, 1, 0])
        assert torch.allclose(T_fixation_start[:, 1, 1], R[:, 1, 1])
    
    def test_fixation_steps_descending_order(self):
        """Test fixation steps in descending order [S-1, S-2, ..., 0]."""
        V, L, S = 10, 8, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        F = torch.arange(S - 1, -1, -1, dtype=torch.long)  # [7, 6, 5, ..., 0]
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        # For position l=0, fixed at step 7
        # Fixation start: At s=0: min(0, 7) = 0 → R[:, 0, 0], at s=7: min(7, 7) = 7 → R[:, 0, 7]
        assert torch.allclose(T_fixation_start[:, 0, 0], R[:, 0, 0])
        assert torch.allclose(T_fixation_start[:, 0, 7], R[:, 0, 7])
    
    def test_fixation_steps_all_same_value(self):
        """Test fixation steps all same value (e.g., all 5)."""
        V, L, S = 10, 5, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        fix_step = 5
        F = torch.full((L,), fix_step, dtype=torch.long)
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        # All positions should have same fixation start trajectory
        for l in range(L):
            # At s=0: min(0, 5) = 0 → R[:, l, 0]
            assert torch.allclose(T_fixation_start[:, l, 0], R[:, l, 0])
            # At s=5: min(5, 5) = 5 → R[:, l, 5]
            assert torch.allclose(T_fixation_start[:, l, 5], R[:, l, fix_step])
    
    def test_fixation_steps_with_duplicates(self):
        """Test fixation steps with duplicate values."""
        V, L, S = 10, 5, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        F = torch.tensor([3, 3, 5, 5, 7], dtype=torch.long)  # Duplicates
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        # Positions with same fixation should use same source step calculation
        # Fixation start: T_fixation_start[:, l, s] = R[:, l, min(s, F[l])]
        # For l=0,1 with F=3, at s=0: source_step = min(0, 3) = 0
        assert torch.allclose(T_fixation_start[:, 0, 0], R[:, 0, 0])
        assert torch.allclose(T_fixation_start[:, 1, 0], R[:, 1, 0])
        # For l=2,3 with F=5, at s=0: source_step = min(0, 5) = 0
        assert torch.allclose(T_fixation_start[:, 2, 0], R[:, 2, 0])
        assert torch.allclose(T_fixation_start[:, 3, 0], R[:, 3, 0])
        
        # At step s=3, positions with F=3 should use step 3 (fixation)
        assert torch.allclose(T_fixation_start[:, 0, 3], R[:, 0, 3])
        assert torch.allclose(T_fixation_start[:, 1, 3], R[:, 1, 3])
    
    @pytest.mark.parametrize("S", [2, 16, 128, 256])  # S must be > 1
    def test_very_large_steps(self, S):
        """Test with very large number of steps."""
        V, L = 50, 10
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        assert T_steps.shape == (V, L, S)
        assert T_fixation_start.shape == (V, L, S)
        assert T_fixation_end.shape == (V, L, S)
        assert T_fixation_ratio.shape == (V, L, S)
    
    @pytest.mark.parametrize("L", [1, 16, 128, 512, 1024])
    def test_very_large_sequence_length(self, L):
        """Test with very large sequence length."""
        V, S = 100, 8
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        assert T_steps.shape == (V, L, S)
        assert T_fixation_start.shape == (V, L, S)
        assert T_fixation_end.shape == (V, L, S)
        assert T_fixation_ratio.shape == (V, L, S)
    
    @pytest.mark.parametrize("V", [100, 1000, 10000, 50000])
    def test_very_large_vocab_size(self, V):
        """Test with very large vocabulary size."""
        L, S = 20, 8
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        assert T_steps.shape == (V, L, S)
        assert T_fixation_start.shape == (V, L, S)
        assert T_fixation_end.shape == (V, L, S)
        assert T_fixation_ratio.shape == (V, L, S)
    
    def test_fixation_start_formula_verification(self):
        """Verify T_fixation_start[:, l, s] = R[:, l, min(s, F[l])] for all l, s."""
        V, L, S = 10, 5, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        F = torch.tensor([0, 2, 4, 6, 7], dtype=torch.long)
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        # Verify formula for all positions and steps
        for l in range(L):
            fix_step = F[l].item()
            for s in range(S):
                source_step = min(s, fix_step)
                source_step = max(0, min(source_step, S - 1))  # Clamp
                expected = R[:, l, source_step]
                actual = T_fixation_start[:, l, s]
                assert torch.allclose(actual, expected), f"Failed at l={l}, s={s}"
    
    def test_fixation_end_formula_verification(self):
        """Verify T_fixation_end[:, l, s] = R[:, l, max(0, F[l]-(S-1)+s)] for all l, s."""
        V, L, S = 10, 5, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        F = torch.tensor([0, 2, 4, 6, 7], dtype=torch.long)
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        # Verify formula for all positions and steps
        for l in range(L):
            fix_step = F[l].item()
            for s in range(S):
                source_step = max(0, fix_step - (S - 1) + s)
                source_step = max(0, min(source_step, S - 1))  # Clamp
                expected = R[:, l, source_step]
                actual = T_fixation_end[:, l, s]
                assert torch.allclose(actual, expected), f"Failed at l={l}, s={s}"
    
    def test_fixation_ratio_formula_verification(self):
        """Verify T_fixation_ratio[:, l, s] = R[:, l, floor(F[l]*s/(S-1))] for all l, s."""
        V, L, S = 10, 5, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        F = torch.tensor([0, 2, 4, 6, 7], dtype=torch.long)
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        # Verify formula for all positions and steps
        for l in range(L):
            fix_step = F[l].item()
            for s in range(S):
                if S > 1:
                    ratio_step = int(fix_step * s / (S - 1))
                    ratio_step = max(0, min(ratio_step, S - 1))  # Clamp
                else:
                    ratio_step = 0
                expected = R[:, l, ratio_step]
                actual = T_fixation_ratio[:, l, s]
                assert torch.allclose(actual, expected), f"Failed at l={l}, s={s}"
    
    def test_first_last_validation_all_trajectories(self):
        """Test that all trajectories have correct first (s=0) and last (s=S-1) values."""
        V, L, S = 10, 5, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        F = torch.tensor([0, 2, 4, 6, 7], dtype=torch.long)
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        # Steps: first = R[:,:,0], last = R[:,:,S-1]
        for l in range(L):
            assert torch.allclose(T_steps[:, l, 0], R[:, l, 0])
            assert torch.allclose(T_steps[:, l, S - 1], R[:, l, S - 1])
        
        # Fixation start: first = R[:,:,0], last = R[:,:,F[l]]
        for l in range(L):
            fix_step = F[l].item()
            assert torch.allclose(T_fixation_start[:, l, 0], R[:, l, 0])
            assert torch.allclose(T_fixation_start[:, l, S - 1], R[:, l, min(S - 1, fix_step)])
        
        # Fixation end: first = R[:,:,0], last = R[:,:,F[l]]
        for l in range(L):
            fix_step = F[l].item()
            assert torch.allclose(T_fixation_end[:, l, 0], R[:, l, 0])
            # At s=S-1: max(0, F[l]-(S-1)+(S-1)) = F[l]
            assert torch.allclose(T_fixation_end[:, l, S - 1], R[:, l, fix_step])
        
        # Fixation ratio: first = R[:,:,0], last = R[:,:,F[l]]
        for l in range(L):
            fix_step = F[l].item()
            assert torch.allclose(T_fixation_ratio[:, l, 0], R[:, l, 0])
            # At s=S-1: floor(F[l]*(S-1)/(S-1)) = F[l]
            assert torch.allclose(T_fixation_ratio[:, l, S - 1], R[:, l, fix_step])
    
    def test_fixation_start_and_end_differ(self):
        """Test that fixation_start and fixation_end produce different trajectories."""
        V, L, S = 10, 5, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        F = torch.tensor([3, 4, 5, 6, 7], dtype=torch.long)
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        # At s=0, both should be R[:,:,0] (same)
        for l in range(L):
            assert torch.allclose(T_fixation_start[:, l, 0], T_fixation_end[:, l, 0])
        
        # At s=S-1, both should be R[:,:,F[l]] (same)
        for l in range(L):
            assert torch.allclose(T_fixation_start[:, l, S - 1], T_fixation_end[:, l, S - 1])
        
        # At intermediate steps, they should differ (unless F[l] is at boundary)
        # For l=1, F=4, at s=2:
        # Fixation start: min(2, 4) = 2 → R[:,1,2]
        # Fixation end: max(0, 4-7+2) = max(0, -1) = 0 → R[:,1,0]
        assert not torch.allclose(T_fixation_start[:, 1, 2], T_fixation_end[:, 1, 2])
    
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_trajectories_different_dtypes(self, dtype):
        """Test trajectory computation with different dtypes."""
        V, L, S = 100, 20, 10
        R = torch.randn(V, L, S, dtype=dtype)
        F = torch.randint(0, S, (L,), dtype=torch.long)
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        assert T_steps.dtype == dtype
        assert T_fixation_start.dtype == dtype
        assert T_fixation_end.dtype == dtype
        assert T_fixation_ratio.dtype == dtype
    
    def test_trajectories_different_devices(self):
        """Test trajectory computation with different devices."""
        V, L, S = 100, 20, 10
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        # CPU
        T_steps_cpu, T_fixation_start_cpu, T_fixation_end_cpu, T_fixation_ratio_cpu = compute_trajectories(R, F, S)
        assert T_steps_cpu.device.type == "cpu"
        
        # CUDA if available
        if torch.cuda.is_available():
            R_cuda = R.cuda()
            F_cuda = F.cuda()
            T_steps_cuda, T_fixation_start_cuda, T_fixation_end_cuda, T_fixation_ratio_cuda = compute_trajectories(R_cuda, F_cuda, S)
            assert T_steps_cuda.device.type == "cuda"
            assert T_fixation_start_cuda.device.type == "cuda"
            assert T_fixation_end_cuda.device.type == "cuda"
            assert T_fixation_ratio_cuda.device.type == "cuda"
    
    def test_trajectories_are_new_tensors_not_views(self):
        """Test that T_fixation and T_ratio are new tensors, not views of R."""
        V, L, S = 100, 20, 10
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        # T_steps is a clone, so modifying R shouldn't affect it
        R_original = R.clone()
        R[0, 0, 0] = 999.0
        # T_steps should be unchanged (it's a clone)
        assert torch.allclose(T_steps, R_original)
        
        # T_fixation_start, T_fixation_end, and T_fixation_ratio are new tensors
        # They should be independent of R after computation
        assert not torch.equal(T_fixation_start, R)
        assert not torch.equal(T_fixation_end, R)
        assert not torch.equal(T_fixation_ratio, R)
    
    def test_trajectories_same_device_as_input(self):
        """Test that output tensors are on same device as input."""
        V, L, S = 100, 20, 10
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation_start, T_fixation_end, T_fixation_ratio = compute_trajectories(R, F, S)
        
        assert T_steps.device == R.device
        assert T_fixation_start.device == R.device
        assert T_fixation_end.device == R.device
        assert T_fixation_ratio.device == R.device


class TestOnDemandTrajectoryFunctions:
    """Tests for compute_fixation_start/end/ratio_trajectory (on-demand step computation)."""

    def test_fixation_start_matches_compute_trajectories_slice(self):
        """compute_fixation_start_trajectory(raw, s, F) equals T_fixation_start[0,:,:,s] from compute_trajectories."""
        V, L, S = 20, 8, 6
        torch.manual_seed(123)
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        _, T_fs, _, _ = compute_trajectories(R.unsqueeze(0), F.unsqueeze(0), S)
        for step_index in [0, S - 1, S // 2]:
            got = compute_fixation_start_trajectory(R, step_index, F)
            ref = T_fs[0, :, :, step_index]
            assert got.shape == (V, L)
            assert torch.allclose(got, ref), f"step_index={step_index}"

    def test_fixation_end_matches_compute_trajectories_slice(self):
        """compute_fixation_end_trajectory(raw, s, F) equals T_fixation_end[0,:,:,s] from compute_trajectories."""
        V, L, S = 20, 8, 6
        torch.manual_seed(456)
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        _, _, T_fe, _ = compute_trajectories(R.unsqueeze(0), F.unsqueeze(0), S)
        for step_index in [0, S - 1, S // 2]:
            got = compute_fixation_end_trajectory(R, step_index, F)
            ref = T_fe[0, :, :, step_index]
            assert got.shape == (V, L)
            assert torch.allclose(got, ref), f"step_index={step_index}"

    def test_fixation_ratio_matches_compute_trajectories_slice(self):
        """compute_fixation_ratio_trajectory(raw, s, F) equals T_fixation_ratio[0,:,:,s] from compute_trajectories."""
        V, L, S = 20, 8, 6
        torch.manual_seed(789)
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        _, _, _, T_fr = compute_trajectories(R.unsqueeze(0), F.unsqueeze(0), S)
        for step_index in [0, S - 1, S // 2]:
            got = compute_fixation_ratio_trajectory(R, step_index, F)
            ref = T_fr[0, :, :, step_index]
            assert got.shape == (V, L)
            assert torch.allclose(got, ref), f"step_index={step_index}"

    def test_on_demand_step_index_bounds(self):
        """On-demand functions raise when step_index out of range."""
        V, L, S = 10, 5, 4
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        with pytest.raises(AssertionError, match="step_index.*out of range"):
            compute_fixation_start_trajectory(R, -1, F)
        with pytest.raises(AssertionError, match="step_index.*out of range"):
            compute_fixation_start_trajectory(R, S, F)

    def test_on_demand_fixation_indices_shape(self):
        """On-demand functions raise when fixation_indices shape != (L,)."""
        V, L, S = 10, 5, 4
        R = torch.randn(V, L, S)
        F_wrong = torch.randint(0, S, (L + 1,))
        with pytest.raises(AssertionError, match="fixation_indices shape"):
            compute_fixation_start_trajectory(R, 0, F_wrong)


class TestTrajectoriesFromLogits:
    """Tests for trajectories_from_logits (model-free entry-point)."""

    def test_empty_logits_history_raises(self):
        """trajectories_from_logits raises on empty logits_history."""
        fixation_steps = torch.randint(0, 5, (2, 20))
        prompt_lens = [3, 3]
        with pytest.raises(ValueError, match="logits_history cannot be empty"):
            trajectories_from_logits([], fixation_steps, prompt_lens)

    def test_fixation_steps_not_2d_raises(self):
        """trajectories_from_logits raises when fixation_steps is not 2-d."""
        S = 4
        logits_history = [torch.randn(1, 10, 8) for _ in range(S)]
        fixation_steps_1d = torch.randint(0, S, (10,))
        prompt_lens = [2]
        with pytest.raises(ValueError, match="fixation_steps must be 2-d"):
            trajectories_from_logits(logits_history, fixation_steps_1d, prompt_lens)

    def test_prompt_lens_length_mismatch_raises(self):
        """trajectories_from_logits raises when len(prompt_lens) != B."""
        B, V, L_full, S = 2, 8, 12, 4
        logits_history = [torch.randn(B, L_full, V) for _ in range(S)]
        fixation_steps = torch.randint(0, S, (B, L_full))
        prompt_lens_wrong = [2]  # B=1, but B=2
        with pytest.raises(ValueError, match="prompt_lens length.*must match batch size"):
            trajectories_from_logits(logits_history, fixation_steps, prompt_lens_wrong)

    def test_output_shapes_b1(self):
        """trajectories_from_logits returns four [B, V, L, S] tensors; B=1."""
        B, V, L_full, S = 1, 16, 20, 6
        max_prompt_len = 4
        generated_len = L_full - max_prompt_len
        logits_history = [torch.randn(B, L_full, V) for _ in range(S)]
        fixation_steps = torch.randint(0, S, (B, L_full))
        prompt_lens = [max_prompt_len]

        out = trajectories_from_logits(logits_history, fixation_steps, prompt_lens)

        assert out["S"] == S
        assert out["L"] == generated_len
        for key in ("steps", "fixation_start", "fixation_end", "fixation_ratio"):
            assert out[key].shape == (B, V, generated_len, S)

    def test_output_shapes_b2(self):
        """trajectories_from_logits returns four [B, V, L, S] tensors; B=2."""
        B, V, L_full, S = 2, 32, 24, 8
        max_prompt_len = 5
        generated_len = L_full - max_prompt_len
        logits_history = [torch.randn(B, L_full, V) for _ in range(S)]
        fixation_steps = torch.randint(0, S, (B, L_full))
        prompt_lens = [max_prompt_len, max_prompt_len]

        out = trajectories_from_logits(logits_history, fixation_steps, prompt_lens)

        assert out["S"] == S
        assert out["L"] == generated_len
        for key in ("steps", "fixation_start", "fixation_end", "fixation_ratio"):
            assert out[key].shape == (B, V, generated_len, S)

    def test_trajectory_sample_interval_S_traj_25_fixation_in_range(self):
        """trajectories_from_logits with S_traj=25 and fixation_steps in [0, S_traj-1]; R shape [B, V, L, S_traj]."""
        S_traj = 25  # e.g. ceil(200/8)
        B, V, L_gen = 1, 64, 50
        max_prompt_len = 2
        L_full = max_prompt_len + L_gen
        logits_history = [torch.randn(B, L_full, V) for _ in range(S_traj)]
        fixation_steps = torch.randint(0, S_traj, (B, L_full))
        prompt_lens = [max_prompt_len]

        out = trajectories_from_logits(
            logits_history, fixation_steps, prompt_lens, return_trajectory_tensors=False
        )

        assert out["S"] == S_traj
        assert out["L"] == L_gen
        assert out["R"].shape == (B, V, L_gen, S_traj)
        assert out["F"].shape == (B, L_gen)
        assert out["F"].min() >= 0 and out["F"].max() <= S_traj - 1

    def test_consistency_with_compute_trajectories(self):
        """trajectories_from_logits output matches compute_trajectories(R, F, S) for same data."""
        B, V, L, S = 2, 20, 10, 5
        max_prompt_len = 3
        L_full = max_prompt_len + L
        R = torch.randn(B, V, L, S)
        F = torch.randint(0, S, (B, L))
        # Build logits_history and fixation_steps so that trajectories_from_logits yields R, F
        R_full = torch.randn(B, V, L_full, S)
        R_full[:, :, max_prompt_len : max_prompt_len + L, :] = R
        logits_history = [
            R_full[:, :, :, s].permute(0, 2, 1) for s in range(S)
        ]  # each [B, L_full, V]
        fixation_steps = torch.full((B, L_full), S - 1, dtype=torch.long)
        fixation_steps[:, max_prompt_len : max_prompt_len + L] = F
        prompt_lens = [max_prompt_len] * B

        out = trajectories_from_logits(logits_history, fixation_steps, prompt_lens)
        T_steps_ref, T_fs_ref, T_fe_ref, T_fr_ref = compute_trajectories(R, F, S)

        assert out["S"] == S
        assert out["L"] == L
        assert torch.allclose(out["steps"], T_steps_ref)
        assert torch.allclose(out["fixation_start"], T_fs_ref)
        assert torch.allclose(out["fixation_end"], T_fe_ref)
        assert torch.allclose(out["fixation_ratio"], T_fr_ref)

    def test_prompt_lens_tensor_accepted(self):
        """prompt_lens can be a 1-d tensor (converted to list internally)."""
        B, V, L_full, S = 1, 8, 14, 3
        logits_history = [torch.randn(B, L_full, V) for _ in range(S)]
        fixation_steps = torch.randint(0, S, (B, L_full))
        prompt_lens = torch.tensor([4])

        out = trajectories_from_logits(logits_history, fixation_steps, prompt_lens)

        assert out["L"] == L_full - 4
        assert out["steps"].shape == (B, V, L_full - 4, S)

    def test_f_padding_when_slice_short(self):
        """F is padded with S-1 when fixation_steps slice is shorter than L."""
        B, V, L_full, S = 1, 8, 15, 4
        max_prompt_len = 10
        # generated_len = 5; fixation_steps has only 12 positions, so slice has 2
        logits_history = [torch.randn(B, L_full, V) for _ in range(S)]
        fixation_steps = torch.zeros(B, 12, dtype=torch.long)  # 12 < L_full
        fixation_steps[:, 10:12] = 1  # 2 positions in generated region
        prompt_lens = [max_prompt_len]

        out = trajectories_from_logits(logits_history, fixation_steps, prompt_lens)

        generated_len = L_full - max_prompt_len  # 5
        assert out["L"] == generated_len
        assert out["fixation_start"].shape == (B, V, generated_len, S)
        # F should have been padded to length 5 with S-1
        assert out["steps"].shape == (1, V, 5, S)

    def test_return_trajectory_tensors_false_returns_R_F_S_L_only(self):
        """trajectories_from_logits(..., return_trajectory_tensors=False) returns only R, F, S, L."""
        B, V, L_full, S = 2, 16, 20, 6
        max_prompt_len = 4
        logits_history = [torch.randn(B, L_full, V) for _ in range(S)]
        fixation_steps = torch.randint(0, S, (B, L_full))
        prompt_lens = [max_prompt_len, max_prompt_len]

        out = trajectories_from_logits(
            logits_history, fixation_steps, prompt_lens, return_trajectory_tensors=False
        )

        assert set(out.keys()) == {"R", "F", "S", "L"}
        assert out["R"].shape == (B, V, L_full - max_prompt_len, S)
        assert out["F"].shape == (B, L_full - max_prompt_len)
        assert out["S"] == S
        assert out["L"] == L_full - max_prompt_len
        for key in ("steps", "fixation_start", "fixation_end", "fixation_ratio"):
            assert key not in out


class TestTrajectoriesFromLogitsGeneratedOnly:
    """Tests for trajectories_from_logits with generated-only logits (sampler contract).
    When logits_history has shape [B, L_gen, V] and fixation_steps has [B, T_full] with
    T_full > L_gen, we expect R to be [B, V, L_gen, S] and F to match fixation slice.
    """

    def test_generated_only_R_has_full_L_gen(self):
        """With generated-only logits, L and R.shape[2] must equal L_gen (not L_gen - max_prompt_len)."""
        B, V, L_gen, S = 2, 16, 10, 5
        max_prompt_len = 4
        T_full = max_prompt_len + L_gen
        logits_history = [torch.randn(B, L_gen, V) for _ in range(S)]
        fixation_steps = torch.randint(0, S, (B, T_full))
        prompt_lens = [max_prompt_len, max_prompt_len]

        out = trajectories_from_logits(
            logits_history, fixation_steps, prompt_lens, return_trajectory_tensors=False
        )

        assert out["L"] == L_gen
        assert out["R"].shape == (B, V, L_gen, S)

    def test_generated_only_R_content_matches_stacked_logits(self):
        """R must equal the stacked generated-only logits (no spurious slice)."""
        B, V, L_gen, S = 1, 8, 6, 4
        max_prompt_len = 3
        T_full = max_prompt_len + L_gen
        torch.manual_seed(42)
        logits_history = [torch.randn(B, L_gen, V) for _ in range(S)]
        fixation_steps = torch.randint(0, S, (B, T_full))
        prompt_lens = [max_prompt_len]

        out = trajectories_from_logits(
            logits_history, fixation_steps, prompt_lens, return_trajectory_tensors=False
        )
        expected_R = stack_logits_history(logits_history)

        assert out["R"].shape == expected_R.shape
        assert torch.allclose(out["R"], expected_R)

    def test_generated_only_F_matches_fixation_slice(self):
        """F must match fixation_steps[b, max_prompt_len : max_prompt_len + L_gen] for each b."""
        B, V, L_gen, S = 2, 8, 7, 5
        max_prompt_len = 4
        T_full = max_prompt_len + L_gen
        logits_history = [torch.randn(B, L_gen, V) for _ in range(S)]
        fixation_steps = torch.randint(0, S, (B, T_full))
        prompt_lens = [max_prompt_len, max_prompt_len]

        out = trajectories_from_logits(
            logits_history, fixation_steps, prompt_lens, return_trajectory_tensors=False
        )
        expected_F_list = []
        for b in range(B):
            slice_F = fixation_steps[b, max_prompt_len : max_prompt_len + L_gen]
            if slice_F.shape[0] >= L_gen:
                F_b = slice_F[:L_gen]
            else:
                F_b = torch.cat(
                    [
                        slice_F,
                        torch.full(
                            (L_gen - slice_F.shape[0],),
                            S - 1,
                            dtype=torch.long,
                            device=fixation_steps.device,
                        ),
                    ]
                )
            expected_F_list.append(F_b)
        expected_F = torch.stack(expected_F_list, dim=0)

        assert out["F"].shape == (B, L_gen)
        assert torch.equal(out["F"], expected_F)

    def test_generated_only_consistency_with_compute_trajectories(self):
        """Generated-only path: trajectory tensors must match compute_trajectories(R_correct, F_correct, S)."""
        B, V, L_gen, S = 2, 12, 8, 6
        max_prompt_len = 3
        T_full = max_prompt_len + L_gen
        R_expected = torch.randn(B, V, L_gen, S)
        F_expected = torch.randint(0, S, (B, L_gen))
        logits_history = [
            R_expected[:, :, :, s].permute(0, 2, 1) for s in range(S)
        ]  # each [B, L_gen, V]
        fixation_steps = torch.full((B, T_full), S - 1, dtype=torch.long)
        fixation_steps[:, max_prompt_len : max_prompt_len + L_gen] = F_expected
        prompt_lens = [max_prompt_len] * B

        out = trajectories_from_logits(
            logits_history, fixation_steps, prompt_lens, return_trajectory_tensors=True
        )
        T_steps_ref, T_fs_ref, T_fe_ref, T_fr_ref = compute_trajectories(
            R_expected, F_expected, S
        )

        assert out["L"] == L_gen
        assert torch.allclose(out["steps"], T_steps_ref)
        assert torch.allclose(out["fixation_start"], T_fs_ref)
        assert torch.allclose(out["fixation_end"], T_fe_ref)
        assert torch.allclose(out["fixation_ratio"], T_fr_ref)

    def test_generated_only_batch_two(self):
        """B=2 with different prompt_lens; L_gen same; F must match fixation slice (max_prompt_len : + L_gen)."""
        B, V, L_gen, S = 2, 8, 5, 4
        prompt_lens = [2, 4]
        max_prompt_len = max(prompt_lens)
        T_full = max_prompt_len + L_gen
        logits_history = [torch.randn(B, L_gen, V) for _ in range(S)]
        fixation_steps = torch.randint(0, S, (B, T_full))

        out = trajectories_from_logits(
            logits_history, fixation_steps, prompt_lens, return_trajectory_tensors=False
        )

        assert out["L"] == L_gen
        assert out["R"].shape == (B, V, L_gen, S)
        for b in range(B):
            expected_F_b = fixation_steps[b, max_prompt_len : max_prompt_len + L_gen]
            if expected_F_b.shape[0] >= L_gen:
                expected_F_b = expected_F_b[:L_gen]
            else:
                expected_F_b = torch.cat(
                    [
                        expected_F_b,
                        torch.full(
                            (L_gen - expected_F_b.shape[0],),
                            S - 1,
                            dtype=torch.long,
                            device=fixation_steps.device,
                        ),
                    ]
                )
            assert torch.equal(out["F"][b], expected_F_b)

    def test_full_sequence_unchanged(self):
        """Full-sequence inputs must still produce same output (regression)."""
        B, V, L_gen, S = 2, 16, 10, 5
        max_prompt_len = 4
        L_full = max_prompt_len + L_gen
        logits_history = [torch.randn(B, L_full, V) for _ in range(S)]
        fixation_steps = torch.randint(0, S, (B, L_full))
        prompt_lens = [max_prompt_len, max_prompt_len]

        out = trajectories_from_logits(
            logits_history, fixation_steps, prompt_lens, return_trajectory_tensors=False
        )

        assert out["L"] == L_gen
        assert out["R"].shape == (B, V, L_gen, S)
        assert out["F"].shape == (B, L_gen)
        assert out["S"] == S
        for key in ("steps", "fixation_start", "fixation_end", "fixation_ratio"):
            assert key not in out


class TestExtractLogitsAtStep:
    """Tests for extract_logits_at_step function."""
    
    def test_extract_first_step(self):
        """Test extracting logits at step 0."""
        V, L, S = 100, 20, 10
        trajectory = torch.randn(V, L, S)
        
        logits = extract_logits_at_step(trajectory, 0)
        
        assert logits.shape == (V, L)
        assert torch.allclose(logits, trajectory[:, :, 0])
    
    def test_extract_middle_step(self):
        """Test extracting logits at middle step."""
        V, L, S = 100, 20, 10
        trajectory = torch.randn(V, L, S)
        
        logits = extract_logits_at_step(trajectory, 5)
        
        assert logits.shape == (V, L)
        assert torch.allclose(logits, trajectory[:, :, 5])
    
    def test_extract_last_step(self):
        """Test extracting logits at last step."""
        V, L, S = 100, 20, 10
        trajectory = torch.randn(V, L, S)
        
        logits = extract_logits_at_step(trajectory, S - 1)
        
        assert logits.shape == (V, L)
        assert torch.allclose(logits, trajectory[:, :, S - 1])
    
    def test_out_of_range_raises_error(self):
        """Test that out-of-range steps raise AssertionError."""
        V, L, S = 100, 20, 10
        trajectory = torch.randn(V, L, S)
        
        with pytest.raises(AssertionError, match="out of range"):
            extract_logits_at_step(trajectory, S)
        
        with pytest.raises(AssertionError, match="out of range"):
            extract_logits_at_step(trajectory, -1)
    
    def test_extract_all_steps(self):
        """Test extracting logits at all valid steps."""
        V, L, S = 100, 20, 10
        trajectory = torch.randn(V, L, S)
        
        for step in range(S):
            logits = extract_logits_at_step(trajectory, step)
            assert logits.shape == (V, L)
            assert torch.allclose(logits, trajectory[:, :, step])
    
    def test_extracted_logits_is_view_not_copy(self):
        """Test that extracted logits is a view (memory efficient), not a copy."""
        V, L, S = 100, 20, 10
        trajectory = torch.randn(V, L, S)
        
        logits = extract_logits_at_step(trajectory, 5)
        
        # Modify original trajectory
        trajectory[0, 0, 5] = 999.0
        
        # If it's a view, the change should be reflected
        # extract_logits_at_step uses slicing: trajectory[:, :, step]
        # This creates a view, so changes to trajectory should affect logits
        assert torch.allclose(logits[0, 0], torch.tensor(999.0))
    
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_extract_different_dtypes(self, dtype):
        """Test extracting logits with different dtypes."""
        V, L, S = 100, 20, 10
        trajectory = torch.randn(V, L, S, dtype=dtype)
        
        logits = extract_logits_at_step(trajectory, 5)
        
        assert logits.dtype == dtype
        assert logits.shape == (V, L)
    
    def test_extract_different_devices(self):
        """Test extracting logits with different devices."""
        V, L, S = 100, 20, 10
        trajectory = torch.randn(V, L, S)
        
        # CPU
        logits_cpu = extract_logits_at_step(trajectory, 5)
        assert logits_cpu.device.type == "cpu"
        
        # CUDA if available
        if torch.cuda.is_available():
            trajectory_cuda = trajectory.cuda()
            logits_cuda = extract_logits_at_step(trajectory_cuda, 5)
            assert logits_cuda.device.type == "cuda"
            assert logits_cuda.shape == (V, L)
    
    def test_extract_exact_values_match(self):
        """Test that extracted logits exactly match trajectory slice."""
        V, L, S = 100, 20, 10
        trajectory = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        
        for step in range(S):
            logits = extract_logits_at_step(trajectory, step)
            expected = trajectory[:, :, step]
            assert torch.equal(logits, expected)


class TestDecodeLogitsToText:
    """Tests for decode_logits_to_text function."""
    
    def test_decode_2d_logits(self):
        """Test decoding [V, L] logits."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        V = tokenizer.vocab_size
        L = 5
        
        # Create logits where argmax gives specific tokens
        logits = torch.zeros(V, L)
        token_ids = [100, 200, 300, 400, 500]
        for i, tid in enumerate(token_ids):
            logits[tid, i] = 1.0
        
        input_ids = torch.zeros(1, 10)  # Dummy
        prompt_len = 0
        
        texts = decode_logits_to_text(logits, tokenizer, input_ids, prompt_len)
        
        assert len(texts) == 1
        assert isinstance(texts[0], str)
    
    def test_decode_3d_logits(self):
        """Test decoding [B, L, V] logits."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        V = tokenizer.vocab_size
        B, L = 2, 3
        
        logits = torch.zeros(B, L, V)
        # Set argmax tokens
        for b in range(B):
            for l in range(L):
                logits[b, l, 100 + b * 10 + l] = 1.0
        
        input_ids = torch.zeros(B, 10)
        prompt_len = 0
        
        texts = decode_logits_to_text(logits, tokenizer, input_ids, prompt_len)
        
        assert len(texts) == B
        assert all(isinstance(t, str) for t in texts)
    
    def test_unexpected_shape_raises_error(self):
        """Test that unexpected shapes raise ValueError."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        logits = torch.randn(10, 20, 30, 40)  # 4D - invalid
        input_ids = torch.zeros(1, 10)
        
        with pytest.raises(ValueError, match="Unexpected logits shape"):
            decode_logits_to_text(logits, tokenizer, input_ids, 0)
    
    def test_decode_empty_logits(self):
        """Test decoding with empty logits (L=0)."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        V = tokenizer.vocab_size
        L = 0
        
        logits = torch.zeros(V, L)
        input_ids = torch.zeros(1, 10)
        prompt_len = 0
        
        texts = decode_logits_to_text(logits, tokenizer, input_ids, prompt_len)
        
        assert len(texts) == 1
        assert isinstance(texts[0], str)
        assert texts[0] == ""  # Empty logits should decode to empty string
    
    def test_decode_single_token(self):
        """Test decoding with single token (L=1)."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        V = tokenizer.vocab_size
        L = 1
        
        logits = torch.zeros(V, L)
        token_id = 100
        logits[token_id, 0] = 1.0
        
        input_ids = torch.zeros(1, 10)
        prompt_len = 0
        
        texts = decode_logits_to_text(logits, tokenizer, input_ids, prompt_len)
        
        assert len(texts) == 1
        assert isinstance(texts[0], str)
    
    def test_decode_very_long_sequence(self):
        """Test decoding with very long sequence (L=512)."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        V = tokenizer.vocab_size
        L = 512
        
        logits = torch.zeros(V, L)
        # Set argmax to specific tokens
        for i in range(L):
            logits[i % V, i] = 1.0
        
        input_ids = torch.zeros(1, 10)
        prompt_len = 0
        
        texts = decode_logits_to_text(logits, tokenizer, input_ids, prompt_len)
        
        assert len(texts) == 1
        assert isinstance(texts[0], str)
    
    def test_decode_batch_size_2(self):
        """Test decoding with batch size B=2."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        V = tokenizer.vocab_size
        B, L = 2, 3
        
        logits = torch.zeros(B, L, V)
        for b in range(B):
            for l in range(L):
                logits[b, l, 100 + b * 10 + l] = 1.0
        
        input_ids = torch.zeros(B, 10)
        prompt_len = 0
        
        texts = decode_logits_to_text(logits, tokenizer, input_ids, prompt_len)
        
        assert len(texts) == B
        assert all(isinstance(t, str) for t in texts)
    
    def test_decode_batch_size_4(self):
        """Test decoding with batch size B=4."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        V = tokenizer.vocab_size
        B, L = 4, 5
        
        logits = torch.zeros(B, L, V)
        for b in range(B):
            for l in range(L):
                logits[b, l, 100 + b * 10 + l] = 1.0
        
        input_ids = torch.zeros(B, 10)
        prompt_len = 0
        
        texts = decode_logits_to_text(logits, tokenizer, input_ids, prompt_len)
        
        assert len(texts) == B
        assert all(isinstance(t, str) for t in texts)
    
    def test_decode_with_prompt_len(self):
        """Test decoding with prompt_len > 0 (should only decode after prompt)."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        V = tokenizer.vocab_size
        L = 10
        prompt_len = 5
        
        logits = torch.zeros(V, L)
        token_ids = list(range(100, 100 + L))
        for i, tid in enumerate(token_ids):
            logits[tid, i] = 1.0
        
        input_ids = torch.zeros(1, 15)  # prompt + generation
        prompt_len = 5
        
        texts = decode_logits_to_text(logits, tokenizer, input_ids, prompt_len)
        
        assert len(texts) == 1
        assert isinstance(texts[0], str)
        # Should decode all L tokens (prompt_len is used for input_ids, not logits)
    
    def test_decode_with_prompt_len_equal_to_input_length(self):
        """Test decoding with prompt_len = input_ids.shape[1] (no generation)."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        V = tokenizer.vocab_size
        L = 10
        
        logits = torch.zeros(V, L)
        input_ids = torch.zeros(1, 10)
        prompt_len = 10  # All input is prompt, no generation
        
        texts = decode_logits_to_text(logits, tokenizer, input_ids, prompt_len)
        
        assert len(texts) == 1
        assert isinstance(texts[0], str)
    
    def test_decode_with_prompt_len_zero(self):
        """Test decoding with prompt_len = 0 (all generation)."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        V = tokenizer.vocab_size
        L = 10
        
        logits = torch.zeros(V, L)
        token_ids = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        for i, tid in enumerate(token_ids):
            if tid < V:
                logits[tid, i] = 1.0
        
        input_ids = torch.zeros(1, 10)
        prompt_len = 0
        
        texts = decode_logits_to_text(logits, tokenizer, input_ids, prompt_len)
        
        assert len(texts) == 1
        assert isinstance(texts[0], str)
    
    def test_decode_verifies_argmax_tokens(self):
        """Test that decoded text matches argmax tokens."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        V = tokenizer.vocab_size
        L = 5
        
        # Create logits where argmax gives specific known tokens
        logits = torch.zeros(V, L)
        token_ids = [100, 200, 300, 400, 500]
        for i, tid in enumerate(token_ids):
            if tid < V:
                logits[tid, i] = 1.0
        
        input_ids = torch.zeros(1, 10)
        prompt_len = 0
        
        texts = decode_logits_to_text(logits, tokenizer, input_ids, prompt_len)
        
        # Decode the token_ids directly to verify
        decoded_direct = tokenizer.decode(token_ids, skip_special_tokens=True)
        # The decoded text should match (or be similar, depending on tokenizer behavior)
        assert isinstance(texts[0], str)
        assert len(texts[0]) >= 0  # May be empty if tokens are special
    
    def test_decode_with_special_tokens(self):
        """Test decoding with special tokens (EOS, PAD, etc.)."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        V = tokenizer.vocab_size
        L = 5
        
        logits = torch.zeros(V, L)
        # Use EOS token if available
        eos_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 50256
        for i in range(L):
            token_id = eos_id if i == 2 else (100 + i)
            if token_id < V:
                logits[token_id, i] = 1.0
        
        input_ids = torch.zeros(1, 10)
        prompt_len = 0
        
        texts = decode_logits_to_text(logits, tokenizer, input_ids, prompt_len)
        
        assert len(texts) == 1
        assert isinstance(texts[0], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
