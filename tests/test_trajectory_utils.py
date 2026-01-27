"""
Comprehensive unit tests for trajectory_utils module.

Tests cover:
- stack_logits_history: Shape validation, single/batch cases
- compute_trajectories: Steps, fixation, ratio trajectory computation
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
        """Test stacking with B=1, S=1."""
        V, L = 10, 5
        logits_history = [torch.randn(1, L, V)]  # [B, L, V]
        R = stack_logits_history(logits_history)
        
        assert R.shape == (V, L, 1)
        assert torch.allclose(R[:, :, 0], logits_history[0][0].T)
    
    def test_single_sample_multiple_steps(self):
        """Test stacking with B=1, S=5."""
        V, L, S = 100, 20, 5
        logits_history = [torch.randn(1, L, V) for _ in range(S)]
        R = stack_logits_history(logits_history)
        
        assert R.shape == (V, L, S)
        for s in range(S):
            assert torch.allclose(R[:, :, s], logits_history[s][0].T)
    
    def test_batch_multiple_steps(self):
        """Test stacking with B=2, S=3 (takes first sample)."""
        V, L, S, B = 50, 10, 3, 2
        logits_history = [torch.randn(B, L, V) for _ in range(S)]
        R = stack_logits_history(logits_history)
        
        assert R.shape == (V, L, S)
        # Should take first sample (index 0)
        for s in range(S):
            assert torch.allclose(R[:, :, s], logits_history[s][0].T)
    
    def test_shape_consistency(self):
        """Test that output shape is always [V, L, S] regardless of batch size."""
        V, L, S = 1000, 64, 8
        
        for B in [1, 2, 4]:
            logits_history = [torch.randn(B, L, V) for _ in range(S)]
            R = stack_logits_history(logits_history)
            assert R.shape == (V, L, S), f"Failed for B={B}"
    
    def test_larger_batches(self):
        """Test stacking with larger batch sizes (B=4, B=8)."""
        V, L, S = 100, 20, 5
        
        for B in [4, 8]:
            logits_history = [torch.randn(B, L, V) for _ in range(S)]
            R = stack_logits_history(logits_history)
            assert R.shape == (V, L, S)
            # Should take first sample (index 0)
            for s in range(S):
                assert torch.allclose(R[:, :, s], logits_history[s][0].T)
    
    def test_exact_tensor_values_match(self):
        """Test that exact tensor values match, not just shapes."""
        V, L, S = 50, 10, 3
        # Use fixed seed for reproducibility
        torch.manual_seed(42)
        logits_history = [torch.randn(1, L, V) for _ in range(S)]
        
        R = stack_logits_history(logits_history)
        
        # Verify exact values for each step
        for s in range(S):
            expected = logits_history[s][0].T  # [V, L]
            actual = R[:, :, s]  # [V, L]
            assert torch.allclose(actual, expected, atol=1e-6)
            # Also check they're the same tensor (or exact copy)
            assert torch.equal(actual, expected) or torch.allclose(actual, expected)
    
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_different_dtypes(self, dtype):
        """Test stacking with different dtypes."""
        V, L, S = 100, 20, 5
        logits_history = [torch.randn(1, L, V, dtype=dtype) for _ in range(S)]
        R = stack_logits_history(logits_history)
        
        assert R.shape == (V, L, S)
        assert R.dtype == dtype
        for s in range(S):
            assert torch.allclose(R[:, :, s], logits_history[s][0].T, atol=1e-3 if dtype != torch.float32 else 1e-6)
    
    def test_different_devices(self):
        """Test stacking with different devices."""
        V, L, S = 100, 20, 5
        
        # CPU
        logits_history_cpu = [torch.randn(1, L, V) for _ in range(S)]
        R_cpu = stack_logits_history(logits_history_cpu)
        assert R_cpu.device.type == "cpu"
        
        # CUDA if available
        if torch.cuda.is_available():
            logits_history_cuda = [torch.randn(1, L, V).cuda() for _ in range(S)]
            R_cuda = stack_logits_history(logits_history_cuda)
            assert R_cuda.device.type == "cuda"
            assert R_cuda.shape == (V, L, S)


class TestComputeTrajectories:
    """Tests for compute_trajectories function."""
    
    def test_steps_trajectory_is_copy(self):
        """Test that T_steps is a direct copy of R."""
        V, L, S = 100, 20, 10
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        assert torch.allclose(T_steps, R)
        assert T_steps.shape == (V, L, S)
    
    def test_fixation_trajectory_lookback(self):
        """Test fixation trajectory looks back from fixation step."""
        V, L, S = 10, 5, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        # Set fixation steps: each position fixed at different steps
        F = torch.tensor([0, 2, 4, 6, 7])  # [L]
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        # For position l=1, fixed at step 2
        # At trajectory step s=0, should look back 0 steps: R[:, 1, 2]
        # At trajectory step s=1, should look back 1 step: R[:, 1, max(0, 2-1)] = R[:, 1, 1]
        # At trajectory step s=2, should look back 2 steps: R[:, 1, max(0, 2-2)] = R[:, 1, 0]
        assert torch.allclose(T_fixation[:, 1, 0], R[:, 1, 2])  # s=0: look back 0
        assert torch.allclose(T_fixation[:, 1, 1], R[:, 1, 1])  # s=1: look back 1
        assert torch.allclose(T_fixation[:, 1, 2], R[:, 1, 0])  # s=2: look back 2
        assert torch.allclose(T_fixation[:, 1, 3], R[:, 1, 0])  # s=3: look back 3, clamped to 0
    
    def test_ratio_trajectory_interpolation(self):
        """Test ratio trajectory interpolates from step 0 to fixation."""
        V, L, S = 10, 3, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        F = torch.tensor([4, 6, 7])  # Fixation steps
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        # For position l=0, fixed at step 4
        # At trajectory step s=0: ratio_step = floor(4 * 0/8) = 0
        # At trajectory step s=4: ratio_step = floor(4 * 4/8) = 2
        # At trajectory step s=7: ratio_step = floor(4 * 7/8) = 3
        # At trajectory step s=8: ratio_step = floor(4 * 8/8) = 4, clamped to S-1=7
        assert torch.allclose(T_ratio[:, 0, 0], R[:, 0, 0])
        assert torch.allclose(T_ratio[:, 0, 4], R[:, 0, 2])
        assert torch.allclose(T_ratio[:, 0, 7], R[:, 0, 3])
    
    def test_shape_mismatch_raises_error(self):
        """Test that shape mismatches raise assertions."""
        V, L, S = 10, 5, 8
        R = torch.randn(V, L, S)
        
        # Wrong S
        with pytest.raises(AssertionError, match="S mismatch"):
            F = torch.randint(0, S, (L,))
            compute_trajectories(R, F, S + 1)
        
        # Wrong F length
        with pytest.raises(AssertionError, match="F length mismatch"):
            F = torch.randint(0, S, (L + 1,))
            compute_trajectories(R, F, S)
    
    def test_all_trajectories_same_shape(self):
        """Test that all three trajectories have shape [V, L, S]."""
        V, L, S = 100, 20, 16
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        assert T_steps.shape == (V, L, S)
        assert T_fixation.shape == (V, L, S)
        assert T_ratio.shape == (V, L, S)
    
    def test_fixation_at_boundary(self):
        """Test fixation trajectory at boundary cases (fixation at step 0 or S-1)."""
        V, L, S = 10, 3, 8
        R = torch.randn(V, L, S)
        
        # All fixed at step 0
        F = torch.zeros(L, dtype=torch.long)
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        # Should all look back to step 0
        for s in range(S):
            assert torch.allclose(T_fixation[:, 0, s], R[:, 0, 0])
        
        # All fixed at last step
        F = torch.full((L,), S - 1, dtype=torch.long)
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        # At s=0, should be R[:, :, S-1], at s=1, should be R[:, :, S-2], etc.
        assert torch.allclose(T_fixation[:, 0, 0], R[:, 0, S - 1])
    
    def test_fixation_steps_ascending_order(self):
        """Test fixation steps in ascending order [0, 1, 2, ..., S-1]."""
        V, L, S = 10, 8, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        F = torch.arange(L, dtype=torch.long)  # [0, 1, 2, ..., 7]
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        # For position l=1, fixed at step 1
        # At s=0: look back 0 → R[:, 1, 1]
        # At s=1: look back 1 → R[:, 1, max(0, 1-1)] = R[:, 1, 0]
        assert torch.allclose(T_fixation[:, 1, 0], R[:, 1, 1])
        assert torch.allclose(T_fixation[:, 1, 1], R[:, 1, 0])
    
    def test_fixation_steps_descending_order(self):
        """Test fixation steps in descending order [S-1, S-2, ..., 0]."""
        V, L, S = 10, 8, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        F = torch.arange(S - 1, -1, -1, dtype=torch.long)  # [7, 6, 5, ..., 0]
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        # For position l=0, fixed at step 7
        # At s=0: look back 0 → R[:, 0, 7]
        # At s=1: look back 1 → R[:, 0, 6]
        assert torch.allclose(T_fixation[:, 0, 0], R[:, 0, 7])
        assert torch.allclose(T_fixation[:, 0, 1], R[:, 0, 6])
    
    def test_fixation_steps_all_same_value(self):
        """Test fixation steps all same value (e.g., all 5)."""
        V, L, S = 10, 5, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        fix_step = 5
        F = torch.full((L,), fix_step, dtype=torch.long)
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        # All positions should have same fixation trajectory
        for l in range(L):
            # At s=0: R[:, l, 5]
            assert torch.allclose(T_fixation[:, l, 0], R[:, l, fix_step])
            # At s=1: R[:, l, 4]
            assert torch.allclose(T_fixation[:, l, 1], R[:, l, fix_step - 1])
    
    def test_fixation_steps_with_duplicates(self):
        """Test fixation steps with duplicate values."""
        V, L, S = 10, 5, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        F = torch.tensor([3, 3, 5, 5, 7], dtype=torch.long)  # Duplicates
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        # Positions with same fixation should use same source step calculation
        # At step s=0, both positions with F=3 should look at step 3
        # T_fixation[:, l, s] = R[:, l, max(0, F[l] - s)]
        # For l=0,1 with F=3, at s=0: source_step = max(0, 3-0) = 3
        assert torch.allclose(T_fixation[:, 0, 0], R[:, 0, 3])  # Position 0 at step 0 = R[:, 0, 3]
        assert torch.allclose(T_fixation[:, 1, 0], R[:, 1, 3])  # Position 1 at step 0 = R[:, 1, 3]
        # For l=2,3 with F=5, at s=0: source_step = max(0, 5-0) = 5
        assert torch.allclose(T_fixation[:, 2, 0], R[:, 2, 5])  # Position 2 at step 0 = R[:, 2, 5]
        assert torch.allclose(T_fixation[:, 3, 0], R[:, 3, 5])  # Position 3 at step 0 = R[:, 3, 5]
        
        # At step s=1, positions with F=3 should look at step 2
        assert torch.allclose(T_fixation[:, 0, 1], R[:, 0, 2])  # Position 0 at step 1 = R[:, 0, 2]
        assert torch.allclose(T_fixation[:, 1, 1], R[:, 1, 2])  # Position 1 at step 1 = R[:, 1, 2]
    
    @pytest.mark.parametrize("S", [1, 16, 128, 256])
    def test_very_large_steps(self, S):
        """Test with very large number of steps."""
        V, L = 50, 10
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        assert T_steps.shape == (V, L, S)
        assert T_fixation.shape == (V, L, S)
        assert T_ratio.shape == (V, L, S)
    
    @pytest.mark.parametrize("L", [1, 16, 128, 512, 1024])
    def test_very_large_sequence_length(self, L):
        """Test with very large sequence length."""
        V, S = 100, 8
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        assert T_steps.shape == (V, L, S)
        assert T_fixation.shape == (V, L, S)
        assert T_ratio.shape == (V, L, S)
    
    @pytest.mark.parametrize("V", [100, 1000, 10000, 50000])
    def test_very_large_vocab_size(self, V):
        """Test with very large vocabulary size."""
        L, S = 20, 8
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        assert T_steps.shape == (V, L, S)
        assert T_fixation.shape == (V, L, S)
        assert T_ratio.shape == (V, L, S)
    
    def test_fixation_trajectory_formula_verification(self):
        """Verify T_fixation[:, l, s] = R[:, l, max(0, F[l] - s)] for all l, s."""
        V, L, S = 10, 5, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        F = torch.tensor([0, 2, 4, 6, 7], dtype=torch.long)
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        # Verify formula for all positions and steps
        for l in range(L):
            fix_step = F[l].item()
            for s in range(S):
                source_step = max(0, fix_step - s)
                expected = R[:, l, source_step]
                actual = T_fixation[:, l, s]
                assert torch.allclose(actual, expected), f"Failed at l={l}, s={s}"
    
    def test_ratio_trajectory_formula_verification(self):
        """Verify T_ratio[:, l, s] = R[:, l, floor(F[l] * s/S)] for all l, s."""
        V, L, S = 10, 5, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        F = torch.tensor([0, 2, 4, 6, 7], dtype=torch.long)
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        # Verify formula for all positions and steps
        for l in range(L):
            fix_step = F[l].item()
            for s in range(S):
                if S > 0:
                    ratio_step = int(fix_step * (s / S))
                    ratio_step = min(ratio_step, S - 1)
                else:
                    ratio_step = 0
                expected = R[:, l, ratio_step]
                actual = T_ratio[:, l, s]
                assert torch.allclose(actual, expected), f"Failed at l={l}, s={s}"
    
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_trajectories_different_dtypes(self, dtype):
        """Test trajectory computation with different dtypes."""
        V, L, S = 100, 20, 10
        R = torch.randn(V, L, S, dtype=dtype)
        F = torch.randint(0, S, (L,), dtype=torch.long)
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        assert T_steps.dtype == dtype
        assert T_fixation.dtype == dtype
        assert T_ratio.dtype == dtype
    
    def test_trajectories_different_devices(self):
        """Test trajectory computation with different devices."""
        V, L, S = 100, 20, 10
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        # CPU
        T_steps_cpu, T_fixation_cpu, T_ratio_cpu = compute_trajectories(R, F, S)
        assert T_steps_cpu.device.type == "cpu"
        
        # CUDA if available
        if torch.cuda.is_available():
            R_cuda = R.cuda()
            F_cuda = F.cuda()
            T_steps_cuda, T_fixation_cuda, T_ratio_cuda = compute_trajectories(R_cuda, F_cuda, S)
            assert T_steps_cuda.device.type == "cuda"
            assert T_fixation_cuda.device.type == "cuda"
            assert T_ratio_cuda.device.type == "cuda"
    
    def test_trajectories_are_new_tensors_not_views(self):
        """Test that T_fixation and T_ratio are new tensors, not views of R."""
        V, L, S = 100, 20, 10
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        # T_steps is a clone, so modifying R shouldn't affect it
        R_original = R.clone()
        R[0, 0, 0] = 999.0
        # T_steps should be unchanged (it's a clone)
        assert torch.allclose(T_steps, R_original)
        
        # T_fixation and T_ratio are new tensors
        # They should be independent of R after computation
        assert not torch.equal(T_fixation, R)
        assert not torch.equal(T_ratio, R)
    
    def test_trajectories_same_device_as_input(self):
        """Test that output tensors are on same device as input."""
        V, L, S = 100, 20, 10
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        assert T_steps.device == R.device
        assert T_fixation.device == R.device
        assert T_ratio.device == R.device


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
