"""
Comprehensive tensor operation tests for trajectory_metrics.

Tests verify exact tensor values, shapes, devices, dtypes for all tensor operations.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.trajectory_utils import (
    stack_logits_history,
    compute_trajectories,
    extract_logits_at_step,
)
from evals.metrics.trajectory_adapters import LogitModelWrapper


class TestTensorValueVerification:
    """Tests that verify exact tensor values, not just shapes."""
    
    def test_stack_logits_history_exact_values(self):
        """Test that stacked logits have exact values matching input."""
        V, L, S = 50, 10, 5
        # Use fixed values for reproducibility
        torch.manual_seed(42)
        logits_history = [torch.randn(1, L, V) for _ in range(S)]
        
        R = stack_logits_history(logits_history)
        
        # Verify exact values for each step
        for s in range(S):
            expected = logits_history[s][0].T  # [V, L]
            actual = R[:, :, s]  # [V, L]
            # Check element-wise equality
            assert torch.allclose(actual, expected, atol=1e-6)
            # Verify specific elements
            assert torch.allclose(actual[0, 0], expected[0, 0])
            assert torch.allclose(actual[-1, -1], expected[-1, -1])
    
    def test_compute_trajectories_t_steps_exact_copy(self):
        """Test that T_steps is exact copy of R (not just close)."""
        V, L, S = 10, 5, 8
        R = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        # T_steps should be exact copy
        assert torch.equal(T_steps, R)
        # Verify they're not the same object (clone creates new tensor)
        assert T_steps is not R
        # But values are identical
        assert torch.allclose(T_steps, R)
    
    def test_compute_trajectories_fixation_exact_formula(self):
        """Test that T_fixation values match exact formula: R[:, l, max(0, F[l] - s)]."""
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
                assert torch.equal(actual, expected), f"Failed at l={l}, s={s}, fix={fix_step}"
    
    def test_compute_trajectories_ratio_exact_formula(self):
        """Test that T_ratio values match exact formula: R[:, l, floor(F[l] * s/S)]."""
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
                assert torch.equal(actual, expected), f"Failed at l={l}, s={s}, fix={fix_step}"
    
    def test_extract_logits_at_step_exact_values(self):
        """Test that extracted logits exactly match trajectory slice."""
        V, L, S = 100, 20, 10
        trajectory = torch.arange(V * L * S, dtype=torch.float32).reshape(V, L, S)
        
        for step in range(S):
            logits = extract_logits_at_step(trajectory, step)
            expected = trajectory[:, :, step]
            # Should be exact match (view, not copy)
            assert torch.equal(logits, expected)
            # Verify it's a view (same underlying data)
            logits[0, 0] = 999.0
            assert trajectory[0, 0, step] == 999.0
    
    def test_logit_model_wrapper_exact_tensor_reference(self):
        """Test that LogitModelWrapper output.logits is same tensor object."""
        B, L, V = 1, 10, 100
        logits = torch.randn(B, L, V)
        device = torch.device("cpu")
        
        wrapper = LogitModelWrapper(logits, device)
        output = wrapper(input_ids=torch.zeros(1, 10))
        
        # output.logits should be the same tensor object
        assert output.logits is logits
        assert output.logits is wrapper.logits
        # Modifying one should affect the other
        logits[0, 0, 0] = 999.0
        assert output.logits[0, 0, 0] == 999.0


class TestTensorDeviceHandling:
    """Tests for tensor device handling across all operations."""
    
    def test_stack_logits_history_preserves_device(self):
        """Test that stack_logits_history preserves device."""
        V, L, S = 100, 20, 5
        
        if torch.cuda.is_available():
            logits_history = [torch.randn(1, L, V).cuda() for _ in range(S)]
            R = stack_logits_history(logits_history)
            assert R.device.type == "cuda"
        else:
            logits_history = [torch.randn(1, L, V) for _ in range(S)]
            R = stack_logits_history(logits_history)
            assert R.device.type == "cpu"
    
    def test_compute_trajectories_preserves_device(self):
        """Test that compute_trajectories preserves device."""
        V, L, S = 100, 20, 10
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        if torch.cuda.is_available():
            R = R.cuda()
            F = F.cuda()
            T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
            assert T_steps.device.type == "cuda"
            assert T_fixation.device.type == "cuda"
            assert T_ratio.device.type == "cuda"
        else:
            T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
            assert T_steps.device.type == "cpu"
            assert T_fixation.device.type == "cpu"
            assert T_ratio.device.type == "cpu"
    
    def test_extract_logits_at_step_preserves_device(self):
        """Test that extract_logits_at_step preserves device."""
        V, L, S = 100, 20, 10
        trajectory = torch.randn(V, L, S)
        
        if torch.cuda.is_available():
            trajectory = trajectory.cuda()
            logits = extract_logits_at_step(trajectory, 5)
            assert logits.device.type == "cuda"
        else:
            logits = extract_logits_at_step(trajectory, 5)
            assert logits.device.type == "cpu"
    
    def test_logit_model_wrapper_preserves_device(self):
        """Test that LogitModelWrapper preserves device."""
        B, L, V = 1, 10, 100
        logits = torch.randn(B, L, V)
        
        if torch.cuda.is_available():
            logits = logits.cuda()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        wrapper = LogitModelWrapper(logits, device)
        assert wrapper.device.type == device.type
        assert wrapper.logits.device.type == device.type


class TestTensorDtypeHandling:
    """Tests for tensor dtype handling across all operations."""
    
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_stack_logits_history_preserves_dtype(self, dtype):
        """Test that stack_logits_history preserves dtype."""
        V, L, S = 100, 20, 5
        logits_history = [torch.randn(1, L, V, dtype=dtype) for _ in range(S)]
        R = stack_logits_history(logits_history)
        assert R.dtype == dtype
    
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_compute_trajectories_preserves_dtype(self, dtype):
        """Test that compute_trajectories preserves dtype."""
        V, L, S = 100, 20, 10
        R = torch.randn(V, L, S, dtype=dtype)
        F = torch.randint(0, S, (L,), dtype=torch.long)
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        assert T_steps.dtype == dtype
        assert T_fixation.dtype == dtype
        assert T_ratio.dtype == dtype
    
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_extract_logits_at_step_preserves_dtype(self, dtype):
        """Test that extract_logits_at_step preserves dtype."""
        V, L, S = 100, 20, 10
        trajectory = torch.randn(V, L, S, dtype=dtype)
        
        logits = extract_logits_at_step(trajectory, 5)
        assert logits.dtype == dtype


class TestTensorMemoryEfficiency:
    """Tests for memory efficiency (views vs copies)."""
    
    def test_extract_logits_at_step_is_view(self):
        """Test that extract_logits_at_step returns a view, not a copy."""
        V, L, S = 100, 20, 10
        trajectory = torch.randn(V, L, S)
        
        logits = extract_logits_at_step(trajectory, 5)
        
        # Modify extracted logits
        original_value = trajectory[0, 0, 5].item()
        logits[0, 0] = 999.0
        
        # If it's a view, trajectory should be modified
        assert trajectory[0, 0, 5].item() == 999.0
    
    def test_t_steps_is_clone_not_view(self):
        """Test that T_steps is a clone (new tensor), not a view."""
        V, L, S = 100, 20, 10
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        # Modify T_steps
        original_value = R[0, 0, 0].item()
        T_steps[0, 0, 0] = 999.0
        
        # R should be unchanged (T_steps is a clone)
        assert R[0, 0, 0].item() == original_value
    
    def test_t_fixation_is_new_tensor(self):
        """Test that T_fixation is a new tensor, not a view."""
        V, L, S = 100, 20, 10
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        # T_fixation should be independent
        original_value = R[0, 0, 0].item()
        T_fixation[0, 0, 0] = 999.0
        
        # R should be unchanged
        assert R[0, 0, 0].item() == original_value
    
    def test_t_ratio_is_new_tensor(self):
        """Test that T_ratio is a new tensor, not a view."""
        V, L, S = 100, 20, 10
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        
        # T_ratio should be independent
        original_value = R[0, 0, 0].item()
        T_ratio[0, 0, 0] = 999.0
        
        # R should be unchanged
        assert R[0, 0, 0].item() == original_value


class TestTensorShapeConsistency:
    """Tests for tensor shape consistency across operations."""
    
    @pytest.mark.parametrize("V,L,S", [(100, 20, 8), (1000, 64, 16), (50000, 256, 128)])
    def test_stack_logits_history_shape_consistency(self, V, L, S):
        """Test shape consistency for various sizes."""
        logits_history = [torch.randn(1, L, V) for _ in range(S)]
        R = stack_logits_history(logits_history)
        assert R.shape == (V, L, S)
    
    @pytest.mark.parametrize("V,L,S", [(100, 20, 8), (1000, 64, 16), (50000, 256, 128)])
    def test_compute_trajectories_shape_consistency(self, V, L, S):
        """Test shape consistency for various sizes."""
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        assert T_steps.shape == (V, L, S)
        assert T_fixation.shape == (V, L, S)
        assert T_ratio.shape == (V, L, S)
    
    @pytest.mark.parametrize("V,L,S", [(100, 20, 8), (1000, 64, 16), (50000, 256, 128)])
    def test_extract_logits_at_step_shape_consistency(self, V, L, S):
        """Test shape consistency for various sizes."""
        trajectory = torch.randn(V, L, S)
        
        for step in range(S):
            logits = extract_logits_at_step(trajectory, step)
            assert logits.shape == (V, L)


class TestTensorEdgeCases:
    """Tests for tensor edge cases (empty, single element, etc.)."""
    
    def test_single_step_single_token(self):
        """Test with S=1, L=1 (minimal case)."""
        V = 100
        logits_history = [torch.randn(1, 1, V)]
        R = stack_logits_history(logits_history)
        assert R.shape == (V, 1, 1)
        
        F = torch.tensor([0], dtype=torch.long)
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, 1)
        assert T_steps.shape == (V, 1, 1)
        assert T_fixation.shape == (V, 1, 1)
        assert T_ratio.shape == (V, 1, 1)
    
    def test_single_vocab_token(self):
        """Test with V=1 (minimal vocab)."""
        V, L, S = 1, 10, 5
        logits_history = [torch.randn(1, L, V) for _ in range(S)]
        R = stack_logits_history(logits_history)
        assert R.shape == (V, L, S)
        
        F = torch.randint(0, S, (L,))
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        assert T_steps.shape == (V, L, S)
    
    def test_very_large_tensors(self):
        """Test with very large tensors (stress test)."""
        V, L, S = 50000, 512, 256  # Large vocab, long sequence, many steps
        logits_history = [torch.randn(1, L, V) for _ in range(S)]
        
        R = stack_logits_history(logits_history)
        assert R.shape == (V, L, S)
        
        F = torch.randint(0, S, (L,))
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        assert T_steps.shape == (V, L, S)
        assert T_fixation.shape == (V, L, S)
        assert T_ratio.shape == (V, L, S)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
