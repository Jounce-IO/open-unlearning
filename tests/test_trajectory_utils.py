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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
