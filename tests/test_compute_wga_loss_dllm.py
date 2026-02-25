"""
Unit tests for compute_wga_loss_dllm (WGA forget loss for diffusion adapters).

Tests:
- Synthetic per_token_nll and masked_indices: sign and value vs hand WGA formula.
- Empty per_token_nll: forget_loss == 0.0 and backward-safe.
"""

import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from trainer.utils import compute_wga_loss_dllm


class TestComputeWgaLossDllm:
    """Synthetic and edge-case tests for compute_wga_loss_dllm."""

    def test_synthetic_matches_hand_formula(self) -> None:
        """forget_loss = -(weight_ce * per_token_nll).mean() with weight_ce = (exp(-per_token_nll)).detach() ** beta."""
        beta = 1.0
        ptnll = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32)
        weight_ce = ((-ptnll).exp().detach()) ** beta
        expected_forget_loss = -(weight_ce * ptnll).mean()

        outputs = type("Outputs", (), {"logits": torch.randn(1, 10, 8), "per_token_nll": ptnll, "masked_indices": None})()
        forget_loss, out = compute_wga_loss_dllm(outputs, beta)
        assert torch.allclose(forget_loss, expected_forget_loss)
        assert out is outputs

    def test_synthetic_negative_forget_loss(self) -> None:
        """WGA forget loss is negative (we maximize forget loss = minimize -forget_loss for gradient ascent)."""
        ptnll = torch.tensor([0.3, 0.8], dtype=torch.float32)

        class Outputs:
            logits = torch.randn(1, 5, 4)
            per_token_nll = ptnll
            masked_indices = None

        forget_loss, _ = compute_wga_loss_dllm(Outputs(), beta=1.0)
        assert forget_loss.item() < 0

    def test_empty_per_token_nll_returns_zero_loss(self) -> None:
        """When per_token_nll is empty (no masked tokens), forget_loss is 0.0 and is a tensor (backward-safe)."""
        class Outputs:
            logits = torch.randn(2, 8, 16)
            per_token_nll = torch.tensor([], dtype=torch.float32)
            masked_indices = torch.zeros(2, 8, dtype=torch.bool)

        forget_loss, out = compute_wga_loss_dllm(Outputs(), beta=1.0)
        assert forget_loss.numel() == 1
        assert forget_loss.item() == 0.0
        assert out.per_token_nll.numel() == 0

    def test_none_per_token_nll_returns_zero_loss(self) -> None:
        """When outputs have no per_token_nll (None), forget_loss is 0.0 (fallback to AR path would use compute_wga_loss)."""
        class Outputs:
            logits = torch.randn(1, 4, 8)

        outputs = Outputs()
        forget_loss, out = compute_wga_loss_dllm(outputs, beta=1.0)
        assert forget_loss.numel() == 1
        assert forget_loss.item() == 0.0
        assert out is outputs
