"""Non-traj probability uses AR step-wise provider when generalized and model is not DiffusionModelAdapter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from evals.metrics.memorization import probability  # noqa: E402
from evals.metrics.utils import evaluate_probability, evaluate_probability_unified  # noqa: E402


class _TinyAR(nn.Module):
    """Minimal causal LM for one forward."""

    def __init__(self, vocab: int = 32, hidden: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.lin = nn.Linear(hidden, vocab)
        self._device = torch.device("cpu")

    @property
    def device(self):
        return self._device

    def forward(self, input_ids, attention_mask=None, labels=None, **_):
        x = self.emb(input_ids)
        logits = self.lin(x)
        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(logits=logits, loss=None)


def test_generalized_ar_model_uses_provider_path() -> None:
    model = _TinyAR()
    model.eval()
    bsz, seq = 2, 6
    input_ids = torch.randint(0, 32, (bsz, seq))
    labels = input_ids.clone()
    labels[:, :2] = -100
    batch = {"input_ids": input_ids, "labels": labels}

    direct = evaluate_probability(model, batch)
    from evals.metrics.step_wise_score import (
        ARStepWiseScoreProvider,
        evaluate_probability_via_provider,
    )

    via = evaluate_probability_via_provider(
        ARStepWiseScoreProvider(), model, batch, ignore_index=-100
    )
    assert len(direct) == len(via)
    for a, b in zip(direct, via, strict=True):
        assert a["prob"] is not None and b["prob"] is not None
        assert abs(a["prob"] - b["prob"]) < 1e-5


def test_evaluate_probability_unified_ar_matches_evaluate_probability() -> None:
    model = _TinyAR()
    model.eval()
    bsz, seq = 2, 6
    input_ids = torch.randint(0, 32, (bsz, seq))
    labels = input_ids.clone()
    labels[:, :2] = -100
    batch = {"input_ids": input_ids, "labels": labels}
    direct = evaluate_probability(model, batch)
    uni = evaluate_probability_unified(
        model,
        batch,
        use_generalized_sequence_probability=True,
        logit_alignment="causal",
    )
    assert len(direct) == len(uni)
    for a, b in zip(direct, uni, strict=True):
        assert a["prob"] is not None and b["prob"] is not None
        assert abs(a["prob"] - b["prob"]) < 1e-5
