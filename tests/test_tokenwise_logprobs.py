"""
Unit tests for tokenwise_logprobs (evals.metrics.utils).

Verifies that when batch input_ids/labels are longer than model output logits,
the function trims to logits length and returns without RuntimeError.
"""

import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.utils import tokenwise_logprobs, tokenwise_logprobs_from_logits
from evals.metrics.trajectory_adapters import LogitModelWrapper
from data.utils import IGNORE_INDEX


def test_tokenwise_logprobs_trim_when_batch_longer_than_logits():
    """tokenwise_logprobs with batch longer than logits: no exception, consistent shapes."""
    bsz, seq_len_logits, V = 1, 32, 100
    # Model returns shorter sequence than batch
    logits = torch.randn(bsz, seq_len_logits, V)
    device = torch.device("cpu")
    model = LogitModelWrapper(logits, device)

    # Batch has longer sequence (e.g. 40) - would cause gather mismatch without trim
    L_batch = 40
    input_ids = torch.randint(0, V, (bsz, L_batch), device=device)
    # Labels: first 10 positions are "labeled" (non-IGNORE), rest ignored
    labels = torch.full((bsz, L_batch), IGNORE_INDEX, dtype=torch.long, device=device)
    labels[0, 1:11] = input_ids[0, 2:12]  # positions 1..10 have labels (answer tokens)
    attention_mask = torch.ones(bsz, L_batch, dtype=torch.long, device=device)

    batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    result = tokenwise_logprobs(model, batch, grad=False, return_labels=False)

    assert isinstance(result, list)
    assert len(result) == bsz
    # After trim, we only have seq_len_logits positions; labels trimmed to 32;
    # actual_indices (non-IGNORE within first 32) are positions 1..10 -> 9 positions (:-1 drops last)
    # So we get 9 log probs per sample
    assert result[0].dim() == 1
    assert result[0].shape[0] == 9  # start_idx-1 : end_idx = 0:10 -> 10 positions, but :-1 in actual_indices -> 9
    assert not torch.isnan(result[0]).any()
    assert not torch.isinf(result[0]).any()


def test_tokenwise_logprobs_from_logits_parity_with_tokenwise_logprobs():
    """tokenwise_logprobs_from_logits returns same as tokenwise_logprobs when model returns those logits."""
    bsz, seq_len, V = 2, 20, 100
    logits = torch.randn(bsz, seq_len, V)
    device = torch.device("cpu")
    model = LogitModelWrapper(logits, device)

    input_ids = torch.randint(0, V, (bsz, seq_len), device=device)
    labels = torch.full((bsz, seq_len), IGNORE_INDEX, dtype=torch.long, device=device)
    labels[0, 2:12] = input_ids[0, 3:13]
    labels[1, 1:8] = input_ids[1, 2:9]
    attention_mask = torch.ones(bsz, seq_len, dtype=torch.long, device=device)
    batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    from_model = tokenwise_logprobs(model, batch, grad=False, return_labels=False)
    from_logits = tokenwise_logprobs_from_logits(batch, logits, return_labels=False)

    assert len(from_model) == len(from_logits) == bsz
    for i in range(bsz):
        assert torch.allclose(from_model[i], from_logits[i]), f"sample {i} mismatch"


def test_tokenwise_logprobs_from_logits_empty_labels():
    """tokenwise_logprobs_from_logits with all IGNORE_INDEX labels returns empty tensors."""
    bsz, seq_len, V = 1, 10, 50
    logits = torch.randn(bsz, seq_len, V)
    labels = torch.full((bsz, seq_len), IGNORE_INDEX, dtype=torch.long)
    batch = {"input_ids": torch.zeros(bsz, seq_len, dtype=torch.long), "labels": labels}
    result = tokenwise_logprobs_from_logits(batch, logits, return_labels=False)
    assert len(result) == 1
    assert result[0].numel() == 0


def test_tokenwise_logprobs_from_logits_return_labels():
    """tokenwise_logprobs_from_logits with return_labels=True returns (log_probs_batch, labels_batch)."""
    bsz, seq_len, V = 1, 15, 80
    logits = torch.randn(bsz, seq_len, V)
    input_ids = torch.randint(0, V, (bsz, seq_len))
    labels = torch.full((bsz, seq_len), IGNORE_INDEX, dtype=torch.long)
    labels[0, 3:10] = input_ids[0, 4:11]
    batch = {"input_ids": input_ids, "labels": labels}
    log_probs, labels_out = tokenwise_logprobs_from_logits(batch, logits, return_labels=True)
    assert len(log_probs) == 1 and len(labels_out) == 1
    assert log_probs[0].shape[0] == 6  # 3..9 minus last from actual_indices[:-1]
    assert labels_out[0].shape[0] == 6
