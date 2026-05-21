"""Tests for target_sequence suffix masking."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

from data.utils import IGNORE_INDEX
from evals.metrics.target_sequence import (
    first_suffix_label_index,
    mask_suffix_template_labels,
    parse_exclude_suffix_template_tokens,
    suffix_template_token_ids,
)


def test_parse_exclude_suffix_default_false() -> None:
    assert parse_exclude_suffix_template_tokens({}) is False
    assert parse_exclude_suffix_template_tokens({"target_sequence": {}}) is False
    assert (
        parse_exclude_suffix_template_tokens(
            {"target_sequence": {"exclude_suffix_template_tokens": True}}
        )
        is True
    )


def test_mask_suffix_template_labels_llada() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )
    suffix_ids = suffix_template_token_ids(tokenizer)
    assert suffix_ids

    # Content + eot + header-like tail
    content = tokenizer.encode("Hello.", add_special_tokens=False)
    eot = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
    tail = tokenizer.encode("<|start_header_id|>", add_special_tokens=False)
    ids = content + eot + tail
    labels = torch.tensor(ids, dtype=torch.long)
    masked = mask_suffix_template_labels(labels, tokenizer)
    idx = first_suffix_label_index(labels, suffix_ids)
    assert idx == len(content)
    assert masked[:idx].tolist() == labels[:idx].tolist()
    assert (masked[idx:] == IGNORE_INDEX).all()


def test_no_suffix_leaves_labels_unchanged() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )
    ids = tokenizer.encode("Only answer text.", add_special_tokens=False)
    labels = torch.tensor(ids, dtype=torch.long)
    masked = mask_suffix_template_labels(labels, tokenizer)
    assert torch.equal(masked, labels)
