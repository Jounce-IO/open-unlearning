"""Tests for FinetuneTrainer.evaluate() with dict of datasets (four-way validation)."""

import sys
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from trainer.base import FinetuneTrainer


class _TinyDataset(Dataset):
    def __init__(self, length=4, seq_len=8, vocab_size=64):
        self.length = length
        input_ids = torch.randint(1, vocab_size, (length, seq_len))
        labels = input_ids.clone()
        labels[:, :2] = -100
        self.data = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones(length, seq_len, dtype=torch.long),
        }

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {k: v[i].clone() for k, v in self.data.items()}


def test_four_way_evaluate_returns_method_and_ce_keys_when_available():
    """When evaluate(eval_dataset=dict) is called, returned metrics include eval_{name}_loss (and _ce when dllm available)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

    # Tiny GPT2 with small vocab to match _TinyDataset
    config = GPT2Config(vocab_size=64, n_positions=16, n_embd=32, n_layer=1, n_head=2)
    model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(tokenizer, "mask_token_id", None) is None:
        tokenizer.mask_token_id = 0

    forget_ds = _TinyDataset(4, 8, 64)
    retain_ds = _TinyDataset(4, 8, 64)
    eval_dict = {"forget": forget_ds, "retain": retain_ds}

    args = TrainingArguments(
        output_dir="/tmp/test_four_way",
        per_device_eval_batch_size=2,
        report_to="none",
    )
    def _fake_rouge_scores(model, tok, batch, **kwargs):
        bsz = batch["input_ids"].shape[0]
        return [
            {"rouge1_recall": 0.1, "rougeL_f1": 0.2, "rougeL_recall": 0.3}
            for _ in range(bsz)
        ]

    with mock.patch(
        "dllm.four_way_rouge.four_way_rouge_scores_for_batch",
        side_effect=_fake_rouge_scores,
    ):
        trainer = FinetuneTrainer(
            model=model,
            args=args,
            train_dataset=_TinyDataset(2, 8, 64),
            eval_dataset=eval_dict,
            tokenizer=tokenizer,
            four_way_rouge_generation_args={"max_new_tokens": 4},
        )
        result = trainer.evaluate(eval_dataset=eval_dict, metric_key_prefix="eval")
    # Four-way path returns EvalLoopOutput to match parent Trainer.evaluate()
    metrics = result.metrics
    assert "eval_forget_loss" in metrics
    assert "eval_retain_loss" in metrics
    assert isinstance(metrics["eval_forget_loss"], (int, float))
    assert isinstance(metrics["eval_retain_loss"], (int, float))
    assert metrics["eval_forget_rougeL_recall"] == 0.3
    assert metrics["eval_retain_rouge1_recall"] == 0.1
    # _ce keys may be present when dllm + DiffusionModelAdapter are available
    for k, v in metrics.items():
        assert isinstance(v, (int, float)), k
