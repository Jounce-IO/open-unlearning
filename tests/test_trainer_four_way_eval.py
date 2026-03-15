"""Tests for FinetuneTrainer.evaluate() with dict of datasets (four-way validation)."""

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from trainer.base import FinetuneTrainer, _scalar_metrics_for_wandb


class _TinyDataset(Dataset):
    def __init__(self, length=4, seq_len=8, vocab_size=64):
        self.length = length
        self.data = {
            "input_ids": torch.randint(1, vocab_size, (length, seq_len)),
            "labels": torch.randint(1, vocab_size, (length, seq_len)),
        }
        self.data["labels"][:, :2] = -100

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
    trainer = FinetuneTrainer(
        model=model,
        args=args,
        train_dataset=_TinyDataset(2, 8, 64),
        eval_dataset=eval_dict,
        tokenizer=tokenizer,
    )
    result = trainer.evaluate(eval_dataset=eval_dict, metric_key_prefix="eval")
    # Four-way path returns EvalLoopOutput to match parent Trainer.evaluate()
    metrics = result.metrics
    assert "eval_forget_loss" in metrics
    assert "eval_retain_loss" in metrics
    assert isinstance(metrics["eval_forget_loss"], (int, float))
    assert isinstance(metrics["eval_retain_loss"], (int, float))
    # _ce keys may be present when dllm + DiffusionModelAdapter are available
    for k, v in metrics.items():
        assert isinstance(v, (int, float)), k
