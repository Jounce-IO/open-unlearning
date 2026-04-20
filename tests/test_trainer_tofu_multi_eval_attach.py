"""FinetuneTrainer uses MDLM-style TOFU multi-eval when runtime is attached."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Dataset
from transformers import Trainer as HFTrainer
from transformers import TrainingArguments
from transformers.trainer_utils import EvalLoopOutput

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from trainer.base import FinetuneTrainer


class _Tiny(Dataset):
    def __init__(self, n: int = 2):
        self.n = n
        self.data = {
            "input_ids": torch.ones(n, 4, dtype=torch.long),
            "labels": torch.ones(n, 4, dtype=torch.long),
        }

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: v[i].clone() for k, v in self.data.items()}


def test_mdlm_style_eval_skips_ou_phase1_and_uses_shared_phase2() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

    from dllm.core.trainers.tofu_multi_eval_phase2 import (
        TofuMultiEvalPhase2Runtime,
        build_runtime_from_flat_method_args,
    )

    cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=32, n_layer=1, n_head=2)
    model = AutoModelForCausalLM.from_config(cfg)
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token_id = tok.eos_token_id
    if getattr(tok, "mask_token_id", None) is None:
        tok.mask_token_id = 0

    rt: TofuMultiEvalPhase2Runtime = build_runtime_from_flat_method_args(
        {
            "tofu_multi_eval_rouge": False,
            "tofu_multi_eval_steps": 1,
            "tofu_multi_eval_rouge_steps": 1,
        }
    )

    args = TrainingArguments(
        output_dir="/tmp/ou_tofu_attach",
        per_device_eval_batch_size=2,
        report_to="none",
    )
    ds = {"forget": _Tiny(2)}
    trainer = FinetuneTrainer(
        model=model,
        args=args,
        train_dataset=_Tiny(2),
        eval_dataset=ds,
        tokenizer=tok,
    )
    trainer.attach_tofu_multi_eval_phase2_runtime(rt)

    phase1_prefixes: list[str] = []

    def fake_hf_evaluate(self, dataset, ignore_keys=None, metric_key_prefix="eval", **kwargs):
        phase1_prefixes.append(metric_key_prefix)
        return {f"{metric_key_prefix}_loss": 0.1}

    p2_metrics = {"eval_forget_loss": 0.2, "eval_forget_loss_ce": 0.3}

    def fake_run_p2(tr, eval_dataset, **kwargs):
        assert kwargs.get("log_metrics") is False
        return EvalLoopOutput(
            predictions=None, label_ids=None, metrics=dict(p2_metrics), num_samples=2
        )

    with (
        patch.object(HFTrainer, "evaluate", fake_hf_evaluate),
        patch(
            "dllm.core.trainers.tofu_multi_eval_phase2.run_tofu_multi_eval_phase2",
            fake_run_p2,
        ),
    ):
        out = trainer._evaluate_four_way(ds, metric_key_prefix="eval")

    assert phase1_prefixes == []
    assert out.metrics.get("eval_forget_loss") == 0.2
    assert out.metrics.get("eval_forget_loss_ce") == 0.3
