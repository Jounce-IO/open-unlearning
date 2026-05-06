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
from trainer import load_trainer


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


def test_load_trainer_derives_ce_defaults_from_adapter_config_when_unset() -> None:
    from transformers import AutoModelForCausalLM, GPT2Config

    cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=32, n_layer=1, n_head=2)
    model = AutoModelForCausalLM.from_config(cfg)
    model.adapter_config = type(
        "AdapterCfg",
        (),
        {
            "loss_weight_type": "linear",
            "loss_normalization_type": "mean",
            "time_epsilon": 0.123,
            "eval_sampler": "mdlm",
            "eval_num_steps": 8,
            "eval_temperature": 0.0,
            "eval_top_p": 0.95,
            "eval_argmax_decoding": False,
            "eval_mask_ratio_generator_type": "cosine",
        },
    )()

    trainer_cfg = {
        "handler": "FinetuneTrainer",
        "args": {"output_dir": "/tmp/ou_runtime_defaults", "report_to": "none"},
        "method_args": {
            "tofu_multi_eval": True,
            "tofu_multi_eval_rouge": False,
            "tofu_multi_eval_steps": 1,
        },
    }
    trainer, _ = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=_Tiny(1),
        eval_dataset={"forget": _Tiny(1)},
        tokenizer=None,
    )
    rt = getattr(trainer, "_dllm_tofu_multi_phase2_runtime")
    assert rt.ce_loss_weight_type == "linear"
    assert rt.ce_loss_normalization_type == "mean"
    assert rt.ce_time_epsilon == 0.123


def test_load_trainer_keeps_explicit_ce_overrides() -> None:
    from transformers import AutoModelForCausalLM, GPT2Config

    cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=32, n_layer=1, n_head=2)
    model = AutoModelForCausalLM.from_config(cfg)
    model.adapter_config = type(
        "AdapterCfg",
        (),
        {
            "loss_weight_type": "scheduler",
            "loss_normalization_type": "batch",
            "time_epsilon": 0.25,
            "eval_sampler": "mdlm",
            "eval_num_steps": 8,
            "eval_temperature": 0.0,
            "eval_top_p": 0.95,
            "eval_argmax_decoding": False,
            "eval_mask_ratio_generator_type": "cosine",
        },
    )()

    trainer_cfg = {
        "handler": "FinetuneTrainer",
        "args": {"output_dir": "/tmp/ou_runtime_overrides", "report_to": "none"},
        "method_args": {
            "tofu_multi_eval": True,
            "tofu_multi_eval_rouge": False,
            "tofu_multi_eval_steps": 1,
            "tofu_multi_eval_ce_loss_weight_type": "uniform",
            "tofu_multi_eval_ce_loss_normalization_type": "batch",
            "tofu_multi_eval_ce_time_epsilon": 0.01,
        },
    }
    trainer, _ = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=_Tiny(1),
        eval_dataset={"forget": _Tiny(1)},
        tokenizer=None,
    )
    rt = getattr(trainer, "_dllm_tofu_multi_phase2_runtime")
    assert rt.ce_loss_weight_type == "uniform"
    assert rt.ce_loss_normalization_type == "batch"
    assert rt.ce_time_epsilon == 0.01


def test_evaluators_path_runs_phase2_even_when_multi_process() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

    from dllm.core.trainers.tofu_multi_eval_phase2 import build_runtime_from_flat_method_args

    cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=32, n_layer=1, n_head=2)
    model = AutoModelForCausalLM.from_config(cfg)
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token_id = tok.eos_token_id

    args = TrainingArguments(
        output_dir="/tmp/ou_tofu_eval_multi",
        per_device_eval_batch_size=1,
        report_to="none",
    )
    trainer = FinetuneTrainer(
        model=model,
        args=args,
        train_dataset=_Tiny(1),
        eval_dataset={"forget": _Tiny(1)},
        tokenizer=tok,
        evaluators={"dummy": type("E", (), {"evaluate": lambda self, **kw: {"x": 1.0}})()},
    )
    trainer.attach_tofu_multi_eval_phase2_runtime(
        build_runtime_from_flat_method_args(
            {
                "tofu_multi_eval": True,
                "tofu_multi_eval_rouge": False,
                "tofu_multi_eval_steps": 1,
            }
        )
    )
    trainer.accelerator = type(
        "AccelStub",
        (),
        {"num_processes": 2, "is_local_main_process": False},
    )()

    with (
        patch(
            "dllm.core.trainers.tofu_multi_eval_phase2.tofu_multi_eval_step_schedule",
            return_value=(True, False),
        ),
        patch(
            "dllm.core.trainers.tofu_multi_eval_phase2.run_tofu_multi_eval_phase2",
            return_value=EvalLoopOutput(
                predictions=None,
                label_ids=None,
                metrics={"eval_forget_loss": 0.5},
                num_samples=1,
            ),
        ) as run_p2,
    ):
        out = trainer.evaluate(metric_key_prefix="eval")

    assert run_p2.called
    assert out.get("eval_forget_loss") == 0.5
