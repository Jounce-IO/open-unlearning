"""FinetuneTrainer four-way eval with real dllm.four_way_rouge (unlearn-shaped path, no mocks)."""

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    TrainingArguments,
)

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from trainer.base import FinetuneTrainer

# Parent workspace package (dllm) — same as other OU tests that import dllm.four_way_rouge.
from dllm.testing.partial_echo_generate import make_partial_echo_output
from dllm.utils.collators import NoAttentionMaskWrapper


def _build_sft_example(
    tokenizer: GPT2TokenizerFast, prompt: str, answer: str
) -> dict[str, torch.Tensor]:
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(" " + answer.strip(), add_special_tokens=False)["input_ids"]
    full = p_ids + a_ids
    input_ids = torch.tensor(full, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    labels[len(p_ids) :] = input_ids[len(p_ids) :].clone()
    return {"input_ids": input_ids, "labels": labels}


class _ListFeatureDataset(Dataset):
    def __init__(self, rows: list[dict[str, torch.Tensor]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        r = self.rows[i]
        return {
            "input_ids": r["input_ids"].tolist(),
            "labels": r["labels"].tolist(),
        }


class _GPT2LMHeadEchoGenerate(torch.nn.Module):
    """Tiny GPT2 for HF compute_loss; generate() partial-echoes gold (ROUGE in (0,1))."""

    def __init__(self, vocab_size: int, pad_token_id: int):
        super().__init__()
        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_positions=256,
            n_embd=32,
            n_layer=1,
            n_head=2,
            n_inner=128,
        )
        self.base = GPT2LMHeadModel(cfg)
        self._pad_id = int(pad_token_id)

    def forward(self, **kwargs):
        return self.base(**kwargs)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pad_token_id=None,
        max_new_tokens: int = 32,
        **kwargs,
    ):
        kwargs.pop("remasking", None)
        return make_partial_echo_output(
            input_ids,
            attention_mask,
            pad_token_id,
            self._pad_id,
            int(max_new_tokens),
            echo_fraction=0.72,
        )


def test_finetune_trainer_four_way_eval_real_rouge_unmocked():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    if getattr(tok, "mask_token_id", None) is None:
        tok.mask_token_id = 0
    pad_id = int(tok.pad_token_id)

    forget_ds = _ListFeatureDataset(
        [
            _build_sft_example(
                tok,
                "Q: Capital of France? A:",
                "Paris is the capital.",
            ),
            _build_sft_example(
                tok,
                "Q: Color of sky? A:",
                "The sky is blue.",
            ),
        ]
    )
    retain_ds = _ListFeatureDataset(
        [
            _build_sft_example(
                tok,
                "Q: Two plus two? A:",
                "The answer is four.",
            ),
        ]
    )
    eval_dict = {"forget": forget_ds, "retain": retain_ds}

    model = _GPT2LMHeadEchoGenerate(tok.vocab_size, pad_id)
    args = TrainingArguments(
        output_dir="/tmp/test_ou_four_way_rouge_e2e",
        per_device_eval_batch_size=2,
        report_to="none",
        remove_unused_columns=False,
    )
    base_collator = DataCollatorForSeq2Seq(
        tok,
        padding=True,
        return_tensors="pt",
        label_pad_token_id=tok.pad_token_id,
    )
    collator = NoAttentionMaskWrapper(base_collator)

    trainer = FinetuneTrainer(
        model=model,
        args=args,
        train_dataset=forget_ds,
        eval_dataset=eval_dict,
        tokenizer=tok,
        data_collator=collator,
        evaluators=None,
        four_way_rouge=True,
        four_way_rouge_remasking="low_confidence",
        four_way_rouge_generation_args={"max_new_tokens": 24, "tokens_per_step": 4},
    )

    result = trainer.evaluate(eval_dataset=eval_dict, metric_key_prefix="eval")
    metrics = result.metrics

    for split in ("forget", "retain"):
        k = f"eval_{split}_rouge1_recall"
        assert k in metrics, metrics.keys()
        assert metrics[k] > 0.0
        assert metrics[k] < 1.0, f"{split}: partial echo should not yield ROUGE 1.0"
        for suffix in ("rougeL_f1", "rougeL_recall"):
            assert f"eval_{split}_{suffix}" in metrics
    for k, v in metrics.items():
        assert isinstance(v, (int, float)), k
