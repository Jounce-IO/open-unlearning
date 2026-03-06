"""
Reproduce truth_ratio correct/wrong indices mismatch locally.
Uses real TOFU HF data, no GPU. Runs the same pre_compute flow as trajectory eval.
"""
from __future__ import annotations

import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, "src"))

def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from data.qa import QAwithDualAnswersDataset
    from data.collators import DataCollatorForSupervisedDataset
    from omegaconf import OmegaConf
    from evals.metrics.trajectory_metrics import _compute_pre_compute_metrics_at_step, _call_metric_at_step
    from evals.metrics import METRICS_REGISTRY
    import torch

    # Real TOFU data (same as job)
    raw = load_dataset("locuslab/TOFU", "forget01_perturbed", split="train")
    raw = raw.select(range(2))
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_args = {"path": "locuslab/TOFU", "name": "forget01_perturbed", "split": "train"}
    dataset = QAwithDualAnswersDataset(
        correct_answer_key="paraphrased_answer",
        wrong_answer_key="perturbed_answer",
        hf_args=hf_args,
        template_args=OmegaConf.create({}),
        tokenizer=tokenizer,
        max_length=256,
    )
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, padding_side="left", index="index")
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=1, collate_fn=collator)
    batch = next(iter(dl))

    # Build batch_template as in trajectory loop: index as int, labels_correct, labels_wrong
    idx_str = str(int(batch["index"].item())) if batch.get("index") is not None else "0"
    L = batch["input_ids"].shape[1]
    batch_template = {
        "input_ids": batch["input_ids"],
        "labels": batch["labels"],
        "attention_mask": batch["attention_mask"],
        "index": batch["index"],
        "labels_correct": batch["labels_correct"] if "labels_correct" in batch else batch["labels"],
        "labels_wrong": batch["labels_wrong"] if "labels_wrong" in batch else None,
    }
    if batch_template["labels_wrong"] is None:
        print("No labels_wrong in batch - dataset may not have dual answers")
        return

    # Mock logits [1, L, V] for _call_metric_at_step (it accepts 2D [V,L] or 3D [1,L,V])
    V = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 50257
    logits = torch.zeros(1, L, V)

    # Inner pre_compute config (same as ks_test -> forget_truth_ratio)
    pre_compute_config = {
        "correct": {
            "access_key": "correct",
            "labels_field": "labels_correct",
            "handler": "probability",
        },
        "wrong": {
            "access_key": "wrong",
            "labels_field": "labels_wrong",
            "handler": "probability",
        },
    }

    print("Calling _compute_pre_compute_metrics_at_step with sample_idx=%r" % idx_str)
    results = _compute_pre_compute_metrics_at_step(
        pre_compute_config=pre_compute_config,
        logits=logits,
        batch_template=batch_template,
        tokenizer=tokenizer,
        sample_labels=batch.get("labels"),
        sample_input_ids=batch["input_ids"],
        sample_prompt_len=0,
        sample_idx=idx_str,
    )

    print("Correct value_by_index keys:", list(results["correct"]["value_by_index"].keys()))
    wrong_val = results["wrong"]
    if isinstance(wrong_val, list):
        print("Wrong[0] value_by_index keys:", list(wrong_val[0]["value_by_index"].keys()))
    else:
        print("Wrong value_by_index keys (dict path):", list(wrong_val["value_by_index"].keys()))
    print("Calling truth_ratio...")
    tr = METRICS_REGISTRY["truth_ratio"]
    out = tr._metric_fn(
        model=None,
        pre_compute={"correct": results["correct"], "wrong": results["wrong"]},
        aggregator="closer_to_1_better",
    )
    print("truth_ratio ok:", out.get("agg_value"))


if __name__ == "__main__":
    main()
