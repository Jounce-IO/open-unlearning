"""
Reproduce truth_ratio correct/wrong indices and "no valid pre_compute" locally.
Uses real TOFU HF data, no GPU. Runs the same pre_compute flow as trajectory eval.
Instrumentation writes to .cursor/debug-0656be.log for hypothesis analysis.
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
    from omegaconf import OmegaConf
    from evals.metrics.trajectory_metrics import _compute_pre_compute_metrics_at_step
    from evals.metrics import METRICS_REGISTRY
    import torch
    from torch.utils.data import DataLoader

    # Real TOFU data (same as job)
    raw = load_dataset("locuslab/TOFU", "forget01_perturbed", split="train")
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
    dl = DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=False)
    V = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 50257
    pre_compute_config = {
        "correct": {"access_key": "correct", "labels_field": "labels_correct", "handler": "probability"},
        "wrong": {"access_key": "wrong", "labels_field": "labels_wrong", "handler": "probability"},
    }
    tr = METRICS_REGISTRY["truth_ratio"]

    # 1) Run over many samples with real data to see if any hit "no valid pre_compute"
    num_samples = min(50, len(dataset))
    print("Running over %d samples (real TOFU forget01_perturbed)..." % num_samples)
    bug_count = 0
    for i, batch in enumerate(dl):
        if i >= num_samples:
            break
        idx_str = str(int(batch["index"].item())) if batch.get("index") is not None else str(i)
        if batch.get("labels_wrong") is None:
            continue
        L = batch["input_ids"].shape[1]
        batch_template = {
            "input_ids": batch["input_ids"],
            "labels": batch["labels"],
            "attention_mask": batch["attention_mask"],
            "index": batch["index"],
            "labels_correct": batch.get("labels_correct", batch["labels"]),
            "labels_wrong": batch["labels_wrong"],
        }
        logits = torch.zeros(1, L, V)
        results = _compute_pre_compute_metrics_at_step(
            pre_compute_config=OmegaConf.create(pre_compute_config),
            logits=logits,
            batch_template=batch_template,
            tokenizer=tokenizer,
            sample_labels=batch.get("labels"),
            sample_input_ids=batch["input_ids"],
            sample_prompt_len=0,
            sample_idx=idx_str,
        )
        out = tr._metric_fn(
            model=None,
            pre_compute={"correct": results["correct"], "wrong": results["wrong"]},
            aggregator="closer_to_1_better",
        )
        if out.get("agg_value") is None:
            bug_count += 1
            print("  sample %s: truth_ratio returned None" % idx_str)
    print("Samples with truth_ratio None: %d / %d" % (bug_count, num_samples))

    # 2) Force bug: wrong[0] has empty value_by_index (as in trajectory when no scores for first option)
    print("Forcing bug case: wrong[0] value_by_index empty...")
    correct_ok = {"value_by_index": {"0": {"prob": 0.5, "avg_loss": -0.693}}, "agg_value": 0.5}
    wrong_bad = [{"value_by_index": {}, "agg_value": None}, {"value_by_index": {"0": {"prob": 0.25, "avg_loss": -1.386}}, "agg_value": 0.25}]
    out_forced = tr._metric_fn(
        model=None,
        pre_compute={"correct": correct_ok, "wrong": wrong_bad},
        aggregator="closer_to_1_better",
    )
    print("Forced case truth_ratio agg_value:", out_forced.get("agg_value"))

    # 3) Force bug: all wrong options have None avg_loss for index (filtered_indices becomes empty)
    print("Forcing bug case: all wrong options None avg_loss for index...")
    wrong_all_none = [{"value_by_index": {"0": {"prob": None, "avg_loss": None}}, "agg_value": None}] * 5
    out_forced2 = tr._metric_fn(
        model=None,
        pre_compute={"correct": correct_ok, "wrong": wrong_all_none},
        aggregator="closer_to_1_better",
    )
    print("Forced case (all wrong None) truth_ratio agg_value:", out_forced2.get("agg_value"))

    print("Done. Check .cursor/debug-0656be.log for hypothesis logs.")


if __name__ == "__main__":
    main()
