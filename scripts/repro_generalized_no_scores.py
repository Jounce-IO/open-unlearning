"""
Reproduce "no scores" in generalized trajectory path locally.
Uses real TOFU data + mock R,F; forces use_generalized_sequence_probability path.
Output is to stdout; no fixed-path file writes.
"""
from __future__ import annotations

import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, "src"))

from data.utils import IGNORE_INDEX


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from data.qa import QAwithDualAnswersDataset
    from data.collators import DataCollatorForSupervisedDataset
    from omegaconf import OmegaConf
    from evals.metrics.trajectory_metrics import (
        _compute_pre_compute_metrics_at_step,
        _batch_template_dual_labels,
    )
    from evals.metrics import METRICS_REGISTRY
    import torch
    from torch.utils.data import DataLoader

    raw = load_dataset("locuslab/TOFU", "forget01_perturbed", split="train")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = QAwithDualAnswersDataset(
        correct_answer_key="paraphrased_answer",
        wrong_answer_key="perturbed_answer",
        hf_args={"path": "locuslab/TOFU", "name": "forget01_perturbed", "split": "train"},
        template_args=OmegaConf.create({}),
        tokenizer=tokenizer,
        max_length=256,
    )
    collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, padding_side="left", index="index"
    )
    dl = DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=False)

    L = 50  # mock trajectory length (must match R, F)
    V = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 50257
    S = 10  # mock steps
    trajectory_config = OmegaConf.create({
        "use_generalized_sequence_probability": True,
        "logit_alignment": "causal",
    })
    pre_compute_config = OmegaConf.create({
        "correct": {"access_key": "correct", "labels_field": "labels_correct", "handler": "probability"},
        "wrong": {"access_key": "wrong", "labels_field": "labels_wrong", "handler": "probability"},
    })
    tr = METRICS_REGISTRY["truth_ratio"]

    # Test L=0 first (can cause no scores: L_use=0 in get_per_position_scores)
    print("Test L=0 (expect no scores -> truth_ratio None)...")
    batch0 = next(iter(dl))
    if batch0.get("labels_wrong") is not None:
        idx_str = str(int(batch0["index"].item()))
        batch_template0 = {"input_ids": batch0["input_ids"], "labels": batch0.get("labels"), "attention_mask": batch0["attention_mask"], "index": batch0["index"]}
        for key in ("labels_correct", "labels_wrong"):
            if key in batch0:
                batch_template0[key] = _batch_template_dual_labels(batch0, 0, key, 0, IGNORE_INDEX)
        R0 = torch.randn(V, 0, S)
        F0 = torch.randint(0, S, (0,))
        sample_traj0 = {"R": R0, "F": F0, "S": S, "L": 0}
        results0 = _compute_pre_compute_metrics_at_step(
            pre_compute_config=pre_compute_config,
            logits=torch.zeros(1, 0, V),
            batch_template=batch_template0,
            tokenizer=tokenizer,
            sample_labels=batch0.get("labels"),
            sample_input_ids=batch0["input_ids"],
            sample_prompt_len=0,
            sample_idx=idx_str,
            trajectory_config=trajectory_config,
            sample_traj=sample_traj0,
            step=0,
        )
        out0 = tr._metric_fn(model=None, pre_compute={"correct": results0["correct"], "wrong": results0["wrong"]}, aggregator="closer_to_1_better")
        print("  L=0 -> truth_ratio agg_value: %s (expect None)" % out0.get("agg_value"))
        assert out0.get("agg_value") is None, "L=0 should yield truth_ratio None"
    print()

    num_samples = min(100, len(dataset))
    print("Running generalized path over %d samples (L=%d)..." % (num_samples, L))
    bug_samples = []

    for i, batch in enumerate(dl):
        if i >= num_samples:
            break
        idx_str = str(int(batch["index"].item())) if batch.get("index") is not None else str(i)
        if batch.get("labels_wrong") is None:
            continue
        # Build batch_template exactly as trajectory loop: labels_* from _batch_template_dual_labels(batch, sample_idx, key, L, IGNORE_INDEX)
        batch_template = {
            "input_ids": batch["input_ids"],
            "labels": batch.get("labels"),
            "attention_mask": batch["attention_mask"],
            "index": batch["index"],
        }
        for key in ("labels_correct", "labels_wrong"):
            if key in batch:
                batch_template[key] = _batch_template_dual_labels(
                    batch, 0, key, L, IGNORE_INDEX
                )
        # Mock R [V, L, S], F [L] for one sample
        R = torch.randn(V, L, S)
        F = torch.randint(0, S, (L,))
        sample_traj = {"R": R, "F": F, "S": S, "L": L}
        logits = torch.zeros(1, L, V)

        results = _compute_pre_compute_metrics_at_step(
            pre_compute_config=pre_compute_config,
            logits=logits,
            batch_template=batch_template,
            tokenizer=tokenizer,
            sample_labels=batch.get("labels"),
            sample_input_ids=batch["input_ids"],
            sample_prompt_len=0,
            sample_idx=idx_str,
            trajectory_config=trajectory_config,
            sample_traj=sample_traj,
            step=0,
        )
        correct_ok = results.get("correct") and (
            (results["correct"].get("agg_value") is not None)
            or (results["correct"].get("value_by_index") and next(iter(results["correct"]["value_by_index"].values()), {}).get("avg_loss") is not None)
        )
        wrong_val = results.get("wrong")
        if isinstance(wrong_val, list):
            wrong_ok = all(
                (w.get("value_by_index") and next(iter(w["value_by_index"].values()), {}).get("avg_loss") is not None)
                for w in wrong_val
            )
        else:
            wrong_ok = wrong_val and (
                wrong_val.get("agg_value") is not None
                or (wrong_val.get("value_by_index") and next(iter(wrong_val["value_by_index"].values()), {}).get("avg_loss") is not None)
            )
        if not correct_ok or not wrong_ok:
            bug_samples.append((idx_str, correct_ok, wrong_ok))
        # truth_ratio on pre_compute
        out = tr._metric_fn(
            model=None,
            pre_compute={"correct": results["correct"], "wrong": results["wrong"]},
            aggregator="closer_to_1_better",
        )
        if out.get("agg_value") is None:
            print("  sample %s: truth_ratio agg_value=None (correct_ok=%s wrong_ok=%s)" % (idx_str, correct_ok, wrong_ok))

    print("Samples with missing correct or wrong pre_compute: %d" % len(bug_samples))
    for idx_str, c, w in bug_samples[:20]:
        print("  index=%s correct_ok=%s wrong_ok=%s" % (idx_str, c, w))
    if len(bug_samples) > 20:
        print("  ... and %d more" % (len(bug_samples) - 20))
    print("Done.")


if __name__ == "__main__":
    main()
