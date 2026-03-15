"""
Debug script: real TOFU data + real code path to see why wrong-side probability
returns None. Prints shapes and any exception.
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
    from evals.metrics.trajectory_metrics import _call_metric_at_step
    from evals.metrics import METRICS_REGISTRY
    import torch
    from torch.utils.data import DataLoader

    _ = load_dataset("locuslab/TOFU", "forget01_perturbed", split="train")  # raw, reserved
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
    batch = next(iter(dl))

    L = batch["input_ids"].shape[1]
    V = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 50257
    logits = torch.zeros(1, L, V)

    # Shapes from real batch
    print("input_ids.shape:", batch["input_ids"].shape)
    print("labels_correct.shape:", batch["labels_correct"].shape if batch.get("labels_correct") is not None else None)
    lw = batch.get("labels_wrong")
    if lw is not None:
        print("labels_wrong.shape:", lw.shape if hasattr(lw, "shape") else [getattr(t, "shape", None) for t in lw] if isinstance(lw, list) else lw)
    else:
        print("labels_wrong: None")

    # Call probability for CORRECT (single labels)
    pre_bt_correct = {**batch, "labels": batch["labels_correct"]}
    prob_metric = METRICS_REGISTRY["probability"]
    try:
        res_correct = _call_metric_at_step(
            metric=prob_metric,
            logits=logits,
            batch_template=pre_bt_correct,
            tokenizer=tokenizer,
            sample_labels=batch.get("labels"),
            sample_input_ids=batch["input_ids"],
            sample_prompt_len=0,
            sample_idx="0",
            metric_config={},
        )
        print("correct result:", type(res_correct), res_correct[0] if isinstance(res_correct, list) and res_correct else res_correct)
    except Exception as e:
        print("correct RAISED:", type(e).__name__, e)

    # Call probability for first WRONG option (same code path as _compute_pre_compute_metrics_at_step)
    if lw is not None:
        if hasattr(lw, "shape") and lw.dim() == 3:
            lab_first = lw[:, 0, :].contiguous()
        elif isinstance(lw, list):
            lab_first = lw[0]
        else:
            lab_first = lw
        print("first wrong labels.shape:", lab_first.shape if hasattr(lab_first, "shape") else None)
        pre_bt_wrong = {**batch, "labels": lab_first}
        try:
            res_wrong = _call_metric_at_step(
                metric=prob_metric,
                logits=logits,
                batch_template=pre_bt_wrong,
                tokenizer=tokenizer,
                sample_labels=batch.get("labels"),
                sample_input_ids=batch["input_ids"],
                sample_prompt_len=0,
                sample_idx="0",
                metric_config={},
            )
            print("wrong result:", type(res_wrong), res_wrong[0] if isinstance(res_wrong, list) and res_wrong else res_wrong)
        except Exception as e:
            print("wrong RAISED:", type(e).__name__, e)
            import traceback
            traceback.print_exc()

def run_one_pre_compute():
    """Run _compute_pre_compute_metrics_at_step once and print correct/wrong structure."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import torch
    from omegaconf import OmegaConf
    from evals.metrics.trajectory_metrics import _compute_pre_compute_metrics_at_step

    load_dataset("locuslab/TOFU", "forget01_perturbed", split="train")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from data.qa import QAwithDualAnswersDataset
    from data.collators import DataCollatorForSupervisedDataset
    from torch.utils.data import DataLoader
    dataset = QAwithDualAnswersDataset(
        correct_answer_key="paraphrased_answer", wrong_answer_key="perturbed_answer",
        hf_args={"path": "locuslab/TOFU", "name": "forget01_perturbed", "split": "train"},
        template_args=OmegaConf.create({}), tokenizer=tokenizer, max_length=256,
    )
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, padding_side="left", index="index")
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=False)))
    L = batch["input_ids"].shape[1]
    V = tokenizer.vocab_size or 50257
    logits = torch.zeros(1, L, V)
    pre_compute_config = OmegaConf.create({
        "correct": {"access_key": "correct", "labels_field": "labels_correct", "handler": "probability"},
        "wrong": {"access_key": "wrong", "labels_field": "labels_wrong", "handler": "probability"},
    })
    batch_template = {
        "input_ids": batch["input_ids"], "labels": batch["labels"],
        "attention_mask": batch["attention_mask"], "index": batch["index"],
        "labels_correct": batch.get("labels_correct", batch["labels"]),
        "labels_wrong": batch["labels_wrong"],
    }
    print("labels_wrong type:", type(batch_template["labels_wrong"]), getattr(batch_template["labels_wrong"], "shape", None))
    results = _compute_pre_compute_metrics_at_step(
        pre_compute_config=pre_compute_config, logits=logits, batch_template=batch_template,
        tokenizer=tokenizer, sample_labels=batch.get("labels"), sample_input_ids=batch["input_ids"],
        sample_prompt_len=0, sample_idx="0",
    )
    print("correct keys:", results["correct"].keys(), "vbi:", list(results["correct"].get("value_by_index", {}).keys()))
    c0 = list(results["correct"].get("value_by_index", {}).values())
    print("correct first entry:", c0[0] if c0 else None)
    wrong_val = results["wrong"]
    print("wrong type:", type(wrong_val), "is_list:", isinstance(wrong_val, list))
    if isinstance(wrong_val, list):
        for i, w in enumerate(wrong_val):
            vbi = w.get("value_by_index", {})
            print("  wrong[%d] vbi:" % i, list(vbi.keys()), list(vbi.values())[0] if vbi else None)
    else:
        vbi = wrong_val.get("value_by_index", {})
        print("wrong vbi:", list(vbi.keys()), list(vbi.values())[0] if vbi else None)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "pre_compute":
        run_one_pre_compute()
    else:
        main()
