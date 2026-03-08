"""
Identify which TOFU samples can cause "no scores" in the generalized path.

Generalized path gets no scores for an option when that option's label row has
no non-IGNORE_INDEX tokens -> get_per_position_scores returns empty -> we store
prob/avg_loss None for that option. That happens when:
- The wrong-option label tensor (after tokenization + collation) is all IGNORE_INDEX.
- Typically: perturbed_answer[k] is empty/whitespace, or tokenization yields no valid labels.

This script (1) scans HF dataset for empty/whitespace perturbed answers,
(2) optionally runs the real dataset+collator and flags samples where any wrong option
    has all-ignore labels.
"""

from __future__ import annotations

import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, "src"))


def main():
    from datasets import load_dataset

    ds = load_dataset("locuslab/TOFU", "forget01_perturbed", split="train")
    n = len(ds)
    empty_perturbed_by_sample = []  # list of (idx, option_indices) where that option is empty/whitespace
    for i in range(n):
        row = ds[i]
        pa = row.get("perturbed_answer")
        if pa is None:
            continue
        if isinstance(pa, list):
            bad = [k for k in range(len(pa)) if not (pa[k] or "").strip()]
        else:
            bad = [] if (pa or "").strip() else [0]
        if bad:
            empty_perturbed_by_sample.append((i, bad))
    print("Samples with at least one empty/whitespace perturbed_answer:")
    if not empty_perturbed_by_sample:
        print("  None found in forget01_perturbed.")
    else:
        for idx, opts in empty_perturbed_by_sample[:50]:
            print(f"  dataset index {idx}: empty option indices {opts}")
        if len(empty_perturbed_by_sample) > 50:
            print(f"  ... and {len(empty_perturbed_by_sample) - 50} more.")
    print(f"Total: {len(empty_perturbed_by_sample)} samples with empty perturbed option(s).")

    # Now check with real tokenizer + dataset: which samples have any wrong-option labels all IGNORE?
    from transformers import AutoTokenizer
    from data.qa import QAwithDualAnswersDataset
    from data.collators import DataCollatorForSupervisedDataset
    from data.utils import IGNORE_INDEX
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

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
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, padding_side="left", index="index")
    dl = DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=False)
    all_ignore_wrong_option = []  # (batch_index, sample_idx_in_batch, option_k)
    for batch_idx, batch in enumerate(dl):
        if batch.get("labels_wrong") is None:
            continue
        lw = batch["labels_wrong"]  # [1, N, L]
        if lw.dim() != 3:
            continue
        N = lw.shape[1]
        for k in range(N):
            row = lw[0, k]
            if (row != IGNORE_INDEX).any().item():
                continue
            all_ignore_wrong_option.append((batch_idx, 0, k))
    print("\nSamples (batch_idx, sample_in_batch, wrong_option_k) with that option's labels all IGNORE_INDEX:")
    if not all_ignore_wrong_option:
        print("  None (collator/dataset may not produce all-ignore for forget01_perturbed).")
    else:
        for t in all_ignore_wrong_option[:50]:
            print(f"  {t}")
        if len(all_ignore_wrong_option) > 50:
            print(f"  ... and {len(all_ignore_wrong_option) - 50} more.")
    print(f"Total: {len(all_ignore_wrong_option)} (batch_idx, sample, option_k) with all-ignore wrong option.")


if __name__ == "__main__":
    main()
