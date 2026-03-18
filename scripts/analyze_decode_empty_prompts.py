"""
Analyze prompts that decode to empty (skip_special_tokens=True).

Finds exact dataset indices, raw dataset content, token IDs, and decoded strings
to determine whether empty decode is due to: (1) empty/question-less data,
(2) extraction bug, or (3) tokenizer behavior.

Run from open-unlearning: python scripts/analyze_decode_empty_prompts.py
  PYTHONPATH=src python scripts/analyze_decode_empty_prompts.py
"""

from __future__ import annotations

import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, "src"))

IGNORE_INDEX = -100


def main():
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from datasets import load_dataset

    from data.qa import QAwithDualAnswersDataset
    from data.collators import DataCollatorForSupervisedDataset
    from evals.metrics.samplers import LengthSortedSampler
    from evals.metrics.trajectory_metrics import _build_prompts_for_sampler

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    print("Loading forget10_perturbed...")
    hf_args = {"path": "locuslab/TOFU", "name": "forget10_perturbed", "split": "train"}
    dataset = QAwithDualAnswersDataset(
        correct_answer_key="paraphrased_answer",
        wrong_answer_key="perturbed_answer",
        hf_args=hf_args,
        template_args={"apply_chat_template": True},
        tokenizer=tokenizer,
        question_key="question",
        max_length=512,
        predict_with_generate=True,
    )
    collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, padding_side="left", index="index"
    )
    sampler = LengthSortedSampler(dataset, length_key="input_ids", descending=True)
    dataloader = DataLoader(
        dataset, batch_size=4, sampler=sampler, collate_fn=collator
    )

    # Collect (idx -> prompt_tokens) via builder (same as test)
    # Also find a batch that contains one of the decode-empty indices to inspect shapes
    all_prompts_by_idx = {}
    batch_debug = None
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        indices = batch.get(
            "index",
            torch.arange(batch_idx * input_ids.shape[0], (batch_idx + 1) * input_ids.shape[0]),
        )
        if batch_idx == 0:
            batch_debug = {
                "input_ids_shape": input_ids.shape,
                "labels_shape": labels.shape if labels is not None else None,
                "indices": [indices[i].item() for i in range(indices.shape[0])],
            }
        prompt_only_input_ids = getattr(dataset, "predict_with_generate", False)
        prompts, prompt_lens, prompt_starts = _build_prompts_for_sampler(
            input_ids, labels, tokenizer, ignore_index=IGNORE_INDEX,
            prompt_only_input_ids=prompt_only_input_ids,
        )
        for i in range(len(prompts)):
            idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
            all_prompts_by_idx[idx] = (prompts[i], prompt_lens[i])

    # Indices that decode to empty
    decode_empty_indices = []
    for idx, (prompt_tokens, plen) in all_prompts_by_idx.items():
        decoded = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        if not decoded.strip():
            decode_empty_indices.append(idx)

    print(f"\n=== Indices whose prompt decodes to empty (skip_special_tokens=True): {len(decode_empty_indices)} ===")
    print(decode_empty_indices)

    # Raw HuggingFace dataset for question text
    raw_ds = load_dataset("locuslab/TOFU", name="forget10_perturbed", split="train")

    # Per-index analysis
    print("\n=== Per-index analysis (first 15 decode-empty indices) ===\n")
    for idx in decode_empty_indices[:15]:
        # Raw dataset row (question, answers)
        raw_row = raw_ds[int(idx)]
        question = raw_row.get("question", "")
        print(f"--- Dataset index {idx} ---")
        print(f"  question (raw): {repr(question[:200])}")
        print(f"  question length: {len(question)} chars, strip empty: {not question.strip()}")

        # What the QA dataset returns (single item, before collation)
        item = dataset[idx]
        raw_input_ids = item["input_ids"]
        if hasattr(raw_input_ids, "tolist"):
            raw_input_ids = raw_input_ids.tolist()
        else:
            raw_input_ids = list(raw_input_ids)
        raw_labels = item.get("labels")
        if raw_labels is not None and hasattr(raw_labels, "tolist"):
            raw_labels = raw_labels.tolist()
        elif raw_labels is not None:
            raw_labels = list(raw_labels)

        print(f"  dataset item input_ids length (before collation): {len(raw_input_ids)}")
        decoded_raw_no_skip = tokenizer.decode(raw_input_ids, skip_special_tokens=False)
        decoded_raw_skip = tokenizer.decode(raw_input_ids, skip_special_tokens=True)
        print(f"  decode(input_ids, skip_special_tokens=False) length: {len(decoded_raw_no_skip)}")
        print(f"  decode(input_ids, skip_special_tokens=True)  length: {len(decoded_raw_skip)}, strip empty: {not decoded_raw_skip.strip()}")
        if len(decoded_raw_no_skip) <= 400:
            print(f"  raw decoded (no skip): {repr(decoded_raw_no_skip[:300])}")

        # What we actually send (from builder, after batching/collation)
        prompt_tokens, plen = all_prompts_by_idx[idx]
        print(f"  builder prompt_tokens length: {plen}")
        decoded_prompt_no_skip = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
        decoded_prompt_skip = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        print(f"  decode(prompt_tokens, skip_special_tokens=True): {repr(decoded_prompt_skip[:200])}")

        # Are they the same?
        if raw_input_ids == prompt_tokens:
            print("  MATCH: prompt_tokens == dataset input_ids (no padding in this batch position)")
        else:
            # Compare lengths and content
            pad_stripped_raw = [t for t in raw_input_ids if t != pad_id]
            if pad_stripped_raw == prompt_tokens:
                print("  MATCH: prompt_tokens == raw input_ids with pad stripped")
            else:
                print(f"  DIFF: raw input_ids (first 20): {raw_input_ids[:20]}")
                print(f"  DIFF: prompt_tokens (first 20): {prompt_tokens[:20]}")
                if len(prompt_tokens) != len(raw_input_ids):
                    print(f"  DIFF: len(raw)={len(raw_input_ids)} len(prompt_tokens)={len(prompt_tokens)}")
        print()

    # Summary: do ALL decode-empty indices have empty or whitespace-only questions?
    print("\n=== Summary: question content for ALL decode-empty indices ===")
    all_empty_question = True
    for idx in decode_empty_indices:
        q = raw_ds[int(idx)].get("question", "")
        if q and q.strip():
            all_empty_question = False
            print(f"  Index {idx} has NON-EMPTY question: {repr(q[:80])}...")
    if all_empty_question:
        print("  ALL decode-empty indices have empty or whitespace-only 'question' in the dataset.")
    else:
        print("  At least one decode-empty index has a non-empty question -> possible extraction/tokenizer bug.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
