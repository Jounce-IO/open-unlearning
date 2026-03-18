"""
Validate prompt extraction locally (no GPU): reproduce the exact batching and
prompt extraction used by trajectory eval, and identify problematic prompts.

Run from open-unlearning: python scripts/validate_prompt_extraction_local.py
  (or from repo root: PYTHONPATH=open-unlearning/src python open-unlearning/scripts/validate_prompt_extraction_local.py)

Uses: forget10_perturbed, batch_size=4, sort_by_length=True, left padding.
Compares inline extraction (prompt_end = first where labels != IGNORE_INDEX,
prompt = input_ids[i, :prompt_end]) with _build_prompts_for_sampler (strips
leading pad, returns prompts, prompt_lens, prompt_starts).

Component analysis: tokenizer, dataset, collator, sampler order, inline vs builder.
"""

from __future__ import annotations

import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, "src"))

IGNORE_INDEX = -100


def analyze_components(
    tokenizer,
    dataset,
    collator,
    dataloader,
    eos_id,
    pad_id,
    get_question,
):
    """Print per-component analysis and return suspects for empty prompts."""
    import torch
    from evals.metrics.trajectory_metrics import _build_prompts_for_sampler

    suspects = []

    # --- Component 1: Tokenizer ---
    print("\n=== Component 1: Tokenizer ===")
    print(f"  pad_token_id={pad_id} eos_token_id={eos_id}")
    sample_ids = dataset[0]["input_ids"][:8]
    print(f"  Decode first 8 tokens of dataset[0]: {tokenizer.decode(sample_ids)}")
    if pad_id is not None:
        print(f"  Decode pad_token_id: {tokenizer.decode([pad_id])!r}")

    # --- Component 2: Dataset (single item) ---
    print("\n=== Component 2: Dataset (single item) ===")
    ex = dataset[0]
    lid = ex["input_ids"]
    lbl = ex.get("labels")
    print(f"  dataset[0] input_ids length={len(lid)}")
    if lbl is not None:
        n_ignore_start = sum(1 for t in lbl if t == IGNORE_INDEX)
        first_non_ignore = next((j for j, t in enumerate(lbl) if t != IGNORE_INDEX), len(lbl))
        print(f"  dataset[0] labels: leading IGNORE count={n_ignore_start} first_non_ignore_idx={first_non_ignore}")
        print(f"  (Inline would use prompt_end={first_non_ignore} -> prompt length={first_non_ignore}; when this item is longest in batch, collator adds no left padding -> same prompt_end=0.)")
    else:
        print("  dataset[0] no labels")

    # --- Component 3: Collator (one batch) ---
    print("\n=== Component 3: Collator (padding) ===")
    print(f"  padding_side={collator.padding_side}")
    batch0 = next(iter(dataloader))
    input_ids = batch0["input_ids"]
    labels = batch0.get("labels")
    indices = batch0.get("index", torch.arange(input_ids.shape[0]))
    B, L = input_ids.shape
    print(f"  First batch shape: B={B} L={L}")
    for i in range(B):
        row = input_ids[i]
        n_pad_left = 0
        for t in row.tolist():
            if t == pad_id or t == eos_id:
                n_pad_left += 1
            else:
                break
        if labels is not None:
            label_row = labels[i]
            first_non_ignore = (label_row != IGNORE_INDEX).nonzero()
            prompt_end = first_non_ignore[0][0].item() if first_non_ignore.numel() else L
            n_ignore_left = int((label_row == IGNORE_INDEX).sum())
            print(f"  sample {i} dataset_idx={indices[i].item()} pad_left~={n_pad_left} labels_ignore_left={n_ignore_left} prompt_end={prompt_end} -> inline_prompt_len={prompt_end}")
            if prompt_end == 0:
                suspects.append(("collator+inline", "sample 0 has prompt_end=0 (no left padding in labels)"))
        else:
            print(f"  sample {i} dataset_idx={indices[i].item()} pad_left~={n_pad_left}")

    # --- Component 4: Sampler order ---
    print("\n=== Component 4: Sampler order (first 2 batches) ===")
    lengths_per_idx = {}
    for idx in range(min(8, len(dataset))):
        lengths_per_idx[idx] = len(dataset[idx]["input_ids"])
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 2:
            break
        inds = batch["input_ids"].shape[0]
        idxs = [batch["index"][i].item() for i in range(inds)]
        lens = [len(dataset[idx]["input_ids"]) for idx in idxs]
        print(f"  batch {batch_idx}: dataset_indices={idxs} input_ids_lengths_before_pad={lens}")
        print(f"    -> sample 0 is longest in batch (length-sorted descending)")

    # --- Component 5: Inline prompt_end logic (batch 0, 1) ---
    print("\n=== Component 5: Inline prompt extraction (prompt_end = first where labels != IGNORE) ===")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 2:
            break
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        indices = batch.get("index", torch.arange(input_ids.shape[0]))
        B = input_ids.shape[0]
        for i in range(B):
            if labels is not None:
                label_mask = labels[i] != IGNORE_INDEX
                prompt_end = (
                    label_mask.nonzero()[0][0].item()
                    if label_mask.any()
                    else input_ids.shape[1]
                )
            else:
                prompt_end = input_ids.shape[1]
            slice_tokens = input_ids[i, :prompt_end].cpu().tolist()
            idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
            print(f"  batch={batch_idx} sample={i} dataset_idx={idx} prompt_end={prompt_end} prompt_len={len(slice_tokens)} prompt_first5={slice_tokens[:5]}")
            if i == 0 and prompt_end == 0:
                suspects.append(("inline_extraction", f"batch {batch_idx} sample 0: prompt_end=0 -> empty list"))

    # --- Component 6: _build_prompts_for_sampler (same batches) ---
    print("\n=== Component 6: _build_prompts_for_sampler (fallback when prompt_end==0) ===")
    prompt_only_input_ids = getattr(dataloader.dataset, "predict_with_generate", False)
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 2:
            break
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        indices = batch.get("index", torch.arange(input_ids.shape[0]))
        prompts_build, prompt_lens_build, _ = _build_prompts_for_sampler(
            input_ids, labels, tokenizer, ignore_index=IGNORE_INDEX,
            prompt_only_input_ids=prompt_only_input_ids,
        )
        for i in range(min(2, len(prompts_build))):
            idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
            p = prompts_build[i]
            pl = prompt_lens_build[i]
            print(f"  batch={batch_idx} sample={i} dataset_idx={idx} prompt_len={pl} first5={p[:5]}")
        if prompt_lens_build[0] > 0:
            print(f"  -> sample 0 gets non-empty prompt via fallback (non_pad tokens)")

    return suspects


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import torch
    from torch.utils.data import DataLoader

    from data.utils import load_hf_dataset, IGNORE_INDEX as DATA_IGNORE
    from data.qa import QAwithDualAnswersDataset
    from data.collators import DataCollatorForSupervisedDataset
    from evals.metrics.samplers import LengthSortedSampler
    from evals.metrics.trajectory_metrics import _build_prompts_for_sampler

    assert DATA_IGNORE == IGNORE_INDEX

    print("Loading tokenizer (GSAI-ML/LLaDA-8B-Instruct, no GPU)...")
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    print(f"  eos_token_id={eos_id} pad_token_id={pad_id}")

    template_args = {"apply_chat_template": True}
    hf_args = {"path": "locuslab/TOFU", "name": "forget10_perturbed", "split": "train"}
    print("Building QAwithDualAnswersDataset (forget10_perturbed)...")
    dataset = QAwithDualAnswersDataset(
        correct_answer_key="paraphrased_answer",
        wrong_answer_key="perturbed_answer",
        hf_args=hf_args,
        template_args=template_args,
        tokenizer=tokenizer,
        question_key="question",
        max_length=512,
        predict_with_generate=True,
    )
    print(f"  len(dataset)={len(dataset)}")

    collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, padding_side="left", index="index"
    )
    batch_size = 4
    sampler = LengthSortedSampler(dataset, length_key="input_ids", descending=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
    )

    raw_ds = load_dataset("locuslab/TOFU", name="forget10_perturbed", split="train")
    def get_question(idx):
        return raw_ds[int(idx)].get("question", "")[:80]

    # --- Component-by-component analysis ---
    suspects = analyze_components(
        tokenizer, dataset, collator, dataloader, eos_id, pad_id, get_question,
    )
    # Rebuild dataloader for full iteration (analyze_components consumed 2 batches)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
    )

    print("\n=== Suspects (from component analysis) ===")
    for comp, msg in suspects:
        print(f"  [{comp}] {msg}")
    if not suspects:
        print("  (none from first 2 batches; see aggregate below)")

    # Exactly as in _generate_trajectories_for_dataloader (inline extraction)
    problematic_inline = []  # (batch_idx, sample_idx, dataset_index, prompt_len, first10, reason)
    problematic_build = []   # same but for _build_prompts_for_sampler output
    all_sample0_inline = []  # (batch_idx, dataset_index, prompt_len, first10)

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        indices = batch.get(
            "index",
            torch.arange(
                batch_idx * input_ids.shape[0],
                (batch_idx + 1) * input_ids.shape[0],
            ),
        )
        B = input_ids.shape[0]
        for i in range(B):
            if labels is not None:
                label_mask = labels[i] != IGNORE_INDEX
                prompt_end = (
                    label_mask.nonzero()[0][0].item()
                    if label_mask.any()
                    else input_ids.shape[1]
                )
            else:
                prompt_end = input_ids.shape[1]
            prompt_inline = input_ids[i, :prompt_end].cpu().tolist()
            prompt_len = len(prompt_inline)
            first10 = prompt_inline[:10] if len(prompt_inline) >= 10 else prompt_inline
            num_leading_eos = 0
            for t in prompt_inline:
                if t == eos_id or t == pad_id:
                    num_leading_eos += 1
                else:
                    break
            idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
            if prompt_len == 0:
                problematic_inline.append((batch_idx, i, idx, 0, first10, "empty_prompt"))
            elif num_leading_eos >= 10 or (first10 and all(t == eos_id or t == pad_id for t in first10)):
                problematic_inline.append((batch_idx, i, idx, prompt_len, first10, "first10_all_eos_or_pad"))
            if i == 0:
                all_sample0_inline.append((batch_idx, idx, prompt_len, first10))

        # Compare with _build_prompts_for_sampler (strips leading pad, returns prompt_starts)
        prompt_only_input_ids = getattr(dataloader.dataset, "predict_with_generate", False)
        prompts_build, prompt_lens_build, _ = _build_prompts_for_sampler(
            input_ids, labels, tokenizer, ignore_index=IGNORE_INDEX,
            prompt_only_input_ids=prompt_only_input_ids,
        )
        for i in range(B):
            pl = prompt_lens_build[i]
            p = prompts_build[i]
            first10_b = p[:10] if len(p) >= 10 else p
            idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
            if pl == 0:
                problematic_build.append((batch_idx, i, idx, 0, first10_b, "empty"))
            elif len(p) >= 10 and all(t == eos_id or t == pad_id for t in first10_b):
                problematic_build.append((batch_idx, i, idx, pl, first10_b, "first10_all_eos_or_pad"))

    # Report
    print("\n--- Inline extraction (same as _generate_trajectories_for_dataloader) ---")
    print(f"Total batches: {len(all_sample0_inline)}")
    print(f"Sample-0 prompt lengths: min={min(x[2] for x in all_sample0_inline)} max={max(x[2] for x in all_sample0_inline)}")
    n_empty_s0 = sum(1 for x in all_sample0_inline if x[2] == 0)
    n_first10_eos_s0 = sum(1 for x in all_sample0_inline if x[2] > 0 and len(x[3]) >= 10 and all(t == eos_id or t == pad_id for t in x[3][:10]))
    print(f"Sample 0: empty prompt count={n_empty_s0} first10_all_eos/pad count={n_first10_eos_s0}")

    print(f"\nProblematic (inline): {len(problematic_inline)}")
    for batch_idx, sample_idx, idx, prompt_len, first10, reason in problematic_inline[:30]:
        print(f"  batch={batch_idx} sample={sample_idx} dataset_index={idx} prompt_len={prompt_len} reason={reason} first10={first10[:5]}... q={get_question(idx)!r}")
    if len(problematic_inline) > 30:
        print(f"  ... and {len(problematic_inline) - 30} more")

    print(f"\nProblematic (_build_prompts_for_sampler): {len(problematic_build)}")
    for batch_idx, sample_idx, idx, prompt_len, first10, reason in problematic_build[:20]:
        print(f"  batch={batch_idx} sample={sample_idx} dataset_index={idx} prompt_len={prompt_len} reason={reason} first10={first10[:5]}...")

    # Dataset indices that end up as sample 0 with bad prompt (for follow-up)
    bad_sample0_indices = [x[1] for x in all_sample0_inline if x[2] == 0 or (len(x[3]) >= 10 and all(t == eos_id or t == pad_id for t in x[3][:10]))]
    print(f"\nDataset indices that are sample 0 with empty or first10_all_eos prompt: {bad_sample0_indices[:50]}")
    if len(bad_sample0_indices) > 50:
        print(f"  ... and {len(bad_sample0_indices) - 50} more")

    print("\n--- Root cause ---")
    print("The trajectory path (_generate_trajectories_for_dataloader) builds prompts INLINE:")
    print("  prompt_end = first index where labels != IGNORE_INDEX; prompt = input_ids[i, :prompt_end]")
    print("With left-padded batches, the longest row (sample 0) often has prompt_end=0 -> empty prompt.")
    print("Other rows get prompt = left padding (EOS) only. _build_prompts_for_sampler has a fallback")
    print("when prompt_end==0 (use non_pad tokens) but the trajectory code does NOT use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
