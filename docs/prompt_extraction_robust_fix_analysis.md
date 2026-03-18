# Robust fix for prompt extraction (no magic-number heuristics)

## 1. Root cause (recap)

- **Data convention (predict_with_generate):**  
  `preprocess_chat_instance(..., predict_with_generate=True)` returns:
  - `input_ids` = **prompt only** (variable length per sample, e.g. 26–35).
  - `labels` = **full conversation** (prompt + response, longer, e.g. 75–80).

- **Collator:**  
  `DataCollatorForSupervisedDataset` pads **independently**:
  - `input_ids` → `pad_sequence(..., padding_value=pad_token_id)` → length = **max(len(input_ids) over batch)**.
  - `labels` → `pad_sequence(..., padding_value=IGNORE_INDEX)` → length = **max(len(labels) over batch)**.

- So we get **different tensor lengths**: e.g. `input_ids.shape = (B, 35)`, `labels.shape = (B, 75)`.

- **Builder today:**  
  `prompt_end = first index where labels[i] != IGNORE_INDEX` (number of leading IGNOREs in that row).  
  Then `prompt = input_ids[i, :prompt_end]` (after strip leading pad).

- **Why this breaks:**
  - For a row with **same** `input_ids` length as the batch max (e.g. 35) but **shorter** `labels` (e.g. 74), labels are padded to 75 so that row gets **one leading IGNORE**. So `prompt_end = 1`.
  - We then take `input_ids[i, :1]` = first token of that row = BOS (no left padding on that row). So we get **one token** instead of the full 35-token prompt.
  - For rows that are left-padded in `input_ids`, `input_ids[i, :prompt_end]` is only padding → after strip we get `[]`; the current “empty → use non_pad” fallback fixes that.
  - So the bug is **not** “empty slice” only; it is “**wrong slice**”: using `prompt_end` from **labels** to slice **input_ids** when the two are padded to **different** lengths, so the slice can be too short (1–2 tokens) and not aligned with the true prompt.

- **Current workaround:**  
  Heuristics in `_build_prompts_for_sampler`: `_STUB_PROMPT_MAX_LEN = 2`, `_MIN_CONTENT_TOKENS = 10`, and “use full non_pad when left-padded or when stub + many content tokens”. These are brittle and not principled.

---

## 2. Principled rule

When the data is in **prompt-only input_ids** mode (i.e. `predict_with_generate`):

- **Semantic:** `input_ids` for each sample contain **only** the prompt (no response).
- So the **correct prompt** for the sampler is exactly the **non-pad tokens** of `input_ids[i]` (order preserved). No slice from `labels` is needed to define the prompt.
- **prompt_starts** (for downstream: `_build_target_sequences_for_sampler`, labels slicing, etc.) must still come from **labels**: `prompt_starts[i] = first index where labels[i] != IGNORE_INDEX`. That is the start of the generation region in the **labels** tensor and is independent of how we build the prompt list.

So the robust rule is:

- **When `prompt_only_input_ids` is True:**  
  For every row `i`, set  
  `prompt = input_ids[i][input_ids[i] != pad_token_id].cpu().tolist()`  
  (and optionally strip leading pad; for “non_pad” there are no leading pad tokens, so it’s redundant).  
  Do **not** use `prompt_end` from labels to slice `input_ids` for defining the prompt.  
  Keep computing `prompt_starts` from labels as today.

- **When `prompt_only_input_ids` is False (training-style):**  
  Keep current logic: `prompt_end` from labels, slice `input_ids[i, :prompt_end]`, strip leading pad.  
  Keep the **single** fallback: **if after strip the prompt is empty and the row has any non-pad tokens, use non_pad for that row.**  
  That handles left-padded training batches without any “stub” or “2 vs 10” heuristics.

This removes all magic numbers and uses the data convention explicitly.

---

## 3. Where to get `prompt_only_input_ids`

- The convention is set at **dataset** construction: e.g. `QAwithDualAnswersDataset(..., predict_with_generate=True)` and `QADataset` / `MMLUUtilityDataset` set `self.predict_with_generate`.
- **trajectory_metrics** has access to the **dataset** when it builds the dataloader: `primary_data`, `data["retain"]`, `secondary_data` (holdout), and `dataloader.dataset` / `holdout_dataloader.dataset`.
- So we can define:  
  `prompt_only_input_ids = getattr(dataset, "predict_with_generate", False)`  
  and pass it down wherever we call `_build_prompts_for_sampler` or `_generate_trajectories_for_dataloader`.

---

## 4. Proposed API and call-site changes

### 4.1 `_build_prompts_for_sampler`

- **Add optional parameter:**  
  `prompt_only_input_ids: bool = False`.
- **When `prompt_only_input_ids` is True:**
  - For each row `i`:  
    `prompt = input_ids[i][input_ids[i] != pad_token_id].cpu().tolist()`  
    (with `pad_token_id` from tokenizer as today; if `pad_token_id` is None, treat all tokens as content: `prompt = input_ids[i].cpu().tolist()`).
  - Still compute `prompt_starts` from labels (unchanged).
  - `prompt_lens[i] = len(prompt)`.
  - **Remove** the heuristics (`_STUB_PROMPT_MAX_LEN`, `_MIN_CONTENT_TOKENS`, `first_is_pad`, `stub_prompt_many_content`) for this path.
- **When `prompt_only_input_ids` is False:**
  - Keep current logic (prompt_end from labels, slice, strip leading pad).
  - Keep **only** the “empty after strip → use non_pad” fallback (no 2/10 rules).

### 4.2 Call sites of `_build_prompts_for_sampler`

| Location | Dataset available? | How to pass `prompt_only_input_ids` |
|----------|--------------------|--------------------------------------|
| **Retain MU loop** (~662) | `data_retain` | `getattr(data_retain, "predict_with_generate", False)` |
| **Main trajectory batch loop** (~2109) | `dataloader.dataset` (primary_data) | `getattr(dataloader.dataset, "predict_with_generate", False)`; pass into builder. |
| **Holdout loop** (~2692) | `holdout_dataloader.dataset` | `getattr(holdout_dataloader.dataset, "predict_with_generate", False)` |
| **_generate_trajectories_for_dataloader** (451) | Only `dataloader` | Either (a) add arg `prompt_only_input_ids` to the function and pass it from callers, or (b) use `getattr(dataloader.dataset, "predict_with_generate", False)` inside the function. Option (b) avoids changing the function signature and is consistent (callers already pass `dataloader`). |

Recommendation: **Option (b)** for `_generate_trajectories_for_dataloader`: inside the function, set  
`prompt_only_input_ids = getattr(dataloader.dataset, "predict_with_generate", False)`  
and pass it to `_build_prompts_for_sampler`. No new parameter on the function.

---

## 5. Downstream semantics (unchanged)

- **prompt_starts** is still “start index of generation in **labels**”. All current uses remain correct:
  - `_build_target_sequences_for_sampler(labels, prompt_starts, L, ...)`.
  - `_per_position_scores_from_R_F_batch(..., prompt_starts, ...)`.
  - Any `labels[i, prompt_starts[i]:]` or similar.
- **prompt_lens** is still “length of the prompt **sent to the sampler**” (len of `prompts[i]`). So:
  - `trajectories_from_logits(..., prompt_lens, ...)`.
  - `effective_lengths_from_eos(sequences, prompt_lens, L, ...)`.
- **transform_prompts** (guardrails) still receives `prompts` and `prompt_lens`; it does not need `prompt_starts`. No change there.

---

## 6. Collator / padding (optional, not required for this fix)

- We could change the collator so that when `predict_with_generate` is set it pads `input_ids` and `labels` to the **same** length (e.g. `max(max(len(input_ids)), max(len(labels)))`). That would align lengths but:
  - For the “longest” row (by input_ids), we’d still have many leading pads in `input_ids` if we pad to labels length, so `prompt_end` from labels would be 0 and we’d still need “use non_pad” for that row.
  - So the **semantic** fix (“when prompt-only, prompt = non_pad(input_ids)”) is still needed and is the single source of truth. Aligning lengths in the collator does not remove the need for that rule and adds complexity (collator would need to know predict_with_generate or a max length). So **do not** change the collator for this fix.

---

## 7. Impacted code (summary)

| File | Change |
|------|--------|
| **open-unlearning/src/evals/metrics/trajectory_metrics.py** | (1) `_build_prompts_for_sampler`: add `prompt_only_input_ids: bool = False`; when True use non_pad path only and drop stub heuristics; when False keep current logic + only empty→non_pad fallback. (2) Retain MU: pass `getattr(data_retain, "predict_with_generate", False)` into builder. (3) Main trajectory batch loop: pass `getattr(dataloader.dataset, "predict_with_generate", False)` into builder. (4) Holdout loop: pass `getattr(holdout_dataloader.dataset, "predict_with_generate", False)` into builder. (5) `_generate_trajectories_for_dataloader`: before calling builder, set `prompt_only_input_ids = getattr(dataloader.dataset, "predict_with_generate", False)` and pass it to `_build_prompts_for_sampler`. |
| **open-unlearning/tests/test_trajectory_metrics.py** | Tests that call `_build_prompts_for_sampler`: add `prompt_only_input_ids=False` for training-style and mixed tests; add `prompt_only_input_ids=True` for predict_with_generate-style tests where we expect non_pad behavior. Assert unchanged outputs. |
| **open-unlearning/tests/test_forget10_prompt_extraction.py** | No change to test logic; it uses the same dataloader/dataset (predict_with_generate=True), so builder will get the flag via dataset and behavior should remain correct (all prompts valid, decode to non-empty). |
| **open-unlearning/scripts/validate_prompt_extraction_local.py** | Optional: pass `prompt_only_input_ids=True` when calling the builder for forget10 (or rely on dataset attribute if we ever call through a dataloader that exposes it). |
| **open-unlearning/scripts/analyze_decode_empty_prompts.py** | No change required; uses same pipeline. |

No changes to **collators**, **data/utils** (except docstring if desired), or **configs**. No new config keys; the flag is derived from the dataset already attached to the dataloader.

---

## 8. Edge cases

- **Dataset without `predict_with_generate`:** `getattr(..., "predict_with_generate", False)` → `False` → current (training-style) behavior. Safe.
- **Tokenizer without `pad_token_id`:** When `prompt_only_input_ids` is True, if `pad_token_id` is None we can define “non_pad” as the full row (no stripping), i.e. `prompt = input_ids[i].cpu().tolist()`. Same as today’s fallback when there is no pad.
- **Labels None:** Builder already handles `labels is None` (prompt_end = full length). With `prompt_only_input_ids` True we ignore that and use non_pad; if labels is None we don’t need prompt_starts for target_sequences (guided path won’t run). So no conflict.

---

## 9. Summary

- **Root cause:** Labels and input_ids are padded to different lengths; using `prompt_end` from labels to slice input_ids can yield a wrong (too short) prompt.
- **Robust fix:** When data is in prompt-only mode (`predict_with_generate`), define the prompt as **non-pad tokens of input_ids**; keep **prompt_starts** from labels for downstream. Remove the 2/10 heuristics; keep only the empty→non_pad fallback for the non–prompt-only path.
- **Implementation:** Add `prompt_only_input_ids` to `_build_prompts_for_sampler`, set from `getattr(dataset, "predict_with_generate", False)` at all call sites (including inside `_generate_trajectories_for_dataloader` via `dataloader.dataset`). No collator or config changes required.
