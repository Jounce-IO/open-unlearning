"""
Pytest: validate TOFU forget10 and forget10_perturbed splits (prompts, labels, answers).

Uses production _build_prompts_for_sampler (same as trajectory path). Validates decode
round-trip: decoded prompt must match the exact original prompt text; labels contain
correct/paraphrased answer; labels_wrong contain perturbed_answer(s) as used by truth_ratio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

IGNORE_INDEX = -100


def _collect_prompts_via_builder_forget10(tokenizer, eos_id, pad_id):
    """Build forget10_perturbed dataloader and collect prompts via _build_prompts_for_sampler.

    Returns (all_prompts_by_idx, problematic, dataset).
    all_prompts_by_idx: dict[dataset_index, (prompt_tokens, prompt_len)]
    problematic: list of (batch_idx, sample_idx, dataset_index, reason)
    dataset: the dataset instance (for comparing to original prompt per index).
    """
    from torch.utils.data import DataLoader

    from data.qa import QAwithDualAnswersDataset
    from data.collators import DataCollatorForSupervisedDataset
    from evals.metrics.samplers import LengthSortedSampler
    from evals.metrics.trajectory_metrics import _build_prompts_for_sampler

    hf_args = {"path": "locuslab/TOFU", "name": "forget10_perturbed", "split": "train"}
    template_args = {"apply_chat_template": True}
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

    all_prompts_by_idx = {}
    problematic = []

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
        prompt_only_input_ids = getattr(dataset, "predict_with_generate", False)
        prompts, prompt_lens, _ = _build_prompts_for_sampler(
            input_ids, labels, tokenizer, ignore_index=IGNORE_INDEX,
            prompt_only_input_ids=prompt_only_input_ids,
        )
        B = len(prompts)
        for i in range(B):
            prompt_tokens = prompts[i]
            prompt_len = len(prompt_tokens)
            first10 = prompt_tokens[:10] if len(prompt_tokens) >= 10 else prompt_tokens
            idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
            all_prompts_by_idx[idx] = (prompt_tokens, prompt_len)
            if prompt_len == 0:
                problematic.append((batch_idx, i, idx, "empty_prompt"))
            elif len(first10) >= 10 and all(
                t == eos_id or t == pad_id for t in first10
            ):
                problematic.append(
                    (batch_idx, i, idx, "first10_all_eos_or_pad")
                )

    return all_prompts_by_idx, problematic, dataset


def _collect_prompts_via_builder_forget10_non_perturbed(tokenizer, eos_id, pad_id):
    """Build forget10 (non-perturbed) dataloader and collect prompts via _build_prompts_for_sampler.

    Uses QADataset with question_key=question, answer_key=answer (same as TOFU_QA_forget).
    Returns (all_prompts_by_idx, problematic, dataset).
    """
    from torch.utils.data import DataLoader

    from data.qa import QADataset
    from data.collators import DataCollatorForSupervisedDataset
    from evals.metrics.samplers import LengthSortedSampler
    from evals.metrics.trajectory_metrics import _build_prompts_for_sampler

    hf_args = {"path": "locuslab/TOFU", "name": "forget10", "split": "train"}
    template_args = {"apply_chat_template": True}
    dataset = QADataset(
        hf_args=hf_args,
        template_args=template_args,
        tokenizer=tokenizer,
        question_key="question",
        answer_key="answer",
        max_length=512,
        predict_with_generate=True,
    )
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
    all_prompts_by_idx = {}
    problematic = []
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
        prompt_only_input_ids = getattr(dataset, "predict_with_generate", False)
        prompts, prompt_lens, _ = _build_prompts_for_sampler(
            input_ids, labels, tokenizer, ignore_index=IGNORE_INDEX,
            prompt_only_input_ids=prompt_only_input_ids,
        )
        B = len(prompts)
        for i in range(B):
            prompt_tokens = prompts[i]
            prompt_len = len(prompt_tokens)
            first10 = prompt_tokens[:10] if len(prompt_tokens) >= 10 else prompt_tokens
            idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
            all_prompts_by_idx[idx] = (prompt_tokens, prompt_len)
            if prompt_len == 0:
                problematic.append((batch_idx, i, idx, "empty_prompt"))
            elif len(first10) >= 10 and all(
                t == eos_id or t == pad_id for t in first10
            ):
                problematic.append(
                    (batch_idx, i, idx, "first10_all_eos_or_pad")
                )
    return all_prompts_by_idx, problematic, dataset


def _build_dataset_and_dataloader_forget10_perturbed(tokenizer):
    """Build forget10_perturbed dataset and dataloader (same config as eval). Returns (dataset, dataloader)."""
    from torch.utils.data import DataLoader

    from data.qa import QAwithDualAnswersDataset
    from data.collators import DataCollatorForSupervisedDataset
    from evals.metrics.samplers import LengthSortedSampler

    hf_args = {"path": "locuslab/TOFU", "name": "forget10_perturbed", "split": "train"}
    template_args = {"apply_chat_template": True}
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
    collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, padding_side="left", index="index"
    )
    sampler = LengthSortedSampler(dataset, length_key="input_ids", descending=True)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        collate_fn=collator,
    )
    return dataset, dataloader


def _build_dataset_and_dataloader_forget10_non_perturbed(tokenizer):
    """Build forget10 dataset and dataloader (same config as eval). Returns (dataset, dataloader)."""
    from torch.utils.data import DataLoader

    from data.qa import QADataset
    from data.collators import DataCollatorForSupervisedDataset
    from evals.metrics.samplers import LengthSortedSampler

    hf_args = {"path": "locuslab/TOFU", "name": "forget10", "split": "train"}
    template_args = {"apply_chat_template": True}
    dataset = QADataset(
        hf_args=hf_args,
        template_args=template_args,
        tokenizer=tokenizer,
        question_key="question",
        answer_key="answer",
        max_length=512,
        predict_with_generate=True,
    )
    collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, padding_side="left", index="index"
    )
    sampler = LengthSortedSampler(dataset, length_key="input_ids", descending=True)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        collate_fn=collator,
    )
    return dataset, dataloader


def _content_from_padded_row(row_tensor, pad_value, left_padded=True):
    """Extract content tokens (non-pad) from a padded row; order preserved. Left padding: content is at the end."""
    if hasattr(row_tensor, "tolist"):
        row = row_tensor.tolist()
    else:
        row = list(row_tensor)
    if left_padded:
        while row and row[0] == pad_value:
            row = row[1:]
    else:
        while row and row[-1] == pad_value:
            row = row[:-1]
    return row


class TestForget10PromptExtraction:
    """All prompts for forget10_perturbed must be non-empty and not EOS/pad-heavy."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer

        t = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
        )
        if t.pad_token_id is None:
            t.pad_token_id = t.eos_token_id
        return t

    @pytest.fixture(scope="class")
    def eos_pad_ids(self, tokenizer):
        return tokenizer.eos_token_id, tokenizer.pad_token_id

    def test_all_forget10_prompts_non_empty_and_not_eos_heavy(
        self, tokenizer, eos_pad_ids
    ):
        """All forget10_perturbed prompts (via _build_prompts_for_sampler) must be non-empty and not EOS/pad-heavy."""
        eos_id, pad_id = eos_pad_ids
        all_prompts_by_idx, problematic, _ = _collect_prompts_via_builder_forget10(
            tokenizer, eos_id, pad_id
        )
        expected_count = 400
        assert len(all_prompts_by_idx) == expected_count, (
            f"Expected {expected_count} samples in forget10_perturbed train, "
            f"got {len(all_prompts_by_idx)}"
        )
        assert not problematic, (
            f"Found {len(problematic)} problematic prompt(s). "
            f"Each must be non-empty and not have first 10 tokens all EOS/pad. "
            f"First 20: {problematic[:20]}"
        )

    def test_all_forget10_prompts_decode_to_meaningful_text(
        self, tokenizer, eos_pad_ids
    ):
        """Decode round-trip: every prompt must decode to non-empty, meaningful text."""
        eos_id, pad_id = eos_pad_ids
        all_prompts_by_idx, problematic, _ = _collect_prompts_via_builder_forget10(
            tokenizer, eos_id, pad_id
        )
        assert len(all_prompts_by_idx) == 400
        decode_failures = []
        for idx, (prompt_tokens, _) in all_prompts_by_idx.items():
            decoded = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            text = decoded.strip()
            if not text:
                decode_failures.append((idx, decoded[:80]))
        assert not decode_failures, (
            f"No prompt may decode to empty (skip_special_tokens=True). "
            f"Found {len(decode_failures)}; first 20: {decode_failures[:20]}"
        )

    def test_all_forget10_prompts_decode_to_original_text(
        self, tokenizer, eos_pad_ids
    ):
        """Decoded prompt must match the exact original prompt text from the dataset."""
        eos_id, pad_id = eos_pad_ids
        all_prompts_by_idx, problematic, dataset = _collect_prompts_via_builder_forget10(
            tokenizer, eos_id, pad_id
        )
        assert not problematic, "Prerequisite: no problematic prompts"
        assert len(all_prompts_by_idx) == 400
        mismatches = []
        for idx, (prompt_tokens, _) in all_prompts_by_idx.items():
            item = dataset[idx]
            if isinstance(item, dict):
                original_input_ids = item["input_ids"]
            else:
                original_input_ids = item[0]["input_ids"]
            if hasattr(original_input_ids, "tolist"):
                original_input_ids = original_input_ids.tolist()
            else:
                original_input_ids = list(original_input_ids)
            expected_text = tokenizer.decode(
                original_input_ids, skip_special_tokens=True
            ).strip()
            actual_text = tokenizer.decode(
                prompt_tokens, skip_special_tokens=True
            ).strip()
            if actual_text != expected_text:
                mismatches.append(
                    (idx, expected_text[:100], actual_text[:100])
                )
        assert not mismatches, (
            f"Decoded prompt must equal original dataset prompt. "
            f"Found {len(mismatches)} mismatch(es); first 10: {mismatches[:10]}"
        )

    def test_all_forget10_labels_contain_original_answer(
        self, tokenizer, eos_pad_ids
    ):
        """Dataset labels must decode to text that contains the correct (paraphrased) answer."""
        eos_id, pad_id = eos_pad_ids
        all_prompts_by_idx, problematic, dataset = _collect_prompts_via_builder_forget10(
            tokenizer, eos_id, pad_id
        )
        assert not problematic, "Prerequisite: no problematic prompts"
        assert len(all_prompts_by_idx) == 400
        correct_key = getattr(dataset, "correct_answer_key", "paraphrased_answer")
        missing = []
        for idx in all_prompts_by_idx:
            item = dataset[idx]
            labels = item["labels"]
            if hasattr(labels, "tolist"):
                labels = labels.tolist()
            else:
                labels = list(labels)
            full_decoded = tokenizer.decode(
                labels, skip_special_tokens=True
            )
            raw_row = dataset.data[idx]
            expected_answer = raw_row[correct_key]
            if isinstance(expected_answer, list):
                expected_answer = expected_answer[0] if expected_answer else ""
            expected_answer = str(expected_answer).strip()
            if not expected_answer or expected_answer not in full_decoded:
                missing.append((idx, expected_answer[:60], full_decoded[:80]))
        assert not missing, (
            f"Labels decode must contain dataset {correct_key!r}. "
            f"Found {len(missing)} missing; first 10: {missing[:10]}"
        )

    def test_all_forget10_perturbed_labels_wrong_contain_perturbed_answer(
        self, tokenizer, eos_pad_ids
    ):
        """labels_wrong (used by truth_ratio) must decode to text containing each perturbed_answer."""
        eos_id, pad_id = eos_pad_ids
        all_prompts_by_idx, problematic, dataset = _collect_prompts_via_builder_forget10(
            tokenizer, eos_id, pad_id
        )
        assert not problematic, "Prerequisite: no problematic prompts"
        wrong_key = getattr(dataset, "wrong_answer_key", "perturbed_answer")
        missing = []
        for idx in all_prompts_by_idx:
            item = dataset[idx]
            labels_wrong = item["labels_wrong"]
            raw_row = dataset.data[idx]
            wrong_answers = raw_row[wrong_key]
            if isinstance(wrong_answers, str):
                wrong_answers = [wrong_answers]
            if not isinstance(labels_wrong, list):
                labels_wrong = [labels_wrong]
            for k, lab in enumerate(labels_wrong):
                if k >= len(wrong_answers):
                    break
                expected = str(wrong_answers[k]).strip()
                if hasattr(lab, "tolist"):
                    lab = lab.tolist()
                else:
                    lab = list(lab)
                decoded = tokenizer.decode(lab, skip_special_tokens=True)
                if not expected or expected not in decoded:
                    missing.append((idx, k, expected[:50], decoded[:60]))
        assert not missing, (
            f"Each labels_wrong[k] decode must contain {wrong_key!r}[k]. "
            f"Found {len(missing)} missing; first 10: {missing[:10]}"
        )


class TestForget10NonPerturbedPromptExtraction:
    """Validate forget10 (non-perturbed) split: prompts and labels as used by QADataset."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer

        t = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
        )
        if t.pad_token_id is None:
            t.pad_token_id = t.eos_token_id
        return t

    @pytest.fixture(scope="class")
    def eos_pad_ids(self, tokenizer):
        return tokenizer.eos_token_id, tokenizer.pad_token_id

    def test_all_forget10_non_perturbed_prompts_non_empty_and_not_eos_heavy(
        self, tokenizer, eos_pad_ids
    ):
        """All forget10 prompts must be non-empty and not EOS/pad-heavy."""
        eos_id, pad_id = eos_pad_ids
        all_prompts_by_idx, problematic, dataset = _collect_prompts_via_builder_forget10_non_perturbed(
            tokenizer, eos_id, pad_id
        )
        assert len(all_prompts_by_idx) == len(dataset), (
            f"Expected {len(dataset)} samples in forget10 train, got {len(all_prompts_by_idx)}"
        )
        assert not problematic, (
            f"Found {len(problematic)} problematic prompt(s). First 20: {problematic[:20]}"
        )

    def test_all_forget10_non_perturbed_prompts_decode_to_original_text(
        self, tokenizer, eos_pad_ids
    ):
        """Decoded prompt must match the exact original prompt text from the dataset."""
        eos_id, pad_id = eos_pad_ids
        all_prompts_by_idx, problematic, dataset = _collect_prompts_via_builder_forget10_non_perturbed(
            tokenizer, eos_id, pad_id
        )
        assert not problematic, "Prerequisite: no problematic prompts"
        mismatches = []
        for idx, (prompt_tokens, _) in all_prompts_by_idx.items():
            item = dataset[idx]
            original_input_ids = item["input_ids"]
            if hasattr(original_input_ids, "tolist"):
                original_input_ids = original_input_ids.tolist()
            else:
                original_input_ids = list(original_input_ids)
            expected_text = tokenizer.decode(
                original_input_ids, skip_special_tokens=True
            ).strip()
            actual_text = tokenizer.decode(
                prompt_tokens, skip_special_tokens=True
            ).strip()
            if actual_text != expected_text:
                mismatches.append(
                    (idx, expected_text[:100], actual_text[:100])
                )
        assert not mismatches, (
            f"Decoded prompt must equal original dataset prompt. "
            f"Found {len(mismatches)} mismatch(es); first 10: {mismatches[:10]}"
        )

    def test_all_forget10_non_perturbed_labels_contain_original_answer(
        self, tokenizer, eos_pad_ids
    ):
        """Dataset labels must decode to text that contains the dataset answer."""
        eos_id, pad_id = eos_pad_ids
        all_prompts_by_idx, problematic, dataset = _collect_prompts_via_builder_forget10_non_perturbed(
            tokenizer, eos_id, pad_id
        )
        assert not problematic, "Prerequisite: no problematic prompts"
        answer_key = getattr(dataset, "answer_key", "answer")
        missing = []
        for idx in all_prompts_by_idx:
            item = dataset[idx]
            labels = item["labels"]
            if hasattr(labels, "tolist"):
                labels = labels.tolist()
            else:
                labels = list(labels)
            full_decoded = tokenizer.decode(labels, skip_special_tokens=True)
            raw_row = dataset.data[idx]
            expected_answer = raw_row[answer_key]
            if isinstance(expected_answer, list):
                expected_answer = expected_answer[0] if expected_answer else ""
            expected_answer = str(expected_answer).strip()
            if not expected_answer or expected_answer not in full_decoded:
                missing.append((idx, expected_answer[:60], full_decoded[:80]))
        assert not missing, (
            f"Labels decode must contain dataset {answer_key!r}. "
            f"Found {len(missing)} missing; first 10: {missing[:10]}"
        )


class TestBatchIndexAndPaddingUsedInEval:
    """Validate batch index and padding (as used in trajectory eval): index correct, padding does not corrupt content."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer

        t = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
        )
        if t.pad_token_id is None:
            t.pad_token_id = t.eos_token_id
        return t

    def test_forget10_perturbed_batch_index_and_padding_preserve_content(
        self, tokenizer
    ):
        """batch['index'][i] must identify the dataset row; padded batch row content must match dataset[idx] (no corruption)."""
        dataset, dataloader = _build_dataset_and_dataloader_forget10_perturbed(
            tokenizer
        )
        pad_id = tokenizer.pad_token_id
        index_errors = []
        input_ids_errors = []
        labels_errors = []
        for batch_idx, batch in enumerate(dataloader):
            assert "index" in batch, "Batch must contain 'index' (used in eval to attribute results)."
            indices = batch["index"]
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            B = input_ids.shape[0]
            for i in range(B):
                idx = indices[i].item() if torch.is_tensor(indices[i]) else int(indices[i])
                if idx < 0 or idx >= len(dataset):
                    index_errors.append((batch_idx, i, idx))
                    continue
                item = dataset[idx]
                orig_input = item["input_ids"]
                orig_labels = item["labels"]
                if hasattr(orig_input, "tolist"):
                    orig_input = orig_input.tolist()
                else:
                    orig_input = list(orig_input)
                if hasattr(orig_labels, "tolist"):
                    orig_labels = orig_labels.tolist()
                else:
                    orig_labels = list(orig_labels)
                batched_input_content = _content_from_padded_row(
                    input_ids[i], pad_id, left_padded=True
                )
                batched_labels_content = _content_from_padded_row(
                    labels[i], IGNORE_INDEX, left_padded=True
                )
                if batched_input_content != orig_input:
                    input_ids_errors.append((batch_idx, i, idx))
                if batched_labels_content != orig_labels:
                    labels_errors.append((batch_idx, i, idx))
        assert not index_errors, (
            f"batch['index'][i] must be valid dataset index. Errors: {index_errors[:15]}"
        )
        assert not input_ids_errors, (
            f"After stripping padding, batch['input_ids'][i] must equal dataset[batch['index'][i]]['input_ids']. "
            f"Errors: {input_ids_errors[:15]}"
        )
        assert not labels_errors, (
            f"After stripping IGNORE_INDEX, batch['labels'][i] must equal dataset[batch['index'][i]]['labels']. "
            f"Errors: {labels_errors[:15]}"
        )

    def test_forget10_perturbed_dataloader_covers_every_index_once(
        self, tokenizer
    ):
        """LengthSortedSampler + dataloader must yield every dataset index exactly once (no drop/duplicate)."""
        dataset, dataloader = _build_dataset_and_dataloader_forget10_perturbed(
            tokenizer
        )
        collected = []
        for batch in dataloader:
            indices = batch["index"]
            for i in range(indices.shape[0]):
                idx = indices[i].item() if torch.is_tensor(indices[i]) else int(indices[i])
                collected.append(idx)
        expected = set(range(len(dataset)))
        assert len(collected) == len(dataset), (
            f"Dataloader must yield len(dataset)={len(dataset)} indices, got {len(collected)}"
        )
        assert set(collected) == expected, (
            f"Indices must be exactly 0..{len(dataset)-1}; missing or duplicate in collected."
        )

    def test_forget10_perturbed_batch_has_labels_correct_and_labels_wrong(
        self, tokenizer
    ):
        """Batches from dual-answer dataset must contain labels_correct and labels_wrong (used by truth_ratio)."""
        _, dataloader = _build_dataset_and_dataloader_forget10_perturbed(
            tokenizer
        )
        for batch in dataloader:
            assert "labels_correct" in batch, (
                "Batch must contain 'labels_correct' for truth_ratio pre_compute."
            )
            assert "labels_wrong" in batch, (
                "Batch must contain 'labels_wrong' for truth_ratio pre_compute."
            )
            break

    def test_forget10_non_perturbed_dataloader_covers_every_index_once(
        self, tokenizer
    ):
        """LengthSortedSampler + dataloader must yield every dataset index exactly once for forget10."""
        dataset, dataloader = _build_dataset_and_dataloader_forget10_non_perturbed(
            tokenizer
        )
        collected = []
        for batch in dataloader:
            indices = batch["index"]
            for i in range(indices.shape[0]):
                idx = indices[i].item() if torch.is_tensor(indices[i]) else int(indices[i])
                collected.append(idx)
        assert len(collected) == len(dataset), (
            f"Dataloader must yield len(dataset)={len(dataset)} indices, got {len(collected)}"
        )
        assert set(collected) == set(range(len(dataset))), (
            "Indices must be exactly 0..len(dataset)-1; missing or duplicate in collected."
        )

    def test_forget10_non_perturbed_batch_index_and_padding_preserve_content(
        self, tokenizer
    ):
        """batch['index'][i] must identify the dataset row; padded batch row content must match dataset[idx] (no corruption)."""
        dataset, dataloader = _build_dataset_and_dataloader_forget10_non_perturbed(
            tokenizer
        )
        pad_id = tokenizer.pad_token_id
        index_errors = []
        input_ids_errors = []
        labels_errors = []
        for batch_idx, batch in enumerate(dataloader):
            assert "index" in batch, "Batch must contain 'index' (used in eval to attribute results)."
            indices = batch["index"]
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            B = input_ids.shape[0]
            for i in range(B):
                idx = indices[i].item() if torch.is_tensor(indices[i]) else int(indices[i])
                if idx < 0 or idx >= len(dataset):
                    index_errors.append((batch_idx, i, idx))
                    continue
                item = dataset[idx]
                orig_input = item["input_ids"]
                orig_labels = item["labels"]
                if hasattr(orig_input, "tolist"):
                    orig_input = orig_input.tolist()
                else:
                    orig_input = list(orig_input)
                if hasattr(orig_labels, "tolist"):
                    orig_labels = orig_labels.tolist()
                else:
                    orig_labels = list(orig_labels)
                batched_input_content = _content_from_padded_row(
                    input_ids[i], pad_id, left_padded=True
                )
                batched_labels_content = _content_from_padded_row(
                    labels[i], IGNORE_INDEX, left_padded=True
                )
                if batched_input_content != orig_input:
                    input_ids_errors.append((batch_idx, i, idx))
                if batched_labels_content != orig_labels:
                    labels_errors.append((batch_idx, i, idx))
        assert not index_errors, (
            f"batch['index'][i] must be valid dataset index. Errors: {index_errors[:15]}"
        )
        assert not input_ids_errors, (
            f"After stripping padding, batch['input_ids'][i] must equal dataset[batch['index'][i]]['input_ids']. "
            f"Errors: {input_ids_errors[:15]}"
        )
        assert not labels_errors, (
            f"After stripping IGNORE_INDEX, batch['labels'][i] must equal dataset[batch['index'][i]]['labels']. "
            f"Errors: {labels_errors[:15]}"
        )
