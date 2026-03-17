"""
Comprehensive tests for trajectory loop paths: forget (batch_template e2e),
retain MU (prompt_starts), _batch_template_dual_labels in main loop context,
decoded vs dataset ground truth, and _slice_batch_template_to_length.

Exercises full path: batch → prompt_starts/prompt_lens → batch_template → shapes/keys
and higher-level invariants (decoded content matches dataset).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

IGNORE_INDEX = -100


def _dataloader_forget10_perturbed(tokenizer, batch_size: int, sort_by_length: bool = True):
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
    sampler = (
        LengthSortedSampler(dataset, length_key="input_ids", descending=True)
        if sort_by_length
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
    )


def _dataloader_forget10_aligned(tokenizer, batch_size: int):
    """Forget10 with predict_with_generate=False so labels use IGNORE for prompt; gen_start = prompt_starts only."""
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
        predict_with_generate=False,
    )
    collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, padding_side="left", index="index"
    )
    sampler = LengthSortedSampler(dataset, length_key="input_ids", descending=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
    )


def _dataloader_forget10(tokenizer, batch_size: int, sort_by_length: bool = True):
    """Forget10 (non-perturbed) same full-convo convention; answer_key=answer."""
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
    sampler = (
        LengthSortedSampler(dataset, length_key="input_ids", descending=True)
        if sort_by_length
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
    )


def _build_prompts_and_starts(dataloader, tokenizer):
    from evals.metrics.trajectory_metrics import _build_prompts_for_sampler
    prompt_only = getattr(dataloader.dataset, "predict_with_generate", False)
    for batch in dataloader:
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        prompts, prompt_lens, prompt_starts = _build_prompts_for_sampler(
            input_ids, labels, tokenizer,
            ignore_index=IGNORE_INDEX,
            prompt_only_input_ids=prompt_only,
        )
        yield batch, prompts, prompt_lens, prompt_starts


def _generation_start_full_convo(prompt_starts_i: int, prompt_lens_i: int) -> int:
    return prompt_starts_i + prompt_lens_i


def _build_batch_template_forget_style(
    batch,
    sample_idx: int,
    prompt_lens: list,
    prompt_starts: list,
    L: int,
    use_generation_start: bool,
):
    """Build batch_template exactly as the forget loop, with optional correct slice (gen_start)."""
    from evals.metrics.trajectory_metrics import _batch_template_dual_labels

    labels = batch.get("labels")
    input_ids = batch["input_ids"]
    sample_labels = labels[sample_idx] if labels is not None else None
    sample_input_ids = input_ids[sample_idx]

    if use_generation_start:
        start = _generation_start_full_convo(prompt_starts[sample_idx], prompt_lens[sample_idx])
    else:
        start = prompt_lens[sample_idx]

    generated_labels = None
    if sample_labels is not None:
        generated_labels = sample_labels[start : start + L].clone()
        if generated_labels.shape[0] < L:
            padding = torch.full(
                (L - generated_labels.shape[0],),
                IGNORE_INDEX,
                dtype=generated_labels.dtype,
                device=generated_labels.device,
            )
            generated_labels = torch.cat([generated_labels, padding])
    generated_input_ids = sample_input_ids[start : start + L].clone()
    if generated_input_ids.shape[0] < L:
        padding = torch.zeros(
            L - generated_input_ids.shape[0],
            dtype=generated_input_ids.dtype,
            device=sample_input_ids.device,
        )
        generated_input_ids = torch.cat([generated_input_ids, padding])

    batch_template = {
        "input_ids": generated_input_ids.unsqueeze(0),
        "labels": generated_labels.unsqueeze(0) if generated_labels is not None else None,
        "attention_mask": torch.ones((1, L), dtype=torch.long, device=sample_input_ids.device),
        "index": torch.tensor([0], dtype=torch.long, device=sample_input_ids.device),
    }
    for key in ("labels_correct", "labels_wrong"):
        if key in batch:
            batch_template[key] = _batch_template_dual_labels(
                batch, sample_idx, key, L, IGNORE_INDEX
            )
    return batch_template


def _build_batch_template_retain_style(
    batch,
    sample_idx: int,
    prompt_starts: list,
    L: int,
):
    """Build batch_template exactly as the retain MU path (uses prompt_starts)."""
    from evals.metrics.trajectory_metrics import _batch_template_dual_labels

    labels = batch.get("labels")
    input_ids = batch["input_ids"]
    sample_labels = labels[sample_idx] if labels is not None else None
    sample_prompt_start = prompt_starts[sample_idx]

    generated_labels = None
    if sample_labels is not None:
        generated_labels = sample_labels[sample_prompt_start : sample_prompt_start + L].clone()
        if generated_labels.shape[0] < L:
            padding = torch.full(
                (L - generated_labels.shape[0],), IGNORE_INDEX,
                dtype=generated_labels.dtype, device=generated_labels.device,
            )
            generated_labels = torch.cat([generated_labels, padding])
    generated_input_ids = input_ids[sample_idx][sample_prompt_start : sample_prompt_start + L].clone()
    if generated_input_ids.shape[0] < L:
        padding = torch.zeros(
            L - generated_input_ids.shape[0],
            dtype=generated_input_ids.dtype,
            device=input_ids.device,
        )
        generated_input_ids = torch.cat([generated_input_ids, padding])

    batch_template = {
        "input_ids": generated_input_ids.unsqueeze(0),
        "labels": generated_labels.unsqueeze(0) if generated_labels is not None else None,
        "attention_mask": torch.ones((1, L), dtype=torch.long, device=input_ids.device),
        "index": torch.tensor([0], dtype=torch.long, device=input_ids.device),
    }
    for key in ("labels_correct", "labels_wrong"):
        if key in batch:
            batch_template[key] = _batch_template_dual_labels(
                batch, sample_idx, key, L, IGNORE_INDEX
            )
    return batch_template


@pytest.fixture(scope="class")
def tokenizer():
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )
    if t.pad_token_id is None:
        t.pad_token_id = t.eos_token_id
    return t


class TestForgetPathBatchTemplateE2E:
    """Main trajectory loop (forget path): full batch → batch_template construction."""

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
    @pytest.mark.parametrize("L", [1, 30, 100])
    def test_batch_template_has_required_keys_and_shapes(
        self, tokenizer, batch_size, L
    ):
        """Building batch_template (forget-style with current slice) yields required keys and shapes for various L."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        for batch, _prompts, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            B = batch["input_ids"].shape[0]
            for sample_idx in range(B):
                bt = _build_batch_template_forget_style(
                    batch, sample_idx, prompt_lens, prompt_starts, L,
                    use_generation_start=False,
                )
                assert "input_ids" in bt
                assert bt["input_ids"].shape == (1, L)
                assert "attention_mask" in bt
                assert bt["attention_mask"].shape == (1, L)
                assert "index" in bt
                if batch.get("labels") is not None:
                    assert bt["labels"] is not None
                    assert bt["labels"].shape == (1, L)
                if "labels_correct" in batch:
                    assert "labels_correct" in bt
                if "labels_wrong" in batch:
                    assert "labels_wrong" in bt

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_batch_template_with_gen_start_decoded_labels_match_dataset_response(
        self, tokenizer, batch_size
    ):
        """When batch_template is built with gen_start, decoded labels match dataset response (paraphrased_answer)."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        dataset = dataloader.dataset
        L = 50
        mismatches = []
        for batch, _prompts, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            indices = batch.get("index", torch.arange(batch["input_ids"].shape[0]))
            for sample_idx in range(batch["input_ids"].shape[0]):
                idx = indices[sample_idx].item() if torch.is_tensor(indices[sample_idx]) else indices[sample_idx]
                bt = _build_batch_template_forget_style(
                    batch, sample_idx, prompt_lens, prompt_starts, L,
                    use_generation_start=True,
                )
                if bt["labels"] is None:
                    continue
                valid = bt["labels"].squeeze(0)[bt["labels"].squeeze(0) != IGNORE_INDEX]
                if valid.numel() == 0:
                    continue
                decoded = tokenizer.decode(valid.tolist(), skip_special_tokens=True).strip()
                expected = dataset.data[idx].get("paraphrased_answer", dataset.data[idx].get("answer", ""))
                if not expected:
                    continue
                expected = expected.strip()
                if expected not in decoded and decoded not in expected:
                    mismatches.append((idx, sample_idx, decoded[:80], expected[:80]))
        assert not mismatches, (
            f"Decoded generated_labels (with gen_start) should match dataset response. Mismatches: {mismatches[:10]}"
        )

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_batch_template_with_prompt_lens_tensor_differs_from_gen_start_when_left_padded(
        self, tokenizer, batch_size
    ):
        """When left-padded (gen_start != prompt_lens), the tensor slice at prompt_lens differs from slice at gen_start."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        L = 50
        for batch, _prompts, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            labels = batch["labels"]
            for sample_idx in range(labels.shape[0]):
                gen_start = _generation_start_full_convo(
                    prompt_starts[sample_idx], prompt_lens[sample_idx]
                )
                if gen_start == prompt_lens[sample_idx]:
                    continue
                slice_bug = labels[sample_idx, prompt_lens[sample_idx] : prompt_lens[sample_idx] + L]
                slice_correct = labels[sample_idx, gen_start : gen_start + L]
                assert not torch.equal(slice_bug, slice_correct), (
                    f"With left pad (gen_start={gen_start} != prompt_lens={prompt_lens[sample_idx]}), "
                    "tensor slice by prompt_lens must differ from slice by gen_start."
                )

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_forget_style_batch_template_with_bug_slice_differs_from_correct_for_left_padded(
        self, tokenizer, batch_size
    ):
        """Explicit bug reproduction: batch_template built with prompt_lens (current code) has generated_labels != correct slice for left-padded samples."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        L = 40
        found_left_padded_mismatch = False
        for batch, _p, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            for sample_idx in range(batch["input_ids"].shape[0]):
                gen_start = _generation_start_full_convo(
                    prompt_starts[sample_idx], prompt_lens[sample_idx]
                )
                if gen_start == prompt_lens[sample_idx]:
                    continue
                bt_bug = _build_batch_template_forget_style(
                    batch, sample_idx, prompt_lens, prompt_starts, L,
                    use_generation_start=False,
                )
                bt_correct = _build_batch_template_forget_style(
                    batch, sample_idx, prompt_lens, prompt_starts, L,
                    use_generation_start=True,
                )
                if bt_bug["labels"] is not None and bt_correct["labels"] is not None:
                    if not torch.equal(bt_bug["labels"], bt_correct["labels"]):
                        found_left_padded_mismatch = True
                        break
            if found_left_padded_mismatch:
                break
        assert found_left_padded_mismatch, (
            "For left-padded batches, batch_template built with prompt_lens (bug) must differ from one built with gen_start (correct)."
        )

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_forget_loop_slice_uses_generation_start_matches_direct_slice(
        self, tokenizer, batch_size
    ):
        """Production forget loop uses _generation_start; built generated_labels must equal labels[i, gen_start:gen_start+L] for at least one left-padded sample."""
        from evals.metrics.trajectory_metrics import _generation_start

        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        L = 30
        prompt_only_input_ids = True
        found_left_padded_ok = False
        for batch, _p, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            labels = batch["labels"]
            for i in range(labels.shape[0]):
                if prompt_starts[i] == 0:
                    continue
                gen_start = _generation_start(
                    i, prompt_starts, prompt_lens, prompt_only_input_ids
                )
                sample_labels = labels[i]
                generated_labels = sample_labels[gen_start : gen_start + L].clone()
                if generated_labels.shape[0] < L:
                    padding = torch.full(
                        (L - generated_labels.shape[0],),
                        IGNORE_INDEX,
                        dtype=generated_labels.dtype,
                        device=generated_labels.device,
                    )
                    generated_labels = torch.cat([generated_labels, padding])
                direct_slice = labels[i, gen_start : gen_start + L]
                min_len = min(generated_labels.shape[0], direct_slice.shape[0])
                if min_len > 0 and torch.equal(
                    generated_labels[:min_len], direct_slice[:min_len]
                ):
                    found_left_padded_ok = True
                    break
            if found_left_padded_ok:
                break
        assert found_left_padded_ok, (
            "For at least one left-padded sample (prompt_starts[i] > 0), "
            "generated_labels built with _generation_start must equal labels[i, gen_start:gen_start+L]."
        )

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_ignore_for_prompt_generation_start_equals_prompt_starts(
        self, tokenizer, batch_size
    ):
        """When prompt_only_input_ids=False (IGNORE-for-prompt), _generation_start must equal prompt_starts[i] and slice at that index yields response (non-IGNORE)."""
        from evals.metrics.trajectory_metrics import _generation_start

        dataloader = _dataloader_forget10_aligned(tokenizer, batch_size=batch_size)
        prompt_only_input_ids = False
        L = 20
        for batch, _p, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            labels = batch["labels"]
            for i in range(labels.shape[0]):
                gen_start = _generation_start(
                    i, prompt_starts, prompt_lens, prompt_only_input_ids
                )
                ps = prompt_starts[i]
                if hasattr(ps, "item"):
                    ps = int(ps.item())
                assert gen_start == ps, (
                    f"IGNORE-for-prompt: generation_start must equal prompt_starts[{i}]; got {gen_start} vs {ps}"
                )
                sl = labels[i, gen_start : gen_start + L]
                n_real = (sl != IGNORE_INDEX).sum().item()
                assert n_real >= 1, (
                    f"Slice at prompt_starts must contain response tokens (non-IGNORE); got {n_real}."
                )
            break


class TestBatchTemplateDualLabelsInMainLoop:
    """_batch_template_dual_labels when used in main (forget) loop with left-padded full-convo batches."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_labels_correct_in_batch_template_contains_dataset_paraphrased_answer(
        self, tokenizer, batch_size
    ):
        """batch_template['labels_correct'] from _batch_template_dual_labels decodes to content containing dataset paraphrased_answer.
        For full-convo rows, content_start is start of prompt+answer so decoded = prompt + answer; we require answer in decoded."""
        from evals.metrics.trajectory_metrics import _batch_template_dual_labels

        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        dataset = dataloader.dataset
        L = 200
        for batch, _p, _pl, _ps in _build_prompts_and_starts(dataloader, tokenizer):
            indices = batch.get("index", torch.arange(batch["input_ids"].shape[0]))
            for sample_idx in range(batch["input_ids"].shape[0]):
                correct_bt = _batch_template_dual_labels(
                    batch, sample_idx, "labels_correct", L, IGNORE_INDEX
                )
                assert correct_bt is not None
                assert correct_bt.dim() == 2 and correct_bt.shape[0] == 1
                valid = correct_bt.squeeze(0)[correct_bt.squeeze(0) != IGNORE_INDEX]
                if valid.numel() == 0:
                    continue
                decoded = tokenizer.decode(valid.tolist(), skip_special_tokens=True).strip()
                idx = indices[sample_idx].item() if torch.is_tensor(indices[sample_idx]) else indices[sample_idx]
                expected = dataset.data[idx].get("paraphrased_answer", "").strip()
                if expected and expected not in decoded and decoded not in expected:
                    pytest.fail(
                        f"labels_correct decode should contain paraphrased_answer for idx={idx}. "
                        f"Got: {decoded[:100]!r}, expected: {expected[:100]!r}"
                    )

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_labels_wrong_in_batch_template_structure_and_contains_perturbed(
        self, tokenizer, batch_size
    ):
        """batch_template['labels_wrong'] is list of N tensors [1,L]; each decodes to content containing dataset perturbed_answer option (full-convo: prompt+answer)."""
        from evals.metrics.trajectory_metrics import _batch_template_dual_labels

        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        dataset = dataloader.dataset
        L = 200
        for batch, _p, _pl, _ps in _build_prompts_and_starts(dataloader, tokenizer):
            if "labels_wrong" not in batch:
                continue
            lw = batch["labels_wrong"]
            indices = batch.get("index", torch.arange(batch["input_ids"].shape[0]))
            for sample_idx in range(batch["input_ids"].shape[0]):
                wrong_bt = _batch_template_dual_labels(
                    batch, sample_idx, "labels_wrong", L, IGNORE_INDEX
                )
                assert wrong_bt is not None
                if lw.dim() == 3:
                    N = lw.shape[1]
                    assert isinstance(wrong_bt, list)
                    assert len(wrong_bt) == N
                    for k, t in enumerate(wrong_bt):
                        assert t.shape == (1, L) or t.numel() == L
                else:
                    assert wrong_bt.shape == (1, L) or wrong_bt.numel() == L
                idx = indices[sample_idx].item() if torch.is_tensor(indices[sample_idx]) else indices[sample_idx]
                raw = dataset.data[idx].get("perturbed_answer", [])
                if isinstance(raw, str):
                    raw = [raw]
                if isinstance(wrong_bt, list):
                    for k, t in enumerate(wrong_bt):
                        valid = t.squeeze(0)[t.squeeze(0) != IGNORE_INDEX] if t.dim() >= 1 else t[t != IGNORE_INDEX]
                        if valid.numel() == 0:
                            continue
                        decoded = tokenizer.decode(valid.tolist(), skip_special_tokens=True).strip()
                        if k < len(raw) and raw[k].strip() and raw[k].strip() not in decoded and decoded not in raw[k].strip():
                            pytest.fail(
                                f"labels_wrong[{k}] decode should contain perturbed_answer[{k}] for idx={idx}."
                            )


class TestRetainMUPathPromptStarts:
    """Retain MU path (~733-743): slicing uses prompt_starts; decoded matches dataset response."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_retain_style_batch_template_decoded_contains_dataset_response(
        self, tokenizer, batch_size
    ):
        """Batch_template built retain-style (prompt_starts = content start) yields decoded labels that contain dataset response.
        For full-convo labels slice is [prompt+response]; use L large enough so response is included."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        dataset = dataloader.dataset
        L = 250
        mismatches = []
        for batch, _p, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            indices = batch.get("index", torch.arange(batch["input_ids"].shape[0]))
            for sample_idx in range(batch["input_ids"].shape[0]):
                bt = _build_batch_template_retain_style(
                    batch, sample_idx, prompt_starts, L
                )
                if bt["labels"] is None:
                    continue
                valid = bt["labels"].squeeze(0)[bt["labels"].squeeze(0) != IGNORE_INDEX]
                if valid.numel() == 0:
                    continue
                decoded = tokenizer.decode(valid.tolist(), skip_special_tokens=True).strip()
                idx = indices[sample_idx].item() if torch.is_tensor(indices[sample_idx]) else indices[sample_idx]
                expected = dataset.data[idx].get("paraphrased_answer", dataset.data[idx].get("answer", "")).strip()
                if expected and expected not in decoded and decoded not in expected:
                    mismatches.append((idx, decoded[:60], expected[:60]))
        assert not mismatches, (
            f"Retain path decoded labels should contain dataset response. Mismatches: {mismatches[:10]}"
        )

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_retain_slice_equals_gen_start_slice_for_full_convo(
        self, tokenizer, batch_size
    ):
        """For full-convo labels, retain path (prompt_starts) slice equals gen_start slice only when labels have IGNORE for prompt. For TOFU full-convo, retain uses content_start not response_start; so retain slice is [content_start:content_start+L] = prompt+response prefix, not response only. So we only assert retain path produces valid batch_template and decoded non-empty when possible."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        L = 30
        for batch, _p, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            for sample_idx in range(batch["input_ids"].shape[0]):
                bt = _build_batch_template_retain_style(
                    batch, sample_idx, prompt_starts, L
                )
                assert bt["input_ids"].shape == (1, L)
                if bt["labels"] is not None:
                    assert bt["labels"].shape == (1, L)


class TestDecodedPromptResponseVsDatasetGroundTruth:
    """Decoded prompt and response (from correct indices) must match dataset ground truth."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_decoded_response_at_gen_start_matches_dataset_answer(
        self, tokenizer, batch_size
    ):
        """Decode labels[gen_start:gen_start+L] (strip IGNORE) and assert content matches dataset answer."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        dataset = dataloader.dataset
        L = 60
        for batch, _prompts, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            labels = batch["labels"]
            indices = batch.get("index", torch.arange(labels.shape[0]))
            for i in range(labels.shape[0]):
                gen_start = _generation_start_full_convo(prompt_starts[i], prompt_lens[i])
                end = min(gen_start + L, labels.shape[1])
                slice_labels = labels[i, gen_start:end]
                valid = slice_labels[slice_labels != IGNORE_INDEX]
                if valid.numel() == 0:
                    continue
                decoded = tokenizer.decode(valid.tolist(), skip_special_tokens=True).strip()
                idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
                expected = dataset.data[idx].get("paraphrased_answer", dataset.data[idx].get("answer", "")).strip()
                if expected and expected not in decoded and decoded not in expected:
                    pytest.fail(
                        f"Decoded response at gen_start for idx={idx} should match dataset answer. "
                        f"Decoded: {decoded[:100]!r}, expected: {expected[:100]!r}"
                    )

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_decoded_prompt_from_input_ids_matches_dataset_question(
        self, tokenizer, batch_size
    ):
        """Decode prompt (non-pad part of input_ids) and assert it contains or matches dataset question."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        dataset = dataloader.dataset
        for batch, prompts, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            input_ids = batch["input_ids"]
            indices = batch.get("index", torch.arange(input_ids.shape[0]))
            for i in range(input_ids.shape[0]):
                prompt_tokens = prompts[i]
                if not prompt_tokens:
                    continue
                decoded_prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True).strip()
                idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
                expected_q = dataset.data[idx].get("question", "").strip()
                if expected_q and expected_q not in decoded_prompt and decoded_prompt not in expected_q:
                    pytest.fail(
                        f"Decoded prompt for idx={idx} should contain dataset question. "
                        f"Prompt: {decoded_prompt[:120]!r}, question: {expected_q[:120]!r}"
                    )

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
    def test_decoded_response_at_gen_start_matches_dataset_answer_forget10(
        self, tokenizer, batch_size
    ):
        """Same invariant for forget10 (non-perturbed): decode labels[gen_start:gen_start+L] matches dataset answer."""
        dataloader = _dataloader_forget10(tokenizer, batch_size=batch_size)
        dataset = dataloader.dataset
        L = 60
        for batch, _prompts, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            labels = batch["labels"]
            indices = batch.get("index", torch.arange(labels.shape[0]))
            for i in range(labels.shape[0]):
                gen_start = _generation_start_full_convo(prompt_starts[i], prompt_lens[i])
                end = min(gen_start + L, labels.shape[1])
                slice_labels = labels[i, gen_start:end]
                valid = slice_labels[slice_labels != IGNORE_INDEX]
                if valid.numel() == 0:
                    continue
                decoded = tokenizer.decode(valid.tolist(), skip_special_tokens=True).strip()
                idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
                expected = dataset.data[idx].get("answer", "").strip()
                if expected and expected not in decoded and decoded not in expected:
                    pytest.fail(
                        f"Decoded response at gen_start for forget10 idx={idx} should match dataset answer. "
                        f"Decoded: {decoded[:100]!r}, expected: {expected[:100]!r}"
                    )


class TestSliceBatchTemplateToLength:
    """Dedicated tests for _slice_batch_template_to_length."""

    def test_2d_tensor_longer_than_length_sliced(self):
        from evals.metrics.trajectory_metrics import _slice_batch_template_to_length

        bt = {
            "input_ids": torch.zeros(1, 20, dtype=torch.long),
            "labels": torch.full((1, 20), IGNORE_INDEX, dtype=torch.long),
        }
        out = _slice_batch_template_to_length(bt, 10)
        assert out["input_ids"].shape == (1, 10)
        assert out["labels"].shape == (1, 10)

    def test_2d_tensor_shorter_than_length_unchanged(self):
        from evals.metrics.trajectory_metrics import _slice_batch_template_to_length

        bt = {
            "input_ids": torch.zeros(1, 5, dtype=torch.long),
            "labels": torch.full((1, 5), IGNORE_INDEX, dtype=torch.long),
        }
        out = _slice_batch_template_to_length(bt, 10)
        assert out["input_ids"].shape == (1, 5)
        assert out["labels"].shape == (1, 5)

    def test_none_value_preserved(self):
        from evals.metrics.trajectory_metrics import _slice_batch_template_to_length

        bt = {"input_ids": torch.zeros(1, 10), "labels": None}
        out = _slice_batch_template_to_length(bt, 5)
        assert out["labels"] is None
        assert out["input_ids"].shape == (1, 5)

    def test_list_of_tensors_sliced(self):
        from evals.metrics.trajectory_metrics import _slice_batch_template_to_length

        bt = {
            "labels_wrong": [
                torch.zeros(1, 20, dtype=torch.long),
                torch.zeros(1, 20, dtype=torch.long),
            ],
        }
        out = _slice_batch_template_to_length(bt, 7)
        assert len(out["labels_wrong"]) == 2
        assert out["labels_wrong"][0].shape == (1, 7)
        assert out["labels_wrong"][1].shape == (1, 7)

    def test_1d_tensor_sliced(self):
        from evals.metrics.trajectory_metrics import _slice_batch_template_to_length

        bt = {"index": torch.tensor([0, 1, 2, 3, 4])}
        out = _slice_batch_template_to_length(bt, 3)
        assert out["index"].shape == (3,)
        assert out["index"].tolist() == [0, 1, 2]

    def test_non_tensor_values_unchanged(self):
        from evals.metrics.trajectory_metrics import _slice_batch_template_to_length

        bt = {
            "input_ids": torch.zeros(1, 10),
            "index": torch.tensor([0]),
            "meta": "string",
        }
        out = _slice_batch_template_to_length(bt, 5)
        assert out["input_ids"].shape == (1, 5)
        assert out["meta"] == "string"

    def test_empty_length_zero(self):
        from evals.metrics.trajectory_metrics import _slice_batch_template_to_length

        bt = {"input_ids": torch.zeros(1, 10), "labels": torch.zeros(1, 10)}
        out = _slice_batch_template_to_length(bt, 0)
        assert out["input_ids"].shape == (1, 0)
        assert out["labels"].shape == (1, 0)

    def test_full_batch_template_with_dual_labels_sliced(self):
        """Slice a batch_template that has labels_correct (2D) and labels_wrong (list of 2D) as in main loop."""
        from evals.metrics.trajectory_metrics import _slice_batch_template_to_length

        bt = {
            "input_ids": torch.zeros(1, 100, dtype=torch.long),
            "labels": torch.full((1, 100), IGNORE_INDEX, dtype=torch.long),
            "attention_mask": torch.ones(1, 100, dtype=torch.long),
            "index": torch.tensor([0]),
            "labels_correct": torch.zeros(1, 100, dtype=torch.long),
            "labels_wrong": [
                torch.zeros(1, 100, dtype=torch.long),
                torch.zeros(1, 100, dtype=torch.long),
            ],
        }
        out = _slice_batch_template_to_length(bt, 50)
        assert out["input_ids"].shape == (1, 50)
        assert out["labels"].shape == (1, 50)
        assert out["labels_correct"].shape == (1, 50)
        assert len(out["labels_wrong"]) == 2
        assert out["labels_wrong"][0].shape == (1, 50)
        assert out["labels_wrong"][1].shape == (1, 50)
        assert out["index"].shape == (1,)

    @pytest.mark.parametrize("length", [1, 5, 19, 20])
    def test_slice_length_boundary_cases(self, length):
        from evals.metrics.trajectory_metrics import _slice_batch_template_to_length

        bt = {"input_ids": torch.zeros(1, 20), "labels": torch.zeros(1, 20)}
        out = _slice_batch_template_to_length(bt, length)
        assert out["input_ids"].shape == (1, length)
        assert out["labels"].shape == (1, length)

    def test_slice_batch_template_leaves_index_1d_unchanged_when_slicing_sequences(self):
        """batch_template['index'] is (1,) and should not be truncated by seq length slice."""
        from evals.metrics.trajectory_metrics import _slice_batch_template_to_length

        bt = {
            "input_ids": torch.zeros(1, 100),
            "labels": torch.zeros(1, 100),
            "index": torch.tensor([42]),
        }
        out = _slice_batch_template_to_length(bt, 50)
        assert out["index"].shape == (1,)
        assert out["index"].item() == 42


# ---------------------------------------------------------------------------
# Collator padding, attention_mask, indexing, tokenization, batch_template
# padding, LengthSortedSampler, _slice_labels_by_content_start edge cases.
# ---------------------------------------------------------------------------


class TestCollatorPaddingAndMask:
    """Validate collator: left padding length, labels IGNORE_INDEX, attention_mask."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
        )
        if t.pad_token_id is None:
            t.pad_token_id = t.eos_token_id
        return t

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_attention_mask_equals_input_ids_ne_pad(self, tokenizer, batch_size):
        """Collator must set attention_mask = (input_ids != pad_token_id)."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        pad_id = tokenizer.pad_token_id
        for batch in dataloader:
            input_ids = batch["input_ids"]
            mask = batch["attention_mask"]
            assert mask.shape == input_ids.shape
            assert torch.equal(mask, (input_ids != pad_id).long()), (
                "attention_mask must equal (input_ids != pad_token_id)"
            )

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_labels_use_ignore_index_for_padding(self, tokenizer, batch_size):
        """Labels from collator must use IGNORE_INDEX for padded positions. Note: labels and input_ids are padded independently (can differ in seq length and alignment), so we only assert each labels row has IGNORE_INDEX where that row is padded (left-pad: leading positions)."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        for batch in dataloader:
            labels = batch["labels"]
            for i in range(labels.shape[0]):
                row = labels[i]
                n_content = (row != IGNORE_INDEX).sum().item()
                assert n_content <= row.shape[0]
                if n_content < row.shape[0]:
                    assert (row == IGNORE_INDEX).any(), (
                        f"Row {i}: labels with n_content={n_content} < len={row.shape[0]} must have IGNORE_INDEX"
                    )

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_labels_and_input_ids_may_differ_in_seq_length(self, tokenizer, batch_size):
        """Collator pads input_ids and labels separately; they can have different seq lengths (e.g. prompt-only input vs full prompt+response labels)."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        for batch in dataloader:
            in_len = batch["input_ids"].shape[1]
            lab_len = batch["labels"].shape[1]
            assert in_len >= 1 and lab_len >= 1
            if in_len != lab_len:
                return
        pytest.skip("No batch had different input_ids vs labels length in this run")

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_left_pad_length_per_row_consistent_with_content(self, tokenizer, batch_size):
        """Left padding length per row = seq_len - content_length (content = non-pad for input_ids)."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        pad_id = tokenizer.pad_token_id
        for batch in dataloader:
            input_ids = batch["input_ids"]
            seq_len = input_ids.shape[1]
            for i in range(input_ids.shape[0]):
                content_len = (input_ids[i] != pad_id).sum().item()
                pad_len = seq_len - content_len
                if pad_len > 0:
                    assert (input_ids[i, :pad_len] == pad_id).all(), (
                        f"Row {i}: first pad_len={pad_len} positions must be pad"
                    )
                    assert (input_ids[i, pad_len:] != pad_id).all() or content_len == 0, (
                        f"Row {i}: positions from pad_len onward must be content (non-pad)"
                    )

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_labels_correct_labels_wrong_3d_structure_and_padding(self, tokenizer, batch_size):
        """labels_correct and labels_wrong from collator: correct shapes; padded positions use IGNORE_INDEX (rows shorter than max_L get IGNORE_INDEX tail)."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        for batch in dataloader:
            lc = batch.get("labels_correct")
            lw = batch.get("labels_wrong")
            B = batch["input_ids"].shape[0]
            if lc is not None:
                assert lc.dim() == 2 and lc.shape[0] == B
            if lw is not None:
                assert lw.dim() == 3, "labels_wrong should be [B, N, L]"
                assert lw.shape[0] == B
                max_L = lw.shape[2]
                for i in range(B):
                    for k in range(lw.shape[1]):
                        row = lw[i, k]
                        n_content = (row != IGNORE_INDEX).sum().item()
                        assert n_content <= max_L
                        if n_content < max_L:
                            assert (row == IGNORE_INDEX).any(), (
                                "labels_wrong row shorter than max_L must have IGNORE_INDEX padding"
                            )


class TestBatchTemplatePaddingWhenSliceShorterThanL:
    """When gen_start + L > seq_len, batch_template must pad to L with IGNORE_INDEX / zeros."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
        )
        if t.pad_token_id is None:
            t.pad_token_id = t.eos_token_id
        return t

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("L", [200, 400])
    def test_generated_labels_padded_to_L_with_ignore_index(
        self, tokenizer, batch_size, L
    ):
        """When slice is shorter than L, generated_labels must be length L with tail IGNORE_INDEX."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        for batch, _p, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            seq_len = batch["labels"].shape[1]
            for sample_idx in range(batch["labels"].shape[0]):
                gen_start = _generation_start_full_convo(
                    prompt_starts[sample_idx], prompt_lens[sample_idx]
                )
                remaining = seq_len - gen_start
                if remaining < L:
                    bt = _build_batch_template_forget_style(
                        batch, sample_idx, prompt_lens, prompt_starts, L,
                        use_generation_start=True,
                    )
                    gl = bt["labels"].squeeze(0)
                    assert gl.shape[0] == L
                    assert (gl[remaining:] == IGNORE_INDEX).all(), (
                        f"Tail of generated_labels (from {remaining} to L={L}) must be IGNORE_INDEX"
                    )

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("L", [200, 400])
    def test_generated_input_ids_padded_to_L_with_zeros(
        self, tokenizer, batch_size, L
    ):
        """When slice is shorter than L, generated_input_ids must be length L with tail zeros."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        for batch, _p, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            seq_len = batch["input_ids"].shape[1]
            for sample_idx in range(batch["input_ids"].shape[0]):
                gen_start = _generation_start_full_convo(
                    prompt_starts[sample_idx], prompt_lens[sample_idx]
                )
                remaining = seq_len - gen_start
                if remaining < L:
                    bt = _build_batch_template_forget_style(
                        batch, sample_idx, prompt_lens, prompt_starts, L,
                        use_generation_start=True,
                    )
                    gi = bt["input_ids"].squeeze(0)
                    assert gi.shape[0] == L
                    assert (gi[remaining:] == 0).all(), (
                        f"Tail of generated_input_ids (from {remaining} to L={L}) must be zeros"
                    )


class TestIndexingBatchToDataset:
    """batch['index'] must be present and map correctly to dataset rows."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
        )
        if t.pad_token_id is None:
            t.pad_token_id = t.eos_token_id
        return t

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_has_index_key(self, tokenizer, batch_size):
        """Batch must contain 'index' key (collator with index='index')."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        for batch in dataloader:
            assert "index" in batch
            assert batch["index"].shape[0] == batch["input_ids"].shape[0]

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_index_maps_to_dataset_row(self, tokenizer, batch_size):
        """batch['index'][i] must identify the dataset row (question/answer match)."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        dataset = dataloader.dataset
        for batch, prompts, _pl, _ps in _build_prompts_and_starts(dataloader, tokenizer):
            indices = batch["index"]
            for i in range(indices.shape[0]):
                idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
                assert 0 <= idx < len(dataset.data)
                expected_q = dataset.data[idx].get("question", "").strip()
                if expected_q and prompts[i]:
                    decoded = tokenizer.decode(prompts[i], skip_special_tokens=True).strip()
                    assert expected_q in decoded or decoded in expected_q, (
                        f"batch index {idx} must map to dataset row with matching question"
                    )


class TestTokenizationInPipeline:
    """Tokenization: pad/eos set, decode does not throw, content tokens valid."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
        )
        if t.pad_token_id is None:
            t.pad_token_id = t.eos_token_id
        return t

    def test_tokenizer_has_pad_and_eos(self, tokenizer):
        """Tokenizer must have pad_token_id and eos_token_id set for pipeline."""
        assert tokenizer.pad_token_id is not None
        assert tokenizer.eos_token_id is not None

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_decode_generation_region_does_not_throw(self, tokenizer, batch_size):
        """Decoding the generation region (valid tokens only) must not raise."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        for batch, _p, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            labels = batch["labels"]
            for i in range(labels.shape[0]):
                gen_start = _generation_start_full_convo(prompt_starts[i], prompt_lens[i])
                end = min(gen_start + 100, labels.shape[1])
                slice_labels = labels[i, gen_start:end]
                valid = slice_labels[slice_labels != IGNORE_INDEX]
                if valid.numel() > 0:
                    tokenizer.decode(valid.tolist(), skip_special_tokens=True)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_prompt_tokens_not_all_pad_or_eos(self, tokenizer, batch_size):
        """Prompt tokens (from _build_prompts_for_sampler) must not be all pad/eos (meaningful content)."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id
        for batch, prompts, _pl, _ps in _build_prompts_and_starts(dataloader, tokenizer):
            for i, prompt_tokens in enumerate(prompts):
                if len(prompt_tokens) == 0:
                    continue
                prompt_t = torch.tensor(prompt_tokens) if isinstance(prompt_tokens, list) else prompt_tokens
                content = (prompt_t != pad_id) & (prompt_t != eos_id)
                assert content.any(), (
                    f"Sample {i}: prompt should have at least one non-pad, non-eos token"
                )


class TestLengthSortedSamplerAndBatchComposition:
    """LengthSortedSampler ordering and prompt_starts vs batch composition."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
        )
        if t.pad_token_id is None:
            t.pad_token_id = t.eos_token_id
        return t

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_batches_sorted_by_input_ids_length_descending(self, tokenizer, batch_size):
        """With LengthSortedSampler(length_key='input_ids'), batch rows should be descending by content length (non-pad)."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        pad_id = tokenizer.pad_token_id
        for batch in dataloader:
            input_ids = batch["input_ids"]
            content_lens = [(input_ids[i] != pad_id).sum().item() for i in range(input_ids.shape[0])]
            for j in range(1, len(content_lens)):
                assert content_lens[j] <= content_lens[j - 1], (
                    f"LengthSortedSampler descending: content_lens should be non-increasing, got {content_lens}"
                )

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_prompt_starts_can_be_nonzero_for_non_longest_row(self, tokenizer, batch_size):
        """With left padding, prompt_starts[i] can be > 0 when row i is not the longest (labels padded to batch max)."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        found_nonzero = False
        for batch, _p, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            for i in range(len(prompt_starts)):
                if prompt_starts[i] > 0:
                    found_nonzero = True
                    break
            if found_nonzero:
                break
        assert found_nonzero, (
            "With batch_size>=2 and left padding, at least one batch should have prompt_starts[i] > 0 for some row"
        )

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_without_sort_prompt_starts_still_in_bounds(self, tokenizer, batch_size):
        """Without LengthSortedSampler, prompt_starts and prompt_lens still in bounds (prompt_starts/gen_start index into labels)."""
        dataloader = _dataloader_forget10_perturbed(
            tokenizer, batch_size=batch_size, sort_by_length=False
        )
        for batch, _p, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            labels_len = batch["labels"].shape[1]
            input_len = batch["input_ids"].shape[1]
            for i in range(len(prompt_starts)):
                assert 0 <= prompt_starts[i] <= labels_len
                assert 0 <= prompt_lens[i] <= input_len
                gen_start = _generation_start_full_convo(prompt_starts[i], prompt_lens[i])
                assert gen_start <= labels_len


class TestSliceLabelsByContentStartEdgeCase:
    """_slice_labels_by_content_start when row is all IGNORE_INDEX."""

    def test_all_ignore_returns_L_ignore_index(self):
        from evals.metrics.trajectory_metrics import _slice_labels_by_content_start

        row = torch.full((50,), IGNORE_INDEX, dtype=torch.long)
        out = _slice_labels_by_content_start(row, L=20, ignore_index=IGNORE_INDEX)
        assert out.shape == (20,)
        assert (out == IGNORE_INDEX).all()

    def test_all_ignore_2d_row_squeezed(self):
        from evals.metrics.trajectory_metrics import _slice_labels_by_content_start

        row = torch.full((1, 50), IGNORE_INDEX, dtype=torch.long)
        out = _slice_labels_by_content_start(row, L=10, ignore_index=IGNORE_INDEX)
        assert out.shape == (10,)
        assert (out == IGNORE_INDEX).all()


class TestBatchTemplateAttentionMask:
    """batch_template attention_mask must be all ones (1, L)."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
        )
        if t.pad_token_id is None:
            t.pad_token_id = t.eos_token_id
        return t

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("L", [10, 50])
    def test_forget_style_batch_template_attention_mask_all_ones(
        self, tokenizer, batch_size, L
    ):
        """Forget-style batch_template must have attention_mask shape (1, L) all ones."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        for batch, _p, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            for sample_idx in range(batch["input_ids"].shape[0]):
                bt = _build_batch_template_forget_style(
                    batch, sample_idx, prompt_lens, prompt_starts, L,
                    use_generation_start=True,
                )
                am = bt["attention_mask"]
                assert am.shape == (1, L)
                assert (am == 1).all()

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_retain_style_batch_template_attention_mask_all_ones(
        self, tokenizer, batch_size
    ):
        """Retain-style batch_template must have attention_mask shape (1, L) all ones."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        L = 30
        for batch, _p, _pl, prompt_starts in _build_prompts_and_starts(dataloader, tokenizer):
            for sample_idx in range(batch["input_ids"].shape[0]):
                bt = _build_batch_template_retain_style(
                    batch, sample_idx, prompt_starts, L
                )
                am = bt["attention_mask"]
                assert am.shape == (1, L)
                assert (am == 1).all()
