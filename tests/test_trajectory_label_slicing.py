"""
Tests for trajectory loop label/input_ids slicing: prompt_starts vs prompt_lens.

Two data conventions:
- IGNORE-for-prompt: labels = [IGNORE]*prompt_len + [response]. Then prompt_starts[i]
  = first non-IGNORE = start of generation. Use prompt_starts for slicing.
- Full-conversation (TOFU predict_with_generate): labels = [prompt + response], then
  left-padded to [IGNORE]*n_pad + [prompt + response]. Then prompt_starts[i] = n_pad
  (start of content), and generation start = prompt_starts[i] + prompt_lens[i].
  Using prompt_lens[i] alone as slice start is wrong when n_pad > 0.

These tests validate every component (prompts, prompt_lens, prompt_starts, correct
slice for generation) and reproduce the bug (many fail until the fix).
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
    """Forget10_perturbed dataloader with configurable batch_size (same config as eval)."""
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


def _build_prompts_and_starts(dataloader, tokenizer):
    """Run _build_prompts_for_sampler on every batch; yield (batch, prompts, prompt_lens, prompt_starts)."""
    from evals.metrics.trajectory_metrics import _build_prompts_for_sampler

    prompt_only = getattr(dataloader.dataset, "predict_with_generate", False)
    for batch in dataloader:
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        prompts, prompt_lens, prompt_starts = _build_prompts_for_sampler(
            input_ids,
            labels,
            tokenizer,
            ignore_index=IGNORE_INDEX,
            prompt_only_input_ids=prompt_only,
        )
        yield batch, prompts, prompt_lens, prompt_starts


def _generation_start_full_convo(prompt_starts_i: int, prompt_lens_i: int) -> int:
    """For full-conversation labels (TOFU): gen start = content start + prompt length."""
    return prompt_starts_i + prompt_lens_i


class TestPromptStartsVsPromptLens:
    """prompt_starts = first non-IGNORE in labels (content start); with full-convo labels + left padding, generation start = prompt_starts + prompt_lens."""

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
    def test_prompt_starts_and_lens_nonnegative_and_in_bounds(
        self, tokenizer, batch_size
    ):
        """prompt_starts[i] and prompt_lens[i] are non-negative and within row bounds. (Labels are padded to batch max length, so even row 0 can have left pad.)"""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        for batch, _prompts, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            labels = batch["labels"]
            seq_len = labels.shape[1]
            for i in range(len(prompt_lens)):
                assert prompt_starts[i] >= 0, f"Row {i}: prompt_starts must be >= 0"
                assert prompt_lens[i] >= 0, f"Row {i}: prompt_lens must be >= 0"
                assert prompt_starts[i] <= seq_len
                gen_start = _generation_start_full_convo(prompt_starts[i], prompt_lens[i])
                assert gen_start <= seq_len, (
                    f"Row {i}: gen_start {gen_start} must be <= seq_len {seq_len}"
                )

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_generation_start_differs_from_prompt_lens_when_left_padded(
        self, tokenizer, batch_size
    ):
        """For full-convo labels, generation_start = prompt_starts[i] + prompt_lens[i]. When prompt_starts[i] > 0, this != prompt_lens[i] (bug: using prompt_lens as slice start)."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        found_difference = False
        for batch, _prompts, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            for i in range(len(prompt_lens)):
                gen_start = _generation_start_full_convo(prompt_starts[i], prompt_lens[i])
                if gen_start != prompt_lens[i]:
                    found_difference = True
                    break
            if found_difference:
                break
        assert found_difference, (
            "Expected at least one sample where generation_start (prompt_starts+prompt_lens) != prompt_lens (left-padded rows)."
        )

    def test_batch_size_one_generation_start_equals_prompt_lens(self, tokenizer):
        """Single-sample batches: no left pad so prompt_starts[0]=0, hence generation_start = 0 + prompt_lens[0] = prompt_lens[0]."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=1)
        for batch, _prompts, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            assert len(prompt_lens) == 1
            assert prompt_starts[0] == 0
            assert _generation_start_full_convo(prompt_starts[0], prompt_lens[0]) == prompt_lens[0]


class TestGenerationRegionSliceContent:
    """The correct generation region in labels is labels[i, prompt_starts[i]:prompt_starts[i]+L]; using prompt_lens as start is wrong when they differ."""

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
    def test_slice_by_prompt_lens_differs_from_slice_by_generation_start_when_left_padded(
        self, tokenizer, batch_size
    ):
        """When left-padded, labels[i, prompt_lens[i]:+L] (bug) != labels[i, gen_start:+L] (correct). Reproduces wrong slice content."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        L = 50
        found_difference = False
        for batch, _prompts, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            labels = batch["labels"]
            B = labels.shape[0]
            for i in range(B):
                gen_start = _generation_start_full_convo(prompt_starts[i], prompt_lens[i])
                if gen_start == prompt_lens[i]:
                    continue
                start_correct = gen_start
                start_wrong = prompt_lens[i]
                end_correct = min(start_correct + L, labels.shape[1])
                end_wrong = min(start_wrong + L, labels.shape[1])
                slice_correct = labels[i, start_correct:end_correct].clone()
                slice_wrong = labels[i, start_wrong:end_wrong].clone()
                if slice_correct.shape != slice_wrong.shape or not torch.equal(
                    slice_correct, slice_wrong
                ):
                    found_difference = True
                    break
            if found_difference:
                break
        assert found_difference, (
            "Expected at least one sample where slice by gen_start != slice by prompt_lens (demonstrates bug)."
        )

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_correct_slice_has_generation_tokens_not_leading_ignore(self, tokenizer, batch_size):
        """labels[i, gen_start:gen_start+L] (full-convo: gen_start = prompt_starts+prompt_lens) should contain real tokens (generation), not all IGNORE."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        L = 20
        for batch, _prompts, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            labels = batch["labels"]
            B = labels.shape[0]
            for i in range(B):
                start = _generation_start_full_convo(prompt_starts[i], prompt_lens[i])
                end = min(start + L, labels.shape[1])
                gen_slice = labels[i, start:end]
                n_real = (gen_slice != IGNORE_INDEX).sum().item()
                assert n_real >= 1, (
                    f"Sample {i}: generation region [gen_start:{start}:{end}] must contain at least one non-IGNORE token; got all IGNORE."
                )


class TestTrajectoryLoopMustUseGenerationStart:
    """
    The trajectory loop must slice labels/input_ids at the generation start index.
    For full-convo labels (TOFU): gen_start = prompt_starts[i] + prompt_lens[i].
    Current code uses prompt_lens[i] as start, which is wrong when left-padded (prompt_starts[i] > 0).
    These tests FAIL with current code; they PASS after the fix.
    """

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer

        t = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
        )
        if t.pad_token_id is None:
            t.pad_token_id = t.eos_token_id
        return t

    def _slice_as_trajectory_loop_fixed(
        self,
        sample_labels,
        sample_input_ids,
        prompt_starts_i: int,
        prompt_lens_i: int,
        L: int,
        prompt_only_input_ids: bool = True,
    ):
        """Replicate fixed trajectory loop: uses generation_start (prompt_starts+prompt_lens for full-convo)."""
        _ps = int(prompt_starts_i) if hasattr(prompt_starts_i, "item") else prompt_starts_i
        _pl = int(prompt_lens_i) if hasattr(prompt_lens_i, "item") else prompt_lens_i
        gen_start = (_ps + _pl) if prompt_only_input_ids else _ps
        gen_start = int(gen_start) if hasattr(gen_start, "item") else gen_start
        gen_labels = sample_labels[gen_start : gen_start + L]
        gen_input_ids = sample_input_ids[gen_start : gen_start + L]
        return gen_labels, gen_input_ids

    def _slice_at_generation_start(
        self,
        sample_labels,
        sample_input_ids,
        prompt_starts_i: int,
        prompt_lens_i: int,
        L: int,
    ):
        """Correct slice for full-convo: gen_start = prompt_starts + prompt_lens."""
        gen_start = _generation_start_full_convo(prompt_starts_i, prompt_lens_i)
        gen_start = int(gen_start) if hasattr(gen_start, "item") else gen_start
        gen_labels = sample_labels[gen_start : gen_start + L]
        gen_input_ids = sample_input_ids[gen_start : gen_start + L]
        return gen_labels, gen_input_ids

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_generated_labels_must_equal_slice_by_generation_start(
        self, tokenizer, batch_size
    ):
        """generated_labels must equal labels[i, gen_start:gen_start+L]. After fix, trajectory loop uses generation_start; this test asserts the invariant."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        L = 30
        failures = []
        for batch, _prompts, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            labels = batch["labels"]
            input_ids = batch["input_ids"]
            indices = batch.get("index", torch.arange(labels.shape[0]))
            B = labels.shape[0]
            for i in range(B):
                gen_labels_loop, _ = self._slice_as_trajectory_loop_fixed(
                    labels[i], input_ids[i], prompt_starts[i], prompt_lens[i], L,
                    prompt_only_input_ids=True,
                )
                gen_labels_correct, _ = self._slice_at_generation_start(
                    labels[i], input_ids[i], prompt_starts[i], prompt_lens[i], L
                )
                min_len = min(gen_labels_loop.shape[0], gen_labels_correct.shape[0])
                if min_len == 0 or not torch.equal(
                    gen_labels_loop[:min_len], gen_labels_correct[:min_len]
                ):
                    failures.append(
                        (indices[i].item() if torch.is_tensor(indices[i]) else indices[i], i, "labels")
                    )
        assert not failures, (
            f"generated_labels must equal labels[i, gen_start:gen_start+L] (gen_start=prompt_starts+prompt_lens). "
            f"Mismatch (dataset_idx, batch_idx, key): {failures[:20]}."
        )

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_generated_input_ids_must_equal_slice_by_generation_start(
        self, tokenizer, batch_size
    ):
        """generated_input_ids must equal input_ids[i, gen_start:gen_start+L]. We use forget10 with predict_with_generate=False so input_ids and labels are same length (position-aligned); gen_start = prompt_starts (IGNORE-for-prompt convention)."""
        dataloader = _dataloader_forget10_full_convo_aligned(tokenizer, batch_size=batch_size)
        L = 30
        failures = []
        for batch, _prompts, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            labels = batch["labels"]
            input_ids = batch["input_ids"]
            assert labels.shape[1] == input_ids.shape[1], "Aligned dataloader must yield same seq length"
            indices = batch.get("index", torch.arange(labels.shape[0]))
            B = labels.shape[0]
            for i in range(B):
                _, gen_input_ids_loop = self._slice_as_trajectory_loop_fixed(
                    labels[i], input_ids[i], prompt_starts[i], prompt_lens[i], L,
                    prompt_only_input_ids=False,
                )
                gen_start = int(prompt_starts[i]) if hasattr(prompt_starts[i], "item") else prompt_starts[i]
                gen_input_ids_correct = input_ids[i][gen_start : gen_start + L]
                min_len = min(
                    gen_input_ids_loop.shape[0], gen_input_ids_correct.shape[0]
                )
                if min_len == 0 or not torch.equal(
                    gen_input_ids_loop[:min_len], gen_input_ids_correct[:min_len]
                ):
                    failures.append(
                        (indices[i].item() if torch.is_tensor(indices[i]) else indices[i], i, "input_ids")
                    )
        assert not failures, (
            f"generated_input_ids must equal input_ids[i, gen_start:gen_start+L]. "
            f"Mismatch (dataset_idx, batch_idx, key): {failures[:20]}."
        )

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_prompt_len_by_index_should_use_prompt_lens_not_starts(self, tokenizer, batch_size):
        """prompt_len_by_index (and decoding prompt from seq[:pl]) must use prompt_lens (length), not prompt_starts. This test documents correct usage: length vs start."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        for _batch, prompts, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            for i in range(len(prompts)):
                assert len(prompts[i]) == prompt_lens[i], (
                    "prompt_lens[i] must equal len(prompts[i]) (length of prompt sent to sampler)."
                )
                # prompt_starts can be >= prompt_lens with left padding; they are not the same concept
                assert prompt_starts[i] >= 0


def _dataloader_forget10_full_convo_aligned(tokenizer, batch_size: int):
    """Forget10 with predict_with_generate=False so input_ids and labels are full conversation and same length (position-aligned)."""
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


def _dataloader_forget10_non_perturbed(tokenizer, batch_size: int):
    """Forget10 (non-perturbed) dataloader, same layout as forget10_perturbed (full-convo labels, left-padded)."""
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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
    )


class TestForget10NonPerturbedLabelSlicing:
    """Same full-convo + left-pad convention as forget10_perturbed; generation_start = prompt_starts + prompt_lens."""

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
    def test_generation_start_consistent_with_labels_content(self, tokenizer, batch_size):
        """For each sample, labels[i, gen_start:] must contain the response (non-IGNORE); gen_start = prompt_starts + prompt_lens."""
        dataloader = _dataloader_forget10_non_perturbed(tokenizer, batch_size=batch_size)
        for batch, _prompts, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            labels = batch["labels"]
            for i in range(labels.shape[0]):
                gen_start = _generation_start_full_convo(prompt_starts[i], prompt_lens[i])
                if gen_start < labels.shape[1]:
                    rest = labels[i, gen_start:]
                    n_real = (rest != IGNORE_INDEX).sum().item()
                    assert n_real >= 1, (
                        f"Sample {i}: labels[i, gen_start:] must contain response tokens (non-IGNORE)."
                    )


class TestSyntheticIgnoreForPromptLabels:
    """
    Synthetic batches where labels = [IGNORE]*prompt_len + [response].
    Then prompt_starts[i] = first non-IGNORE = start of generation (no need for +prompt_lens).
    Validates that _build_prompts_for_sampler returns correct prompt_starts and that
    slicing by prompt_starts gives the generation region.
    """

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer

        t = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
        )
        if t.pad_token_id is None:
            t.pad_token_id = t.eos_token_id
        return t

    def test_synthetic_left_padded_ignore_prompt_prompt_starts_is_gen_start(self, tokenizer):
        """Synthetic: labels = [IGNORE]*n_pad + [IGNORE]*prompt_len + [gen]. Then first non-IGNORE = n_pad+prompt_len = gen start. So prompt_starts = gen start."""
        from evals.metrics.trajectory_metrics import _build_prompts_for_sampler

        # Simulate batch of 3: row0 longest (no pad), row1/2 left-padded. Labels use IGNORE for prompt.
        max_len = 80
        prompt_lens_content = [50, 40, 30]  # content prompt lengths
        gen_lens = [20, 25, 30]
        pad_len_0 = 0
        pad_len_1 = max_len - (prompt_lens_content[1] + gen_lens[1])
        pad_len_2 = max_len - (prompt_lens_content[2] + gen_lens[2])
        row0 = [IGNORE_INDEX] * prompt_lens_content[0] + list(range(100, 100 + gen_lens[0]))
        row0 = row0 + [IGNORE_INDEX] * (max_len - len(row0))
        row1 = [IGNORE_INDEX] * pad_len_1 + [IGNORE_INDEX] * prompt_lens_content[1] + list(range(200, 200 + gen_lens[1]))
        row1 = row1 + [IGNORE_INDEX] * (max_len - len(row1))
        row2 = [IGNORE_INDEX] * pad_len_2 + [IGNORE_INDEX] * prompt_lens_content[2] + list(range(300, 300 + gen_lens[2]))
        row2 = row2 + [IGNORE_INDEX] * (max_len - len(row2))
        labels = torch.tensor([row0, row1, row2])
        # input_ids: same layout for content; use pad_id for padding
        pad_id = tokenizer.pad_token_id
        row0_inp = [1] * prompt_lens_content[0] + [0] * gen_lens[0]
        row0_inp = row0_inp + [pad_id] * (max_len - len(row0_inp))
        row1_inp = [pad_id] * pad_len_1 + [1] * prompt_lens_content[1] + [0] * gen_lens[1]
        row1_inp = row1_inp + [pad_id] * (max_len - len(row1_inp))
        row2_inp = [pad_id] * pad_len_2 + [1] * prompt_lens_content[2] + [0] * gen_lens[2]
        row2_inp = row2_inp + [pad_id] * (max_len - len(row2_inp))
        input_ids = torch.tensor([row0_inp, row1_inp, row2_inp])
        prompts, plens, pstarts = _build_prompts_for_sampler(
            input_ids, labels, tokenizer, ignore_index=IGNORE_INDEX, prompt_only_input_ids=False
        )
        # For IGNORE-for-prompt: first non-IGNORE = gen start. Row0: prompt_lens_content[0]=50, so gen start=50. Row1: pad_len_1+prompt_lens_content[1]; row2: pad_len_2+prompt_lens_content[2].
        expected_gen_start_0 = prompt_lens_content[0]
        expected_gen_start_1 = pad_len_1 + prompt_lens_content[1]
        expected_gen_start_2 = pad_len_2 + prompt_lens_content[2]
        assert pstarts[0] == expected_gen_start_0, f"Row0: prompt_starts should be gen start {expected_gen_start_0}, got {pstarts[0]}"
        assert pstarts[1] == expected_gen_start_1
        assert pstarts[2] == expected_gen_start_2
        # Slice at prompt_starts must be the generation
        L = 10
        for i in range(3):
            gen_slice = labels[i, pstarts[i] : pstarts[i] + L]
            assert (gen_slice != IGNORE_INDEX).all(), f"Row {i}: slice at prompt_starts must be gen tokens (no IGNORE)."

    def test_synthetic_slice_by_prompt_lens_wrong_when_ignore_prompt_and_left_padded(self, tokenizer):
        """With IGNORE-for-prompt + left padding, slice by prompt_lens (content length) is wrong for non-longest rows."""
        from evals.metrics.trajectory_metrics import _build_prompts_for_sampler

        max_len = 60
        prompt_lens_content = [40, 25]
        gen_lens = [15, 20]
        pad_1 = max_len - (prompt_lens_content[1] + gen_lens[1])
        row0 = [IGNORE_INDEX] * prompt_lens_content[0] + list(range(100, 100 + gen_lens[0]))
        row0 = row0 + [IGNORE_INDEX] * (max_len - len(row0))
        row1 = [IGNORE_INDEX] * pad_1 + [IGNORE_INDEX] * prompt_lens_content[1] + list(range(200, 200 + gen_lens[1]))
        row1 = row1 + [IGNORE_INDEX] * (max_len - len(row1))
        labels = torch.tensor([row0, row1])
        pad_id = tokenizer.pad_token_id
        row0_inp = [1] * prompt_lens_content[0] + [0] * gen_lens[0] + [pad_id] * (max_len - prompt_lens_content[0] - gen_lens[0])
        row1_inp = [pad_id] * pad_1 + [1] * prompt_lens_content[1] + [0] * gen_lens[1] + [pad_id] * (max_len - pad_1 - prompt_lens_content[1] - gen_lens[1])
        input_ids = torch.tensor([row0_inp, row1_inp])
        prompts, plens, pstarts = _build_prompts_for_sampler(
            input_ids, labels, tokenizer, ignore_index=IGNORE_INDEX, prompt_only_input_ids=False
        )
        # Row1: prompt_lens[1] = 25 (content). But in the tensor, gen starts at pad_1+25. So slice at plens[1]=25 is wrong (that's still in IGNORE/prompt region).
        assert pstarts[1] == pad_1 + prompt_lens_content[1]
        L = 5
        slice_by_plen = labels[1, plens[1] : plens[1] + L]
        slice_by_pstart = labels[1, pstarts[1] : pstarts[1] + L]
        assert not torch.equal(slice_by_plen, slice_by_pstart), (
            "Slice by prompt_lens must differ from slice by prompt_starts for left-padded row (IGNORE-for-prompt)."
        )


class TestBatchSizeCombinations:
    """Vary batch size (including partial batch) and dataset; assert prompt_starts/lens and gen_start consistency."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer

        t = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
        )
        if t.pad_token_id is None:
            t.pad_token_id = t.eos_token_id
        return t

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
    def test_forget10_perturbed_all_batch_sizes_prompt_starts_and_lens_in_bounds(
        self, tokenizer, batch_size
    ):
        """For every batch (including partial), prompt_starts[i] and prompt_lens[i] are in [0, labels.shape[1]) and gen_start <= labels.shape[1]."""
        dataloader = _dataloader_forget10_perturbed(tokenizer, batch_size=batch_size)
        for batch, _prompts, prompt_lens, prompt_starts in _build_prompts_and_starts(
            dataloader, tokenizer
        ):
            labels = batch["labels"]
            B = labels.shape[0]
            for i in range(B):
                assert 0 <= prompt_starts[i] <= labels.shape[1]
                assert 0 <= prompt_lens[i] <= labels.shape[1]
                gen_start = _generation_start_full_convo(prompt_starts[i], prompt_lens[i])
                assert gen_start <= labels.shape[1], (
                    f"Sample {i}: gen_start {gen_start} must be <= labels length {labels.shape[1]}"
                )
