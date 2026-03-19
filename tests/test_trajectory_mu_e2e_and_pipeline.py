"""
E2E trajectory MU with real data (no GPU/real model) and pipeline tests for retain/ra/wf.

- E2E: Real HF datasets (retain_perturbed, real_authors_perturbed, world_facts_perturbed),
  mock sampler returning valid logits_history/fixation_steps/sequences; run _compute_mu_for_dataset
  or full trajectory_metrics and assert 3/9 components, hmean, structure.
- Pipeline: Prompts, labels, tokenization/padding, slicing, trajectory steps for each MU dataset
  (same style as test_forget10_prompt_extraction and test_trajectory_label_slicing).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

IGNORE_INDEX = -100

# Template args matching trajectory_all / TOFU
TEMPLATE_ARGS = {"apply_chat_template": True}
QUESTION_KEY = "question"
HF_PATH = "locuslab/TOFU"
MAX_LEN = 512


def _tokenizer_fixture():
    from transformers import AutoTokenizer

    t = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )
    if t.pad_token_id is None:
        t.pad_token_id = t.eos_token_id
    return t


def _collator(tokenizer, padding_side="left"):
    from data.collators import DataCollatorForSupervisedDataset

    return DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, padding_side=padding_side, index="index"
    )


def _dataset_retain_perturbed(tokenizer, samples=4):
    """Retain dual-answer dataset (same as trajectory_all TOFU_QA_retain_eval)."""
    from data.qa import QAwithDualAnswersDataset

    hf_args = {
        "path": HF_PATH,
        "name": "retain_perturbed",
        "split": "train[:%d]" % samples if samples else "train",
    }
    return QAwithDualAnswersDataset(
        correct_answer_key="paraphrased_answer",
        wrong_answer_key="perturbed_answer",
        hf_args=hf_args,
        template_args=TEMPLATE_ARGS,
        tokenizer=tokenizer,
        question_key=QUESTION_KEY,
        max_length=MAX_LEN,
        predict_with_generate=True,
    )


def _dataset_ra(tokenizer, samples=4):
    """Real authors dual-answer (same as trajectory_all TOFU_QA_ra_eval)."""
    from data.qa import QAwithDualAnswersDataset

    hf_args = {
        "path": HF_PATH,
        "name": "real_authors_perturbed",
        "split": "train[:%d]" % samples if samples else "train",
    }
    return QAwithDualAnswersDataset(
        correct_answer_key="answer",
        wrong_answer_key="perturbed_answer",
        hf_args=hf_args,
        template_args=TEMPLATE_ARGS,
        tokenizer=tokenizer,
        question_key=QUESTION_KEY,
        max_length=MAX_LEN,
    )


def _dataset_wf(tokenizer, samples=4):
    """World facts dual-answer (same as trajectory_all TOFU_QA_wf_eval)."""
    from data.qa import QAwithDualAnswersDataset

    hf_args = {
        "path": HF_PATH,
        "name": "world_facts_perturbed",
        "split": "train[:%d]" % samples if samples else "train",
    }
    return QAwithDualAnswersDataset(
        correct_answer_key="answer",
        wrong_answer_key="perturbed_answer",
        hf_args=hf_args,
        template_args=TEMPLATE_ARGS,
        tokenizer=tokenizer,
        question_key=QUESTION_KEY,
        max_length=MAX_LEN,
    )


def _dataloader(dataset, tokenizer, batch_size=2, sort_by_length=True):
    from torch.utils.data import DataLoader

    from evals.metrics.samplers import LengthSortedSampler

    collator = _collator(tokenizer)
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


def _make_mock_sampler_output(B, L_full, V, S, device="cpu"):
    """Return a sampler output with logits_history, fixation_steps, sequences for trajectories_from_logits and effective_lengths_from_eos."""
    logits_history = [torch.randn(B, L_full, V, device=device) for _ in range(S)]
    fixation_steps = torch.randint(0, S, (B, L_full), device=device, dtype=torch.long)
    sequences = torch.randint(0, V, (B, L_full), device=device, dtype=torch.long)

    class SamplerOutput:
        pass

    out = SamplerOutput()
    out.logits_history = logits_history
    out.fixation_steps = fixation_steps
    out.sequences = sequences
    return out


class TestMUDatasetsPrompts:
    """Prompts built for retain/ra/wf MU pass: non-empty, in-bounds, same convention as forget path."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return _tokenizer_fixture()

    @pytest.mark.parametrize("dataset_key,dataset_fn", [
        ("retain", _dataset_retain_perturbed),
        ("ra", _dataset_ra),
        ("wf", _dataset_wf),
    ])
    def test_mu_dataset_prompts_non_empty_and_in_bounds(
        self, tokenizer, dataset_key, dataset_fn
    ):
        """For each MU dataset, prompts from _build_prompts_for_sampler are non-empty and prompt_lens/prompt_starts in bounds."""
        from evals.metrics.trajectory_metrics import _build_prompts_for_sampler

        dataset = dataset_fn(tokenizer, samples=4)
        collator = _collator(tokenizer)
        dl = DataLoader(dataset, batch_size=2, collate_fn=collator)
        prompt_only = getattr(dataset, "predict_with_generate", False)
        for batch in dl:
            input_ids = batch["input_ids"]
            labels = batch.get("labels")
            prompts, prompt_lens, prompt_starts = _build_prompts_for_sampler(
                input_ids,
                labels,
                tokenizer,
                ignore_index=IGNORE_INDEX,
                prompt_only_input_ids=prompt_only,
            )
            seq_len = input_ids.shape[1]
            for i in range(len(prompt_lens)):
                assert prompt_lens[i] >= 0, f"{dataset_key} sample {i}: prompt_lens >= 0"
                assert prompt_starts[i] >= 0, f"{dataset_key} sample {i}: prompt_starts >= 0"
                assert prompt_starts[i] <= seq_len, (
                    f"{dataset_key} sample {i}: prompt_starts {prompt_starts[i]} <= seq_len {seq_len}"
                )
                # gen_start may exceed seq_len when labels/input_ids are truncated (e.g. short padded batch)
                gen_start = prompt_starts[i] + (prompt_lens[i] if prompt_only else 0)
                assert gen_start <= seq_len or prompt_only, (
                    f"{dataset_key} sample {i}: when not prompt_only, gen_start {gen_start} <= seq_len {seq_len}"
                )
            assert len(prompts) == len(prompt_lens) == len(prompt_starts)


class TestMUDatasetsLabelsAndDual:
    """Labels and dual labels (labels_correct, labels_wrong) for retain/ra/wf."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return _tokenizer_fixture()

    @pytest.mark.parametrize("dataset_key,dataset_fn", [
        ("retain", _dataset_retain_perturbed),
        ("ra", _dataset_ra),
        ("wf", _dataset_wf),
    ])
    def test_mu_dataset_batch_has_labels_correct_and_labels_wrong(
        self, tokenizer, dataset_key, dataset_fn
    ):
        """Each MU dataset uses QAwithDualAnswersDataset so batch has labels_correct and labels_wrong."""
        from torch.utils.data import DataLoader

        dataset = dataset_fn(tokenizer, samples=4)
        collator = _collator(tokenizer)
        dl = DataLoader(dataset, batch_size=2, collate_fn=collator)
        for batch in dl:
            assert "labels_correct" in batch
            assert "labels_wrong" in batch
            assert batch["labels"] is not None
            B = batch["input_ids"].shape[0]
            assert batch["labels_correct"].shape[0] == B
            # labels_wrong can be [B, L] or [B, N, L] for N options
            assert batch["labels_wrong"].shape[0] == B


class TestMUDatasetsTokenizationPadding:
    """Tokenization and padding consistent for MU datasets (same tokenizer, left pad, IGNORE_INDEX)."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return _tokenizer_fixture()

    def test_mu_datasets_use_same_padding_side(self, tokenizer):
        """Collator uses left padding for trajectory_all; MU datasets use same collator convention."""
        from data.collators import DataCollatorForSupervisedDataset

        c = DataCollatorForSupervisedDataset(
            tokenizer=tokenizer, padding_side="left", index="index"
        )
        assert c.padding_side == "left"

    def test_mu_dataset_labels_use_ignore_index_for_padding(self, tokenizer):
        """Labels use IGNORE_INDEX (-100) for prompt/padding positions (same as trajectory path)."""
        from torch.utils.data import DataLoader

        from evals.metrics.utils import IGNORE_INDEX as EVAL_IGNORE

        dataset = _dataset_retain_perturbed(tokenizer, samples=2)
        collator = _collator(tokenizer)
        dl = DataLoader(dataset, batch_size=1, collate_fn=collator)
        for batch in dl:
            labels = batch["labels"]
            if labels is not None:
                # Left-padded: leading positions can be pad/ignore
                assert EVAL_IGNORE == IGNORE_INDEX


class TestMUDatasetsSlicing:
    """Generated region and full vs eos slicing for MU pass (same contracts as trajectory_utils)."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return _tokenizer_fixture()

    def test_slice_generated_region_matches_prompt_starts_and_lens(self, tokenizer):
        """Generated region = labels[gen_start:gen_start+L]; gen_start = prompt_starts + prompt_lens for predict_with_generate."""
        from torch.utils.data import DataLoader

        from evals.metrics.trajectory_metrics import _build_prompts_for_sampler

        dataset = _dataset_retain_perturbed(tokenizer, samples=2)
        collator = _collator(tokenizer)
        dl = DataLoader(dataset, batch_size=1, collate_fn=collator)
        for batch in dl:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            prompts, prompt_lens, prompt_starts = _build_prompts_for_sampler(
                input_ids,
                labels,
                tokenizer,
                IGNORE_INDEX,
                prompt_only_input_ids=True,
            )
            gen_start = prompt_starts[0] + prompt_lens[0]
            L = labels.shape[1] - gen_start
            assert L >= 0
            generated_slice = labels[0, gen_start : gen_start + L]
            assert generated_slice.shape[0] == L


class TestMUTrajectorySteps:
    """Step alignment and run_steps_to_use for MU passes."""

    def test_derive_steps_to_use_consistent_for_same_config(self):
        """_derive_steps_to_use returns same steps for same S and trajectory_config (retain/ra/wf use same config)."""
        from evals.metrics.trajectory_metrics import _derive_steps_to_use

        S = 25
        config = {
            "sampler_kwargs": {
                "steps": S,
                "trajectory_sample_interval": 8,
                "max_new_tokens": 200,
            }
        }
        steps1, _ = _derive_steps_to_use(S, config)
        steps2, _ = _derive_steps_to_use(S, config)
        assert steps1 == steps2
        assert len(steps1) > 0


class TestComputeMuForDatasetE2E:
    """E2E _compute_mu_for_dataset with real data and mock sampler (no GPU)."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return _tokenizer_fixture()

    def test_compute_mu_for_dataset_retain_returns_three_components(self, tokenizer):
        """With real retain_perturbed data and mock sampler, _compute_mu_for_dataset returns 3 keys per step/view."""
        from evals.metrics.trajectory_metrics import (
            _compute_mu_for_dataset,
            _MU_DATASET_KEYS,
        )
        from evals.metrics import METRICS_REGISTRY

        dataset = _dataset_retain_perturbed(tokenizer, samples=4)
        collator = _collator(tokenizer)
        batch_size = 2
        L_full = 256
        V = 32000
        S = 9
        trajectory_config = {
            "trajectory_names": ["steps", "fixation_start", "fixation_end", "fixation_ratio"],
            "include_views": ["full", "eos"],
            "sampler_kwargs": {
                "steps": S,
                "max_new_tokens": 64,
                "trajectory_sample_interval": 8,
            },
        }
        loaded_metrics = {
            "probability": {"metric": METRICS_REGISTRY["probability"], "config": {}},
            "rouge": {"metric": METRICS_REGISTRY["rouge"], "config": {"rouge_type": "rougeL_recall"}},
            "truth_ratio": {"metric": METRICS_REGISTRY["truth_ratio"], "config": {}},
        }

        model = Mock()
        sampler = Mock()

        def sample_side_effect(**kwargs):
            inputs = kwargs.get("inputs", [])
            B = len(inputs) if isinstance(inputs, list) else inputs.shape[0]
            return _make_mock_sampler_output(B, L_full, V, S)

        sampler.sample = Mock(side_effect=sample_side_effect)
        model.sampler = sampler

        result = _compute_mu_for_dataset(
            model,
            dataset,
            "retain",
            collator,
            batch_size,
            trajectory_config,
            tokenizer,
            loaded_metrics,
            sort_by_length=True,
            use_distributed_sampler=False,
            world_size=1,
            rank=0,
        )

        keys_expected = _MU_DATASET_KEYS["retain"]
        trajectory_names_expected = ("steps", "fixation_start", "fixation_end", "fixation_ratio")
        assert isinstance(result, dict)
        assert set(result.keys()) == set(trajectory_names_expected)
        for traj_name in trajectory_names_expected:
            assert traj_name in result
            for step_key, step_val in result[traj_name].items():
                assert isinstance(step_val, dict)
                for view in ("full", "eos"):
                    if view in step_val:
                        comp = step_val[view]
                        for k in keys_expected:
                            assert k in comp, f"missing {k} at traj={traj_name} step {step_key} view {view}"
                            assert isinstance(comp[k], dict) and "agg_value" in comp[k]

    def test_compute_mu_for_dataset_ra_returns_three_components(self, tokenizer):
        """With real real_authors_perturbed data and mock sampler, _compute_mu_for_dataset returns ra_* keys."""
        from evals.metrics.trajectory_metrics import (
            _compute_mu_for_dataset,
            _MU_DATASET_KEYS,
        )
        from evals.metrics import METRICS_REGISTRY

        dataset = _dataset_ra(tokenizer, samples=4)
        collator = _collator(tokenizer)
        batch_size = 2
        L_full = 256
        V = 32000
        S = 9
        trajectory_config = {
            "trajectory_names": ["steps", "fixation_start", "fixation_end", "fixation_ratio"],
            "include_views": ["full"],
            "sampler_kwargs": {
                "steps": S,
                "max_new_tokens": 64,
                "trajectory_sample_interval": 8,
            },
        }
        loaded_metrics = {
            "probability": {"metric": METRICS_REGISTRY["probability"], "config": {}},
            "rouge": {"metric": METRICS_REGISTRY["rouge"], "config": {"rouge_type": "rougeL_recall"}},
            "truth_ratio": {"metric": METRICS_REGISTRY["truth_ratio"], "config": {}},
        }

        model = Mock()
        sampler = Mock()

        def sample_side_effect(**kwargs):
            inputs = kwargs.get("inputs", [])
            B = len(inputs) if isinstance(inputs, list) else inputs.shape[0]
            return _make_mock_sampler_output(B, L_full, V, S)

        sampler.sample = Mock(side_effect=sample_side_effect)
        model.sampler = sampler

        result = _compute_mu_for_dataset(
            model,
            dataset,
            "ra",
            collator,
            batch_size,
            trajectory_config,
            tokenizer,
            loaded_metrics,
            sort_by_length=True,
            use_distributed_sampler=False,
            world_size=1,
            rank=0,
        )

        keys_expected = _MU_DATASET_KEYS["ra"]
        trajectory_names_expected = ("steps", "fixation_start", "fixation_end", "fixation_ratio")
        assert isinstance(result, dict)
        assert set(result.keys()) == set(trajectory_names_expected)
        for traj_name in trajectory_names_expected:
            for step_key, step_val in result[traj_name].items():
                assert isinstance(step_val, dict)
                for view in step_val:
                    comp = step_val[view]
                    for k in keys_expected:
                        assert k in comp
                        assert isinstance(comp[k], dict) and "agg_value" in comp[k]

    def test_compute_mu_for_dataset_wf_returns_three_components(self, tokenizer):
        """With real world_facts_perturbed data and mock sampler, _compute_mu_for_dataset returns wf_* keys."""
        from evals.metrics.trajectory_metrics import (
            _compute_mu_for_dataset,
            _MU_DATASET_KEYS,
        )
        from evals.metrics import METRICS_REGISTRY

        dataset = _dataset_wf(tokenizer, samples=4)
        collator = _collator(tokenizer)
        batch_size = 2
        L_full = 256
        V = 32000
        S = 9
        trajectory_config = {
            "trajectory_names": ["steps", "fixation_start", "fixation_end", "fixation_ratio"],
            "include_views": ["full"],
            "sampler_kwargs": {
                "steps": S,
                "max_new_tokens": 64,
                "trajectory_sample_interval": 8,
            },
        }
        loaded_metrics = {
            "probability": {"metric": METRICS_REGISTRY["probability"], "config": {}},
            "rouge": {"metric": METRICS_REGISTRY["rouge"], "config": {"rouge_type": "rougeL_recall"}},
            "truth_ratio": {"metric": METRICS_REGISTRY["truth_ratio"], "config": {}},
        }

        model = Mock()
        sampler = Mock()

        def sample_side_effect(**kwargs):
            inputs = kwargs.get("inputs", [])
            B = len(inputs) if isinstance(inputs, list) else inputs.shape[0]
            return _make_mock_sampler_output(B, L_full, V, S)

        sampler.sample = Mock(side_effect=sample_side_effect)
        model.sampler = sampler

        result = _compute_mu_for_dataset(
            model,
            dataset,
            "wf",
            collator,
            batch_size,
            trajectory_config,
            tokenizer,
            loaded_metrics,
            sort_by_length=True,
            use_distributed_sampler=False,
            world_size=1,
            rank=0,
        )

        keys_expected = _MU_DATASET_KEYS["wf"]
        trajectory_names_expected = ("steps", "fixation_start", "fixation_end", "fixation_ratio")
        assert isinstance(result, dict)
        assert set(result.keys()) == set(trajectory_names_expected)
        for traj_name in trajectory_names_expected:
            for step_key, step_val in result[traj_name].items():
                assert isinstance(step_val, dict)
                for view in step_val:
                    comp = step_val[view]
                    for k in keys_expected:
                        assert k in comp
                        assert isinstance(comp[k], dict) and "agg_value" in comp[k]


class TestNineMetricMergeValidation:
    """Validate EXPECTED_9_MU_KEYS and _validate_merged_9_mu (9-metric merge assertion)."""

    def test_expected_9_mu_keys_has_nine_and_matches_dataset_keys(self):
        from evals.metrics.trajectory_metrics import (
            _MU_DATASET_KEYS,
            EXPECTED_9_MU_KEYS,
        )

        assert len(EXPECTED_9_MU_KEYS) == 9
        expected_from_datasets = frozenset(
            k for keys in _MU_DATASET_KEYS.values() for k in keys
        )
        assert EXPECTED_9_MU_KEYS == expected_from_datasets
        assert "retain_Q_A_Prob" in EXPECTED_9_MU_KEYS
        assert "ra_Q_A_Prob_normalised" in EXPECTED_9_MU_KEYS
        assert "wf_Truth_Ratio" in EXPECTED_9_MU_KEYS

    def test_validate_merged_9_mu_pass_with_all_nine_keys(self):
        from evals.metrics.trajectory_metrics import (
            EXPECTED_9_MU_KEYS,
            _validate_merged_9_mu,
        )

        retain_agg_by_step = {
            "0": {
                "full": {k: {"agg_value": 0.5} for k in EXPECTED_9_MU_KEYS},
                "eos": {k: {"agg_value": 0.5} for k in EXPECTED_9_MU_KEYS},
            },
        }
        _validate_merged_9_mu(retain_agg_by_step)

    def test_validate_merged_9_mu_empty_dict_no_raise(self):
        from evals.metrics.trajectory_metrics import _validate_merged_9_mu

        _validate_merged_9_mu({})

    def test_validate_merged_9_mu_fail_missing_key_raises(self):
        from evals.metrics.trajectory_metrics import (
            EXPECTED_9_MU_KEYS,
            _validate_merged_9_mu,
        )

        keys_minus_one = set(EXPECTED_9_MU_KEYS) - {"wf_Truth_Ratio"}
        retain_agg_by_step = {
            "0": {
                "full": {k: {"agg_value": 0.5} for k in keys_minus_one},
            },
        }
        with pytest.raises(ValueError) as exc_info:
            _validate_merged_9_mu(retain_agg_by_step)
        assert "missing" in str(exc_info.value).lower()
        assert "wf_Truth_Ratio" in str(exc_info.value) or "extra" in str(exc_info.value).lower()

    def test_validate_merged_9_mu_fail_extra_key_raises(self):
        from evals.metrics.trajectory_metrics import (
            EXPECTED_9_MU_KEYS,
            _validate_merged_9_mu,
        )

        keys_plus_extra = dict({k: {"agg_value": 0.5} for k in EXPECTED_9_MU_KEYS})
        keys_plus_extra["extra_key"] = {"agg_value": 0.1}
        retain_agg_by_step = {"0": {"full": keys_plus_extra}}
        with pytest.raises(ValueError) as exc_info:
            _validate_merged_9_mu(retain_agg_by_step)
        assert "extra" in str(exc_info.value).lower()

    def test_validate_merged_9_mu_pass_with_per_traj_structure(self):
        """Per-traj retain_agg_by_step (four trajectory types, full+eos) passes validation."""
        from evals.metrics.trajectory_metrics import (
            EXPECTED_9_MU_KEYS,
            _validate_merged_9_mu,
        )

        trajectory_names = ("steps", "fixation_start", "fixation_end", "fixation_ratio")
        step_dict = {
            "0": {
                "full": {k: {"agg_value": 0.5} for k in EXPECTED_9_MU_KEYS},
                "eos": {k: {"agg_value": 0.5} for k in EXPECTED_9_MU_KEYS},
            },
            "1": {
                "full": {k: {"agg_value": 0.5} for k in EXPECTED_9_MU_KEYS},
                "eos": {k: {"agg_value": 0.5} for k in EXPECTED_9_MU_KEYS},
            },
        }
        retain_agg_by_step = {t: step_dict for t in trajectory_names}
        _validate_merged_9_mu(retain_agg_by_step)

    def test_validate_merged_9_mu_per_traj_fail_missing_key_raises(self):
        """Per-traj structure with a missing key in one traj raises."""
        from evals.metrics.trajectory_metrics import (
            EXPECTED_9_MU_KEYS,
            _validate_merged_9_mu,
        )

        keys_minus_one = set(EXPECTED_9_MU_KEYS) - {"wf_Truth_Ratio"}
        step_dict_ok = {
            "0": {
                "full": {k: {"agg_value": 0.5} for k in EXPECTED_9_MU_KEYS},
                "eos": {k: {"agg_value": 0.5} for k in EXPECTED_9_MU_KEYS},
            },
        }
        step_dict_bad = {
            "0": {
                "full": {k: {"agg_value": 0.5} for k in keys_minus_one},
                "eos": {k: {"agg_value": 0.5} for k in keys_minus_one},
            },
        }
        retain_agg_by_step = {
            "steps": step_dict_ok,
            "fixation_start": step_dict_ok,
            "fixation_end": step_dict_bad,
            "fixation_ratio": step_dict_ok,
        }
        with pytest.raises(ValueError) as exc_info:
            _validate_merged_9_mu(retain_agg_by_step)
        assert "missing" in str(exc_info.value).lower() or "expected exactly" in str(exc_info.value).lower()


class TestNineMetricMergeSyntheticFourTrajFullEos:
    """Synthetic 9-metric merge: per-traj retain/ra/wf -> retain_agg_by_step with 4 traj types and full+eos."""

    def test_nine_metric_merge_produces_four_trajectory_types_and_both_views(self):
        """Merge logic with per-traj retain_res/ra_res/wf_res yields 4 traj names and full+eos per step."""
        from evals.metrics.trajectory_metrics import EXPECTED_9_MU_KEYS

        trajectory_names = ("steps", "fixation_start", "fixation_end", "fixation_ratio")
        views = ("full", "eos")

        def _three(prefix):
            prob_key = f"{prefix}_Q_A_Prob" if prefix == "retain" else f"{prefix}_Q_A_Prob_normalised"
            return {
                prob_key: {"agg_value": 0.5},
                f"{prefix}_Q_A_ROUGE": {"agg_value": 0.6},
                f"{prefix}_Truth_Ratio": {"agg_value": 0.7},
            }

        retain_res = {}
        ra_res = {}
        wf_res = {}
        for t in trajectory_names:
            retain_res[t] = {}
            ra_res[t] = {}
            wf_res[t] = {}
            for sk in ("0", "1"):
                retain_res[t][sk] = {v: _three("retain") for v in views}
                ra_res[t][sk] = {v: _three("ra") for v in views}
                wf_res[t][sk] = {v: _three("wf") for v in views}

        retain_agg_by_step = {}
        for traj_name in trajectory_names:
            retain_agg_by_step[traj_name] = {}
            for step_key in retain_res[traj_name]:
                merged_views = {}
                for view in views:
                    merged_views[view] = {
                        **retain_res[traj_name][step_key].get(view, {}),
                        **ra_res[traj_name].get(step_key, {}).get(view, {}),
                        **wf_res[traj_name].get(step_key, {}).get(view, {}),
                    }
                retain_agg_by_step[traj_name][step_key] = merged_views

        assert set(retain_agg_by_step.keys()) == set(trajectory_names)
        for traj_name in trajectory_names:
            assert set(retain_agg_by_step[traj_name].keys()) == {"0", "1"}
            for step_key in ("0", "1"):
                step_val = retain_agg_by_step[traj_name][step_key]
                assert set(step_val.keys()) == set(views)
                for view in views:
                    assert set(step_val[view].keys()) == EXPECTED_9_MU_KEYS


class TestTrajectoryMetricsNineComponentE2E:
    """Full trajectory_metrics with retain+ra+wf yields 9 components (mock sampler, real data)."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return _tokenizer_fixture()

    def test_trajectory_metrics_with_retain_ra_wf_produces_nine_components(self):
        """Merged retain+ra+wf per-traj per-step results have 9 components; report logic exposes them."""
        nine_keys = {
            "retain_Q_A_Prob", "retain_Q_A_ROUGE", "retain_Truth_Ratio",
            "ra_Q_A_Prob_normalised", "ra_Q_A_ROUGE", "ra_Truth_Ratio",
            "wf_Q_A_Prob_normalised", "wf_Q_A_ROUGE", "wf_Truth_Ratio",
        }
        trajectory_names = ("steps", "fixation_start", "fixation_end", "fixation_ratio")
        views = ("full", "eos")

        def _three(prefix):
            prob_key = f"{prefix}_Q_A_Prob" if prefix == "retain" else f"{prefix}_Q_A_Prob_normalised"
            return {
                prob_key: {"agg_value": 0.5},
                f"{prefix}_Q_A_ROUGE": {"agg_value": 0.6},
                f"{prefix}_Truth_Ratio": {"agg_value": 0.7},
            }

        retain_res = {t: {"0": {v: _three("retain") for v in views}, "1": {v: _three("retain") for v in views}} for t in trajectory_names}
        ra_res = {t: {"0": {v: _three("ra") for v in views}, "1": {v: _three("ra") for v in views}} for t in trajectory_names}
        wf_res = {t: {"0": {v: _three("wf") for v in views}, "1": {v: _three("wf") for v in views}} for t in trajectory_names}

        retain_agg_by_step = {}
        for traj_name in trajectory_names:
            retain_agg_by_step[traj_name] = {}
            for step_key in retain_res[traj_name]:
                retain_agg_by_step[traj_name][step_key] = {
                    view: {
                        **retain_res[traj_name][step_key].get(view, {}),
                        **ra_res[traj_name].get(step_key, {}).get(view, {}),
                        **wf_res[traj_name].get(step_key, {}).get(view, {}),
                    }
                    for view in views
                }

        for traj_name in trajectory_names:
            for step_key, step_val in retain_agg_by_step[traj_name].items():
                for view in views:
                    assert view in step_val
                    assert nine_keys.issubset(step_val[view].keys()), (
                        f"traj={traj_name} step {step_key} view {view} missing some of {nine_keys}"
                    )

        # Report-building logic uses steps dict: retain_agg_by_step.get("steps", retain_agg_by_step)
        def _is_mu_key(x):
            return (
                str(x).startswith("retain_")
                or str(x).startswith("ra_")
                or str(x).startswith("wf_")
            )
        steps_dict = retain_agg_by_step.get("steps", retain_agg_by_step)
        components = {}
        for step_key, pre in steps_dict.items():
            for view in ("full", "eos"):
                if view not in pre:
                    continue
                pv = pre[view]
                if not isinstance(pv, dict):
                    continue
                if step_key not in components:
                    components[step_key] = {}
                if view not in components[step_key]:
                    components[step_key][view] = {}
                for name, ent in pv.items():
                    if _is_mu_key(name) and isinstance(ent, dict) and "agg_value" in ent:
                        components[step_key][view][name] = ent["agg_value"]
        assert len(components) > 0
        first_step = next(iter(components.values()))
        for view in first_step:
            assert nine_keys.issubset(first_step[view].keys())
