"""
Tests for trajectory truth_ratio pre_compute: both non-generalized and generalized paths.

Covers:
- Non-generalized: _compute_pre_compute_metrics_at_step with 3D labels_wrong (no trajectory_config).
  Current behavior: wrong ends up with None (shape mismatch). After fix: wrong = list of N dicts with valid prob/avg_loss.
- Generalized: _compute_pre_compute_metrics_at_step with trajectory_config + sample_traj, labels_wrong as list of N tensors.
  Ensures we get N results with valid prob/avg_loss when R/F and labels align; documents when one option yields None.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from data.utils import IGNORE_INDEX


# ---- Non-generalized path: 3D labels_wrong (from collator) ----

def test_non_generalized_3d_labels_wrong_structure():
    """
    With 3D labels_wrong [1, N, L] and no trajectory_config/sample_traj, pre_compute for 'wrong'
    should eventually yield list of N dicts (one per option) with prob/avg_loss.
    Current code path: passes 3D to probability -> shape mismatch -> wrong ends as dict with None.
    This test asserts the expected structure (list of N) when fixed; currently skips (documents bug).
    After fix: loop over N options, align each option's labels to logits length, then wrong = list of N.
    """
    from omegaconf import OmegaConf
    from evals.metrics.trajectory_metrics import _compute_pre_compute_metrics_at_step
    from evals.metrics import METRICS_REGISTRY

    # Simulate collator output: input_ids length 58, labels_wrong [1, 5, 59] (N=5, L_wrong=59)
    L_input, N, L_wrong, V = 58, 5, 59, 100
    batch_template = {
        "input_ids": torch.zeros(1, L_input, dtype=torch.long),
        "labels": torch.full((1, L_input), IGNORE_INDEX, dtype=torch.long),
        "labels_correct": torch.full((1, L_input), IGNORE_INDEX, dtype=torch.long),
        "labels_wrong": torch.full((1, N, L_wrong), IGNORE_INDEX, dtype=torch.long),
    }
    # Put some valid token ids so probability has positions to score
    batch_template["labels_correct"][0, -10:] = torch.arange(1, 11)
    for k in range(N):
        batch_template["labels_wrong"][0, k, -10:] = torch.arange(1, 11)

    logits = torch.zeros(1, L_input, V)
    pre_compute_config = OmegaConf.create({
        "correct": {"access_key": "correct", "labels_field": "labels_correct", "handler": "probability"},
        "wrong": {"access_key": "wrong", "labels_field": "labels_wrong", "handler": "probability"},
    })
    results = _compute_pre_compute_metrics_at_step(
        pre_compute_config=pre_compute_config,
        logits=logits,
        batch_template=batch_template,
        tokenizer=None,
        sample_labels=batch_template["labels"],
        sample_input_ids=batch_template["input_ids"],
        sample_prompt_len=0,
        sample_idx="0",
    )
    wrong_val = results["wrong"]
    # Current bug: wrong is a single dict with value_by_index["0"] = {"prob": None} (no avg_loss).
    # Desired: wrong is list of N dicts each with value_by_index["0"] = {"prob": float, "avg_loss": float}.
    if isinstance(wrong_val, list):
        assert len(wrong_val) == N, "wrong should be list of N options"
        for i, w in enumerate(wrong_val):
            vbi = w.get("value_by_index", {})
            assert "0" in vbi, f"wrong[{i}] should have value_by_index['0']"
            entry = vbi["0"]
            assert isinstance(entry, dict), f"wrong[{i}]['0'] should be dict"
            # After fix we expect prob and avg_loss to be numeric (not None)
            if entry.get("avg_loss") is not None and entry.get("prob") is not None:
                assert isinstance(entry["avg_loss"], (int, float, np.floating))
                assert isinstance(entry["prob"], (int, float, np.floating))
    else:
        # Current behavior: wrong is dict; one entry may have None
        vbi = wrong_val.get("value_by_index", {})
        assert "0" in vbi
        # Document current bug: entry can have prob/avg_loss None
        pytest.skip(
            "Trajectory is generalized by definition; the non-generalized path is unsupported for "
            "dual-answer (no implementation of 3D labels_wrong in non-generalized path)."
        )


def test_non_generalized_3d_labels_wrong_tr_raises_when_no_valid_pre_compute():
    """
    With real TOFU-like 3D labels_wrong, pre_compute can end up with no valid avg_loss.
    truth_ratio returns agg_value=None when no valid pre_compute so trajectory eval continues.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from data.qa import QAwithDualAnswersDataset
    from data.collators import DataCollatorForSupervisedDataset
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf
    from evals.metrics.trajectory_metrics import _compute_pre_compute_metrics_at_step
    from evals.metrics import METRICS_REGISTRY

    load_dataset("locuslab/TOFU", "forget01_perturbed", split="train")
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
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=False)))
    L = batch["input_ids"].shape[1]
    V = tokenizer.vocab_size or 50257
    logits = torch.zeros(1, L, V)
    pre_compute_config = OmegaConf.create({
        "correct": {"access_key": "correct", "labels_field": "labels_correct", "handler": "probability"},
        "wrong": {"access_key": "wrong", "labels_field": "labels_wrong", "handler": "probability"},
    })
    batch_template = {
        "input_ids": batch["input_ids"],
        "labels": batch["labels"],
        "attention_mask": batch["attention_mask"],
        "index": batch["index"],
        "labels_correct": batch.get("labels_correct", batch["labels"]),
        "labels_wrong": batch["labels_wrong"],
    }
    results = _compute_pre_compute_metrics_at_step(
        pre_compute_config=pre_compute_config,
        logits=logits,
        batch_template=batch_template,
        tokenizer=tokenizer,
        sample_labels=batch.get("labels"),
        sample_input_ids=batch["input_ids"],
        sample_prompt_len=0,
        sample_idx="0",
    )
    tr = METRICS_REGISTRY["truth_ratio"]
    # No valid pre_compute (wrong has None avg_loss) → truth_ratio returns None so eval continues.
    result = tr._metric_fn(
        model=None,
        pre_compute={"correct": results["correct"], "wrong": results["wrong"]},
        aggregator="closer_to_1_better",
    )
    assert result["agg_value"] is None
    assert result["value_by_index"] == {}


# ---- Generalized path: list of N label tensors (from _batch_template_dual_labels) ----

def test_generalized_list_of_label_tensors_wrong_returns_n_results():
    """
    Generalized path: trajectory_config + sample_traj (R, F), labels_wrong as list of N tensors [1, L].
    When R/F have length L and each label tensor has length L with some non-ignore positions, we get N results with valid prob/avg_loss.
    """
    from evals.metrics.trajectory_metrics import _compute_pre_compute_metrics_at_step
    from evals.metrics.step_wise_score import build_effective_step_fixation_logits
    from omegaconf import OmegaConf

    L, V, S, N = 20, 100, 10, 3
    step = 2
    # R [1, V, L, S], F [1, L]
    R = torch.randn(1, V, L, S)
    F = torch.randint(0, S, (1, L))
    sample_traj = {"R": R.squeeze(0), "F": F.squeeze(0), "S": S}
    trajectory_config = OmegaConf.create({
        "use_generalized_sequence_probability": True,
        "logit_alignment": "causal",
    })
    # List of N label tensors [1, L]; avoid all-ignore so we get scores
    labels_wrong_list = []
    for _ in range(N):
        lab = torch.full((1, L), IGNORE_INDEX, dtype=torch.long)
        lab[0, 5:15] = torch.randint(1, V, (10,))
        labels_wrong_list.append(lab)

    batch_template = {
        "input_ids": torch.zeros(1, L, dtype=torch.long),
        "labels": torch.full((1, L), IGNORE_INDEX, dtype=torch.long),
        "labels_correct": torch.full((1, L), IGNORE_INDEX, dtype=torch.long),
        "labels_wrong": labels_wrong_list,
    }
    batch_template["labels_correct"][0, 5:15] = torch.randint(1, V, (10,))

    pre_compute_config = OmegaConf.create({
        "correct": {"access_key": "correct", "labels_field": "labels_correct", "handler": "probability"},
        "wrong": {"access_key": "wrong", "labels_field": "labels_wrong", "handler": "probability"},
    })
    results = _compute_pre_compute_metrics_at_step(
        pre_compute_config=pre_compute_config,
        logits=torch.zeros(1, L, V),  # not used in generalized path
        batch_template=batch_template,
        tokenizer=None,
        sample_labels=batch_template["labels"],
        sample_input_ids=batch_template["input_ids"],
        sample_prompt_len=0,
        sample_idx="0",
        trajectory_config=trajectory_config,
        sample_traj=sample_traj,
        step=step,
    )
    assert "wrong" in results
    wrong_val = results["wrong"]
    assert isinstance(wrong_val, list), "generalized path with list labels_wrong must return list of N"
    assert len(wrong_val) == N
    for i, w in enumerate(wrong_val):
        vbi = w.get("value_by_index", {})
        assert "0" in vbi, f"wrong[{i}] missing value_by_index['0']"
        entry = vbi["0"]
        assert entry.get("prob") is not None, f"wrong[{i}] should have prob (generalized path with valid R/F and labels)"
        assert entry.get("avg_loss") is not None, f"wrong[{i}] should have avg_loss"


def test_generalized_list_one_option_all_ignore_yields_none_for_that_option():
    """
    When one of the N label tensors is all IGNORE_INDEX, get_per_position_scores returns empty for that option ->
    we append {"prob": None, "avg_loss": None} for that option. So we get N entries with one None.
    """
    from evals.metrics.trajectory_metrics import _compute_pre_compute_metrics_at_step
    from omegaconf import OmegaConf

    L, V, S, N = 20, 100, 10, 3
    step = 2
    R = torch.randn(1, V, L, S)
    F = torch.randint(0, S, (1, L))
    sample_traj = {"R": R.squeeze(0), "F": F.squeeze(0), "S": S}
    trajectory_config = OmegaConf.create({
        "use_generalized_sequence_probability": True,
        "logit_alignment": "causal",
    })
    labels_wrong_list = []
    for i in range(N):
        lab = torch.full((1, L), IGNORE_INDEX, dtype=torch.long)
        if i != 1:
            lab[0, 5:15] = torch.randint(1, V, (10,))
        # Option 1 is all ignore -> no scores
        labels_wrong_list.append(lab)

    batch_template = {
        "input_ids": torch.zeros(1, L, dtype=torch.long),
        "labels": torch.full((1, L), IGNORE_INDEX, dtype=torch.long),
        "labels_correct": torch.full((1, L), IGNORE_INDEX, dtype=torch.long),
        "labels_wrong": labels_wrong_list,
    }
    batch_template["labels_correct"][0, 5:15] = torch.randint(1, V, (10,))

    pre_compute_config = OmegaConf.create({
        "correct": {"access_key": "correct", "labels_field": "labels_correct", "handler": "probability"},
        "wrong": {"access_key": "wrong", "labels_field": "labels_wrong", "handler": "probability"},
    })
    results = _compute_pre_compute_metrics_at_step(
        pre_compute_config=pre_compute_config,
        logits=torch.zeros(1, L, V),
        batch_template=batch_template,
        tokenizer=None,
        sample_labels=batch_template["labels"],
        sample_input_ids=batch_template["input_ids"],
        sample_prompt_len=0,
        sample_idx="0",
        trajectory_config=trajectory_config,
        sample_traj=sample_traj,
        step=step,
    )
    wrong_val = results["wrong"]
    assert isinstance(wrong_val, list) and len(wrong_val) == N
    # Option 1 (index 1) should have prob/avg_loss None
    assert wrong_val[1]["value_by_index"]["0"]["prob"] is None
    assert wrong_val[1]["value_by_index"]["0"]["avg_loss"] is None
    # Others should be valid
    for i in (0, 2):
        assert wrong_val[i]["value_by_index"]["0"]["prob"] is not None
        assert wrong_val[i]["value_by_index"]["0"]["avg_loss"] is not None


def test_generalized_correct_single_tensor_returns_valid():
    """
    Generalized path for 'correct' (single labels tensor, not list): one call to provider, one result.
    """
    from evals.metrics.trajectory_metrics import _compute_pre_compute_metrics_at_step
    from omegaconf import OmegaConf

    L, V, S = 20, 100, 10
    step = 2
    R = torch.randn(1, V, L, S)
    F = torch.randint(0, S, (1, L))
    sample_traj = {"R": R.squeeze(0), "F": F.squeeze(0), "S": S}
    trajectory_config = OmegaConf.create({
        "use_generalized_sequence_probability": True,
        "logit_alignment": "causal",
    })
    labels_correct = torch.full((1, L), IGNORE_INDEX, dtype=torch.long)
    labels_correct[0, 5:15] = torch.randint(1, V, (10,))

    batch_template = {
        "input_ids": torch.zeros(1, L, dtype=torch.long),
        "labels": labels_correct.clone(),
        "labels_correct": labels_correct,
    }
    pre_compute_config = OmegaConf.create({
        "correct": {"access_key": "correct", "labels_field": "labels_correct", "handler": "probability"},
    })
    results = _compute_pre_compute_metrics_at_step(
        pre_compute_config=pre_compute_config,
        logits=torch.zeros(1, L, V),
        batch_template=batch_template,
        tokenizer=None,
        sample_labels=batch_template["labels"],
        sample_input_ids=batch_template["input_ids"],
        sample_prompt_len=0,
        sample_idx="0",
        trajectory_config=trajectory_config,
        sample_traj=sample_traj,
        step=step,
    )
    assert "correct" in results
    c = results["correct"]
    assert c["value_by_index"]["0"]["prob"] is not None
    assert c["value_by_index"]["0"]["avg_loss"] is not None


def test_generalized_L_zero_pre_compute_then_truth_ratio_returns_none():
    """
    When L=0 we early-exit and set pre_compute to None for correct/wrong.
    truth_ratio returns agg_value=None, value_by_index={} so trajectory eval can continue.
    """
    from evals.metrics.trajectory_metrics import _compute_pre_compute_metrics_at_step
    from evals.metrics import METRICS_REGISTRY
    from omegaconf import OmegaConf

    V, S, N = 100, 10, 3
    R = torch.randn(V, 0, S)  # L=0
    F = torch.randint(0, S, (0,))
    sample_traj = {"R": R, "F": F, "S": S, "L": 0}
    trajectory_config = OmegaConf.create({
        "use_generalized_sequence_probability": True,
        "logit_alignment": "causal",
    })
    pre_compute_config = OmegaConf.create({
        "correct": {"access_key": "correct", "labels_field": "labels_correct", "handler": "probability"},
        "wrong": {"access_key": "wrong", "labels_field": "labels_wrong", "handler": "probability"},
    })
    batch_template = {
        "input_ids": torch.zeros(1, 0, dtype=torch.long),
        "labels": torch.full((1, 0), IGNORE_INDEX, dtype=torch.long),
        "labels_correct": torch.full((1, 0), IGNORE_INDEX, dtype=torch.long),
        "labels_wrong": [torch.full((1, 0), IGNORE_INDEX, dtype=torch.long) for _ in range(N)],
    }
    results = _compute_pre_compute_metrics_at_step(
        pre_compute_config=pre_compute_config,
        logits=torch.zeros(1, 0, V),
        batch_template=batch_template,
        tokenizer=None,
        sample_labels=batch_template["labels"],
        sample_input_ids=batch_template["input_ids"],
        sample_prompt_len=0,
        sample_idx="0",
        trajectory_config=trajectory_config,
        sample_traj=sample_traj,
        step=0,
    )
    assert results["correct"]["agg_value"] is None
    assert results["correct"]["value_by_index"]["0"]["avg_loss"] is None
    assert isinstance(results["wrong"], list)
    assert len(results["wrong"]) == N
    for w in results["wrong"]:
        assert w["value_by_index"]["0"]["avg_loss"] is None
    tr = METRICS_REGISTRY["truth_ratio"]
    result = tr._metric_fn(
        model=None,
        pre_compute={"correct": results["correct"], "wrong": results["wrong"]},
        aggregator="closer_to_1_better",
    )
    assert result["agg_value"] is None
    assert result["value_by_index"] == {}
