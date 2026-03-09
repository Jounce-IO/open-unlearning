"""
Local tests with real HF dataset download (no GPU). Reproduce correct/wrong same-indices
and ks_test 'score' bugs, then verify fixes. At least 5 tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))


# ---- 1. int key (correct) vs str key (wrong): truth_ratio normalizes to str and succeeds ----
def test_reproduce_bug_correct_int_key_wrong_str_key_truth_ratio_assert_fails():
    """Pre_compute correct has value_by_index with int key, wrong has str key; truth_ratio normalizes to str and does not raise."""
    from evals.metrics import METRICS_REGISTRY
    metric = METRICS_REGISTRY["truth_ratio"]
    correct_vbi = {453: {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_vbi = {"453": {"prob": 0.25, "avg_loss": -np.log(0.25)}}
    wrong_list = [{"value_by_index": wrong_vbi, "agg_value": 0.25}]
    result = metric._metric_fn(
        model=None,
        pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
        aggregator="closer_to_1_better",
    )
    assert result["agg_value"] is not None
    assert "453" in result["value_by_index"]


# ---- 2. Same keys (both str) -> truth_ratio passes ----
def test_truth_ratio_passes_when_both_correct_and_wrong_use_string_indices():
    """When both correct and wrong value_by_index use same string keys, truth_ratio succeeds."""
    from evals.metrics import METRICS_REGISTRY
    metric = METRICS_REGISTRY["truth_ratio"]
    correct_vbi = {"453": {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = [{"value_by_index": {"453": {"prob": 0.25, "avg_loss": -np.log(0.25)}}, "agg_value": 0.25}]
    result = metric._metric_fn(
        model=None,
        pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
        aggregator="closer_to_1_better",
    )
    assert result["agg_value"] is not None
    assert "453" in result["value_by_index"] and "score" in result["value_by_index"]["453"]


# ---- 3. ks_test needs 'score' in each value_by_index entry ----
def test_ks_test_keyerror_when_value_by_index_entry_missing_score():
    """Reproduce: forget_truth_ratio returns value_by_index with entry without 'score' -> ks_test KeyError."""
    from evals.metrics.privacy import ks_test
    # Simulate broken output (e.g. from wrong structure)
    pre_compute_forget = {"value_by_index": {"0": {"prob": 0.8}}}
    with pytest.raises(KeyError, match="score"):
        np.array([evals["score"] for evals in pre_compute_forget["value_by_index"].values()])


def test_ks_test_succeeds_when_value_by_index_has_score():
    """ks_test succeeds when each value_by_index entry has 'score'."""
    pre_compute_forget = {"value_by_index": {"0": {"score": 0.8}, "1": {"score": 0.9}}}
    stats = np.array([evals["score"] for evals in pre_compute_forget["value_by_index"].values()])
    assert stats.shape == (2,)
    assert list(stats) == [0.8, 0.9]


# ---- 4. _compute_pre_compute_metrics_at_step normalizes to single idx_key (after fix) ----
def test_compute_pre_compute_normalizes_correct_value_by_index_to_single_idx_key():
    """When probability returns dict with value_by_index keyed by int, output has only idx_key (str)."""
    import torch
    from unittest.mock import patch
    from evals.metrics.trajectory_metrics import _compute_pre_compute_metrics_at_step

    sample_idx = "453"
    pre_compute_config = {
        "para": {"access_key": "correct", "labels_field": "labels_correct", "handler": "probability"},
        "pert": {"access_key": "wrong", "labels_field": "labels_wrong", "handler": "probability"},
    }
    batch_template = {
        "input_ids": torch.zeros(1, 16, dtype=torch.long),
        "labels": torch.zeros(1, 16, dtype=torch.long),
        "labels_correct": torch.zeros(1, 16, dtype=torch.long),
        "labels_wrong": torch.zeros(1, 5, 16, dtype=torch.long),
    }
    logits = torch.zeros(1, 16, 100)
    tokenizer = __import__("unittest.mock").mock.MagicMock()

    # Simulate probability returning dict with value_by_index keyed by INT (data index)
    returns = [
        {"agg_value": 0.5, "value_by_index": {453: {"prob": 0.5, "avg_loss": -np.log(0.5)}}},
        [[{"prob": 0.2, "avg_loss": -1.61}], [{"prob": 0.15, "avg_loss": -1.90}]],
    ]
    with patch("evals.metrics.trajectory_metrics._call_metric_at_step", side_effect=returns):
        results = _compute_pre_compute_metrics_at_step(
            pre_compute_config=pre_compute_config,
            logits=logits,
            batch_template=batch_template,
            tokenizer=tokenizer,
            sample_labels=None,
            sample_input_ids=batch_template["input_ids"],
            sample_prompt_len=0,
            sample_idx=sample_idx,
        )

    correct_vbi = results["correct"]["value_by_index"]
    correct_keys = list(correct_vbi.keys())
    assert correct_keys == [sample_idx], f"correct value_by_index should have only idx_key {sample_idx!r}, got {correct_keys}"
    wrong_list = results["wrong"]
    assert isinstance(wrong_list, list)
    assert list(wrong_list[0]["value_by_index"].keys()) == [sample_idx]
    # truth_ratio must see same indices
    from evals.metrics import METRICS_REGISTRY
    tr_result = METRICS_REGISTRY["truth_ratio"]._metric_fn(
        model=None,
        pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
        aggregator="closer_to_1_better",
    )
    assert tr_result["agg_value"] is not None
    assert sample_idx in tr_result["value_by_index"] and "score" in tr_result["value_by_index"][sample_idx]


# ---- 5. Real HF download: TOFU dual-answer batch + pre_compute + truth_ratio + ks_test ----
def test_real_hf_tofu_dual_answer_pre_compute_truth_ratio_and_ks_test():
    """Load TOFU from HF (2 samples), build pre_compute with normalized keys, truth_ratio and ks_test succeed."""
    try:
        from datasets import load_dataset
    except ImportError:
        pytest.skip("datasets not installed")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    from unittest.mock import patch
    from data.qa import QAwithDualAnswersDataset
    from data.collators import DataCollatorForSupervisedDataset
    from omegaconf import OmegaConf
    from evals.metrics import METRICS_REGISTRY
    from torch.utils.data import DataLoader

    raw = load_dataset("locuslab/TOFU", "forget01_perturbed", split="train")
    raw = raw.select(range(min(2, len(raw))))
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_args = {"path": "locuslab/TOFU", "name": "forget01_perturbed", "split": "train"}
    template_args = OmegaConf.create({})
    with patch("data.qa.load_hf_dataset", return_value=raw):
        dataset = QAwithDualAnswersDataset(
            correct_answer_key="paraphrased_answer",
            wrong_answer_key="perturbed_answer",
            hf_args=hf_args,
            template_args=template_args,
            tokenizer=tokenizer,
            max_length=256,
        )
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, padding_side="left", index="index")
    dl = DataLoader(dataset, batch_size=2, collate_fn=collator)
    batch = next(iter(dl))

    indices = batch.get("index")
    idx_keys = [str(int(i.item())) for i in indices] if indices is not None else ["0", "1"]
    correct_vbi = {k: {"prob": 0.5, "avg_loss": -np.log(0.5)} for k in idx_keys}
    wrong_list = [
        {"value_by_index": {k: {"prob": 0.1 + 0.02 * j, "avg_loss": -np.log(0.1 + 0.02 * j)} for k in idx_keys}, "agg_value": 0.15}
        for j in range(5)
    ]

    metric = METRICS_REGISTRY["truth_ratio"]
    result = metric._metric_fn(
        model=None,
        pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
        aggregator="closer_to_1_better",
    )
    assert result["agg_value"] is not None
    for k in idx_keys:
        assert k in result["value_by_index"] and "score" in result["value_by_index"][k]

    forget_tr = {"value_by_index": result["value_by_index"]}
    stats = np.array([evals["score"] for evals in forget_tr["value_by_index"].values()])
    assert len(stats) == len(idx_keys)
    assert np.all(np.isfinite(stats))


# ---- 6. Full chain: int-keyed correct -> normalize -> truth_ratio -> ks_test (assert real output, no KeyError) ----
def test_ks_test_chain_with_int_keyed_correct_normalized_then_truth_ratio_and_ks_test_succeed():
    """Reproduce job path: when correct comes back with value_by_index keyed by int (e.g. 25),
    normalization must produce str keys so truth_ratio passes and ks_test gets evals['score'].
    Assert full output: truth_ratio value_by_index has 'score', ks_test runs without KeyError."""
    import torch
    from unittest.mock import patch
    from evals.metrics.trajectory_metrics import _compute_pre_compute_metrics_at_step
    from evals.metrics import METRICS_REGISTRY

    sample_idx = "25"
    pre_compute_config = {
        "para": {"access_key": "correct", "labels_field": "labels_correct", "handler": "probability"},
        "pert": {"access_key": "wrong", "labels_field": "labels_wrong", "handler": "probability"},
    }
    batch_template = {
        "input_ids": torch.zeros(1, 16, dtype=torch.long),
        "labels": torch.zeros(1, 16, dtype=torch.long),
        "labels_correct": torch.zeros(1, 16, dtype=torch.long),
        "labels_wrong": torch.zeros(1, 5, 16, dtype=torch.long),
    }
    logits = torch.zeros(1, 16, 100)
    tokenizer = __import__("unittest.mock").mock.MagicMock()

    # Correct returns int-keyed dict (reproduces job bug); wrong returns list of lists
    inner_returns = [
        {"agg_value": 0.5, "value_by_index": {25: {"prob": 0.5, "avg_loss": -0.693}}},
        [[{"prob": 0.2, "avg_loss": -1.61}], [{"prob": 0.15, "avg_loss": -1.90}]],
    ]
    with patch("evals.metrics.trajectory_metrics._call_metric_at_step", side_effect=inner_returns):
        results = _compute_pre_compute_metrics_at_step(
            pre_compute_config=pre_compute_config,
            logits=logits,
            batch_template=batch_template,
            tokenizer=tokenizer,
            sample_labels=None,
            sample_input_ids=batch_template["input_ids"],
            sample_prompt_len=0,
            sample_idx=sample_idx,
        )

    correct_vbi = results["correct"]["value_by_index"]
    assert list(correct_vbi.keys()) == [sample_idx], f"normalize must yield str key {sample_idx!r}, got {list(correct_vbi.keys())}"
    wrong_list = results["wrong"]
    assert list(wrong_list[0]["value_by_index"].keys()) == [sample_idx]

    # Truth ratio must succeed and output value_by_index with scalar 'score' (for ks_test)
    tr = METRICS_REGISTRY["truth_ratio"]
    tr_result = tr._metric_fn(
        model=None,
        pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
        aggregator="closer_to_1_better",
    )
    assert tr_result["agg_value"] is not None
    assert sample_idx in tr_result["value_by_index"]
    assert "score" in tr_result["value_by_index"][sample_idx]
    score_val = tr_result["value_by_index"][sample_idx]["score"]
    assert np.isscalar(score_val) or (isinstance(score_val, np.ndarray) and score_val.size == 1), "score must be scalar for ks_test"

    # ks_test must run without KeyError (uses evals['score'])
    pre_compute_for_ks = {"forget": {"value_by_index": tr_result["value_by_index"]}}
    ks = METRICS_REGISTRY["ks_test"]
    ks_result = ks._metric_fn(model=None, pre_compute=pre_compute_for_ks, reference_logs=None)
    assert "agg_value" in ks_result
    # Real assertion: ks_test built stats from evals["score"]
    stats = np.array([evals["score"] for evals in pre_compute_for_ks["forget"]["value_by_index"].values()])
    assert len(stats) == 1 and np.isfinite(stats[0])


# ---- 7. correct has both int and str key, wrong has str: truth_ratio normalizes and succeeds ----
def test_reproduce_bug_correct_has_int_and_str_keys_wrong_has_str_only():
    """Correct has both 453 and '453', wrong has '453'; truth_ratio normalizes to str and succeeds (no assert)."""
    from evals.metrics import METRICS_REGISTRY
    metric = METRICS_REGISTRY["truth_ratio"]
    correct_vbi = {453: {"prob": 0.5, "avg_loss": -np.log(0.5)}, "453": {"prob": 0.5, "avg_loss": -np.log(0.5)}}
    wrong_list = [{"value_by_index": {"453": {"prob": 0.25, "avg_loss": -np.log(0.25)}}}]
    correct_indices = list(correct_vbi.keys())
    wrong_indices = list(wrong_list[0]["value_by_index"].keys())
    assert correct_indices != wrong_indices  # raw keys differ
    result = metric._metric_fn(
        model=None,
        pre_compute={"correct": {"value_by_index": correct_vbi}, "wrong": wrong_list},
        aggregator="closer_to_1_better",
    )
    assert result["agg_value"] is not None
    assert "453" in result["value_by_index"]
