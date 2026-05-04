"""
Tests for eval hotspot optimizations (avg_losses, tokenizer, rouge).

Locks current behavior and asserts optimized behavior for:
- evaluate_probability: structure and values (prob, avg_loss per sample)
- _compute_prob_from_fixation_logits: same format and values
- eval_rouge_recall_batch: same scores for fixed (gen, gt) pairs
- eval_text_similarity: keys and value shapes, batch_decode usage
"""

import torch
from unittest.mock import Mock

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.utils import (
    evaluate_probability,
    _batch_to_device,
    _tensor_to_list_of_floats,
    eval_rouge_recall_batch,
    eval_rouge_recall_batch_worker_multi_steps,
    eval_text_similarity,
)
from evals.metrics.step_wise_score import compute_prob_packed_shifted_segments
from evals.metrics.trajectory_metrics import _compute_prob_from_fixation_logits


class TestTensorToListOfFloats:
    """Tests for _tensor_to_list_of_floats helper (single GPU→CPU transfer)."""

    def test_1d_tensor_returns_list_of_floats(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        out = _tensor_to_list_of_floats(t)
        assert out == [1.0, 2.0, 3.0]

    def test_float32_preserved(self):
        t = torch.tensor([0.5, 0.25], dtype=torch.float32)
        out = _tensor_to_list_of_floats(t)
        assert out == [0.5, 0.25]

    def test_bfloat16_converted(self):
        t = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        out = _tensor_to_list_of_floats(t)
        assert out == [1.0, 2.0]


class TestEvaluateProbabilityStructureAndValues:
    """Test evaluate_probability returns same structure and values for a fixed batch."""

    def test_returns_list_of_dicts_with_prob_and_avg_loss(self):
        batch_size = 2
        seq_len = 5
        vocab_size = 10
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[:, 0] = -100  # IGNORE_INDEX for first token
        model = Mock()
        model.device = torch.device("cpu")
        # output.logits shape (B, seq_len, V) for logits[..., :-1, :] and labels[..., 1:]
        model.return_value = Mock(logits=torch.randn(batch_size, seq_len, vocab_size))
        model.side_effect = lambda **kw: model.return_value
        batch = {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "labels": labels,
            "attention_mask": torch.ones(batch_size, seq_len),
        }
        result = evaluate_probability(model, batch)
        assert isinstance(result, list)
        assert len(result) == batch_size
        for item in result:
            assert isinstance(item, dict)
            assert "prob" in item
            assert "avg_loss" in item
            assert isinstance(item["prob"], (int, float))
            assert isinstance(item["avg_loss"], (int, float))

    def test_prob_and_avg_loss_consistent(self):
        """prob = exp(-avg_loss) for each sample."""
        import math
        batch_size = 2
        seq_len = 6
        vocab_size = 8
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[:, 0] = -100
        model = Mock()
        model.device = torch.device("cpu")
        model.return_value = Mock(logits=torch.randn(batch_size, seq_len, vocab_size))
        model.side_effect = lambda **kw: model.return_value
        batch = {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "labels": labels,
            "attention_mask": torch.ones(batch_size, seq_len),
        }
        result = evaluate_probability(model, batch)
        for item in result:
            assert abs(item["prob"] - math.exp(-item["avg_loss"])) < 1e-5

    def test_batch_to_device_handles_list_of_tensors(self):
        """_batch_to_device moves list of tensors to device without calling .to() on list."""
        device = torch.device("cpu")
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0, 4.0])
        batch = {"a": t1, "b": [t1, t2], "c": torch.tensor(5)}
        out = _batch_to_device(batch, device)
        assert out["a"].device == device
        assert isinstance(out["b"], list)
        assert len(out["b"]) == 2
        assert out["b"][0].device == device
        assert out["b"][1].device == device
        assert out["c"].device == device

    def test_evaluate_probability_with_list_labels_wrong_trajectory_path(self):
        """Trajectory batch_template has labels_wrong as list of N tensors [1, L]; no AttributeError."""
        batch_size = 1
        seq_len = 5
        vocab_size = 10
        n_wrong_options = 3
        # Single-sample batch (trajectory per-sample): labels_wrong = list of N tensors [1, L]
        labels_correct = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels_correct[:, 0] = -100
        labels_wrong_list = [
            torch.randint(0, vocab_size, (batch_size, seq_len)) for _ in range(n_wrong_options)
        ]
        for lab in labels_wrong_list:
            lab[:, 0] = -100
        model = Mock()
        model.device = torch.device("cpu")
        model.return_value = Mock(logits=torch.randn(batch_size, seq_len, vocab_size))
        model.side_effect = lambda **kw: model.return_value
        batch = {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "labels": labels_correct,
            "labels_wrong": labels_wrong_list,
            "attention_mask": torch.ones(batch_size, seq_len),
        }
        result = evaluate_probability(model, batch)
        # Should return list of N result lists (one per wrong option), each list has 1 dict (batch_size=1).
        assert isinstance(result, list)
        assert len(result) == n_wrong_options
        for option_results in result:
            assert isinstance(option_results, list)
            assert len(option_results) == 1
            assert "prob" in option_results[0]
            assert "avg_loss" in option_results[0]
            assert option_results[0]["prob"] is not None
            assert option_results[0]["avg_loss"] is not None

    def test_evaluate_probability_labels_field_list_returns_n_option_results(self):
        """When labels_field is in fn_args and batch[labels_field] is list of tensors, return N option results."""
        batch_size = 1
        seq_len = 4
        vocab_size = 8
        n_opts = 2
        labels_list = [
            torch.randint(0, vocab_size, (batch_size, seq_len)) for _ in range(n_opts)
        ]
        for lab in labels_list:
            lab[:, 0] = -100
        model = Mock()
        model.device = torch.device("cpu")
        model.return_value = Mock(logits=torch.randn(batch_size, seq_len, vocab_size))
        model.side_effect = lambda **kw: model.return_value
        batch = {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "labels": labels_list[0],
            "labels_wrong": labels_list,
            "attention_mask": torch.ones(batch_size, seq_len),
        }
        result = evaluate_probability(model, batch, labels_field="labels_wrong")
        assert isinstance(result, list)
        assert len(result) == n_opts
        for option_results in result:
            assert len(option_results) == 1
            assert "prob" in option_results[0] and "avg_loss" in option_results[0]


class TestComputeProbFromFixationLogits:
    """Test _compute_prob_from_fixation_logits returns same format and values."""

    def test_returns_list_of_dicts_with_prob_and_avg_loss(self):
        B, T, V = 2, 4, 8
        fixation_logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T + 1))
        labels[:, 0] = -100
        device = torch.device("cpu")
        result = _compute_prob_from_fixation_logits(
            fixation_logits, labels, device, ignore_index=-100
        )
        assert isinstance(result, list)
        assert len(result) == B
        for item in result:
            assert isinstance(item, dict)
            assert "prob" in item
            assert "avg_loss" in item
            assert isinstance(item["prob"], (int, float))
            assert isinstance(item["avg_loss"], (int, float))

    def test_prob_equals_exp_neg_avg_loss(self):
        B, T, V = 1, 3, 5
        fixation_logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T + 1))
        labels[:, 0] = -100
        device = torch.device("cpu")
        result = _compute_prob_from_fixation_logits(
            fixation_logits, labels, device, ignore_index=-100
        )
        import math
        for item in result:
            assert abs(item["prob"] - math.exp(-item["avg_loss"])) < 1e-5


class TestComputeProbPackedShiftedSegments:
    """Packed shifted CE matches rectangular batch and per-segment sequential CE."""

    def test_packed_matches_rectangular_batch(self):
        torch.manual_seed(0)
        B, T, V = 4, 5, 11
        fixation_logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T + 1))
        labels[:, 0] = -100
        device = torch.device("cpu")
        rect = _compute_prob_from_fixation_logits(
            fixation_logits, labels, device, ignore_index=-100
        )
        segs_log = [fixation_logits[b : b + 1] for b in range(B)]
        segs_lab = [labels[b : b + 1] for b in range(B)]
        packed = compute_prob_packed_shifted_segments(
            segs_log, segs_lab, device, ignore_index=-100
        )
        assert len(packed) == B
        for a, b in zip(rect, packed, strict=True):
            assert abs(a["prob"] - b["prob"]) < 1e-5
            assert abs(a["avg_loss"] - b["avg_loss"]) < 1e-5

    def test_packed_ragged_matches_sequential(self):
        """Variable segment lengths (eos-style): packed == loop of single-segment calls."""
        torch.manual_seed(1)
        V = 9
        device = torch.device("cpu")
        seg_logits = []
        seg_labels = []
        expected = []
        for Tb in (3, 5, 2, 6):
            log = torch.randn(1, Tb, V)
            lab = torch.randint(0, V, (1, Tb))
            lab[:, 0] = -100
            seg_logits.append(log)
            seg_labels.append(lab)
            expected.extend(
                _compute_prob_from_fixation_logits(
                    log, lab, device, ignore_index=-100
                )
            )
        packed = compute_prob_packed_shifted_segments(
            seg_logits, seg_labels, device, ignore_index=-100
        )
        assert len(packed) == len(expected)
        for a, b in zip(packed, expected, strict=True):
            assert abs(a["prob"] - b["prob"]) < 1e-5
            assert abs(a["avg_loss"] - b["avg_loss"]) < 1e-5

    def test_packed_row_chunking_matches_monolithic(self):
        """Row-batched CE inside compute_prob_packed_shifted_segments matches one-shot CE."""
        torch.manual_seed(2)
        V = 17
        device = torch.device("cpu")
        seg_logits = []
        seg_labels = []
        for Tb in (4, 9, 3, 7):
            log = torch.randn(1, Tb, V)
            lab = torch.randint(0, V, (1, Tb))
            lab[:, 0] = -100
            seg_logits.append(log)
            seg_labels.append(lab)
        ref = compute_prob_packed_shifted_segments(
            seg_logits, seg_labels, device, ignore_index=-100, max_ce_logits_rows=10_000
        )
        for cap in (1, 2, 3, 7, 64):
            packed = compute_prob_packed_shifted_segments(
                seg_logits, seg_labels, device, ignore_index=-100, max_ce_logits_rows=cap
            )
            assert len(packed) == len(ref)
            for a, b in zip(ref, packed, strict=True):
                assert abs(float(a["prob"]) - float(b["prob"])) < 1e-6
                assert abs(float(a["avg_loss"]) - float(b["avg_loss"])) < 1e-5


class TestEvalRougeRecallBatch:
    """Test eval_rouge_recall_batch: same (gen, gt) pairs -> same scores; use_stemmer option."""

    def test_same_pairs_same_scores(self):
        gen_list = ["the cat sat on the mat", "hello world"]
        gt_list = ["the cat sat on the mat", "hello world"]
        out1 = eval_rouge_recall_batch(gen_list, gt_list, use_stemmer=True)
        out2 = eval_rouge_recall_batch(gen_list, gt_list, use_stemmer=True)
        assert len(out1) == len(out2) == 2
        for a, b in zip(out1, out2):
            assert a["rouge1_recall"] == b["rouge1_recall"]
            assert a["rougeL_f1"] == b["rougeL_f1"]
            assert a["rougeL_recall"] == b["rougeL_recall"]

    def test_returns_expected_keys(self):
        out = eval_rouge_recall_batch(["a"], ["a"], use_stemmer=True)
        assert len(out) == 1
        assert "rouge1_recall" in out[0]
        assert "rougeL_f1" in out[0]
        assert "rougeL_recall" in out[0]
        assert isinstance(out[0]["rouge1_recall"], (int, float))
        assert isinstance(out[0]["rougeL_f1"], (int, float))
        assert isinstance(out[0]["rougeL_recall"], (int, float))

    def test_use_stemmer_false_valid_structure(self):
        out = eval_rouge_recall_batch(["the cat"], ["the cat"], use_stemmer=False)
        assert len(out) == 1
        assert "rouge1_recall" in out[0]
        assert "rougeL_f1" in out[0]
        assert "rougeL_recall" in out[0]
        assert 0 <= out[0]["rouge1_recall"] <= 1
        assert 0 <= out[0]["rougeL_f1"] <= 1
        assert 0 <= out[0]["rougeL_recall"] <= 1

    def test_scorer_reuse_same_result(self):
        """Passing a scorer gives same result as creating one internally."""
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        out_with = eval_rouge_recall_batch(["x y z"], ["x y z"], scorer=scorer)
        out_without = eval_rouge_recall_batch(["x y z"], ["x y z"], use_stemmer=True)
        assert out_with[0]["rouge1_recall"] == out_without[0]["rouge1_recall"]
        assert out_with[0]["rougeL_f1"] == out_without[0]["rougeL_f1"]
        assert out_with[0]["rougeL_recall"] == out_without[0]["rougeL_recall"]

    def test_worker_multi_steps_matches_per_step_batches(self):
        gen_per_step = [
            ["the cat sat", "hello world"],
            ["partial cat sat", "hello there"],
        ]
        gt_list = ["the cat sat on mat", "hello world today"]
        combined = eval_rouge_recall_batch_worker_multi_steps(
            gen_per_step, gt_list, use_stemmer=True
        )
        assert len(combined) == len(gen_per_step)
        for k, gens in enumerate(gen_per_step):
            ref = eval_rouge_recall_batch(gens, gt_list, use_stemmer=True)
            assert len(combined[k]) == len(ref)
            for row_a, row_b in zip(combined[k], ref, strict=True):
                assert row_a.keys() == row_b.keys()
                for key in row_a:
                    assert row_a[key] == row_b[key]


class TestEvalTextSimilarityBatchDecode:
    """Test eval_text_similarity uses batch_decode only (no per-sample decode loops)."""

    def test_returns_expected_keys_and_shapes(self):
        model = Mock()
        model.device = torch.device("cpu")
        model.generate = Mock(return_value=torch.randint(0, 100, (2, 20)))
        tokenizer = Mock()
        tokenizer.eos_token_id = 0
        tokenizer.decode = Mock(return_value="")  # for stopwords
        def batch_decode_ret(x, **kw):
            n = x.shape[0] if hasattr(x, "shape") else len(x)
            return ["input"] * n
        tokenizer.batch_decode = Mock(side_effect=batch_decode_ret)
        batch = {
            "input_ids": torch.randint(0, 10, (2, 5)),
            "labels": torch.randint(0, 10, (2, 10)),
            "attention_mask": torch.ones(2, 5),
        }
        from omegaconf import OmegaConf
        gen_args = OmegaConf.create({"max_new_tokens": 5})
        result = eval_text_similarity(model, tokenizer, batch, gen_args)
        assert isinstance(result, list)
        assert len(result) == 2
        for item in result:
            assert "rouge1_recall" in item
            assert "rougeL_f1" in item
            assert "rougeL_recall" in item
            assert "input" in item
            assert "ground_truth" in item
            assert "generation" in item

    def test_batch_decode_called_batched(self):
        """batch_decode is called with batched inputs (not per-sample)."""
        model = Mock()
        model.device = torch.device("cpu")
        model.generate = Mock(return_value=torch.randint(0, 100, (2, 15)))
        tokenizer = Mock()
        tokenizer.eos_token_id = 0
        tokenizer.decode = Mock(return_value="")
        batch_decode_calls = []
        def record_batch_decode(ids, **kw):
            batch_decode_calls.append(len(ids) if hasattr(ids, "__len__") else 1)
            return ["text"] * (len(ids) if hasattr(ids, "__len__") else 1)
        tokenizer.batch_decode = Mock(side_effect=record_batch_decode)
        batch = {
            "input_ids": torch.randint(0, 10, (2, 5)),
            "labels": torch.randint(0, 10, (2, 10)),
            "attention_mask": torch.ones(2, 5),
        }
        from omegaconf import OmegaConf
        gen_args = OmegaConf.create({"max_new_tokens": 5})
        eval_text_similarity(model, tokenizer, batch, gen_args)
        # Expect 3 batch_decode calls: input_ids, tokens (labels), output slice
        assert tokenizer.batch_decode.called
        assert len(batch_decode_calls) >= 2  # at least input and one more batch

    def test_accepts_plain_dict_generation_args(self):
        """After Hydra compose + to_container(metric_cfg), generation_args is a dict (not DictConfig)."""
        model = Mock()
        model.device = torch.device("cpu")
        model.generate = Mock(return_value=torch.randint(0, 100, (2, 20)))
        tokenizer = Mock()
        tokenizer.eos_token_id = 0
        tokenizer.decode = Mock(return_value="")

        def batch_decode_ret(x, **kw):
            n = x.shape[0] if hasattr(x, "shape") else len(x)
            return ["input"] * n

        tokenizer.batch_decode = Mock(side_effect=batch_decode_ret)
        batch = {
            "input_ids": torch.randint(0, 10, (2, 5)),
            "labels": torch.randint(0, 10, (2, 10)),
            "attention_mask": torch.ones(2, 5),
        }
        gen_args = {"max_new_tokens": 5}
        result = eval_text_similarity(model, tokenizer, batch, gen_args)
        assert isinstance(result, list) and len(result) == 2
        assert gen_args == {"max_new_tokens": 5}
