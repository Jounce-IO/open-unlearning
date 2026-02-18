"""
Tests for MIA batch-by-batch streaming: process_batch, compute_batch_values_from_logits,
mia_auc_streaming parity with mia_auc.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from data.utils import IGNORE_INDEX
from evals.metrics.utils import tokenwise_logprobs_from_logits
from evals.metrics.trajectory_adapters import DualLogitModelWrapper, LogitModelWrapper
from evals.metrics.mia.all_attacks import Attack
from evals.metrics.mia.min_k import MinKProbAttack
from evals.metrics.mia.utils import (
    batch_to_cpu,
    mia_auc,
    mia_auc_from_score_dicts,
    mia_auc_streaming,
    MIAStreamingAccumulator,
    process_mia_batch_worker,
)


def _make_batch(bsz, seq_len, V, device="cpu"):
    input_ids = torch.randint(0, V, (bsz, seq_len), device=device)
    labels = torch.full((bsz, seq_len), IGNORE_INDEX, dtype=torch.long, device=device)
    labels[:, 1 : seq_len - 1] = input_ids[:, 2:seq_len]
    indices = torch.arange(bsz, device=device)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones(bsz, seq_len, dtype=torch.long, device=device),
        "index": indices,
    }


class TestProcessBatch:
    def test_process_batch_returns_scores_per_index(self):
        """process_batch(batch, batch_values) returns {str(idx): {"score": float}}."""
        bsz, seq_len, V = 2, 10, 50
        logits = torch.randn(bsz, seq_len, V)
        batch = _make_batch(bsz, seq_len, V)
        batch_values = tokenwise_logprobs_from_logits(batch, logits)

        model = LogitModelWrapper(logits, torch.device("cpu"))
        attack = MinKProbAttack(model=model, data=[], collator=lambda x: x, batch_size=1, k=0.4)
        out = attack.process_batch(batch, batch_values)

        assert isinstance(out, dict)
        assert set(out.keys()) == {"0", "1"}
        for idx in ("0", "1"):
            assert "score" in out[idx]
            assert isinstance(out[idx]["score"], (float, np.floating))

    def test_process_batch_merged_matches_attack_value_by_index(self):
        """Merged process_batch output matches value_by_index from attack() for same data/logits."""
        bsz, seq_len, V = 3, 12, 80
        logits = torch.randn(bsz, seq_len, V)
        device = torch.device("cpu")
        batch = _make_batch(bsz, seq_len, V)

        logits_by_key = {
            "forget": {str(i): logits[i].T for i in range(bsz)},
            "holdout": {str(i): logits[i].T for i in range(bsz)},
        }
        wrapper = DualLogitModelWrapper(logits_by_key, device)
        data_forget = [{"input_ids": batch["input_ids"][i : i + 1], "labels": batch["labels"][i : i + 1], "attention_mask": batch["attention_mask"][i : i + 1], "index": torch.tensor([i])} for i in range(bsz)]

        def collator(samples):
            return {
                "input_ids": torch.cat([s["input_ids"] for s in samples]),
                "labels": torch.cat([s["labels"] for s in samples]),
                "attention_mask": torch.cat([s["attention_mask"] for s in samples]),
                "index": torch.cat([s["index"] for s in samples]),
            }

        wrapper.set_dataset_key("forget")
        attack = MinKProbAttack(model=wrapper, data=data_forget, collator=collator, batch_size=bsz, k=0.4)
        full_result = attack.attack()
        full_value_by_index = full_result["value_by_index"]

        batch_values = tokenwise_logprobs_from_logits(batch, logits)
        attack2 = MinKProbAttack(model=wrapper, data=[], collator=collator, batch_size=1, k=0.4)
        batch_scores = attack2.process_batch(batch, batch_values)

        for idx in batch_scores:
            assert idx in full_value_by_index
            assert abs(batch_scores[idx]["score"] - full_value_by_index[idx]["score"]) < 1e-5


class TestComputeBatchValuesFromLogits:
    def test_min_k_compute_batch_values_from_logits_matches_tokenwise_from_logits(self):
        """MinKProbAttack.compute_batch_values_from_logits matches tokenwise_logprobs_from_logits."""
        bsz, seq_len, V = 2, 15, 100
        logits = torch.randn(bsz, seq_len, V)
        batch = _make_batch(bsz, seq_len, V)

        from_util = tokenwise_logprobs_from_logits(batch, logits)
        attack = MinKProbAttack(model=None, data=[], collator=lambda x: x, batch_size=1, k=0.4)
        from_attack = attack.compute_batch_values_from_logits(batch, logits)

        assert len(from_util) == len(from_attack)
        for i in range(bsz):
            assert torch.allclose(from_util[i], from_attack[i])

    def test_process_batch_with_compute_batch_values_from_logits_matches_model_path(self):
        """process_batch(batch, compute_batch_values_from_logits(batch, logits)) matches model path."""
        bsz, seq_len, V = 2, 10, 50
        logits = torch.randn(bsz, seq_len, V)
        batch = _make_batch(bsz, seq_len, V)
        device = torch.device("cpu")
        model = LogitModelWrapper(logits, device)

        attack = MinKProbAttack(model=model, data=[], collator=lambda x: x, batch_size=1, k=0.4)
        batch_values_model = attack.compute_batch_values(batch)
        batch_values_logits = attack.compute_batch_values_from_logits(batch, logits)

        for i in range(bsz):
            assert torch.allclose(batch_values_model[i], batch_values_logits[i])

        out_model = attack.process_batch(batch, batch_values_model)
        out_logits = attack.process_batch(batch, batch_values_logits)
        for idx in out_model:
            assert abs(out_model[idx]["score"] - out_logits[idx]["score"]) < 1e-5

    def test_min_k_compute_batch_values_from_per_position_scores(self):
        """MinKProbAttack.compute_batch_values_from_per_position_scores returns log-probs and compute_score is consistent."""
        per_position_scores = [[0.5, 0.25, 0.1], [0.8, 0.6]]
        batch = _make_batch(2, 5, 10)
        attack = MinKProbAttack(model=None, data=[], collator=lambda x: x, batch_size=1, k=0.4)
        batch_values = attack.compute_batch_values_from_per_position_scores(batch, per_position_scores)
        assert len(batch_values) == 2
        assert batch_values[0].shape[0] == 3
        assert batch_values[1].shape[0] == 2
        import numpy as np
        np.testing.assert_allclose(batch_values[0].numpy(), -np.log([0.5, 0.25, 0.1]), rtol=1e-5)
        scores = attack.process_batch(batch, batch_values)
        assert len(scores) == 2
        assert all("score" in v for v in scores.values())


class TestMiaAucFromScoreDicts:
    """Tests for aggregation from score dicts (value_by_index for forget and holdout)."""

    def test_aggregation_returns_same_shape_as_mia_auc(self):
        """mia_auc_from_score_dicts returns dict with forget, holdout, auc, agg_value."""
        forget_vbi = {"0": {"score": 0.1}, "1": {"score": 0.2}}
        holdout_vbi = {"0": {"score": 0.8}, "1": {"score": 0.9}}
        out = mia_auc_from_score_dicts(forget_vbi, holdout_vbi)
        assert "forget" in out and "holdout" in out
        assert out["forget"]["value_by_index"] == forget_vbi
        assert out["holdout"]["value_by_index"] == holdout_vbi
        assert "auc" in out and "agg_value" in out
        assert out["auc"] == out["agg_value"]
        forget_scores = [forget_vbi[k]["score"] for k in sorted(forget_vbi)]
        holdout_scores = [holdout_vbi[k]["score"] for k in sorted(holdout_vbi)]
        expected_auc = roc_auc_score(
            [0] * len(forget_scores) + [1] * len(holdout_scores),
            forget_scores + holdout_scores,
        )
        assert abs(out["auc"] - expected_auc) < 1e-9

    def test_aggregation_agg_value_forget_holdout(self):
        """forget and holdout have agg_value as mean of their scores."""
        forget_vbi = {"0": {"score": 1.0}, "1": {"score": 3.0}}
        holdout_vbi = {"2": {"score": 2.0}}
        out = mia_auc_from_score_dicts(forget_vbi, holdout_vbi)
        assert out["forget"]["agg_value"] == 2.0
        assert out["holdout"]["agg_value"] == 2.0


def _batch_logits_iter(data_list, collator, batch_size, logits_by_idx):
    """Yield (batch, logits) for each batch from data_list. logits_by_idx: idx_str -> [L,V]."""
    for start in range(0, len(data_list), batch_size):
        batch_samples = data_list[start : start + batch_size]
        batch = collator(batch_samples)
        indices = batch["index"].cpu().numpy().tolist()
        indices = [indices] if isinstance(indices, int) else indices
        logits_list = [logits_by_idx[str(idx)].unsqueeze(0) for idx in indices]
        logits = torch.cat(logits_list, dim=0)
        yield batch, logits


class TestMiaAucStreaming:
    """Tests for mia_auc_streaming parity with mia_auc."""

    def test_mia_auc_streaming_same_result_as_mia_auc(self):
        """mia_auc_streaming gives same AUC and value_by_index as mia_auc for same data/logits."""
        bsz_f, bsz_h, seq_len, V = 4, 3, 10, 50
        device = torch.device("cpu")
        torch.manual_seed(42)
        logits_f = torch.randn(bsz_f, seq_len, V)
        logits_h = torch.randn(bsz_h, seq_len, V)

        data_f = [
            {
                "input_ids": torch.randint(0, V, (seq_len,)),
                "labels": torch.full((seq_len,), IGNORE_INDEX, dtype=torch.long),
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
                "index": torch.tensor(i),
            }
            for i in range(bsz_f)
        ]
        for i in range(bsz_f):
            data_f[i]["labels"][1 : seq_len - 1] = data_f[i]["input_ids"][2:seq_len]
        data_h = [
            {
                "input_ids": torch.randint(0, V, (seq_len,)),
                "labels": torch.full((seq_len,), IGNORE_INDEX, dtype=torch.long),
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
                "index": torch.tensor(bsz_f + i),
            }
            for i in range(bsz_h)
        ]
        for i in range(bsz_h):
            data_h[i]["labels"][1 : seq_len - 1] = data_h[i]["input_ids"][2:seq_len]

        def collator(samples):
            return {
                "input_ids": torch.stack([s["input_ids"] for s in samples]),
                "labels": torch.stack([s["labels"] for s in samples]),
                "attention_mask": torch.stack([s["attention_mask"] for s in samples]),
                "index": torch.stack([s["index"] for s in samples]),
            }

        logits_by_key = {
            "forget": {str(i): logits_f[i].T for i in range(bsz_f)},
            "holdout": {str(bsz_f + i): logits_h[i].T for i in range(bsz_h)},
        }
        wrapper = DualLogitModelWrapper(logits_by_key, device)
        wrapper.set_dataset_key("forget")
        out_mia = mia_auc(
            MinKProbAttack,
            wrapper,
            {"forget": data_f, "holdout": data_h},
            collator=collator,
            batch_size=4,
            k=0.4,
        )

        forget_iter = _batch_logits_iter(data_f, collator, 2, {str(i): logits_f[i] for i in range(bsz_f)})
        holdout_iter = _batch_logits_iter(
            data_h, collator, 2, {str(bsz_f + i): logits_h[i] for i in range(bsz_h)}
        )
        out_stream = mia_auc_streaming(
            MinKProbAttack,
            forget_iter,
            holdout_iter,
            collator=collator,
            batch_size=2,
            device=device,
            k=0.4,
        )

        assert abs(out_stream["auc"] - out_mia["auc"]) < 1e-5
        assert abs(out_stream["agg_value"] - out_mia["agg_value"]) < 1e-5
        for idx in out_mia["forget"]["value_by_index"]:
            assert idx in out_stream["forget"]["value_by_index"]
            assert abs(
                out_stream["forget"]["value_by_index"][idx]["score"]
                - out_mia["forget"]["value_by_index"][idx]["score"]
            ) < 1e-5
        for idx in out_mia["holdout"]["value_by_index"]:
            assert idx in out_stream["holdout"]["value_by_index"]
            assert abs(
                out_stream["holdout"]["value_by_index"][idx]["score"]
                - out_mia["holdout"]["value_by_index"][idx]["score"]
            ) < 1e-5


class TestProcessMiaBatchWorker:
    """process_mia_batch_worker (for worker-pool offload) matches in-process accumulator."""

    def test_process_mia_batch_worker_matches_accumulator_add_forget_batch(self):
        """Worker returns same score dict as MIAStreamingAccumulator.add_forget_batch."""
        bsz, seq_len, V = 2, 10, 50
        logits = torch.randn(bsz, seq_len, V)
        batch = _make_batch(bsz, seq_len, V)
        device = torch.device("cpu")

        def collator(x):
            return x

        acc = MIAStreamingAccumulator(
            MinKProbAttack,
            collator=collator,
            batch_size=bsz,
            device=device,
            k=0.4,
        )
        acc.add_forget_batch(batch, logits)
        expected = acc.forget_value_by_index

        batch_cpu = batch_to_cpu(batch)
        logits_cpu = logits.cpu()
        got = process_mia_batch_worker(
            "min_k",
            {"k": 0.4},
            batch_size=bsz,
            batch_cpu=batch_cpu,
            logits_cpu=logits_cpu,
        )
        assert set(got.keys()) == set(expected.keys())
        for idx in expected:
            assert abs(got[idx]["score"] - expected[idx]["score"]) < 1e-5
