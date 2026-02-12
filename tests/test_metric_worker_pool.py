"""
TDD tests for metric worker pool offload (ROUGE in process pool).

Tests define desired behavior; implementation in utils.py and trajectory_metrics.py
makes them pass (Red → Green).
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))


def _trajectory_metrics_rouge_mock_setup(S=4, L_gen=10, V=100):
    """Shared mock setup for trajectory_metrics with rouge: sampler, model, tokenizer, dataset, collator."""
    full_len = 5 + L_gen
    logits_history = [torch.randn(1, full_len, V) for _ in range(S)]
    fixation_steps = torch.randint(0, S, (1, full_len))

    class MockSamplerOutput:
        def __init__(self, sequences, histories, logits_history, fixation_steps):
            self.sequences = sequences
            self.histories = histories
            self.logits_history = logits_history
            self.fixation_steps = fixation_steps

    sampler = Mock()
    sampler.sample = Mock(
        return_value=MockSamplerOutput(
            sequences=torch.randint(0, V, (1, full_len)),
            histories=None,
            logits_history=logits_history,
            fixation_steps=fixation_steps,
        )
    )
    model = Mock()
    model.sampler = sampler
    tokenizer = Mock()
    tokenizer.decode = Mock(return_value="decoded")
    tokenizer.eos_token_id = 2

    class MockDataset:
        def __init__(self):
            self.data = [
                {
                    "input_ids": torch.randint(0, V, (full_len,)),
                    "labels": torch.randint(0, V, (full_len,)),
                }
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    def mock_collator(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "indices": torch.tensor([0]),
        }

    base_trajectory_config = {
        "return_logits": True,
        "return_fixation_steps": True,
        "sampler_kwargs": {
            "steps": S,
            "max_new_tokens": L_gen,
            "trajectory_sample_interval": 8,
        },
    }
    return model, tokenizer, MockDataset(), mock_collator, base_trajectory_config, S


class TestWorkerParity:
    """1.1 Worker parity: eval_rouge_recall_batch_worker returns same as eval_rouge_recall_batch(..., scorer=None)."""

    def test_eval_rouge_recall_batch_worker_matches_eval_rouge_recall_batch(self):
        from evals.metrics.utils import eval_rouge_recall_batch, eval_rouge_recall_batch_worker

        gen_outputs = ["the cat sat", "the dog ran"]
        ground_truths = ["the cat sat down", "the dog ran fast"]
        worker_result = eval_rouge_recall_batch_worker(
            gen_outputs, ground_truths, use_stemmer=True
        )
        expected = eval_rouge_recall_batch(
            gen_outputs, ground_truths, use_stemmer=True, scorer=None
        )
        assert len(worker_result) == len(expected)
        for w, e in zip(worker_result, expected):
            assert set(w.keys()) == set(e.keys())
            for k in w:
                assert w[k] == e[k], f"{k}: {w[k]} != {e[k]}"


class TestSynchronousPathWhenPoolSizeZero:
    """1.2 When metric_worker_pool_size=0, eval_rouge_recall_batch is called in main process."""

    def test_trajectory_metrics_with_pool_size_zero_calls_eval_rouge_in_main(self):
        from evals.metrics.trajectory_metrics import trajectory_metrics

        (
            model,
            tokenizer,
            dataset,
            collator,
            base_config,
            S,
        ) = _trajectory_metrics_rouge_mock_setup()
        trajectory_config = {**base_config, "metric_worker_pool_size": 0}

        raw_fn = (
            trajectory_metrics._metric_fn
            if hasattr(trajectory_metrics, "_metric_fn")
            else trajectory_metrics
        )
        with patch(
            "evals.metrics.trajectory_metrics.eval_rouge_recall_batch"
        ) as mock_rouge_batch:
            mock_rouge_batch.return_value = [
                {"rouge1_recall": 0.5, "rougeL_f1": 0.6, "rougeL_recall": 0.7}
                for _ in range(S)
            ]
            raw_fn(
                model,
                metric_name="trajectory_metrics",
                cache={},
                metrics=["rouge"],
                data=dataset,
                collators=collator,
                tokenizer=tokenizer,
                batch_size=1,
                trajectory_config=trajectory_config,
            )
            assert mock_rouge_batch.call_count >= 1


class TestPoolPathParity:
    """1.3 Pool path produces same ROUGE results as synchronous path."""

    def test_trajectory_metrics_pool_size_2_same_results_as_pool_size_0(self):
        from evals.metrics.trajectory_metrics import trajectory_metrics

        (
            model,
            tokenizer,
            dataset,
            collator,
            base_config,
            _,
        ) = _trajectory_metrics_rouge_mock_setup()
        # Do not patch eval_rouge_recall_batch: sync runs in main process, pool runs
        # in workers (no patch there). Both must run real ROUGE to get comparable results.
        raw_fn = (
            trajectory_metrics._metric_fn
            if hasattr(trajectory_metrics, "_metric_fn")
            else trajectory_metrics
        )

        result_sync = raw_fn(
            model,
            metric_name="trajectory_metrics",
            cache={},
            metrics=["rouge"],
            data=dataset,
            collators=collator,
            tokenizer=tokenizer,
            batch_size=1,
            trajectory_config={**base_config, "metric_worker_pool_size": 0},
        )

        result_pool = raw_fn(
            model,
            metric_name="trajectory_metrics",
            cache={},
            metrics=["rouge"],
            data=dataset,
            collators=collator,
            tokenizer=tokenizer,
            batch_size=1,
            trajectory_config={**base_config, "metric_worker_pool_size": 2},
        )

        def _to_comparable(val):
            if hasattr(val, "tolist"):
                return val.tolist()
            if isinstance(val, list):
                return [_to_comparable(x) for x in val]
            return val

        assert result_sync is not None and result_pool is not None
        assert set(result_sync.keys()) == set(result_pool.keys())

        def get_agg_value(result):
            if "agg_value" in result and isinstance(result["agg_value"], dict):
                return result["agg_value"]
            for k, v in result.items():
                if k == "trajectory_step_metadata":
                    continue
                if isinstance(v, dict) and "agg_value" in v:
                    return v["agg_value"]
            return None

        agg_sync = get_agg_value(result_sync)
        agg_pool = get_agg_value(result_pool)
        assert agg_sync is not None and agg_pool is not None
        assert set(agg_sync.keys()) == set(agg_pool.keys())
        for traj_name in agg_sync:
            assert set(agg_sync[traj_name].keys()) == set(
                agg_pool[traj_name].keys()
            )
            for metric_name in agg_sync[traj_name]:
                v_sync = _to_comparable(agg_sync[traj_name][metric_name])
                v_pool = _to_comparable(agg_pool[traj_name][metric_name])
                assert v_sync == v_pool, (
                    f"{traj_name} {metric_name}: sync != pool"
                )


class TestPoolPathDoesNotCallRougeInMain:
    """1.4 When metric_worker_pool_size > 0, eval_rouge_recall_batch is not called in main process."""

    def test_trajectory_metrics_with_pool_size_2_does_not_call_eval_rouge_in_main(self):
        from evals.metrics.trajectory_metrics import trajectory_metrics

        (
            model,
            tokenizer,
            dataset,
            collator,
            base_config,
            _,
        ) = _trajectory_metrics_rouge_mock_setup()
        trajectory_config = {**base_config, "metric_worker_pool_size": 2}

        raw_fn = (
            trajectory_metrics._metric_fn
            if hasattr(trajectory_metrics, "_metric_fn")
            else trajectory_metrics
        )
        with patch(
            "evals.metrics.trajectory_metrics.eval_rouge_recall_batch"
        ) as mock_rouge_batch:
            raw_fn(
                model,
                metric_name="trajectory_metrics",
                cache={},
                metrics=["rouge"],
                data=dataset,
                collators=collator,
                tokenizer=tokenizer,
                batch_size=1,
                trajectory_config=trajectory_config,
            )
            assert mock_rouge_batch.call_count == 0


class TestExecutorShutdownOnExit:
    """1.5 When executor is created, shutdown(wait=True) is called on exit."""

    def test_trajectory_metrics_shuts_down_executor_when_pool_used(self):
        from concurrent.futures import ProcessPoolExecutor

        from evals.metrics.trajectory_metrics import trajectory_metrics

        (
            model,
            tokenizer,
            dataset,
            collator,
            base_config,
            _,
        ) = _trajectory_metrics_rouge_mock_setup()
        trajectory_config = {**base_config, "metric_worker_pool_size": 1}

        raw_fn = (
            trajectory_metrics._metric_fn
            if hasattr(trajectory_metrics, "_metric_fn")
            else trajectory_metrics
        )

        real_executor_class = ProcessPoolExecutor
        created_executors = []

        class CapturingProcessPoolExecutor(real_executor_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created_executors.append(self)

            def shutdown(self, wait=True, *, cancel_futures=False):
                if not hasattr(self, "_shutdown_called"):
                    self._shutdown_called = True
                return super().shutdown(wait=wait, cancel_futures=cancel_futures)

        with patch(
            "evals.metrics.trajectory_metrics.ProcessPoolExecutor",
            CapturingProcessPoolExecutor,
        ):
            raw_fn(
                model,
                metric_name="trajectory_metrics",
                cache={},
                metrics=["rouge"],
                data=dataset,
                collators=collator,
                tokenizer=tokenizer,
                batch_size=1,
                trajectory_config=trajectory_config,
            )

        assert len(created_executors) >= 1
        for ex in created_executors:
            assert getattr(ex, "_shutdown_called", False), (
                "Executor shutdown was not called"
            )
