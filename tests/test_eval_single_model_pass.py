"""Tests for single model-pass and sample-once evaluation optimization.

When multiple trajectory_metrics are configured, the Evaluator should:
1. Load shared data once (get_datasets/get_collators called once).
2. Invoke trajectory_metrics once with all sub-metrics (one model pass, not N).
3. Sampler.sample called once per item (not N times per item).
4. Full trajectories not accumulated; only metric-sized outputs retained.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


def _minimal_trajectory_metric_cfg(metrics_list, display_names, datasets_key="datasets"):
    """Minimal config for one trajectory_metrics entry."""
    return {
        "handler": "trajectory_metrics",
        "metrics": metrics_list,
        "metric_display_names": display_names,
        "datasets": {"TOFU_QA_forget": {"handler": "QADataset", "args": {}}}
        if datasets_key else None,
        "collators": {"DataCollator": {"handler": "DataCollatorForSupervisedDataset", "args": {}}},
        "batch_size": 1,
        "trajectory_config": {"logits_source": "sampler", "return_logits": True},
    }


@pytest.fixture
def eval_cfg_two_trajectory_metrics():
    """Eval config with 2 metrics, both handler trajectory_metrics."""
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "handler": "TOFUEvaluator",
        "output_dir": "/tmp/test_eval",
        "metrics": {
            "traj_prob": _minimal_trajectory_metric_cfg(
                ["probability"],
                ["display_prob"],
            ),
            "traj_rouge": _minimal_trajectory_metric_cfg(
                ["rouge"],
                ["display_rouge"],
            ),
        },
        "samples": 2,
        "overwrite": True,
    })


@pytest.fixture
def eval_cfg_single_trajectory_metric():
    """Eval config with 1 metric (no coalescing)."""
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "handler": "TOFUEvaluator",
        "output_dir": "/tmp/test_eval",
        "metrics": {
            "traj_only": _minimal_trajectory_metric_cfg(
                ["probability"],
                ["display_only"],
            ),
        },
        "samples": 2,
        "overwrite": True,
    })


def test_coalesced_run_calls_get_datasets_once(eval_cfg_two_trajectory_metrics):
    """When 2+ trajectory metrics are configured, get_datasets is called once (shared data)."""
    from evals import get_evaluators

    get_datasets_calls = []

    def count_get_datasets(*args, **kwargs):
        get_datasets_calls.append(1)
        return MagicMock()

    with patch("evals.base.get_datasets", side_effect=count_get_datasets):
        with patch("evals.base.get_collators", return_value=MagicMock()):
            evaluators = get_evaluators({"tofu_trajectory": eval_cfg_two_trajectory_metrics})
            evaluator = evaluators["tofu_trajectory"]
            from evals.metrics import METRICS_REGISTRY
            traj_metric = METRICS_REGISTRY.get("trajectory_metrics")
            if traj_metric is None:
                pytest.skip("trajectory_metrics not registered")

            def fake_trajectory_metrics(model, **kwargs):
                return {
                    "display_prob": {"agg_value": 0.5, "value_by_index": {}},
                    "display_rouge": {"agg_value": 0.6, "value_by_index": {}},
                }

            with patch.object(traj_metric, "_metric_fn", side_effect=fake_trajectory_metrics):
                mock_model = MagicMock()
                mock_model.config._name_or_path = "test-model"
                evaluator.evaluate(mock_model, output_dir="/tmp/test_eval_out")

    assert len(get_datasets_calls) == 1, "get_datasets should be called once for coalesced run"


def test_coalesced_run_invokes_trajectory_metrics_once(eval_cfg_two_trajectory_metrics):
    """When 2+ trajectory metrics are configured, trajectory_metrics is invoked once (not twice)."""
    from evals import get_evaluators
    from evals.metrics import METRICS_REGISTRY

    traj_metric = METRICS_REGISTRY.get("trajectory_metrics")
    if traj_metric is None:
        pytest.skip("trajectory_metrics not registered")

    trajectory_metrics_call_count = []

    def count_and_return(model, **kwargs):
        trajectory_metrics_call_count.append(1)
        return {
            "display_prob": {"agg_value": 0.5, "value_by_index": {}},
            "display_rouge": {"agg_value": 0.6, "value_by_index": {}},
        }

    with patch("evals.base.get_datasets", return_value=MagicMock()):
        with patch("evals.base.get_collators", return_value=MagicMock()):
            evaluators = get_evaluators({"tofu_trajectory": eval_cfg_two_trajectory_metrics})
            evaluator = evaluators["tofu_trajectory"]

            with patch.object(traj_metric, "_metric_fn", side_effect=count_and_return):
                mock_model = MagicMock()
                mock_model.config._name_or_path = "test-model"
                evaluator.evaluate(mock_model, output_dir="/tmp/test_eval_out")

    assert len(trajectory_metrics_call_count) == 1, (
        "trajectory_metrics should be invoked once when 2 metrics are coalesced"
    )


def test_coalesced_run_sampler_sample_called_once_per_item(eval_cfg_two_trajectory_metrics):
    """When 2+ trajectory metrics are coalesced, sampler.sample is called once per item (not N times per item)."""
    from evals import get_evaluators
    from evals.metrics.trajectory_metrics import trajectory_metrics as raw_trajectory_metrics

    V, L_gen, S = 100, 10, 8
    prompt_len = 5
    full_len = prompt_len + L_gen
    num_items = 2

    class MockSamplerOutput:
        def __init__(self, sequences, histories, logits_history, fixation_steps):
            self.sequences = sequences
            self.histories = histories
            self.logits_history = logits_history
            self.fixation_steps = fixation_steps

    sampler = Mock()
    sampler.sample.side_effect = [
        MockSamplerOutput(
            sequences=torch.randint(0, V, (1, full_len)),
            histories=None,
            logits_history=[torch.randn(1, full_len, V) for _ in range(S)],
            fixation_steps=torch.randint(0, S, (1, full_len)),
        )
        for _ in range(num_items)
    ]

    mock_model = MagicMock()
    mock_model.sampler = sampler
    mock_model.config._name_or_path = "test-model"
    if hasattr(mock_model, "adapter_config"):
        delattr(mock_model, "adapter_config")

    class MockDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.data = [
                {
                    "input_ids": torch.randint(0, V, (full_len,)),
                    "labels": torch.randint(0, V, (full_len,)),
                }
                for _ in range(num_items)
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    def mock_collator(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "index": torch.tensor(list(range(len(batch)))),
        }

    shared_dataset = MockDataset()

    def return_shared_data(*args, **kwargs):
        return shared_dataset

    with patch("evals.base.get_datasets", side_effect=return_shared_data):
        with patch("evals.base.get_collators", return_value=mock_collator):
            evaluators = get_evaluators({"tofu_trajectory": eval_cfg_two_trajectory_metrics})
            evaluator = evaluators["tofu_trajectory"]

            tokenizer = Mock()
            tokenizer.decode = lambda x, **kwargs: "decoded"

            evaluator.evaluate(
                mock_model,
                output_dir="/tmp/test_eval_out",
                tokenizer=tokenizer,
                template_args=None,
            )

    assert sampler.sample.call_count == num_items, (
        f"sampler.sample should be called once per item ({num_items} items), got {sampler.sample.call_count}"
    )


def test_trajectory_metrics_returns_only_metric_outputs_not_full_trajectories():
    """trajectory_metrics result retains only metric-sized outputs (value_by_index empty), not full trajectories."""
    from evals.metrics import METRICS_REGISTRY

    raw_trajectory_metrics = METRICS_REGISTRY["trajectory_metrics"]._metric_fn

    V, L_gen, S = 50, 6, 4
    full_len = 5 + L_gen
    num_items = 2

    class MockSamplerOutput:
        def __init__(self, sequences, histories, logits_history, fixation_steps):
            self.sequences = sequences
            self.histories = histories
            self.logits_history = logits_history
            self.fixation_steps = fixation_steps

    sampler = Mock()
    sampler.sample.side_effect = [
        MockSamplerOutput(
            sequences=torch.randint(0, V, (1, full_len)),
            histories=None,
            logits_history=[torch.randn(1, full_len, V) for _ in range(S)],
            fixation_steps=torch.randint(0, S, (1, full_len)),
        )
        for _ in range(num_items)
    ]

    model = Mock()
    model.sampler = sampler

    class MockDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.data = [
                {"input_ids": torch.randint(0, V, (full_len,)), "labels": torch.randint(0, V, (full_len,))}
                for _ in range(num_items)
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    def mock_collator(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "index": torch.tensor(list(range(len(batch)))),
        }

    tokenizer = Mock()
    tokenizer.decode = lambda x, **kwargs: "decoded"

    result = raw_trajectory_metrics(
        model,
        metrics=["probability"],
        data=MockDataset(),
        collators=mock_collator,
        tokenizer=tokenizer,
        batch_size=1,
        trajectory_config={
            "return_logits": True,
            "return_fixation_steps": True,
            "sampler_kwargs": {"steps": S, "max_new_tokens": L_gen},
        },
    )

    assert result is not None
    if "agg_value" in result:
        agg = result["agg_value"]
        assert isinstance(agg, dict)
        for traj_name, metrics_dict in agg.items():
            assert isinstance(metrics_dict, dict)
            for metric_name, arr in metrics_dict.items():
                assert arr is not None and (hasattr(arr, "size") and arr.size > 0 or len(arr) > 0)
    if "value_by_index" in result:
        assert result["value_by_index"] == {}, (
            "trajectory_metrics should not retain full per-sample trajectories; value_by_index should be empty"
        )
    if isinstance(result, dict):
        for v in result.values():
            if isinstance(v, dict) and "value_by_index" in v:
                assert v["value_by_index"] == {}, (
                    "per-metric value_by_index should be empty (only metric-sized outputs retained)"
                )


def test_prepare_kwargs_uses_injected_data_and_collators():
    """UnlearningMetric.prepare_kwargs_evaluate_metric skips get_datasets/get_collators when data/collators provided."""
    from evals.metrics.base import UnlearningMetric

    def dummy_fn(model, **kwargs):
        return {"agg_value": 0.0}

    metric = UnlearningMetric(name="test", metric_fn=dummy_fn)
    mock_model = MagicMock()

    # When data and collators are already in kwargs, they should not be replaced
    shared_data = MagicMock()
    shared_collators = MagicMock()
    kwargs = {
        "data": shared_data,
        "collators": shared_collators,
        "tokenizer": None,
        "template_args": None,
    }

    with patch.object(metric, "get_datasets") as mock_get_datasets:
        with patch.object(metric, "get_collators") as mock_get_collators:
            result = metric.prepare_kwargs_evaluate_metric(mock_model, "test", **kwargs)

    assert result.get("data") is shared_data
    assert result.get("collators") is shared_collators
    mock_get_datasets.assert_not_called()
    mock_get_collators.assert_not_called()
