"""prepare_kwargs_evaluate_metric injects use_generalized_sequence_probability into pre_compute."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from evals.metrics.base import (  # noqa: E402
    resolve_use_generalized_sequence_probability,
    UnlearningMetric,
)


def test_resolve_eval_default() -> None:
    ev = OmegaConf.create({"use_generalized_sequence_probability": False})
    flat = {"batch_size": 1}
    assert resolve_use_generalized_sequence_probability(ev, flat) is False


def test_resolve_metric_overrides_eval() -> None:
    ev = OmegaConf.create({"use_generalized_sequence_probability": True})
    flat = {"use_generalized_sequence_probability": False}
    assert resolve_use_generalized_sequence_probability(ev, flat) is False


def test_resolve_default_true_when_missing() -> None:
    ev = OmegaConf.create({})
    flat = {}
    assert resolve_use_generalized_sequence_probability(ev, flat) is True


def test_prepare_kwargs_injects_flag_for_pre_compute(monkeypatch) -> None:
    """pre_compute branch merges parent resolved generalized flag into nested cfg."""
    from evals.metrics.memorization import probability as prob_singleton

    calls = []

    def fake_evaluate(self, model, metric_name, cache, **kw):
        calls.append(kw.get("use_generalized_sequence_probability"))
        return {"agg_value": 0.0, "value_by_index": {}}

    monkeypatch.setattr(UnlearningMetric, "evaluate", fake_evaluate)

    inner = UnlearningMetric(
        name="probability", metric_fn=prob_singleton._metric_fn
    )
    outer = UnlearningMetric(name="truth_ratio", metric_fn=lambda **x: x)
    outer.set_pre_compute_metrics({"correct": inner})

    eval_cfg = OmegaConf.create(
        {"use_generalized_sequence_probability": False, "logit_alignment": "causal"}
    )
    pre_cfg = OmegaConf.create(
        {
            "correct": {
                "handler": "probability",
                "access_key": "correct",
                "batch_size": 1,
            }
        }
    )
    kwargs = {
        "eval_cfg": eval_cfg,
        "pre_compute": pre_cfg,
        "batch_size": 1,
    }
    outer.prepare_kwargs_evaluate_metric(
        MagicMock(),
        "truth_ratio",
        cache={},
        **kwargs,
    )
    assert calls == [False]
