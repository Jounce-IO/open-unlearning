"""Contract tests for effective_parity JSON fields (FR-012)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.effective_parity import attach_effective_parity_to_cache


class _EvalCfg:
    def __init__(self, decoupling: dict) -> None:
        self._dec = decoupling

    def get(self, key: str):
        if key == "decoupling":
            return self._dec
        return None


def test_attach_effective_parity_includes_hash_and_schema() -> None:
    cache: dict = {}
    dec = {
        "trajectory_mode": "disabled",
        "benchmark": "tofu",
        "split": "forget10",
        "feature_profile_hash": "abc123",
        "applicability_statuses": "trajectory_pass:not_applicable,evaluation_mode:applicable",
    }
    model = SimpleNamespace(
        adapter_config=SimpleNamespace(
            evaluation_mode="guided_native",
            tokens_per_step=4,
            max_new_tokens=200,
            trajectory_sample_interval=8,
        )
    )
    attach_effective_parity_to_cache(cache, _EvalCfg(dec), model)
    ep = cache["effective_parity"]
    assert ep["schema_version"] == 1
    assert len(ep["effective_parity_hash"]) == 64
    pr = ep["effective_parity_snapshot"]["parity_relevant"]
    assert pr["evaluation_mode"] == "guided_native"
    assert pr["tokens_per_step"] == 4
    assert "applicability_statuses" in pr


def test_attach_effective_parity_idempotent() -> None:
    cache: dict = {"effective_parity": {"schema_version": 99}}
    attach_effective_parity_to_cache(
        cache,
        _EvalCfg({"trajectory_mode": "disabled"}),
        SimpleNamespace(adapter_config=None),
    )
    assert cache["effective_parity"]["schema_version"] == 99


def test_attach_effective_parity_skips_without_decoupling() -> None:
    cache: dict = {}
    attach_effective_parity_to_cache(cache, _EvalCfg(None), SimpleNamespace())
    assert "effective_parity" not in cache
