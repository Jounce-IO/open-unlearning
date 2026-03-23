"""Hydra: non-traj eval defaults for generalized sequence probability (006)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
sys.path.insert(0, str(REPO_ROOT / "src"))


def _compose_eval(overrides: list[str]):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR.resolve()),
        job_name="test",
    ):
        return compose(config_name="eval.yaml", overrides=overrides)


def test_tofu_non_traj_default_use_generalized_true() -> None:
    cfg = _compose_eval(
        [
            "experiment=eval/tofu/default",
            "model=LLaDA-8B-Instruct",
            "task_name=test",
            "eval=tofu",
            "eval.tofu.forget_split=forget10",
            "eval.tofu.holdout_split=holdout10",
        ]
    )
    assert cfg.eval.tofu.use_generalized_sequence_probability is True
    assert cfg.eval.tofu.logit_alignment == "causal"


def test_muse_non_traj_default_use_generalized_true() -> None:
    cfg = _compose_eval(
        [
            "experiment=eval/muse/default",
            "model=LLaDA-8B-Instruct",
            "task_name=test",
            "eval=muse",
        ]
    )
    assert cfg.eval.muse.use_generalized_sequence_probability is True
    assert cfg.eval.muse.logit_alignment == "causal"
