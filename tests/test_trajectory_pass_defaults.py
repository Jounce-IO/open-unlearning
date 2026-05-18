"""Hydra compose: trajectory TR passes use OU TOFU_QA_* dataset wiring."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from pathlib import Path

OU_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    "pass_pkg,expected_dataset_key,answer_key",
    [
        (
            "trajectory_pass_retain_guided_tr_pert",
            "TOFU_QA_retain_pert",
            "perturbed_answer",
        ),
        (
            "trajectory_pass_ra_guided_tr_pert",
            "TOFU_QA_ra_pert",
            "perturbed_answer",
        ),
        (
            "trajectory_pass_retain_guided_tr_para",
            "TOFU_QA_retain_para",
            "paraphrased_answer",
        ),
        (
            "trajectory_pass_ra_guided_tr_correct",
            "TOFU_QA_ra",
            "answer",
        ),
    ],
)
def test_trajectory_pass_resolved_dataset(
    pass_pkg: str, expected_dataset_key: str, answer_key: str
) -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(OU_CONFIGS),
    ):
        cfg = compose(
            config_name="eval",
            overrides=[
                "eval=tofu_trajectory",
                f"eval/tofu_metrics@eval.tofu_trajectory.tofu_metrics={pass_pkg}",
            ],
        )
    node = cfg.eval.tofu_trajectory.metrics[pass_pkg]
    ds_cfg = OmegaConf.to_container(node.datasets, resolve=True)
    assert isinstance(ds_cfg, dict)
    matched = [
        (name, spec)
        for name, spec in ds_cfg.items()
        if expected_dataset_key in str(name) or expected_dataset_key in str(spec)
    ]
    assert matched, f"{pass_pkg}: no dataset matching {expected_dataset_key!r}"
    for _name, spec in matched:
        args = spec.get("args") or {}
        assert args.get("answer_key") == answer_key
