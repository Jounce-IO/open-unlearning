"""Regression (008-traj-capture-parity): coalesced trajectory_all must forward include_metrics.

Without this, Hydra include_metrics=[probability] is ignored and full trajectory MU runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

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


def _get_evaluators_from_cfg(cfg):
    from omegaconf import OmegaConf, open_dict

    eval_cfgs = cfg.eval
    with open_dict(eval_cfgs):
        has_handler = eval_cfgs.get("handler") is not None
    if has_handler:
        eval_cfgs = {"eval": eval_cfgs}
    if not hasattr(eval_cfgs, "items") or not callable(getattr(eval_cfgs, "items", None)):
        eval_cfgs = OmegaConf.create(dict(eval_cfgs))
    from evals import get_evaluators

    return get_evaluators(eval_cfgs)


def _require_trajectory_metrics() -> None:
    try:
        from evals.metrics import METRICS_REGISTRY

        if "trajectory_metrics" not in METRICS_REGISTRY:
            pytest.skip("trajectory_metrics not registered")
    except Exception:
        pytest.skip("evals not importable")


@pytest.mark.integration
def test_coalesced_trajectory_forwards_include_metrics_to_evaluate_metric(
    tmp_path: Path,
) -> None:
    """YAML include_metrics must reach trajectory_metrics kwargs (coalesced Evaluator path)."""
    _require_trajectory_metrics()
    cfg = _compose_eval(
        [
            "experiment=eval/tofu/default",
            "model=LLaDA-8B-Instruct",
            "task_name=test",
            "eval=tofu_trajectory",
            "eval.tofu_trajectory.forget_split=forget10",
            "eval.tofu_trajectory.holdout_split=holdout10",
            "eval.tofu_trajectory.samples=2",
            "eval.tofu_trajectory.metrics.trajectory_all.include_metrics=[probability]",
            f"paths.output_dir={tmp_path}",
        ]
    )
    evaluators = _get_evaluators_from_cfg(cfg)
    ev = next(iter(evaluators.values()))
    traj_metric = ev.metrics["trajectory_all"]
    captured: dict = {}
    orig_em = traj_metric.evaluate_metric

    def _capture_evaluate_metric(model, metric_name, **kw):
        captured["include_metrics"] = kw.get("include_metrics")
        return {"trajectory_forget_Q_A_Prob": {"agg_value": 0.42}}

    traj_metric.evaluate_metric = _capture_evaluate_metric
    try:
        mock_model = MagicMock()
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.config = MagicMock()
        mock_model.config._name_or_path = "test/stub"
        tokenizer = MagicMock()
        ev.evaluate(mock_model, output_dir=str(tmp_path), overwrite=True, tokenizer=tokenizer)
    finally:
        traj_metric.evaluate_metric = orig_em

    im = captured.get("include_metrics")
    assert im is not None, "include_metrics was not passed through coalesced merged_args"
    assert list(im) == ["probability"]


@pytest.mark.integration
def test_coalesced_include_metrics_probability_loads_forget_dataset_only(
    tmp_path: Path,
) -> None:
    """Subset include_metrics must not pull retain/holdout/ra/wf into get_datasets (008 T008)."""
    _require_trajectory_metrics()
    cfg = _compose_eval(
        [
            "experiment=eval/tofu/default",
            "model=LLaDA-8B-Instruct",
            "task_name=test",
            "eval=tofu_trajectory",
            "eval.tofu_trajectory.forget_split=forget10",
            "eval.tofu_trajectory.holdout_split=holdout10",
            "eval.tofu_trajectory.samples=2",
            "eval.tofu_trajectory.metrics.trajectory_all.include_metrics=[probability]",
            f"paths.output_dir={tmp_path}",
        ]
    )
    evaluators = _get_evaluators_from_cfg(cfg)
    ev = next(iter(evaluators.values()))
    traj_metric = ev.metrics["trajectory_all"]
    captured: dict = {}
    orig_gd = traj_metric.get_datasets

    def _wrap_gd(dataset_cfgs=None, **kwargs):
        from omegaconf import OmegaConf

        if dataset_cfgs is None:
            captured["access_keys"] = []
            return orig_gd(dataset_cfgs=dataset_cfgs, **kwargs)
        if OmegaConf.is_config(dataset_cfgs):
            dc = OmegaConf.to_container(dataset_cfgs, resolve=True) or {}
        else:
            dc = dict(dataset_cfgs) if hasattr(dataset_cfgs, "items") else {}
        aks = sorted(
            str(v.get("access_key", n))
            for n, v in dc.items()
            if isinstance(v, dict)
        )
        captured["access_keys"] = aks
        return orig_gd(dataset_cfgs=dataset_cfgs, **kwargs)

    traj_metric.get_datasets = _wrap_gd
    orig_em = traj_metric.evaluate_metric

    def _stub_em(model, metric_name, **kw):
        return {"trajectory_forget_Q_A_Prob": {"agg_value": 0.42}}

    traj_metric.evaluate_metric = _stub_em
    try:
        mock_model = MagicMock()
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.config = MagicMock()
        mock_model.config._name_or_path = "test/stub"
        tokenizer = MagicMock()
        ev.evaluate(mock_model, output_dir=str(tmp_path), overwrite=True, tokenizer=tokenizer)
    finally:
        traj_metric.get_datasets = orig_gd
        traj_metric.evaluate_metric = orig_em

    assert captured.get("access_keys") == ["forget"]
