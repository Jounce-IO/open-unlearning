"""008 US1 / T006: dataset-scope parity preflight (unified preset = T008).

Today `trajectory_all` composes five access_key slots (forget/holdout/retain/ra/wf) while
standard `eval=tofu` with a single metric uses a smaller loader graph. T008 should make
paired capture on/off runs share the same dataset identity.
"""

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


def _trajectory_all_access_keys(cfg) -> frozenset[str]:
    traj = cfg.eval.tofu_trajectory.metrics.trajectory_all
    ds = traj.datasets
    out: list[str] = []
    for k in ds:
        node = ds[k]
        ak = node.get("access_key") if hasattr(node, "get") else None
        if ak is not None:
            out.append(str(ak))
    return frozenset(out)


def _metric_node_access_keys(cfg, node: str) -> frozenset[str]:
    m = cfg.eval.tofu_trajectory.metrics[node]
    ds = m.datasets
    out: list[str] = []
    for k in ds:
        node_cfg = ds[k]
        ak = node_cfg.get("access_key") if hasattr(node_cfg, "get") else None
        if ak is not None:
            out.append(str(ak))
    return frozenset(out)


def _metric_node_include_metrics(cfg, node: str) -> tuple[str, ...]:
    m = cfg.eval.tofu_trajectory.metrics[node]
    inc = m.get("include_metrics") if hasattr(m, "get") else None
    if inc is None:
        return ()
    return tuple(str(x) for x in list(inc))


def test_tofu_trajectory_all_composes_five_dataset_access_keys() -> None:
    """trajectory_all YAML still lists five access_keys; coalesced load can subset via include_metrics (T008)."""
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
        ]
    )
    keys = _trajectory_all_access_keys(cfg)
    assert keys == frozenset({"forget", "holdout", "retain", "ra", "wf"})


def test_paired_muse_traj_vs_no_traj_knowmem_dataset_access_keys_match() -> None:
    """MUSE: same trajectory_muse_knowmem dataset graph for capture on vs off (dllm unified argv)."""
    base_m = [
        "experiment=eval/muse/default",
        "model=LLaDA-8B-Instruct",
        "task_name=test",
        "eval.muse_trajectory.data_split=News",
        "eval.muse_trajectory.samples=2",
    ]

    def _knowmem_access_keys(cfg) -> frozenset[str]:
        kn = cfg.eval.muse_trajectory.metrics.trajectory_muse_knowmem
        ds = kn.datasets
        out: list[str] = []
        for k in ds:
            node = ds[k]
            ak = node.get("access_key") if hasattr(node, "get") else None
            if ak is not None:
                out.append(str(ak))
        return frozenset(out)

    traj_m = _compose_eval(["eval=muse_trajectory", *base_m])
    nt_m = _compose_eval(
        ["eval=muse_trajectory", "eval.muse_trajectory.trajectory_capture=false", *base_m]
    )
    assert _knowmem_access_keys(traj_m) == _knowmem_access_keys(nt_m)


def test_tofu_single_eval_path_compose_matches_legacy_forget_unguided() -> None:
    """010: eval=tofu_trajectory + tofu_metrics override composes like legacy forget preset."""
    base = [
        "experiment=eval/tofu/default",
        "model=LLaDA-8B-Instruct",
        "task_name=test",
        "eval.tofu_trajectory.forget_split=forget10",
        "eval.tofu_trajectory.holdout_split=holdout10",
        "eval.tofu_trajectory.samples=2",
        "+eval.tofu_trajectory.metrics.trajectory_pass_forget_unguided.include_metrics=[probability]",
    ]
    legacy = _compose_eval(["eval=tofu_trajectory_forget_unguided", *base])
    modern = _compose_eval(
        [
            "eval=tofu_trajectory",
            "eval/tofu_metrics@eval.tofu_trajectory.tofu_metrics=trajectory_pass_forget_unguided",
            *base,
        ]
    )
    node = "trajectory_pass_forget_unguided"
    assert _metric_node_access_keys(legacy, node) == _metric_node_access_keys(modern, node)
    assert _metric_node_include_metrics(legacy, node) == _metric_node_include_metrics(
        modern, node
    )


def test_tofu_single_eval_path_compose_matches_legacy_forget_guided_native() -> None:
    base = [
        "experiment=eval/tofu/default",
        "model=LLaDA-8B-Instruct",
        "task_name=test",
        "eval.tofu_trajectory.forget_split=forget10",
        "eval.tofu_trajectory.holdout_split=holdout10",
        "eval.tofu_trajectory.samples=2",
        "+eval.tofu_trajectory.metrics.trajectory_pass_forget_guided_native.include_metrics=[probability]",
    ]
    legacy = _compose_eval(["eval=tofu_trajectory_forget_guided_native", *base])
    modern = _compose_eval(
        [
            "eval=tofu_trajectory",
            "eval/tofu_metrics@eval.tofu_trajectory.tofu_metrics=trajectory_pass_forget_guided_native",
            *base,
        ]
    )
    node = "trajectory_pass_forget_guided_native"
    assert _metric_node_access_keys(legacy, node) == _metric_node_access_keys(modern, node)
    assert _metric_node_include_metrics(legacy, node) == _metric_node_include_metrics(
        modern, node
    )


def test_paired_tofu_traj_vs_no_traj_dataset_access_keys_match() -> None:
    """Same trajectory_all YAML graph for capture on vs off (dllm maps no-traj to tofu_trajectory + capture false)."""
    base = [
        "experiment=eval/tofu/default",
        "model=LLaDA-8B-Instruct",
        "task_name=test",
        "eval.tofu_trajectory.forget_split=forget10",
        "eval.tofu_trajectory.holdout_split=holdout10",
        "eval.tofu_trajectory.samples=2",
        "eval.tofu_trajectory.metrics.trajectory_all.include_metrics=[probability]",
    ]
    traj_cfg = _compose_eval(["eval=tofu_trajectory", *base])
    nt_cfg = _compose_eval(["eval=tofu_trajectory", "eval.tofu_trajectory.trajectory_capture=false", *base])
    assert _trajectory_all_access_keys(traj_cfg) == _trajectory_all_access_keys(nt_cfg)
