"""Real-config eval tests from within open-unlearning: Hydra compose + get_evaluators.

Loads real configs from this repo's configs/ (experiment=eval/tofu/default,
eval/muse/default), passes Hydra validation, and asserts get_evaluators succeeds.
Covers TOFU/MUSE trajectory with all/some metrics and with/without reference.
Run from repo root: uv run pytest open-unlearning/tests/test_eval_real_configs_hydra.py -v
Or from open-unlearning: uv run pytest tests/test_eval_real_configs_hydra.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
sys.path.insert(0, str(REPO_ROOT / "src"))


def _compose_eval(overrides: list[str]):
    """Compose eval config from open-unlearning configs. Returns DictConfig."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR.resolve()),
        job_name="test",
    ):
        return compose(config_name="eval.yaml", overrides=overrides)


def _get_evaluators_from_cfg(cfg):
    """Normalize cfg.eval and call get_evaluators (same logic as eval.py / dllm run_validate_config)."""
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


def _canonical_retain_json_path(tmp_path: Path) -> Path:
    """Write minimal canonical retain reference JSON and return path."""
    p = tmp_path / "retain_ref.json"
    data = {
        "mia_min_k": {"agg_value": 0.1},
        "forget_truth_ratio": {
            "value_by_index": {"0": {"score": 0.5}},
            "agg_value": 0.5,
        },
    }
    p.write_text(json.dumps(data))
    return p


class TestTofuRealConfigsHydra:
    """TOFU: real Hydra compose from open-unlearning configs + get_evaluators."""

    def test_tofu_trajectory_hydra_accepts_plus_diffusion_adapter_max_new_tokens(self) -> None:
        """Compose with +model.diffusion_adapter.max_new_tokens (dllm eval CLI injects this).
        Base model config has no diffusion_adapter, so override must use + to append."""
        cfg = _compose_eval([
            "experiment=eval/tofu/default",
            "model=LLaDA-8B-Instruct",
            "task_name=test",
            "eval=tofu_trajectory",
            "eval.tofu_trajectory.forget_split=forget10",
            "eval.tofu_trajectory.holdout_split=holdout10",
            "eval.tofu_trajectory.samples=2",
            "+model.diffusion_adapter.max_new_tokens=200",
        ])
        assert hasattr(cfg.model, "diffusion_adapter")
        assert cfg.model.diffusion_adapter.max_new_tokens == 200

    def test_tofu_trajectory_all_metrics_no_reference(self) -> None:
        _require_trajectory_metrics()
        cfg = _compose_eval([
            "experiment=eval/tofu/default",
            "model=LLaDA-8B-Instruct",
            "task_name=test",
            "eval=tofu_trajectory",
            "eval.tofu_trajectory.forget_split=forget10",
            "eval.tofu_trajectory.holdout_split=holdout10",
            "eval.tofu_trajectory.samples=2",
        ])
        evaluators = _get_evaluators_from_cfg(cfg)
        assert len(evaluators) >= 1
        ev = next(iter(evaluators.values()))
        assert ev.name == "TOFU"
        assert "trajectory_all" in ev.metrics

    def test_tofu_trajectory_some_metrics_no_reference(self) -> None:
        _require_trajectory_metrics()
        cfg = _compose_eval([
            "experiment=eval/tofu/default",
            "model=LLaDA-8B-Instruct",
            "task_name=test",
            "eval=tofu_trajectory",
            "eval.tofu_trajectory.forget_split=forget10",
            "eval.tofu_trajectory.holdout_split=holdout10",
            "eval.tofu_trajectory.samples=2",
            "eval.tofu_trajectory.metrics.trajectory_all.include_metrics=[probability,rouge,privleak]",
        ])
        evaluators = _get_evaluators_from_cfg(cfg)
        assert len(evaluators) >= 1
        ev = next(iter(evaluators.values()))
        assert ev.name == "TOFU"

    def test_tofu_trajectory_with_reference(self, tmp_path: Path) -> None:
        _require_trajectory_metrics()
        ref_path = _canonical_retain_json_path(tmp_path)
        cfg = _compose_eval([
            "experiment=eval/tofu/default",
            "model=LLaDA-8B-Instruct",
            "task_name=test",
            "eval=tofu_trajectory",
            "eval.tofu_trajectory.forget_split=forget10",
            "eval.tofu_trajectory.holdout_split=holdout10",
            "eval.tofu_trajectory.samples=2",
            f"eval.tofu_trajectory.retain_logs_path={ref_path}",
        ])
        evaluators = _get_evaluators_from_cfg(cfg)
        assert len(evaluators) >= 1
        ev = next(iter(evaluators.values()))
        assert str(ev.eval_cfg.get("retain_logs_path")) == str(ref_path)

    def test_tofu_non_trajectory_samples_and_retain_reference_mode_compose(self) -> None:
        """Standard eval=tofu accepts eval.tofu.samples and retain_reference_mode (dllm CLI parity)."""
        cfg = _compose_eval([
            "experiment=eval/tofu/default",
            "model=LLaDA-8B-Instruct",
            "task_name=test",
            "eval=tofu",
            "forget_split=forget10",
            "holdout_split=holdout10",
            "eval.tofu.samples=5",
            "eval.tofu.retain_reference_mode=true",
        ])
        assert cfg.eval.tofu.samples == 5
        assert cfg.eval.tofu.retain_reference_mode is True
        evaluators = _get_evaluators_from_cfg(cfg)
        ev = next(iter(evaluators.values()))
        assert ev.name == "TOFU"
        assert ev.eval_cfg.get("samples") == 5
        assert ev.eval_cfg.get("retain_reference_mode") is True

    def test_tofu_non_trajectory_retain_logs_path_via_global(self, tmp_path: Path) -> None:
        """Experiment wires retain_logs_path into eval.tofu.retain_logs_path."""
        ref_path = _canonical_retain_json_path(tmp_path)
        cfg = _compose_eval([
            "experiment=eval/tofu/default",
            "model=LLaDA-8B-Instruct",
            "task_name=test",
            "eval=tofu",
            "forget_split=forget10",
            "holdout_split=holdout10",
            f"retain_logs_path={ref_path}",
        ])
        assert str(cfg.eval.tofu.retain_logs_path) == str(ref_path)
        evaluators = _get_evaluators_from_cfg(cfg)
        ev = next(iter(evaluators.values()))
        assert str(ev.eval_cfg.get("retain_logs_path")) == str(ref_path)

    def test_tofu_non_trajectory_forget_quality_interpolated_path_null_when_no_retain_logs(
        self,
    ) -> None:
        r"""forget_quality YAML uses path: ${eval.tofu.retain_logs_path}; null when unset."""
        cfg = _compose_eval([
            "experiment=eval/tofu/default",
            "model=LLaDA-8B-Instruct",
            "task_name=test",
            "eval=tofu",
            "forget_split=forget10",
            "holdout_split=holdout10",
            "eval.tofu.retain_reference_mode=true",
        ])
        fq = cfg.eval.tofu.metrics.forget_quality
        assert fq.reference_logs.retain_model_logs.path is None
        evaluators = _get_evaluators_from_cfg(cfg)
        ev = next(iter(evaluators.values()))
        assert ev.eval_cfg.get("retain_reference_mode") is True

    def test_tofu_non_trajectory_privleak_interpolated_path_null(self) -> None:
        cfg = _compose_eval([
            "experiment=eval/tofu/default",
            "model=LLaDA-8B-Instruct",
            "task_name=test",
            "eval=tofu",
            "forget_split=forget10",
            "holdout_split=holdout10",
        ])
        pl = cfg.eval.tofu.metrics.privleak
        assert pl.reference_logs.retain_model_logs.path is None
        _get_evaluators_from_cfg(cfg)


class TestMuseRealConfigsHydra:
    """MUSE: real Hydra compose from open-unlearning configs + get_evaluators."""

    def test_muse_trajectory_all_metrics_no_reference(self) -> None:
        _require_trajectory_metrics()
        cfg = _compose_eval([
            "experiment=eval/muse/default",
            "model=LLaDA-8B-Instruct",
            "task_name=test",
            "eval=muse_trajectory",
            "eval.muse_trajectory.data_split=News",
            "eval.muse_trajectory.samples=2",
        ])
        evaluators = _get_evaluators_from_cfg(cfg)
        assert len(evaluators) >= 1
        ev = next(iter(evaluators.values()))
        assert ev.name == "MUSE"
        assert "trajectory_muse_knowmem" in ev.metrics
        assert "trajectory_muse_verbmem" in ev.metrics

    def test_muse_trajectory_some_metrics_no_reference(self) -> None:
        _require_trajectory_metrics()
        cfg = _compose_eval([
            "experiment=eval/muse/default",
            "model=LLaDA-8B-Instruct",
            "task_name=test",
            "eval=muse_trajectory",
            "eval.muse_trajectory.data_split=Books",
            "eval.muse_trajectory.samples=2",
            "eval.muse_trajectory.metrics.trajectory_muse_knowmem.include_metrics=[forget_knowmem_rouge]",
            "eval.muse_trajectory.metrics.trajectory_muse_verbmem.include_metrics=[privleak]",
        ])
        evaluators = _get_evaluators_from_cfg(cfg)
        assert len(evaluators) >= 1
        ev = next(iter(evaluators.values()))
        assert ev.name == "MUSE"

    def test_muse_trajectory_with_reference(self, tmp_path: Path) -> None:
        _require_trajectory_metrics()
        ref_path = _canonical_retain_json_path(tmp_path)
        cfg = _compose_eval([
            "experiment=eval/muse/default",
            "model=LLaDA-8B-Instruct",
            "task_name=test",
            "eval=muse_trajectory",
            "eval.muse_trajectory.data_split=News",
            "eval.muse_trajectory.samples=2",
            f"eval.muse_trajectory.retain_logs_path={ref_path}",
        ])
        evaluators = _get_evaluators_from_cfg(cfg)
        assert len(evaluators) >= 1
        ev = next(iter(evaluators.values()))
        assert str(ev.eval_cfg.get("retain_logs_path")) == str(ref_path)

    def test_muse_non_trajectory_samples_retain_mode_compose(self) -> None:
        cfg = _compose_eval([
            "experiment=eval/muse/default",
            "model=LLaDA-8B-Instruct",
            "task_name=test",
            "eval=muse",
            "data_split=News",
            "eval.muse.samples=3",
            "eval.muse.retain_reference_mode=true",
        ])
        assert cfg.eval.muse.samples == 3
        assert cfg.eval.muse.retain_reference_mode is True
        evaluators = _get_evaluators_from_cfg(cfg)
        ev = next(iter(evaluators.values()))
        assert ev.name == "MUSE"
        assert ev.eval_cfg.get("samples") == 3

    def test_muse_non_trajectory_retain_logs_path_via_global(self, tmp_path: Path) -> None:
        ref_path = _canonical_retain_json_path(tmp_path)
        cfg = _compose_eval([
            "experiment=eval/muse/default",
            "model=LLaDA-8B-Instruct",
            "task_name=test",
            "eval=muse",
            "data_split=Books",
            f"retain_logs_path={ref_path}",
        ])
        assert str(cfg.eval.muse.retain_logs_path) == str(ref_path)
        evaluators = _get_evaluators_from_cfg(cfg)
        ev = next(iter(evaluators.values()))
        assert str(ev.eval_cfg.get("retain_logs_path")) == str(ref_path)
