"""Model preparation for eval: optionally wrap with DiffusionModelAdapter when trajectory metrics are used."""

from __future__ import annotations

import logging
from omegaconf import DictConfig, OmegaConf

from evals.__init__ import _is_single_evaluator_config

logger = logging.getLogger(__name__)


def _eval_cfg_uses_trajectory_metrics(eval_cfg: DictConfig) -> bool:
    """True if this eval config has any metric with handler trajectory_metrics."""
    metrics = eval_cfg.get("metrics")
    if not metrics:
        return False
    for _name, m_cfg in (metrics.items() if hasattr(metrics, "items") else []):
        if getattr(m_cfg, "handler", None) == "trajectory_metrics":
            return True
    return False


def _cfg_has_trajectory_metrics(cfg: DictConfig) -> bool:
    """True if any evaluator in cfg.eval uses trajectory_metrics."""
    eval_cfgs = cfg.get("eval")
    if eval_cfgs is None:
        return False
    if _is_single_evaluator_config(eval_cfgs):
        return _eval_cfg_uses_trajectory_metrics(eval_cfgs)
    for _name, eval_cfg in (eval_cfgs.items() if hasattr(eval_cfgs, "items") else []):
        if eval_cfg.get("handler") is None:
            continue
        if _eval_cfg_uses_trajectory_metrics(eval_cfg):
            return True
    return False


def _cfg_model_is_diffusion(cfg: DictConfig) -> bool:
    """True if model config indicates diffusion (LLaDA) or use_diffusion_adapter."""
    if cfg.get("model") is None:
        return False
    model_cfg = cfg.model
    if OmegaConf.select(model_cfg, "use_diffusion_adapter") is True:
        return True
    path = OmegaConf.select(model_cfg, "model_args.pretrained_model_name_or_path")
    if path is None:
        return False
    return "llada" in str(path).lower()


def should_wrap_model_for_trajectory(cfg: DictConfig) -> bool:
    """Return True when eval uses trajectory_metrics and model is diffusion (e.g. LLaDA)."""
    return _cfg_has_trajectory_metrics(cfg) and _cfg_model_is_diffusion(cfg)


def prepare_model_for_eval(cfg: DictConfig, model, tokenizer):
    """Optionally wrap model with DiffusionModelAdapter when trajectory metrics are used."""
    if not should_wrap_model_for_trajectory(cfg):
        return model
    try:
        from dllm.integrations.open_unlearning_adapter import DiffusionModelAdapter
    except ImportError as e:
        logger.error(
            "Trajectory metrics require DiffusionModelAdapter (dllm). "
            "Install dllm or run from the dllm workspace. %s",
            e,
        )
        raise RuntimeError(
            "Trajectory metrics require model.sampler (DiffusionModelAdapter). "
            "Install dllm and ensure LLaDA/diffusion model is used with trajectory eval."
        ) from e
    return DiffusionModelAdapter(model, tokenizer)
