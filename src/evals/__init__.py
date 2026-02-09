from typing import Dict, Any
from omegaconf import DictConfig
from evals.tofu import TOFUEvaluator
from evals.muse import MUSEEvaluator
from evals.lm_eval import LMEvalEvaluator

EVALUATOR_REGISTRY: Dict[str, Any] = {}


def _register_evaluator(evaluator_class):
    EVALUATOR_REGISTRY[evaluator_class.__name__] = evaluator_class


def get_evaluator(name: str, eval_cfg: DictConfig, **kwargs):
    evaluator_handler_name = eval_cfg.get("handler")
    assert evaluator_handler_name is not None, ValueError(f"{name} handler not set")
    eval_handler = EVALUATOR_REGISTRY.get(evaluator_handler_name)
    if eval_handler is None:
        raise NotImplementedError(
            f"{evaluator_handler_name} not implemented or not registered"
        )
    return eval_handler(eval_cfg, **kwargs)


def _is_single_evaluator_config(eval_cfgs: DictConfig) -> bool:
    """True if eval_cfgs is the direct content of one eval group (handler, metrics, ...), not a dict of named configs."""
    handler = eval_cfgs.get("handler")
    if handler is None:
        return False
    # Handler must be a string (e.g. "TOFUEvaluator"), not a nested config
    if not isinstance(handler, str):
        return False
    # If any direct child is a DictConfig with its own "handler", this is a dict of evaluators
    try:
        for key in eval_cfgs:
            if key == "handler":
                continue
            child = eval_cfgs.get(key)
            if hasattr(child, "get") and child.get("handler") is not None:
                return False
    except Exception:
        pass
    return True


def get_evaluators(eval_cfgs: DictConfig, **kwargs):
    evaluators = {}
    # When Hydra composes with a single eval choice (e.g. eval=tofu_trajectory_multi), cfg.eval
    # can be the merged content (handler, metrics, ...) with no wrapper key. Treat as one evaluator.
    if _is_single_evaluator_config(eval_cfgs):
        eval_cfgs = {"eval": eval_cfgs}
    for eval_name, eval_cfg in eval_cfgs.items():
        # Skip fragment configs that don't have a handler (e.g. experiment merge adds tofu: {forget_split}).
        if eval_cfg.get("handler") is None:
            continue
        evaluators[eval_name] = get_evaluator(eval_name, eval_cfg, **kwargs)
    return evaluators


# Register Your benchmark evaluators
_register_evaluator(TOFUEvaluator)
_register_evaluator(MUSEEvaluator)
_register_evaluator(LMEvalEvaluator)
