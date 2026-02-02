"""Tests for eval.py main loop - ensures LMEvalEvaluator (no .metrics) doesn't crash.

LMEvalEvaluator uses tasks, not metrics. The eval.py loop calls
log_retain_logs_path_none_if_needed(evaluator.metrics, ...) for each evaluator.
Without hasattr guard, LMEvalEvaluator raises AttributeError.
"""

import sys
from pathlib import Path

from omegaconf import OmegaConf

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.privacy import log_retain_logs_path_none_if_needed


def test_lmeval_evaluator_has_no_metrics_attr():
    """LMEvalEvaluator has tasks, not metrics - loop must guard with hasattr."""
    from evals import get_evaluators

    # WMDP config produces LMEvalEvaluator
    eval_cfg = OmegaConf.create({
        "handler": "LMEvalEvaluator",
        "tasks": ["wmdp_cyber"],
        "output_dir": "/tmp/test",
        "overwrite": False,
        "simple_evaluate_args": {},
    })
    evaluators = get_evaluators({"lm_eval": eval_cfg})

    assert "lm_eval" in evaluators
    ev = evaluators["lm_eval"]
    assert not hasattr(ev, "metrics"), "LMEvalEvaluator should not have .metrics"
    assert hasattr(ev, "tasks")


def test_eval_loop_handles_lmeval_without_attribute_error():
    """Simulate eval.py main loop: iterating evaluators with log_retain guard must not raise."""
    from evals import get_evaluators

    eval_cfg = OmegaConf.create({
        "handler": "LMEvalEvaluator",
        "tasks": ["wmdp_cyber"],
        "output_dir": "/tmp/test",
        "overwrite": False,
        "simple_evaluate_args": {},
    })
    evaluators = get_evaluators({"lm_eval": eval_cfg})

    # Same logic as eval.py main() - must not raise AttributeError
    for evaluator_name, evaluator in evaluators.items():
        if hasattr(evaluator, "metrics"):
            log_retain_logs_path_none_if_needed(
                "start of run",
                evaluator.metrics,
                evaluator.eval_cfg.get("retain_logs_path"),
            )
