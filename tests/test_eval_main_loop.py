"""Tests for eval.py main loop - ensures LMEvalEvaluator (no .metrics) doesn't crash.

LMEvalEvaluator uses tasks, not metrics. The eval.py loop calls
log_retain_logs_path_none_if_needed(evaluator.metrics, ...) for each evaluator.
Without hasattr guard, LMEvalEvaluator raises AttributeError.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

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


def test_evaluator_passes_eval_cfg_to_metric_fn_in_per_metric_loop():
    """Evaluator must pass eval_cfg (and thus retain_reference_mode) to metric_fn so trajectory_metrics can write by_step keys."""
    from evals import get_evaluators

    eval_cfg = OmegaConf.create({
        "handler": "TOFUEvaluator",
        "output_dir": "/tmp/test_eval_cfg",
        "overwrite": True,
        "metrics": {
            "trajectory_all": {
                "handler": "trajectory_metrics",
                "datasets": {},
                "metrics": ["privleak", "truth_ratio"],
            }
        },
    })
    with patch("evals.base.get_metrics") as get_metrics_mock:
        mock_metric_fn = Mock(return_value={"agg_value": 0.5})
        get_metrics_mock.return_value = {"trajectory_all": mock_metric_fn}
        evaluators = get_evaluators({"tofu_trajectory": eval_cfg})
    ev = evaluators["tofu_trajectory"]
    mock_model = MagicMock()
    with patch.object(ev, "load_logs_from_file", return_value={}), patch.object(ev, "save_logs"), patch.object(ev, "prepare_model", return_value=mock_model):
        ev.evaluate(mock_model)
    mock_metric_fn.assert_called_once()
    call_kwargs = mock_metric_fn.call_args[1]
    assert "eval_cfg" in call_kwargs, "metric_fn must receive eval_cfg so trajectory_metrics can write by_step keys in retain_reference_mode"
    assert call_kwargs["eval_cfg"] is ev.eval_cfg
