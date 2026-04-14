"""Regression: ROUGE display leg is guidance-free in binding table."""

from evals.metrics.trajectory_pass_binding import DISPLAY_METRIC_BINDING, get_pass_spec


def test_forget_rouge_is_guidance_free_in_binding():
    regime, _ = DISPLAY_METRIC_BINDING["trajectory_forget_Q_A_ROUGE"]
    assert regime == "guidance_free"


def test_forget_unguided_pass_uses_unguided_mode():
    assert get_pass_spec("forget__unguided").evaluation_mode == "unguided"
