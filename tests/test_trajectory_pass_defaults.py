"""Default guided variant for guided passes (guided_native / guided_skew)."""

from evals.metrics.trajectory_pass_binding import get_pass_spec


def test_forget_guided_native_default():
    assert get_pass_spec("forget__guided_native").evaluation_mode == "guided_native"
