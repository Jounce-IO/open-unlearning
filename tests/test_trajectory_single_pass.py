"""Single-pass trajectory filter integration (no model)."""

from evals.metrics.trajectory_pass_binding import get_pass_spec


def test_pass_specs_cover_cli_documented_passes():
    for pid in (
        "forget__unguided",
        "forget__guided_native",
        "forget__guided_skew",
    ):
        s = get_pass_spec(pid)
        assert s.pass_id == pid
