"""Tests for scripts/validate_trajectory_metric_coverage_from_log.py"""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_trajectory_metric_coverage_from_log.py"

_METRICS = [
    "probability",
    "rouge",
    "extraction_strength",
    "truth_ratio",
    "ks_test",
    "hm_aggregate",
    "privleak",
]
_TRAJS = ["steps", "fixation_start", "fixation_end", "fixation_ratio"]


def _full_synthetic_log() -> str:
    lines = []
    for traj in _TRAJS:
        for view in ("full", "eos"):
            for m in _METRICS:
                lines.append(
                    f"TRAJECTORY_METRIC_COVERAGE view={view} traj={traj} metric={m} "
                    "array_len=24 finite_values=24"
                )
    lines.append(
        "TRAJECTORY_STEP_META num_trajectory_steps=24 step_values_count=24 "
        "probability_on_steps_traj_len=24 lengths_match=True"
    )
    return "\n".join(lines) + "\n"


def _run(log: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        input=log,
        text=True,
        capture_output=True,
        cwd=str(SCRIPT.parents[1]),
    )


def test_validator_ok_full_grid():
    r = _run(_full_synthetic_log())
    assert r.returncode == 0, r.stderr + r.stdout
    assert "OK:" in r.stdout


def test_validator_require_mu():
    log = _full_synthetic_log()
    log += (
        "TRAJECTORY_MU_SUBMETRIC_STEPS mu_aggregate_steps=24 first_step=0 last_step=23\n"
        "TRAJECTORY_MU_SUBMETRIC_COVERAGE step=0 view=full submetric_count=3 "
        "submetrics=['retain_Q_A_Prob', 'retain_Q_A_ROUGE', 'retain_Truth_Ratio']\n"
        "TRAJECTORY_MU_SUBMETRIC_COVERAGE step=0 view=eos submetric_count=3 "
        "submetrics=['retain_Q_A_Prob', 'retain_Q_A_ROUGE', 'retain_Truth_Ratio']\n"
    )
    r = _run(log, "--require-mu")
    assert r.returncode == 0, r.stderr + r.stdout


def test_validator_detects_missing_traj():
    # Only steps traj: default validator expects 4 trajectory types
    short = "\n".join(
        f"TRAJECTORY_METRIC_COVERAGE view={v} traj=steps metric={m} array_len=5 finite_values=5"
        for v in ("full", "eos")
        for m in _METRICS
    )
    short += "\nTRAJECTORY_STEP_META num_trajectory_steps=5 step_values_count=5 probability_on_steps_traj_len=5 lengths_match=True\n"
    r2 = _run(short)
    assert r2.returncode == 1
    assert "MISSING" in r2.stdout


def test_validator_no_debug_exit_2():
    r = _run("INFO only\n")
    assert r.returncode == 2
