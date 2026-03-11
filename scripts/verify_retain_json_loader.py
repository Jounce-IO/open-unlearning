#!/usr/bin/env python3
"""Prove that the reference_logs loader (base.py) can load the retain results JSON.
Run from repo root: uv run python scripts/verify_retain_json_loader.py
"""
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

# Retain results path (relative to dllm repo root when run from open-unlearning)
retain_json = repo_root.parent / "reports/003-retain-eval-fixes/eval-retain99-20260311-144244/results.json"
if not retain_json.exists():
    retain_json = Path("reports/003-retain-eval-fixes/eval-retain99-20260311-144244/results.json")
if not retain_json.exists():
    raise SystemExit(f"Retain JSON not found: {retain_json}")

from evals.metrics.base import UnlearningMetric


def _dummy_fn(**kwargs):
    return {"agg_value": 0.5}


def main():
    metric = UnlearningMetric("trajectory_all", _dummy_fn)
    reference_logs_config = {
        "retain_model_logs": {
            "path": str(retain_json),
            "include": {
                "mia_min_k": {"access_key": "retain"},
                "forget_truth_ratio": {"access_key": "retain"},
                "retain_Q_A_Prob": {"access_key": "retain_Q_A_Prob"},
                "retain_Q_A_ROUGE": {"access_key": "retain_Q_A_ROUGE"},
                "retain_Truth_Ratio": {"access_key": "retain_Truth_Ratio"},
            },
        },
    }
    kwargs = metric.prepare_kwargs_evaluate_metric(
        None,
        "trajectory_all",
        {},
        reference_logs=reference_logs_config,
    )
    assert "reference_logs" in kwargs, "reference_logs not in kwargs"
    ref = kwargs["reference_logs"]
    assert "retain_model_logs" in ref, "retain_model_logs not in reference_logs"
    rml = ref["retain_model_logs"]

    # Per-step data (for step-matched ks_test / privleak)
    assert "retain_forget_tr_by_step" in rml, "retain_forget_tr_by_step missing"
    assert "retain_mia_by_step" in rml, "retain_mia_by_step missing"
    tr_by_step = rml["retain_forget_tr_by_step"]
    mia_by_step = rml["retain_mia_by_step"]

    # At least one step key present (reader will look up by step value e.g. "6")
    assert len(tr_by_step) > 0, "forget_truth_ratio_by_step is empty"
    assert len(mia_by_step) > 0, "mia_min_k_by_step is empty"
    step_key = next(iter(tr_by_step))
    step_data = tr_by_step[step_key]
    assert "value_by_index" in step_data, f"Step {step_key} missing value_by_index"
    scores = [v.get("score") for v in step_data["value_by_index"].values() if isinstance(v, dict)]
    assert any(s is not None for s in scores), f"Step {step_key} has no scores (ks_test needs these)"

    # retain slot (for aggregate ks_test when step-matched not used)
    if "retain" in rml and rml["retain"] is not None:
        r = rml["retain"]
        if isinstance(r, dict) and "value_by_index" in r:
            print("retain.value_by_index present (aggregate FQ usable)")
        else:
            print("retain present (agg_value or other)")

    print("OK: loader loaded retain JSON; retain_forget_tr_by_step, retain_mia_by_step present and usable.")
    print(f"  Steps in forget_truth_ratio_by_step: {list(tr_by_step.keys())[:10]}{'...' if len(tr_by_step) > 10 else ''}")
    print(f"  Steps in mia_min_k_by_step: {list(mia_by_step.keys())[:10]}{'...' if len(mia_by_step) > 10 else ''}")


if __name__ == "__main__":
    main()
