#!/usr/bin/env python3
"""
Validate trajectory eval DEBUG logs: every metric ran with consistent trajectory lengths.

Requires LOGLEVEL=DEBUG so the evaluator emits:
  TRAJECTORY_METRIC_COVERAGE view=... traj=... metric=... array_len=... finite_values=...
  TRAJECTORY_MU_SUBMETRIC_COVERAGE (when hm_aggregate runs)
  TRAJECTORY_STEP_META

Examples:
  kubectl logs job/my-eval -n dllm 2>&1 | uv run python scripts/validate_trajectory_metric_coverage_from_log.py
  uv run python scripts/validate_trajectory_metric_coverage_from_log.py /path/to/pod.log
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict

COVERAGE_RE = re.compile(
    r"TRAJECTORY_METRIC_COVERAGE view=(\S+) traj=(\S+) metric=(\S+) array_len=(\d+) finite_values=(\d+)(?: missing=(\d+))?"
)
MU_COVERAGE_RE = re.compile(
    r"TRAJECTORY_MU_SUBMETRIC_COVERAGE step=(\S+) view=(\S+) submetric_count=(\d+) submetrics=(.+)$"
)
MU_STEPS_RE = re.compile(
    r"TRAJECTORY_MU_SUBMETRIC_STEPS mu_aggregate_steps=(\d+)"
)
STEP_META_RE = re.compile(
    r"TRAJECTORY_STEP_META num_trajectory_steps=(\d+) step_values_count=(\d+) "
    r"probability_on_steps_traj_len=(\d+) lengths_match=(\S+)"
)

DEFAULT_METRICS = [
    "probability",
    "rouge",
    "extraction_strength",
    "truth_ratio",
    "ks_test",
    "hm_aggregate",
    "privleak",
]
DEFAULT_TRAJS = ["steps", "fixation_start", "fixation_end", "fixation_ratio"]
DEFAULT_VIEWS = ["full", "eos"]


def parse_log(text: str) -> tuple[dict, list[tuple], dict, list[str]]:
    """Returns (coverage, mu_rows, step_meta, errors)."""
    coverage: dict[tuple[str, str, str], tuple[int, int, bool]] = {}
    mu_rows: list[tuple[str, str, int, list[str]]] = []
    step_meta: dict[str, str] = {}
    errors: list[str] = []

    for line in text.splitlines():
        m = COVERAGE_RE.search(line)
        if m:
            view, traj, metric, alen, fin, missing = m.groups()
            a, f = int(alen), int(fin)
            is_missing = missing == "1"
            coverage[(view, traj, metric)] = (a, f, is_missing)
            continue
        m = MU_COVERAGE_RE.search(line)
        if m:
            step, view, count_s, sub_s = m.groups()
            try:
                subs = ast.literal_eval(sub_s.strip())
                if not isinstance(subs, list):
                    subs = []
            except (SyntaxError, ValueError):
                errors.append(f"Could not parse submetrics line: {line[:120]}")
                subs = []
            mu_rows.append((step, view, int(count_s), subs))
            continue
        m = MU_STEPS_RE.search(line)
        if m:
            step_meta["mu_aggregate_steps"] = m.group(1)
            continue
        m = STEP_META_RE.search(line)
        if m:
            step_meta["num_trajectory_steps"] = m.group(1)
            step_meta["step_values_count"] = m.group(2)
            step_meta["probability_len"] = m.group(3)
            step_meta["lengths_match"] = m.group(4)
    return coverage, mu_rows, step_meta, errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "logfile",
        nargs="?",
        type=argparse.FileType("r"),
        default=None,
        help="Log file (default: stdin)",
    )
    ap.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated expected metric names",
    )
    ap.add_argument(
        "--trajs",
        default=",".join(DEFAULT_TRAJS),
        help="Comma-separated trajectory names",
    )
    ap.add_argument(
        "--views",
        default=",".join(DEFAULT_VIEWS),
        help="Comma-separated views",
    )
    ap.add_argument(
        "--require-mu",
        action="store_true",
        help="Require TRAJECTORY_MU_SUBMETRIC_* lines (hm_aggregate run)",
    )
    ap.add_argument(
        "--min-mu-submetrics",
        type=int,
        default=3,
        help="Minimum submetrics per MU line when --require-mu",
    )
    args = ap.parse_args()
    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    trajs = [x.strip() for x in args.trajs.split(",") if x.strip()]
    views = [x.strip() for x in args.views.split(",") if x.strip()]

    text = (args.logfile or sys.stdin).read()
    if args.logfile:
        args.logfile.close()

    if "TRAJECTORY_METRIC_COVERAGE" not in text:
        print(
            "ERROR: No TRAJECTORY_METRIC_COVERAGE lines found. "
            "Run eval with LOGLEVEL=DEBUG (e.g. job.env.LOGLEVEL=DEBUG).",
            file=sys.stderr,
        )
        return 2

    coverage, mu_rows, step_meta, parse_errors = parse_log(text)
    for e in parse_errors:
        print(f"WARN: {e}", file=sys.stderr)

    failed = False
    for view in views:
        for traj in trajs:
            lens: set[int] = set()
            for metric in metrics:
                key = (view, traj, metric)
                if key not in coverage:
                    print(f"MISSING: no log line for view={view} traj={traj} metric={metric}")
                    failed = True
                    continue
                alen, fin, missing = coverage[key]
                if missing or alen == 0:
                    print(
                        f"MISSING_DATA: view={view} traj={traj} metric={metric} "
                        f"array_len={alen} finite={fin}"
                    )
                    failed = True
                    continue
                if fin == 0:
                    print(
                        f"ALL_NAN: view={view} traj={traj} metric={metric} "
                        f"array_len={alen} finite_values=0"
                    )
                    failed = True
                lens.add(alen)
            if len(lens) > 1:
                print(
                    f"LEN_MISMATCH: view={view} traj={traj} array_lens={sorted(lens)} "
                    f"(metrics should share trajectory depth)"
                )
                failed = True

    if step_meta.get("lengths_match") == "False":
        print(
            f"STEP_META_MISMATCH: num_trajectory_steps={step_meta.get('num_trajectory_steps')} "
            f"probability_len={step_meta.get('probability_len')}"
        )
        failed = True
    elif step_meta.get("lengths_match") == "True":
        print(
            f"OK STEP_META: steps={step_meta.get('num_trajectory_steps')} "
            f"step_values={step_meta.get('step_values_count')}"
        )

    if args.require_mu:
        if not mu_rows:
            print("MISSING: TRAJECTORY_MU_SUBMETRIC_COVERAGE (hm_aggregate / retain MU pass)")
            failed = True
        else:
            for step, view, n_sub, subs in mu_rows:
                if n_sub < args.min_mu_submetrics:
                    print(
                        f"MU_SUBMETRIC_SHORT: step={step} view={view} count={n_sub} "
                        f"(expected >= {args.min_mu_submetrics})"
                    )
                    failed = True
            print(f"OK MU: {len(mu_rows)} submetric coverage lines, steps={step_meta.get('mu_aggregate_steps')}")

    if not failed:
        n = len([(v, t, m) for v in views for t in trajs for m in metrics])
        print(f"OK: {n} metric×view×traj combinations present with data")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
