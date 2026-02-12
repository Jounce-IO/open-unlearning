#!/usr/bin/env python3
"""
Benchmark all ROUGE backends and write a report (mean time, relative to baseline, drop-in?).

Usage:
  uv run python open-unlearning/scripts/benchmark_rouge.py [--output report.md] [--iterations N]

When run from repo root (e.g. in K8s job from /app), ensure open-unlearning/src is on PYTHONPATH
or run from open-unlearning with PYTHONPATH=src.
"""

import argparse
import platform
import sys
import time
from pathlib import Path

# Allow importing from open-unlearning/src when run from main repo (e.g. /app)
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
_src = _repo_root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from evals.metrics.rouge_impl.rouge_backends import ROUGE_BACKENDS, get_backend
from evals.metrics.rouge_impl.golden_data import ROUGE_GOLDEN_PAIRS, get_golden_gen_gt_lists


def _golden_pairs():
    return ROUGE_GOLDEN_PAIRS


def _gen_gt_lists():
    return get_golden_gen_gt_lists()


# Backends that are drop-in replacements (parity with stemmer baseline)
DROP_IN_BACKENDS = {"baseline", "minimal_python", "two_row_lcs", "batch_cached", "numpy_lcs"}
# Backends that are benchmark-only (different scores)
BENCHMARK_ONLY_BACKENDS = {"no_stemmer"}


def run_benchmark(
    iterations: int = 50,
    warmup: int = 2,
) -> list[tuple[str, float, bool, str]]:
    """
    Run each backend repeatedly; return list of (name, mean_time_sec, drop_in, notes).
    """
    gen_list, gt_list = _gen_gt_lists()
    n_pairs = len(gen_list)
    results = []
    baseline_time = None

    for name, fn in ROUGE_BACKENDS:
        if name == "numpy_lcs":
            try:
                import numpy
            except ImportError:
                results.append((name, -1.0, True, "skipped (numpy not available)"))
                continue
        # Warmup
        for _ in range(warmup):
            fn(gen_list, gt_list, use_stemmer=True, scorer=None)
        # Timed runs
        start = time.perf_counter()
        for _ in range(iterations):
            fn(gen_list, gt_list, use_stemmer=True, scorer=None)
        elapsed = time.perf_counter() - start
        mean_time = elapsed / iterations
        if baseline_time is None and name == "baseline":
            baseline_time = mean_time
        drop_in = name in DROP_IN_BACKENDS
        if name in BENCHMARK_ONLY_BACKENDS:
            notes = "No; different scores (benchmark only)."
        elif drop_in:
            notes = "Yes"
        else:
            notes = "Yes"
        results.append((name, mean_time, drop_in, notes))

    # Fill relative time
    out = []
    for name, mean_time, drop_in, notes in results:
        if mean_time < 0:
            out.append((name, mean_time, drop_in, notes))
            continue
        rel = mean_time / baseline_time if baseline_time else 1.0
        if name in BENCHMARK_ONLY_BACKENDS:
            notes = "No; different scores (benchmark only)."
        elif drop_in:
            notes = "Yes"
        out.append((name, mean_time, drop_in, notes))
    return out, baseline_time


def write_report(
    results: list[tuple[str, float, bool, str]],
    baseline_time: float,
    output_path: str | Path | None,
    iterations: int,
) -> str:
    """Format Markdown report and write to path or return as string."""
    lines = [
        "# ROUGE backends benchmark report",
        "",
        f"**Machine:** {platform.node()}",
        f"**Python:** {sys.version.split()[0]}",
        f"**Platform:** {platform.platform()}",
        "",
        f"**Golden pairs:** {len(_golden_pairs())}",
        f"**Iterations per backend:** {iterations}",
        "",
        "| Backend | Mean time (s) | Time relative to baseline | Drop-in replacement? | Notes |",
        "|---------|---------------|---------------------------|------------------------|-------|",
    ]
    for name, mean_time, drop_in, notes in results:
        if mean_time < 0:
            rel = "-"
            mean_s = "-"
        else:
            rel = f"{mean_time / baseline_time:.4f}" if baseline_time else "1.0"
            mean_s = f"{mean_time:.6f}"
        drop = "Yes" if drop_in and name not in BENCHMARK_ONLY_BACKENDS else "No"
        if name in BENCHMARK_ONLY_BACKENDS:
            notes = "Different scores (benchmark only)."
        lines.append(f"| {name} | {mean_s} | {rel} | {drop} | {notes} |")
    report = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Benchmark ROUGE backends")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output Markdown file path")
    parser.add_argument("--iterations", "-n", type=int, default=50, help="Number of iterations per backend")
    args = parser.parse_args()
    results, baseline_time = run_benchmark(iterations=args.iterations)
    report = write_report(results, baseline_time, args.output, args.iterations)
    print(report)
    if args.output:
        print(f"\nWrote report to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
