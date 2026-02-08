# Benchmarking trajectory evaluation (single model-pass optimization)

This document describes how to compare evaluation runtime with and without the single model-pass optimization using py-spy.

## Goal

When the evaluator is configured with **multiple trajectory metrics** (e.g. 7 separate entries), the optimization coalesces them into **one** trajectory_metrics run with all sub-metrics. That yields:

- **One model pass per item** (instead of N passes when N metrics)
- **One data load** (shared data passed into the single run)
- **Up to ~Nx speedup** (e.g. ~7x when 7 metrics)

## Py-spy profiling

### Command

Limit evaluation to **~100 samples** for faster comparison:

```bash
# Set samples via config (e.g. eval.tofu_trajectory.samples=100)
# Then run eval under py-spy:

uvx py-spy record -f raw -o /tmp/pyspy_eval.raw --subprocesses --full-filenames -r 10 -- <your_eval_command>
```

Example with a typical dllm eval invocation:

```bash
uvx py-spy record -f raw -o /tmp/pyspy_eval.raw --subprocesses --full-filenames -r 10 -- \
  python -m eval experiment=eval/tofu model=... eval.tofu_trajectory.samples=100
```

### Compare main vs branch

1. **On main (without optimization):** Use a config that registers **7 separate** trajectory metric entries (each with one sub-metric). Run the same eval command with `eval.tofu_trajectory.samples=100`. Record wall-clock time and optionally the py-spy output path.

2. **On feature branch (with optimization):** Use the same config (7 separate trajectory metrics). Run the same eval command with `eval.tofu_trajectory.samples=100`. The evaluator will coalesce them into one trajectory_metrics run.

3. **Compare:** Wall-clock time on branch should be up to ~7x lower than on main when 7 metrics are configured. Optionally compare py-spy raw outputs (e.g. time in sampler vs in metric computation).

### Notes

- With a **single** metric entry (e.g. `trajectory_all` with all 7 sub-metrics), main and branch behave the same: one trajectory_metrics call, one model pass per item. The speedup appears when the config has **multiple** trajectory metric entries that are coalesced on the branch.
- The optional script `scripts/run_eval_pyspy.sh` (if present) wraps the above for a standard eval + 100 samples.
