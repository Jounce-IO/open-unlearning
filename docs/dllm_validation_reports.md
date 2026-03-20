# dLLM validation reports (TOFU E2E)

The **dllm** repo stores captured evaluator outputs under `reports/main/<run-id>/` (`results.json`, `summary.md`) for retain / unlearn validation jobs (e.g. TOFU forget10, 5 samples, trajectory vs non-trajectory).

- **Index:** [dllm `docs/validation-e2e-runs.md`](https://github.com/Jounce-IO/dllm/blob/docs/validation-e2e-tofu-s5-reports/docs/validation-e2e-runs.md) (tracked run table + links to PRs and report paths; use `main` in the URL after the dllm PR merges).
- **Semantics:** retain reference `TOFU_EVAL.json` shape and flat metric keys are produced by the Open Unlearning evaluator; loaders live in `evals.metrics.base` (`load_and_validate_reference`).

This file is only a **pointer** so submodule readers can find the canonical documentation on the main dllm tree.
