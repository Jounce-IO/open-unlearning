# TOFU independent trajectory passes (Jounce-IO fork)

This fork splits OU scalar TOFU metrics into **one Hydra metric package per inference pass** (`configs/eval/tofu_metrics/trajectory_pass_*.yaml`), each with a stable **`trajectory_pass_id`** for merge in the main **dllm** repo.

**Downstream merge doc:** [dllm `docs/tofu-trajectory-multi-pass-merge.md`](https://github.com/Jounce-IO/dllm/blob/feat/tofu-retain-sft-trajectory/docs/tofu-trajectory-multi-pass-merge.md) (path on branch `feat/tofu-retain-sft-trajectory`).

---

## `retain_sft__unguided`

- **Config:** `trajectory_pass_retain_sft_unguided.yaml`
- **Pass id:** `retain_sft__unguided`
- **Dataset:** `QADataset` on `locuslab/TOFU`, `hf_args.name: ${eval.tofu_trajectory.retain_sft_split}` (`retain90` / `retain95` / `retain99`)
- **Gold:** `answer` (SFT six-split parity)
- **Metric key:** `trajectory_retain_sft_Q_A_ROUGE`
- **Not in Model Utility** (ROUGE-only optional pass for plots vs `trajectory_retain_Q_A_ROUGE`)

**Contrast `retain__unguided`:** also `QADataset` + **`answer`** on split **`retain_perturbed`** → `trajectory_retain_Q_A_ROUGE` (OU MU retain ROUGE leg). The distinction is **which HF subset/rows**, not paraphrase vs answer gold.

---

## Fourteen-pass MU bundle (compute)

Aligned with OU non-trajectory TR/prob wiring:

- Forget/retain: `*__guided_tr_para` / `*__guided_tr_pert` on `TOFU_QA_*_para` / `*_pert` dataset configs
- RA/WF: `*__guided_tr_correct` / `*__guided_tr_pert` on `real_authors_perturbed` / `world_facts_perturbed`
- Unguided ROUGE: `QADataset` + `answer` for all four splits

**Deprecated:** `*__guided_native` (`QAwithDualAnswersDataset`) — use split TR legs + dllm merge synthesis instead.

---

## `value_by_index` and PERT aggregation

### Persistence

Evaluator **`save_logs`** keeps **`value_by_index`** by default so dllm can merge passes offline.

Trajectory metrics write per-sample step series into `value_by_index` when the metric requests persistence (`persist_value_by_index` in metric config / trajectory handler).

### PERT / multi-option wrong answers

For `*__guided_tr_pert` with list `perturbed_answer`:

1. Collator nests options under `"0"`, `"1"`, …
2. Trajectory eval runs **mini-batches** per option (same pattern as `run_batchwise_evals`).
3. Metric output includes **`probability_wrong_sum`** (sum of wrong-option probs) used by dllm **`synthesize_prob_normalised_metrics`**.

### PERT probability aggregation order

`probability_wrong_sum` aggregation runs **after all trajectory types** for the pass (fix for partial KeyError when mixing traj types).

### Merge append for per-index rows

When synthesizing TR at merge time, dllm **appends** per-index rows from PERT passes into merged `value_by_index` (fix for dropped indices on PERT-only samples).

---

## Config index

| Pass id | YAML stem |
|---------|-----------|
| `forget__unguided` | `trajectory_pass_forget_unguided` |
| `forget__guided_prob` | `trajectory_pass_forget_guided_prob` |
| `forget__guided_tr_para` / `*_pert` | `trajectory_pass_forget_guided_tr_para` / `*_pert` |
| `retain__unguided` | `trajectory_pass_retain_unguided` |
| `retain__guided_prob` | `trajectory_pass_retain_guided_prob` |
| `retain__guided_tr_*` | `trajectory_pass_retain_guided_tr_*` |
| `retain_sft__unguided` | `trajectory_pass_retain_sft_unguided` |
| `ra__unguided` | `trajectory_pass_ra_unguided` |
| `ra__guided_tr_correct` / `*_pert` | `trajectory_pass_ra_guided_tr_*` |
| `wf__unguided` | `trajectory_pass_wf_unguided` |
| `wf__guided_tr_correct` / `*_pert` | `trajectory_pass_wf_guided_tr_*` |

Override split: `eval.tofu_trajectory.forget_split`, `eval.tofu_trajectory.retain_sft_split`, samples, sampler kwargs in `configs/eval/tofu_trajectory.yaml`.
