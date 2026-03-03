# Input-Output Baselines for Unlearning Evaluation

This document describes inference-time input/output transformations used to evaluate unlearning with **guardrails** and **in-context unlearning (ICUL)**. The same checkpoint can be run with or without these baselines; trajectory metrics score the effective answer.

## References

- **Guardrail baselines:** Thaker et al., *Guardrail Baselines for Unlearning in LLMs*, arXiv 2403.03329.
- **ICUL:** Pawelczyk et al., *In-Context Unlearning*, ICML 2024, arXiv 2310.07579.

## Four Guardrails

1. **Prompt prefix** — Prepend an instruction (e.g. “Do not reveal …”) to each prompt; tokenized and prepended before sampling.
2. **Input filter** — (Stub.) Option `wmdp_style` reserved for future query blocking.
3. **Output filter (keyword)** — If the decoded output contains any of `keyword_list`, replace it with `output_filter_safe_reply` (default: “I don’t know.”). Not applied when `benchmark == "wmdp"` (categorical answers).
4. **ICUL** — In-context unlearning: prepend a block of forget (flipped) and retain (correct) examples to the prompt, then run the model as usual.

## ICUL Methodology

- **Three-step format:** [Forget 1 + wrong answer] … [Forget K + wrong] then [Retain 1 + correct] … [Retain L + correct], then the eval prompt.
- **Train-only:** Pools are loaded from train-designated splits only (TOFU: forget_split/retain_split; MUSE: forget_qa_icl/retain_qa_icl; WMDP: train split).
- **TOFU:** Current eval sample is excluded from the forget block (by `dataset_index`).
- **WMDP:** Same ICUL methodology with **categorical answers**: question line ends with “Answer:” then a single letter (A/B/C/D). Forget = one random wrong letter; retain = correct letter.

## Configuration

Config lives under `trajectory_config.guardrail` (or `input_output_baseline`). Schema:

- `benchmark`: `tofu` | `muse` | `wmdp` (for ICUL and for output-filter bypass when wmdp).
- `prompt_prefix`: Optional string prepended to every prompt.
- `input_filter`: Optional (e.g. `wmdp_style`); currently stub.
- `output_filter`: `none` | `keyword`.
- `keyword_list`: List of strings (or single string) for keyword output filter.
- `output_filter_safe_reply`: Replacement when keyword matches (default: “I don’t know.”).
- `icul`: `{ enabled: bool, K?: int, L?: int }`. When `enabled` and `benchmark` is tofu/muse/wmdp, pools are loaded once per run via `load_icul_pools` and injected as `icul_forget_pool` and `icul_retain_pool`.

Pools are loaded from datasets: TOFU (`locuslab/TOFU`), MUSE (`muse-bench/MUSE-{data_split}`), WMDP (`cais/wmdp`).

## How to Run

- **Without guardrails:** Omit `guardrail` (or leave it empty). Trajectory metrics behave as before.
- **With guardrails:** Set `trajectory_config.guardrail` in your experiment/config with the desired keys. For ICUL, set `icul.enabled: true` and `benchmark`; pools are loaded automatically from train splits.

## Code

- **Module:** `src/evals/guardrails.py` — `transform_prompts`, `transform_output_text`, `load_icul_pools`.
- **Integration:** `src/evals/metrics/trajectory_metrics.py` — after `_build_prompts_for_sampler`, `transform_prompts` is applied when guardrail config is present; in the text-based metric path, `transform_output_text` is applied to decoded text before ROUGE etc.
