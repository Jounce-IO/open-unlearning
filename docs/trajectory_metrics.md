# Trajectory Metrics for dLLM Unlearning Evaluation

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Architecture](#architecture)
4. [Usage Guide](#usage-guide)
5. [Configuration](#configuration)
6. [Implementation Details](#implementation-details)
7. [Output Format](#output-format)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Examples](#examples)

## Overview

Trajectory metrics enable evaluation of unlearning methods for diffusion language models (dLLMs) by examining how metrics evolve over the diffusion trajectory. Unlike autoregressive models that generate tokens sequentially, dLLMs use iterative denoising, making it possible to analyze metric values at each diffusion step.

### Key Features

- **Step-by-step analysis**: Compute metrics at each diffusion step (0 to S-1)
- **Three trajectory types**: Steps, fixation, and ratio trajectories
- **Dynamic metric loading**: Supports any metric from the open-unlearning framework
- **Pre-compute metrics support**: Automatically computes pre-compute metrics at each step
- **Multiple metric support**: Works with both logit-based and text-based metrics
- **Memory efficient**: Extracts only generated portion from full sequence
- **Flexible configuration**: Easy to add to existing evaluations

### Use Cases

- **Unlearning evaluation**: Understand how unlearning affects model behavior at each step
- **Diffusion analysis**: Study how denoising progresses through the trajectory
- **Model debugging**: Identify at which steps models make critical decisions
- **Research**: Analyze the relationship between diffusion steps and metric values

## Core Concepts

### 1. Logits Tensor R

**Shape:** `[V, L, S]`

- **V**: Vocabulary size
- **L**: Generated sequence length (excluding prompt)
- **S**: Number of diffusion steps

The logits tensor `R` captures the model's probability distribution over the vocabulary at each position and step during generation.

**Key Implementation Detail**: The sampler returns logits for the **full sequence** (prompt + generated), but trajectory metrics extract only the **generated portion** (`L` tokens) for analysis.

### 2. Fixation Steps Tensor F

**Shape:** `[L]`

- **F[l]**: The diffusion step at which token at position `l` was fixed (committed)

Fixation steps are recorded directly in the sampler when tokens are committed during the diffusion process. This is different from autoregressive models where tokens are generated sequentially.

**Example**: If `F[5] = 10`, the token at position 5 was fixed at diffusion step 10.

### 3. Trajectory Tensors

Three trajectory tensors, each of shape `[V, L, S]`:

#### Steps Trajectory

**Definition:** `T_steps[v, l, s] = R[v, l, s]`

Direct copy of the logits tensor. Shows raw logits at each step without transformation.

#### Fixation Trajectory

**Definition:** `T_fixation[v, l, s] = R[v, l, max(0, F[l] - s)]`

For each token position, looks back `s` steps from the fixation step. Creates a trajectory centered around when each token was fixed.

**Example**: If token at position 5 was fixed at step 10:
- `s=0`: Uses logits from step 10 (fixation step)
- `s=1`: Uses logits from step 9 (1 step before fixation)
- `s=2`: Uses logits from step 8 (2 steps before fixation)

#### Ratio Trajectory

**Definition:** `T_ratio[v, l, s] = R[v, l, floor(F[l] * (s / S))]`

Interpolates from step 0 to the fixation step proportionally. Creates a smooth interpolation trajectory.

**Example**: If token at position 5 was fixed at step 10 and `S=20`:
- `s=0`: Uses logits from step 0
- `s=10`: Uses logits from step 5 (halfway to fixation)
- `s=19`: Uses logits from step 9 (near fixation)

### 4. Metrics Computation

For each trajectory type and step:

1. Extract logits `[V, L]` at step `s` from trajectory
2. For **logit-based metrics** (e.g., `probability`, `exact_memorization`):
   - Wrap logits in `LogitModelWrapper` to make them callable as `model(**batch)`
   - Call metric function with batch containing labels
   - Extract metric value from result
3. For **text-based metrics** (e.g., `rouge`, `bleu`):
   - Decode logits to text via argmax
   - Call metric function with generated and ground truth text
   - Extract metric value from result

## Architecture

### Component Overview

```
┌─────────────────┐
│   Sampler       │
│  (MDLMSampler)  │
│                 │
│ - Generates     │
│ - Returns:      │
│   • logits_     │
│     history     │
│   • fixation_   │
│     steps       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ trajectory_     │
│   metrics()     │
│                 │
│ 1. Stack R      │
│ 2. Extract F    │
│ 3. Compute      │
│    trajectories │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  For each       │
│  trajectory &   │
│  step:          │
│                 │
│ • Extract       │
│   logits        │
│ • Compute       │
│   metrics       │
│ • Store results │
└─────────────────┘
```

### Key Modules

1. **`trajectory_utils.py`**: Core trajectory computation
   - `trajectories_from_logits()`: Model-free entry-point (logits + fixation → four trajectory tensors); no model or sampler; testable with saved tensors
   - `stack_logits_history()`: Convert list of logits to tensor
   - `compute_trajectories()`: Compute four trajectory types from R, F, S
   - `extract_logits_at_step()`: Extract logits at specific step

2. **`trajectory_adapters.py`**: Metric computation adapters
   - `LogitModelWrapper`: Wraps logits to be callable as model
   - `compute_logit_metric_at_step()`: Compute logit-based metrics
   - `compute_text_metric_at_step()`: Compute text-based metrics

3. **`trajectory_metrics.py`**: Main metric function
   - Shape validation and extraction
   - Generated portion extraction (prompt vs generated)
   - Label alignment
   - Results aggregation

## Usage Guide

### Basic Usage

Add trajectory metrics to any evaluation:

```bash
cd /workspaces/dllm/open-unlearning

python src/eval.py \
  --config-name=eval.yaml \
  eval=tofu \
  model=LLaDA-8B-Instruct \
  +eval.tofu.metrics.trajectory_metrics=_global_.eval.metrics.trajectory_metrics \
  +eval.tofu.metrics.trajectory_metrics.handler=trajectory_metrics \
  task_name=trajectory_eval
```

### With Custom Settings

```bash
python src/eval.py \
  --config-name=eval.yaml \
  eval=tofu \
  model=LLaDA-8B-Instruct \
  +eval.tofu.metrics.trajectory_metrics=_global_.eval.metrics.trajectory_metrics \
  +eval.tofu.metrics.trajectory_metrics.handler=trajectory_metrics \
  +eval.tofu.metrics.trajectory_metrics.trajectory_config.sampler_kwargs.steps=32 \
  +eval.tofu.metrics.trajectory_metrics.trajectory_config.sampler_kwargs.max_new_tokens=64 \
  task_name=trajectory_eval
```

### Using dllm Job CLI (K8s)

```bash
cd /workspaces/dllm

uv run dllm job trajectory-eval \
  --set gpu.type=A100-40 \
  -- dllm eval trajectory-eval --model GSAI-ML/LLaDA-8B-Instruct --benchmark tofu
```

## Configuration

### Minimal Configuration (List Format)

```yaml
trajectory_metrics:
  handler: trajectory_metrics
  metrics:
    - probability  # Simple list of metric names
  trajectory_config:
    return_logits: true
    return_fixation_steps: true
```

### Configuration with Metric Configs (Dict Format)

```yaml
trajectory_metrics:
  handler: trajectory_metrics
  metrics:
    probability: {}  # Simple metric, no config needed
    exact_memorization: {}  # Another simple metric
    truth_ratio:  # Metric with pre_compute
      aggregator: closer_to_1_better
      pre_compute:
        probability:  # Pre-compute metric
          access_key: correct
        probability:  # Can reuse same metric
          access_key: wrong
  trajectory_config:
    return_logits: true
    return_fixation_steps: true
```

### Full Configuration

```yaml
# @package _global_.eval.metrics.trajectory_metrics

defaults:
  - ../../data/datasets@datasets: MUSE_forget_knowmem
  - ../../collator@collators: DataCollatorForSupervisedDatasetwithIndex

handler: trajectory_metrics
batch_size: 1  # Start with 1 for memory efficiency

# Metrics can be specified as:
# 1. Simple list: ["probability", "exact_memorization"]
# 2. Dict with configs: {"probability": {}, "truth_ratio": {"aggregator": "..."}}
metrics:
  - probability
  - exact_memorization
  # - rouge  # Uncomment for text-based metrics

trajectory_config:
  logits_source: sampler  # or "external"
  return_logits: true  # Required: enables logits tracking
  return_fixation_steps: true  # Required: enables fixation tracking
  include_views: [full, eos]  # full = all positions 0..L; eos = positions 0..L_eff-1 only. Default both.
  sampler_kwargs:  # Additional sampler arguments
    steps: 32  # Number of diffusion steps
    temperature: 0.0
    max_new_tokens: 64  # Max tokens to generate

datasets:
  MUSE_forget_knowmem:
    args:
      hf_args:
        path: muse-bench/MUSE-News
      predict_with_generate: True

collators:
  DataCollatorForSupervisedDataset:
    args:
      padding_side: left  # For generation
```

### Views: full and eos

Trajectory metrics can be computed for two **views** of the same trajectory:

- **full**: Uses all positions `0..L` (full generated length). Captures "leakage" and behavior over the entire sequence.
- **eos**: Uses positions `0..L_eff-1` only, where `L_eff` is the effective length up to and including the first EOS token in the generated region. Captures the "real" response before padding or post-EOS tokens.

Configure with `trajectory_config.include_views: [full, eos]` (default), `[full]`, or `[eos]`. When both are included, results are returned with `agg_value["full"]` and `agg_value["eos"]` (and similarly `step_distribution["full"]`, `step_distribution["eos"]`). The dllm CLI flag `--trajectory-views both|full|eos` overrides this.

### Dynamic Metric Loading

Trajectory metrics support **any metric** from the open-unlearning framework. Metrics are loaded dynamically from `METRICS_REGISTRY` at runtime.

**Available Metrics:**
- `probability` - Token-level probability (lowest memory)
- `exact_memorization` - Exact match rate
- `truth_ratio` - Requires pre_compute metrics
- `rouge` - Text-based similarity
- `extraction_strength` - Extraction strength metric
- And all other registered metrics

**Pre-compute Metrics:**
Metrics that require pre-compute (like `truth_ratio`) automatically compute their dependencies at each trajectory step:

```yaml
metrics:
  truth_ratio:
    aggregator: closer_to_1_better
    pre_compute:
      probability:  # Computed at each step
        access_key: correct
      probability:  # Can reuse same metric with different access_key
        access_key: wrong
```

The system will:
1. Compute each pre-compute metric at each step
2. Structure results with `access_key` names
3. Pass them to the main metric
4. Return trajectory results organized by step

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `handler` | str | required | Must be `"trajectory_metrics"` |
| `metrics` | list[str] or dict | required | List of metric names OR dict mapping names to configs |
| `batch_size` | int | 1 | Batch size for evaluation |
| `trajectory_config.return_logits` | bool | true | Enable logits tracking in sampler |
| `trajectory_config.return_fixation_steps` | bool | true | Enable fixation step tracking |
| `trajectory_config.include_views` | list | `[full, eos]` | Which views to compute: **full** (all positions 0..L, leakage), **eos** (positions 0..L_eff-1 only, real response up to first EOS). Use `[full]`, `[eos]`, or both. Default both. |
| `trajectory_config.sampler_kwargs.trajectory_sample_interval` | int | 8 | Interval mode only; defaults to 8 when omitted. Every-step mode is not used. |
| `trajectory_config.sampler_kwargs.steps` | int | 32 | Number of diffusion steps |
| `trajectory_config.sampler_kwargs.max_new_tokens` | int | 64 | Max tokens to generate |

### Metric Configuration

**List Format (Simple):**
```yaml
metrics:
  - probability
  - exact_memorization
```

**Dict Format (With Configs):**
```yaml
metrics:
  probability: {}  # No config needed
  truth_ratio:
    aggregator: closer_to_1_better
    pre_compute:
      probability:
        access_key: correct
      probability:
        access_key: wrong
```

**Pre-compute Metrics:**
- Pre-compute metrics are automatically computed at each trajectory step
- Results are structured with `access_key` names
- Supports nested pre-compute (pre-compute metrics can have their own pre-compute)
- Preserves all fields (e.g., `avg_loss` for `truth_ratio` compatibility)

## Implementation Details

### Sampler Integration

Samplers (MDLMSampler, BD3LMSampler) are modified to:

1. **Store logits**: When `return_logits=True`:
   ```python
   logits_history.append(logits.clone())  # [B, T, V] where T = prompt + generated
   ```

2. **Track fixation steps**: When tokens are committed:
   ```python
   fixation_steps[transfer_index] = global_step  # [B, T]
   ```

3. **Return in SamplerOutput**:
   ```python
   return SamplerOutput(
       sequences=x,
       histories=histories,
       logits_history=logits_history,  # list of [B, T, V]
       fixation_steps=fixation_steps,   # [B, T]
   )
   ```

### Prompt extraction and data formats

Trajectory metrics support two data conventions so TOFU, MUSE, and WMDP work without dataset-specific logic:

1. **Labels mark prompt:** `labels` use `IGNORE_INDEX` (-100) for the prompt prefix. The prompt is taken as `input_ids[:, :prompt_end]` where `prompt_end` is the first index where `labels != IGNORE_INDEX` (or the full sequence length if all labels are IGNORE). This is the typical training-style setup (e.g. WMDP with `PretrainingDataset`).

2. **Prompt-only input_ids:** With `predict_with_generate=True` (e.g. TOFU and MUSE), preprocessing may produce `input_ids` that contain only the prompt and `labels` that are the full conversation token ids with no IGNORE positions. In that case `prompt_end` from labels would be 0. The code uses a fallback: when `prompt_end == 0` and the tokenizer has a `pad_token_id`, the prompt is taken as the **non-pad tokens** of `input_ids` (in order). That yields the correct prompt for the sampler without dataset-specific branches.

Both the main batch loop and the privleak holdout batch use the same logic (via `_build_prompts_for_sampler`).

### Generated Portion Extraction

**Critical Implementation Detail**: The sampler returns logits for the **full sequence** (prompt + generated), but trajectory metrics need only the **generated portion**.

```python
# Stack logits: [V, T_full, S] where T_full = prompt_len + generated_len
R_full = stack_logits_history(logits_history)

# Extract only generated portion
max_prompt_len = max(prompt_lens)
generated_len = T_full - max_prompt_len
R = R_full[:, max_prompt_len:max_prompt_len + generated_len, :]  # [V, L, S]
```

Similarly for fixation steps and labels:
```python
# Extract fixation steps for generated region
F = fixation_steps[0][max_prompt_len:max_prompt_len + L]  # [L]

# Extract labels for generated region
generated_labels = sample_labels[sample_prompt_len:sample_prompt_len + L]  # [L]
```

### Shape Alignment

The key challenge is ensuring logits and labels have matching shapes:

1. **Logits from trajectory**: `[V, L]` where `L` = generated length
2. **Labels**: `[L]` where `L` = generated length
3. **After metric processing**:
   - `evaluate_probability` does: `logits[..., :-1, :]` → `[1, L-1, V]`
   - And: `labels[..., 1:]` → `[1, L-1]`
   - CrossEntropyLoss expects: `[1, V, L-1]` and `[1, L-1]` ✓

### Metric Computation Flow

```python
# For each trajectory type and step:
for traj_name, trajectory in trajectories.items():  # steps, fixation, ratio
    for step in range(S):
        # 1. Extract logits at this step
        logits = extract_logits_at_step(trajectory, step)  # [V, L]
        
        # 2. For logit-based metrics
        if _is_logit_based_metric(metric_name):
            # Convert to [1, L, V]
            logits = logits.transpose(0, 1).unsqueeze(0)
            
            # Wrap in LogitModelWrapper
            model_wrapper = LogitModelWrapper(logits, device)
            
            # Create batch with labels
            batch = {
                "input_ids": torch.zeros(1, L),  # Dummy
                "labels": generated_labels.unsqueeze(0),  # [1, L]
                "attention_mask": torch.ones(1, L),
            }
            
            # Compute metric
            result = metric_fn(model=model_wrapper, batch=batch)
        
        # 3. For text-based metrics
        else:
            # Decode logits to text
            texts = decode_logits_to_text(logits, tokenizer, ...)
            
            # Compute metric
            result = metric_fn(texts=texts, ground_truths=ground_truths, ...)
```

## Output Format

The `trajectory_metrics` function returns a dictionary following the standard metric format. When `include_views` is `[full, eos]` (default), `agg_value` and `step_distribution` are **nested by view**:

```python
{
    "agg_value": {
        # When include_views = [full, eos] (default), one key per view
        "full": {
            "steps": {
                "probability": np.array([0.5, 0.6, 0.7, ...]),  # [S] - mean at each step
                "exact_memorization": np.array([0.8, 0.85, 0.9, ...]),  # [S]
                ...
            },
            "fixation_start": {...},
            "fixation_end": {...},
            "fixation_ratio": {...}
        },
        "eos": {
            # Same structure as "full", but computed over positions 0..L_eff-1 only
            "steps": {"probability": np.array([...]), ...},
            "fixation_start": {...},
            "fixation_end": {...},
            "fixation_ratio": {...}
        }
    },
    # step_distribution has the same nesting: step_distribution["full"], step_distribution["eos"]
    "step_distribution": {
        "full": { "steps": {...}, "fixation_start": {...}, ... },
        "eos": { "steps": {...}, "fixation_start": {...}, ... }
    },
    # Legacy flat shape when include_views has a single view (e.g. [full] only):
    # "agg_value": { "steps": {...}, "fixation_start": {...}, ... }
    "value_by_index": {
        # Per-sample values
        "0": {
            "trajectories": {
                "steps": {
                    "step_0": {
                        "probability": 0.5,
                        "exact_memorization": 0.8,
                        ...
                    },
                    "step_1": {...},
                    ...
                    "step_S-1": {...}
                },
                "fixation": {
                    "step_0": {...},
                    ...
                },
                "ratio": {
                    "step_0": {...},
                    ...
                }
            }
        },
        "1": {...},  # Next sample
        ...
    }
}
```

## Best Practices

### Memory Management

1. **Use small batch sizes**: Start with `batch_size=1` for trajectory metrics
2. **Reduce steps for testing**: Use `steps=16` or `steps=32` instead of 128
3. **Limit generation length**: Use `max_new_tokens=32` or `max_new_tokens=64` for testing
4. **Selective metric computation**: Only compute metrics you need

### Performance Optimization

1. **Reduce number of steps**: Compute metrics at every Nth step if needed
2. **Parallelize metric computation**: Use multiple GPUs if available
3. **Cache decoded text**: For text-based metrics, cache decoded text per step
4. **Use smaller models**: For testing, use smaller models or model variants

### Configuration Tips

1. **Start minimal**: Begin with one metric (`probability`) and small steps
2. **Gradually increase**: Add more metrics and steps as needed
3. **Monitor memory**: Watch GPU memory usage and adjust batch size
4. **Test locally first**: Run small tests locally before deploying to K8s

## Troubleshooting

### Common Issues

#### "Model does not have a sampler"

**Cause**: Model is not a diffusion model or not wrapped with `DiffusionModelAdapter`.

**Solution**:
- Ensure you're using a diffusion model (LLaDA, Dream, etc.)
- Check that `DiffusionModelAdapter` is being used (auto-detected for diffusion models)

#### "No logits_history returned"

**Cause**: `return_logits` is not set to `true` or sampler doesn't support it.

**Solution**:
- Set `trajectory_config.return_logits: true` in config
- Ensure sampler supports `return_logits` parameter (MDLMSampler, BD3LMSampler)

#### "Expected target size [1, X], got [1, Y]"

**Cause**: Shape mismatch between logits and labels. Usually means generated portion extraction failed.

**Solution**:
- Check that prompt length extraction is correct
- Verify that `L` (generated length) matches between logits and labels
- Ensure `max_prompt_len` is computed correctly

#### "Key 'trajectory_metrics' is not in struct"

**Cause**: Config not properly added to eval config.

**Solution**:
- Use `+eval.tofu.metrics.trajectory_metrics=_global_.eval.metrics.trajectory_metrics`
- Explicitly set handler: `+eval.tofu.metrics.trajectory_metrics.handler=trajectory_metrics`

#### Memory errors (OOM)

**Cause**: Too much memory usage from storing logits for all steps.

**Solution**:
- Reduce `batch_size` to 1
- Reduce `steps` (e.g., 16 instead of 128)
- Reduce `max_new_tokens` (e.g., 32 instead of 128)
- Use gradient checkpointing if available

### Debugging Tips

1. **Check shapes**: Print shapes of `R`, `F`, `L`, `S` to verify correctness
2. **Verify extraction**: Check that generated portion extraction is correct
3. **Test with one sample**: Use `batch_size=1` and `max_samples=1` for debugging
4. **Use unit tests**: Run `pytest tests/test_trajectory_*.py` to verify logic

## Examples

### Example 1: Basic Trajectory Evaluation

```bash
python src/eval.py \
  --config-name=eval.yaml \
  eval=tofu \
  model=LLaDA-8B-Instruct \
  +eval.tofu.metrics.trajectory_metrics=_global_.eval.metrics.trajectory_metrics \
  +eval.tofu.metrics.trajectory_metrics.handler=trajectory_metrics \
  +eval.tofu.metrics.trajectory_metrics.metrics=[probability] \
  +eval.tofu.metrics.trajectory_metrics.trajectory_config.sampler_kwargs.steps=16 \
  +eval.tofu.metrics.trajectory_metrics.trajectory_config.sampler_kwargs.max_new_tokens=32 \
  task_name=trajectory_basic
```

### Example 2: Full Trajectory Analysis

```bash
python src/eval.py \
  --config-name=eval.yaml \
  eval=tofu \
  model=LLaDA-8B-Instruct \
  +eval.tofu.metrics.trajectory_metrics=_global_.eval.metrics.trajectory_metrics \
  +eval.tofu.metrics.trajectory_metrics.handler=trajectory_metrics \
  +eval.tofu.metrics.trajectory_metrics.metrics=[probability,exact_memorization,rouge] \
  +eval.tofu.metrics.trajectory_metrics.trajectory_config.sampler_kwargs.steps=64 \
  +eval.tofu.metrics.trajectory_metrics.trajectory_config.sampler_kwargs.max_new_tokens=128 \
  task_name=trajectory_full
```

### Example 3: Quick Testing

```bash
python src/eval.py \
  --config-name=eval.yaml \
  eval=trajectory_test \
  model=LLaDA-8B-Instruct \
  eval.trajectory_test.metrics.trajectory_metrics.batch_size=1 \
  eval.trajectory_test.metrics.trajectory_metrics.metrics=[probability] \
  eval.trajectory_test.metrics.trajectory_metrics.trajectory_config.sampler_kwargs.steps=8 \
  eval.trajectory_test.metrics.trajectory_metrics.trajectory_config.sampler_kwargs.max_new_tokens=16 \
  data.max_samples=2 \
  task_name=trajectory_quick_test
```

## Related Documentation

- [Trajectory Metrics Specification](./trajectory_metrics_specification.md) - Detailed mathematical specification
- [Testing Trajectory Metrics](./testing_trajectory_metrics.md) - Testing guide
- [Unit Tests](../tests/README.md) - Unit test documentation

## References

- LLaDA Paper: [Link to paper]
- Diffusion Models: [Link to diffusion models paper]
- Open Unlearning Framework: [Link to framework]
