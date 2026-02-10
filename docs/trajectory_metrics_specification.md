# Trajectory-Based Metrics Specification for dLLM Unlearning Evaluation

## Overview

This document specifies the trajectory-based metrics evaluation system for diffusion language models (dLLMs). The system computes metrics at each diffusion step across four trajectory types, enabling analysis of how metrics evolve during the denoising process.

## Trajectory Examples

The following example illustrates how different trajectory types map diffusion steps to R steps. In this example:
- 5 token positions (rows)
- Fixation steps: F[0]=6, F[1]=3, F[2]=3, F[3]=9, F[4]=11
- `=` represents a diffusion step, `F` marks the fixation step, `[=]` marks the current trajectory step

### Steps Trajectory
```
Step 0:  [=]=====F-----
         [=]==F--------
         [=]==F--------
         [=]========F--
         [=]===========F

Step K:  =====[=]F-----
         ===F-[-]------
         ===F-[-]------
         =====[=]===F--
         =====[=]=====F

Step S-1: ======F-----
          ===F--------
          ===F--------
          =========F--
          ===========F
```

### Fixation Start Trajectory
```
Step 0:  [=]=====F-----
         [=]==F--------
         [=]==F--------
         [=]========F--
         [=]===========F

Step K:  =====[=]F-----
         ===[F]--------
         ===[F]--------
         =====[=]===F--
         =====[=]=====F

Step S-1: ======[F]-----
          ===[F]--------
          ===[F]--------
          =========[F]--
          ============[F]
```

### Fixation End Trajectory
```
Step 0:  [=]=====F-----
         [=]==F--------
         [=]==F--------
         [=]========F--
         [=]==========F

Step K:  =[=]====F-----
         [=]==F--------
         [=]==F--------
         ====[=]====F--
         ======[=]====F

Step S-1: ======[F]-----
          ===[F]--------
          ===[F]--------
          =========[F]--
          ============[F]

Step S-2: =====[=]F-----
          ==[=]F--------
          ==[=]F--------
          ========[=]F--
          ===========[=]F
```

### Fixation Ratio Trajectory
```
Step 0:  [=]=====F-----
         [=]==F--------
         [=]==F--------
         [=]========F--
         [=]===========F

Step K:  ==[=]==F-----
         =[=]=F--------
         =[=]=F--------
         ====[=]====F--
         =====[=]=====F

Step S-1: ======[F]-----
          ===[F]--------
          ===[F]--------
          =========[F]--
          ============[F]
```

## Definitions

### 1. Logits Tensor R

**Shape:** `[V, L, S]`

- **V**: Vocabulary size
- **L**: Output length (number of generated tokens)
- **S**: Number of diffusion steps

**Definition:**
- `R[v, l, s]` = logit for vocabulary token `v` at position `l` at diffusion step `s`
- Captures the model's logit distribution over the vocabulary at each position and step

**Source:**
- Computed by samplers during generation when `return_logits=True`
- Stored in `SamplerOutput.logits_history` as a list of `[B, L, V]` tensors (one per step)
- Stacked into `[V, L, S]` tensor using `stack_logits_history()`

### 2. Fixation Steps Tensor F

**Shape:** `[L]`

- **F[l]**: The diffusion step at which token at position `l` was fixed (committed)
- `0 ≤ F[l] < S` for all `l ∈ [0, L-1]`

**Computation:**
- Computed directly in the sampler during generation
- When tokens are committed (e.g., `x[transfer_index] = x0[transfer_index]`), the current global step is recorded: `fixation_steps[transfer_index] = global_step`
- Stored in `SamplerOutput.fixation_steps` as `[B, T]` tensor
- Extracted to `[L]` for trajectory computation (for single sample, takes first batch)

**Key Properties:**
- Fixation step is recorded **at the moment** tokens are committed
- No post-processing needed - step is known when token is fixed
- Positions that are never fixed (always mask) default to `S-1` (last step)

### 3. Trajectory Tensors

Four trajectory tensors, each of shape `[V, L, S]`:

All fixation trajectories satisfy: **first (s=0) = R step 0**, **last (s=S-1) = R step F[l]**.

#### 3.1 Steps Trajectory

**Definition:**
```
T_steps[v, l, s] = R[v, l, s]
```

**Semantics:**
- Direct copy of the logits tensor R
- Shows raw logits at each step without transformation
- At `s=0`: Uses logits from step 0
- At `s=S-1`: Uses logits from step S-1

#### 3.2 Fixation Start Trajectory

**Definition:**
```
T_fixation_start[v, l, s] = R[v, l, min(s, F[l])]
```

**Semantics:**
- Trajectory from step 0 to fixation step F[l]
- At `s=0`: Uses logits from step 0
- At `s≤F[l]`: Uses logits from step s (increases linearly)
- At `s>F[l]`: Uses logits from step F[l] (clamped to fixation step)
- At `s=S-1`: Uses logits from step F[l] (fixation step)

**Example:**
- If token at position 5 was fixed at step 7 (`F[5] = 7`) and `S=10`:
  - `s=0`: Uses logits from step 0
  - `s=3`: Uses logits from step 3
  - `s=7`: Uses logits from step 7 (fixation)
  - `s=9`: Uses logits from step 7 (clamped to fixation)

#### 3.3 Fixation End Trajectory

**Definition:**
```
T_fixation_end[v, l, s] = R[v, l, max(0, F[l] - (S-1) + s)]
```

**Semantics:**
- Trajectory from step 0 to fixation step F[l]
- At `s=0`: Uses logits from step 0 (since F[l] - (S-1) + 0 ≤ 0 when F[l] < S)
- As `s` increases: Uses logits from step `F[l] - (S-1) + s` (increases toward fixation)
- At `s=S-1`: Uses logits from step F[l] (fixation step)
- Source step is clamped to valid range [0, S-1]

**Example:**
- If token at position 5 was fixed at step 7 (`F[5] = 7`) and `S=10`:
  - `s=0`: Uses logits from step `max(0, 7-9+0) = 0`
  - `s=3`: Uses logits from step `max(0, 7-9+3) = 1`
  - `s=7`: Uses logits from step `max(0, 7-9+7) = 5`
  - `s=9`: Uses logits from step `max(0, 7-9+9) = 7` (fixation)

#### 3.4 Fixation Ratio Trajectory

**Definition:**
```
T_fixation_ratio[v, l, s] = R[v, l, floor(F[l] * s / (S-1))]
```

**Semantics:**
- Linear interpolation from step 0 to fixation step F[l]
- At `s=0`: Uses logits from step 0
- At `s=S-1`: Uses logits from step F[l] (fixation step)
- Creates a smooth linear interpolation trajectory
- Assumes S > 1

**Example:**
- If token at position 5 was fixed at step 7 (`F[5] = 7`) and `S=10`:
  - `s=0`: Uses logits from step `floor(7 * 0/9) = 0`
  - `s=3`: Uses logits from step `floor(7 * 3/9) = 2`
  - `s=6`: Uses logits from step `floor(7 * 6/9) = 4`
  - `s=9`: Uses logits from step `floor(7 * 9/9) = 7` (fixation)

### 4. Metrics Tensor M

**Shape:** `[4, S, M]`

- **First dimension**: Trajectory type (0=steps, 1=fixation_start, 2=fixation_end, 3=fixation_ratio)
- **Second dimension**: Diffusion step `s` (0 to S-1)
- **Third dimension**: Metric index (one per requested metric)

**Computation:**
For each trajectory type `t ∈ {0, 1, 2, 3}` and step `s ∈ [0, S-1]`:

1. Extract logits `[V, L]` at step `s` from trajectory `t`
2. For logit-based metrics:
   - Wrap logits in `LogitModelWrapper`
   - Call metric function (e.g., `evaluate_probability`)
   - Extract metric value from result
3. For text-based metrics:
   - Decode logits to text via argmax
   - Call metric function (e.g., `eval_text_similarity`)
   - Extract metric value from result
4. Store in `M[t, s, m]` where `m` is the metric index

## Output Format

The `trajectory_metrics` function returns a dictionary following the standard metric format:

```python
{
    "agg_value": {
        # Aggregated values across all samples, steps, and trajectories
        "steps": {
            "probability": np.array([...]),  # [S] - mean across samples at each step
            "rouge": np.array([...]),        # [S] - mean across samples at each step
            ...
        },
        "fixation_start": {...},    # Same structure
        "fixation_end": {...},      # Same structure
        "fixation_ratio": {...}     # Same structure
    },
    "value_by_index": {},  # Empty - per-sample trajectories not stored (memory optimization)
    "step_distribution": {
        # Per-step distribution stats (same nesting as agg_value) for box-and-whisker / interval plots
        "steps": {
            "probability": {
                "mean": np.array([...]),   # [S] - same as agg_value for this metric
                "std": np.array([...]),   # [S]
                "median": np.array([...]),
                "p25": np.array([...]),
                "p75": np.array([...]),
                "min": np.array([...]),
                "max": np.array([...]),
                "ci_low": np.array([...]),   # 95% CI (mean - 1.96*SE)
                "ci_high": np.array([...]),  # 95% CI (mean + 1.96*SE)
            },
            ...
        },
        "fixation_start": {...},    # Same structure
        "fixation_end": {...},
        "fixation_ratio": {...}
    }
}
```

Use `step_distribution` for visualizations: box-and-whisker (p25, median, p75, min, max per step) or line + shaded interval (mean with ci_low/ci_high or mean ± std).

## Implementation Details

### Sampler Integration

Samplers (MDLMSampler, BD3LMSampler) are modified to:

1. **Store logits**: When `return_logits=True`, store logits after each forward pass:
   ```python
   logits_history.append(logits.clone())  # [B, L, V]
   ```

2. **Track fixation steps**: When tokens are committed:
   ```python
   fixation_steps[transfer_index] = global_step
   ```

3. **Return in SamplerOutput**:
   ```python
   return SamplerOutput(
       sequences=x,
       histories=histories,
       logits_history=logits_history,  # list of [B, L, V]
       fixation_steps=fixation_steps,   # [B, T]
   )
   ```

### Trajectory Computation

1. **Stack logits**: Convert `logits_history` list to `R [V, L, S]` tensor
2. **Extract F**: Extract fixation steps `F [L]` from `fixation_steps [B, T]`
3. **Compute trajectories**: Use `compute_trajectories(R, F, S)` to get four trajectory tensors (steps, fixation_start, fixation_end, fixation_ratio)

### Metric Computation

For each trajectory type and step:

1. **Extract logits**: `logits = trajectory[:, :, s]` → `[V, L]`
2. **Reshape**: `logits = logits.unsqueeze(0)` → `[1, L, V]` (add batch dimension)
3. **Compute metric**:
   - **Logit-based**: Use `LogitModelWrapper` to provide logits to metric function
   - **Text-based**: Decode logits to text, then call metric function
4. **Extract value**: Get metric value from result dict
5. **Store**: Add to results structure

## Configuration

Example configuration:

```yaml
trajectory_metrics:
  handler: trajectory_metrics
  metrics:
    - probability
    - exact_memorization
    - rouge
    - classifier_prob
  trajectory_config:
    logits_source: sampler  # or "external"
    return_logits: true  # Sampler config
    return_fixation_steps: true  # Sampler config
    sampler_kwargs:  # Additional sampler arguments
      steps: 128
      temperature: 0.0
```

## Usage

The trajectory metrics can be used like any other metric in the evaluation framework:

```python
from evals import Evaluator

evaluator = Evaluator(name="trajectory_eval", eval_cfg=config)
results = evaluator.evaluate(model=model, tokenizer=tokenizer)
```

Results will include trajectory metrics with values at each step for each trajectory type.

## Notes

- **Memory considerations**: 
  - Per-sample trajectory data is not stored (only aggregated values are returned)
  - Storing logits for all steps can be memory-intensive. Consider:
    - Computing metrics only at selected steps
    - Using smaller batch sizes
    - Optional quantization of logits
  
- **Performance**: Computing metrics at every step for every trajectory can be slow. Consider:
  - Computing metrics only at selected steps
  - Parallelizing metric computation
  - Caching decoded text for text-based metrics

- **Batched generation**: Current implementation handles single-sample trajectories. For batched generation, trajectories are computed per sample or averaged.
