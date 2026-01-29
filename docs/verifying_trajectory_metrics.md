# Verifying Trajectory Metrics Correctness

This guide provides multiple methods to verify that trajectory metrics are calculated correctly.

## 1. Run Unit Tests

The most reliable way to verify correctness is to run the comprehensive test suite:

```bash
cd open-unlearning
pytest tests/test_trajectory_metrics.py -v
pytest tests/test_trajectory_text_metrics.py -v
pytest tests/test_trajectory_utils.py -v
```

These tests verify:
- ✅ Shape handling for logits and trajectories
- ✅ Text-based metric detection and handling
- ✅ Logit-based metric computation
- ✅ Argument passing and error handling
- ✅ Integration with both probability and ROUGE metrics

## 2. Value Range Checks

### Probability Metric
- **Expected range**: `[0, 1]` (probabilities are normalized)
- **Typical values**: Small positive values (1e-4 to 1e-1) for intermediate diffusion steps
- **Check**: All values should be non-negative and ≤ 1.0

```python
# In your results analysis
prob_values = results["agg_value"]["steps"]["probability"]
assert (prob_values >= 0).all(), "Probability values must be non-negative"
assert (prob_values <= 1.0).all(), "Probability values must be ≤ 1.0"
```

### ROUGE Metric
- **Expected range**: `[0, 1]` (ROUGE scores are normalized)
- **Typical values**: 
  - Early steps: Very low (0 to 1e-4) - text is noisy/incomplete
  - Later steps: Higher (1e-4 to 1e-2) - text quality improves
  - Final steps: Should be highest (but still may be low if diffusion is incomplete)
- **Check**: All values should be non-negative and ≤ 1.0

```python
rouge_values = results["agg_value"]["steps"]["rouge"]
assert (rouge_values >= 0).all(), "ROUGE values must be non-negative"
assert (rouge_values <= 1.0).all(), "ROUGE values must be ≤ 1.0"
```

## 3. Trajectory Pattern Verification

The four trajectory types should show different patterns:

### Steps Trajectory (`T_steps`)
- **Pattern**: Should show gradual improvement over steps
- **Check**: Values should generally increase (or stabilize) as step increases
- **Exception**: Early steps may be noisy

```python
steps_prob = results["agg_value"]["steps"]["probability"]
# Check that later steps are generally better than early steps
# (allowing for some noise)
assert steps_prob[-1] >= steps_prob[0] * 0.5, "Final step should be better than initial"
```

### Fixation Start Trajectory (`T_fixation_start`)
- **Pattern**: Values increase from step 0 to fixation step, then plateau
- **Check**: Should show monotonic increase until fixation step, then constant

### Fixation End Trajectory (`T_fixation_end`)
- **Pattern**: Values increase from step 0 to fixation step
- **Check**: Should show gradual increase toward fixation step

### Fixation Ratio Trajectory (`T_fixation_ratio`)
- **Pattern**: Smooth linear interpolation from step 0 to fixation step
- **Check**: Should show smooth linear progression (less noisy than steps)

## 4. Compare with Non-Trajectory Baseline

Run the same metrics **without** trajectory computation to get baseline values:

```yaml
# configs/eval/muse_metrics/baseline_probability.yaml
probability:
  handler: probability
  data:
    dataset: muse-bench/MUSE-News
    split: forget_qa[:5]
  batch_size: 1
```

Compare:
- **Trajectory final step** should be similar to **baseline** (within reasonable tolerance)
- If final step values differ significantly, investigate why

```python
# Compare trajectory final step with baseline
trajectory_final = results["agg_value"]["steps"]["probability"][-1]
baseline_value = baseline_results["probability"]["agg_value"]

# Should be within 10% (allowing for sampling differences)
tolerance = 0.1
assert abs(trajectory_final - baseline_value) / baseline_value < tolerance
```

## 5. Inspect Decoded Text

For text-based metrics (ROUGE), verify that decoded text makes sense:

```python
# Add debug logging to trajectory_metrics.py
# In _handle_text_based_metric, after decoding:
logger.debug(f"Step {step}: Decoded text: {gen_text[:100]}")
logger.debug(f"Step {step}: Ground truth: {ground_truth[:100]}")
```

**Expected behavior**:
- **Early steps**: Decoded text should be gibberish or incomplete
- **Middle steps**: Text should start making sense but be incomplete
- **Later steps**: Text should be more coherent and closer to ground truth

## 6. Sanity Checks on Results Structure

Verify the output structure matches expectations:

```python
def verify_trajectory_results(results):
    """Verify trajectory metrics results structure."""
    assert "agg_value" in results
    assert "value_by_index" in results
    
    agg = results["agg_value"]
    assert "steps" in agg
    assert "fixation_start" in agg
    assert "fixation_end" in agg
    assert "fixation_ratio" in agg
    
    # Check that all trajectory types have the same metrics
    metrics_in_steps = set(agg["steps"].keys())
    metrics_in_fixation_start = set(agg["fixation_start"].keys())
    metrics_in_fixation_end = set(agg["fixation_end"].keys())
    metrics_in_fixation_ratio = set(agg["fixation_ratio"].keys())
    
    assert metrics_in_steps == metrics_in_fixation_start == metrics_in_fixation_end == metrics_in_fixation_ratio, \
        "All trajectory types should have same metrics"
    
    # Check array lengths match number of steps
    S = len(agg["steps"]["probability"])  # Assuming probability exists
    for traj_type in ["steps", "fixation_start", "fixation_end", "fixation_ratio"]:
        for metric_name, metric_values in agg[traj_type].items():
            assert len(metric_values) == S, \
                f"{traj_type}.{metric_name} should have length {S}, got {len(metric_values)}"
    
    # Note: value_by_index no longer contains per-sample trajectories (memory optimization)
    # Only agg_value is returned
```

## 7. Cross-Metric Consistency

If running multiple metrics, check for consistency:

```python
# Probability and ROUGE should correlate (higher prob → higher ROUGE)
prob_values = results["agg_value"]["steps"]["probability"]
rouge_values = results["agg_value"]["steps"]["rouge"]

# Compute correlation (should be positive)
correlation = np.corrcoef(prob_values, rouge_values)[0, 1]
assert correlation > -0.5, "Probability and ROUGE should not be strongly anti-correlated"
```

## 8. Manual Inspection Script

Create a script to inspect results:

```python
#!/usr/bin/env python3
"""Inspect trajectory metrics results for correctness."""

import json
import numpy as np
from pathlib import Path

def inspect_results(results_file: str):
    """Inspect trajectory metrics results."""
    with open(results_file) as f:
        results = json.load(f)
    
    if "trajectory_metrics" not in results:
        print("❌ No trajectory_metrics in results")
        return
    
    tm = results["trajectory_metrics"]
    agg = tm["agg_value"]
    
    print("=== Trajectory Metrics Inspection ===\n")
    
    # Check structure
    print("✅ Structure check:")
    for traj_type in ["steps", "fixation", "ratio"]:
        if traj_type in agg:
            print(f"  - {traj_type}: ✓")
            for metric_name in agg[traj_type]:
                values = np.array(agg[traj_type][metric_name])
                print(f"    - {metric_name}: {len(values)} steps")
                print(f"      Range: [{values.min():.6f}, {values.max():.6f}]")
                print(f"      Mean: {values.mean():.6f}")
        else:
            print(f"  - {traj_type}: ❌ Missing")
    
    # Check value ranges
    print("\n✅ Value range checks:")
    for traj_type in ["steps", "fixation", "ratio"]:
        for metric_name, values in agg[traj_type].items():
            values = np.array(values)
            if metric_name in ["probability", "rouge"]:
                if (values < 0).any():
                    print(f"  ❌ {traj_type}.{metric_name}: Has negative values")
                elif (values > 1.0).any():
                    print(f"  ❌ {traj_type}.{metric_name}: Has values > 1.0")
                else:
                    print(f"  ✓ {traj_type}.{metric_name}: All values in [0, 1]")
    
    # Check patterns
    print("\n✅ Pattern checks:")
    for traj_type in ["steps", "fixation", "ratio"]:
        for metric_name, values in agg[traj_type].items():
            values = np.array(values)
            if len(values) > 1:
                # Check for NaN or Inf
                if np.isnan(values).any():
                    print(f"  ❌ {traj_type}.{metric_name}: Contains NaN")
                elif np.isinf(values).any():
                    print(f"  ❌ {traj_type}.{metric_name}: Contains Inf")
                else:
                    print(f"  ✓ {traj_type}.{metric_name}: No NaN/Inf")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inspect_results.py <results.json>")
        sys.exit(1)
    inspect_results(sys.argv[1])
```

## 9. Regression Testing

Save known-good results and compare:

```python
# Save baseline results
baseline = {
    "steps": {"probability": [...], "rouge": [...]},
    "fixation": {...},
    "ratio": {...}
}
with open("baseline_results.json", "w") as f:
    json.dump(baseline, f)

# Compare new results
def compare_with_baseline(new_results, baseline_file):
    with open(baseline_file) as f:
        baseline = json.load(f)
    
    for traj_type in ["steps", "fixation", "ratio"]:
        for metric_name in baseline[traj_type]:
            new_vals = np.array(new_results["agg_value"][traj_type][metric_name])
            baseline_vals = np.array(baseline[traj_type][metric_name])
            
            # Check shape
            assert new_vals.shape == baseline_vals.shape
            
            # Check values are similar (within tolerance)
            diff = np.abs(new_vals - baseline_vals)
            max_diff = diff.max()
            if max_diff > 0.01:  # 1% tolerance
                print(f"⚠️  {traj_type}.{metric_name}: Max diff = {max_diff:.6f}")
            else:
                print(f"✓ {traj_type}.{metric_name}: Matches baseline")
```

## 10. Visual Inspection

Plot the trajectory values to visually inspect patterns:

```python
import matplotlib.pyplot as plt

def plot_trajectories(results):
    """Plot trajectory metrics over steps."""
    agg = results["agg_value"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, traj_type in enumerate(["steps", "fixation", "ratio"]):
        ax = axes[idx]
        for metric_name, values in agg[traj_type].items():
            ax.plot(values, label=metric_name, marker='o')
        ax.set_xlabel("Step")
        ax.set_ylabel("Metric Value")
        ax.set_title(f"{traj_type.capitalize()} Trajectory")
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("trajectory_plots.png")
    print("✓ Saved trajectory_plots.png")
```

## Quick Verification Checklist

After running trajectory metrics, verify:

- [ ] All unit tests pass
- [ ] Probability values are in [0, 1]
- [ ] ROUGE values are in [0, 1]
- [ ] No NaN or Inf values
- [ ] All trajectory types have same metrics
- [ ] Array lengths match number of steps
- [ ] Per-sample results have correct structure
- [ ] Final step values are reasonable (compare with baseline if available)
- [ ] Trajectory patterns make sense (steps improve, fixation centered, ratio smooth)

## Common Issues and Fixes

### All ROUGE values are 0
- **Cause**: Text decoding issue or ground truth extraction problem
- **Fix**: Check decoded text in logs, verify labels are extracted correctly

### Probability values are all the same
- **Cause**: Logits not being updated correctly across steps
- **Fix**: Verify `logits_history` contains different values at each step

### Values are NaN or Inf
- **Cause**: Division by zero or invalid logits
- **Fix**: Check for empty sequences, verify logits are valid

### Trajectory arrays have wrong length
- **Cause**: Step counting mismatch
- **Fix**: Verify `S` (number of steps) matches sampler output
