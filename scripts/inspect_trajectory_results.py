#!/usr/bin/env python3
"""Inspect trajectory metrics results for correctness.

Usage:
    python scripts/inspect_trajectory_results.py <results.json>
    
Example:
    python scripts/inspect_trajectory_results.py saves/eval/trajectory_test/MUSE_EVAL.json
"""

import json
import numpy as np
import sys
from pathlib import Path


def inspect_results(results_file: str):
    """Inspect trajectory metrics results."""
    results_path = Path(results_file)
    if not results_path.exists():
        print(f"❌ File not found: {results_file}")
        return False
    
    with open(results_path) as f:
        results = json.load(f)
    
    if "trajectory_metrics" not in results:
        print("❌ No 'trajectory_metrics' key in results")
        print(f"Available keys: {list(results.keys())}")
        return False
    
    tm = results["trajectory_metrics"]
    
    if "agg_value" not in tm:
        print("❌ No 'agg_value' in trajectory_metrics")
        return False
    
    agg = tm["agg_value"]
    
    print("=" * 60)
    print("Trajectory Metrics Inspection")
    print("=" * 60)
    print()
    
    # Check structure
    print("📋 Structure Check:")
    all_ok = True
    for traj_type in ["steps", "fixation_start", "fixation_end", "fixation_ratio"]:
        if traj_type in agg:
            print(f"  ✓ {traj_type}: Present")
            if not isinstance(agg[traj_type], dict):
                print(f"    ❌ {traj_type} is not a dict")
                all_ok = False
                continue
            for metric_name in agg[traj_type]:
                values = np.array(agg[traj_type][metric_name])
                if not isinstance(values, np.ndarray):
                    values = np.array(values)
                print(f"    - {metric_name}: {len(values)} steps")
        else:
            print(f"  ❌ {traj_type}: Missing")
            all_ok = False
    
    if not all_ok:
        print("\n⚠️  Structure issues found!")
        return False
    
    print("\n✅ Structure is correct")
    print()
    
    # Check value ranges
    print("📊 Value Range Checks:")
    for traj_type in ["steps", "fixation_start", "fixation_end", "fixation_ratio"]:
        for metric_name, values in agg[traj_type].items():
            values = np.array(values)
            
            # Check for NaN/Inf
            if np.isnan(values).any():
                print(f"  ❌ {traj_type}.{metric_name}: Contains NaN")
                all_ok = False
            elif np.isinf(values).any():
                print(f"  ❌ {traj_type}.{metric_name}: Contains Inf")
                all_ok = False
            else:
                # Check range based on metric type
                if metric_name in ["probability", "rouge"]:
                    if (values < 0).any():
                        print(f"  ❌ {traj_type}.{metric_name}: Has negative values")
                        print(f"     Min: {values.min():.6e}")
                        all_ok = False
                    elif (values > 1.0).any():
                        print(f"  ❌ {traj_type}.{metric_name}: Has values > 1.0")
                        print(f"     Max: {values.max():.6e}")
                        all_ok = False
                    else:
                        print(f"  ✓ {traj_type}.{metric_name}: All values in [0, 1]")
                        print(f"     Range: [{values.min():.6e}, {values.max():.6e}]")
                        print(f"     Mean: {values.mean():.6e}, Std: {values.std():.6e}")
    
    print()
    
    # Check patterns
    print("📈 Pattern Analysis:")
    for traj_type in ["steps", "fixation_start", "fixation_end", "fixation_ratio"]:
        for metric_name, values in agg[traj_type].items():
            values = np.array(values)
            if len(values) > 1:
                # Check monotonicity (for steps trajectory)
                if traj_type == "steps":
                    increasing = np.sum(np.diff(values) > 0)
                    decreasing = np.sum(np.diff(values) < 0)
                    print(f"  {traj_type}.{metric_name}:")
                    print(f"    - Increasing steps: {increasing}/{len(values)-1}")
                    print(f"    - Decreasing steps: {decreasing}/{len(values)-1}")
                    print(f"    - First value: {values[0]:.6e}, Last value: {values[-1]:.6e}")
                    if values[-1] > values[0]:
                        print(f"    ✓ Overall improvement: {((values[-1] - values[0]) / values[0] * 100):.1f}%")
    
    print()
    
    # Check consistency across trajectories
    print("🔄 Cross-Trajectory Consistency:")
    required_trajectories = ["steps", "fixation_start", "fixation_end", "fixation_ratio"]
    if all(traj in agg for traj in required_trajectories):
        steps_metrics = set(agg["steps"].keys())
        fixation_start_metrics = set(agg["fixation_start"].keys())
        fixation_end_metrics = set(agg["fixation_end"].keys())
        fixation_ratio_metrics = set(agg["fixation_ratio"].keys())
        
        if steps_metrics == fixation_start_metrics == fixation_end_metrics == fixation_ratio_metrics:
            print(f"  ✓ All trajectories have same metrics: {steps_metrics}")
        else:
            print(f"  ❌ Metric mismatch:")
            print(f"     Steps: {steps_metrics}")
            print(f"     Fixation start: {fixation_start_metrics}")
            print(f"     Fixation end: {fixation_end_metrics}")
            print(f"     Fixation ratio: {fixation_ratio_metrics}")
            all_ok = False
        
        # Check array lengths
        if steps_metrics:
            first_metric = list(steps_metrics)[0]
            S_steps = len(agg["steps"][first_metric])
            S_fixation_start = len(agg["fixation_start"][first_metric])
            S_fixation_end = len(agg["fixation_end"][first_metric])
            S_fixation_ratio = len(agg["fixation_ratio"][first_metric])
            
            if S_steps == S_fixation_start == S_fixation_end == S_fixation_ratio:
                print(f"  ✓ All trajectories have same length: {S_steps} steps")
            else:
                print(f"  ❌ Length mismatch:")
                print(f"     Steps: {S_steps}, Fixation start: {S_fixation_start}, Fixation end: {S_fixation_end}, Fixation ratio: {S_fixation_ratio}")
                all_ok = False
    
    print()
    
    # Summary
    print("=" * 60)
    if all_ok:
        print("✅ All checks passed! Results look correct.")
        return True
    else:
        print("❌ Some checks failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    results_file = sys.argv[1]
    success = inspect_results(results_file)
    sys.exit(0 if success else 1)
