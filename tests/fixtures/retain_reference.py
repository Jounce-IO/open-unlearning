"""Retain reference JSON fixtures for loader and metric tests.

Per contracts/retain-reference-json.md and research.md:
- retain_reference_mode form: scalar aggregate + by_step keys (mia_min_k_by_step, forget_truth_ratio_by_step).
- short form: nested aggregate only (e.g. mia_min_k with agg_value as dict by view/traj).
"""

# Retain reference mode form: scalar aggregate + by_step (for loader contract test 1, E2E).
RETAIN_REFERENCE_MODE = {
    "mia_min_k": {"agg_value": 0.27, "auc": 0.27},
    "forget_truth_ratio": {"agg_value": 0.85, "value_by_index": {"0": {"score": 0.85}}},
    "mia_min_k_by_step": {
        "0": {"agg_value": 0.25},
        "1": {"agg_value": 0.28},
        "2": {"agg_value": 0.27},
    },
    "forget_truth_ratio_by_step": {
        "0": {"value_by_index": {"0": {"score": 0.84}, "1": {"score": 0.86}}},
        "1": {"value_by_index": {"0": {"score": 0.85}}},
        "2": {"value_by_index": {"0": {"score": 0.85}}},
    },
}

# Short form: nested aggregate only (view/traj) for loader normalization / _required_but_missing tests.
NESTED_AGGREGATE_ONLY = {
    "mia_min_k": {
        "agg_value": {
            "full": {"steps": {"privleak": 0.42}},
        },
    },
    "forget_truth_ratio": {
        "agg_value": {
            "full": {"steps": {"truth_ratio": 0.83}},
        },
    },
}

# Nested but extractable: one view/traj with numeric value (loader should normalize to scalar).
NESTED_AGGREGATE_EXTRACTABLE = {
    "mia_min_k": {
        "agg_value": {
            "full": {"steps": {"privleak": 0.42}},
        },
    },
}

# Nested and not extractable (no numeric, no auc): loader should set _required_but_missing.
NESTED_AGGREGATE_NOT_EXTRACTABLE = {
    "mia_min_k": {
        "agg_value": {"foo": "bar"},
    },
}

# Real report shape: mia_min_k has scalar; forget_truth_ratio has trajectory_forget_Truth_Ratio-style
# agg_value (full.steps.truth_ratio list) which current extractor does not normalize.
# With trajectory_all include order (mia_min_k then forget_truth_ratio), last wins → slot gets
# forget_truth_ratio → no scalar → slot bug (privleak gets nothing).
REAL_REPORT_SHAPE_BOTH_KEYS = {
    "mia_min_k": {"agg_value": 0.2725, "auc": 0.2725, "holdout": {}, "forget": {}},
    "trajectory_forget_Truth_Ratio": {
        "agg_value": {
            "full": {"steps": {"truth_ratio": [0.76, 0.71, 0.69, 0.68]}},
        },
        "value_by_index": {"0": {"score": 0.77}},
    },
}
