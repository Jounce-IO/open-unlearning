"""Full vs eos trajectory semantics for hm_aggregate retain MU and privleak streaming."""

import torch

from evals.metrics import METRICS_REGISTRY
from evals.metrics.mia.min_k import MinKProbAttack
from evals.metrics.mia.utils import MIAStreamingAccumulator
from evals.metrics.trajectory_metrics import _call_metric_at_step


def test_hm_aggregate_call_uses_trajectory_view_when_retain_nested():
    """hm_aggregate must use retain_agg_by_step[step][view], not the same pre for both views."""
    metric = METRICS_REGISTRY["hm_aggregate"]
    pre_full = {
        "retain_Q_A_Prob": {"agg_value": 0.9},
        "retain_Q_A_ROUGE": {"agg_value": 0.9},
    }
    pre_eos = {
        "retain_Q_A_Prob": {"agg_value": 0.1},
        "retain_Q_A_ROUGE": {"agg_value": 0.1},
    }
    retain_agg = {"0": {"full": pre_full, "eos": pre_eos}}
    logits = torch.zeros(1, 2, 8)
    batch_t = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "index": torch.tensor([0]),
    }
    r_full = _call_metric_at_step(
        metric=metric,
        logits=logits,
        batch_template=batch_t,
        tokenizer=None,
        metric_config={},
        sample_idx="0",
        step_index=0,
        retain_agg_by_step=retain_agg,
        trajectory_view="full",
    )
    r_eos = _call_metric_at_step(
        metric=metric,
        logits=logits,
        batch_template=batch_t,
        tokenizer=None,
        metric_config={},
        sample_idx="0",
        step_index=0,
        retain_agg_by_step=retain_agg,
        trajectory_view="eos",
    )
    assert r_full["agg_value"] is not None and r_eos["agg_value"] is not None
    assert abs(r_full["agg_value"] - r_eos["agg_value"]) > 1e-6


def test_hm_aggregate_nested_requires_trajectory_view():
    """Per-view retain_agg: missing trajectory_view must raise (no default to full)."""
    import pytest

    metric = METRICS_REGISTRY["hm_aggregate"]
    retain_agg = {
        "0": {
            "full": {"retain_Q_A_Prob": {"agg_value": 0.5}},
            "eos": {"retain_Q_A_Prob": {"agg_value": 0.5}},
        }
    }
    logits = torch.zeros(1, 2, 8)
    batch_t = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "index": torch.tensor([0]),
    }
    with pytest.raises(ValueError, match="trajectory_view"):
        _call_metric_at_step(
            metric,
            logits,
            batch_t,
            metric_config={},
            sample_idx="0",
            step_index=0,
            retain_agg_by_step=retain_agg,
        )


def test_hm_aggregate_same_view_when_full_equals_eos_pre():
    """When L_eff == L, full and eos retain pre_match; hm_aggregate matches."""
    metric = METRICS_REGISTRY["hm_aggregate"]
    pre = {"retain_Q_A_Prob": {"agg_value": 0.5}, "retain_Q_A_ROUGE": {"agg_value": 0.5}}
    retain_agg = {"0": {"full": pre, "eos": pre}}
    logits = torch.zeros(1, 2, 8)
    batch_t = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "index": torch.tensor([0]),
    }
    r_full = _call_metric_at_step(
        metric, logits, batch_t, metric_config={}, sample_idx="0", step_index=0,
        retain_agg_by_step=retain_agg, trajectory_view="full",
    )
    r_eos = _call_metric_at_step(
        metric, logits, batch_t, metric_config={}, sample_idx="0", step_index=0,
        retain_agg_by_step=retain_agg, trajectory_view="eos",
    )
    assert abs(r_full["agg_value"] - r_eos["agg_value"]) < 1e-9


def test_privleak_streaming_full_vs_eos_aggregate_differs_when_tail_differs():
    """EOS-truncated per_position scores omit tail; forget mean min-k score differs full vs eos."""
    import numpy as np

    collator = lambda x: x
    k = 0.3
    acc_full = MIAStreamingAccumulator(
        MinKProbAttack, collator, 4, torch.device("cpu"), k=k
    )
    acc_eos = MIAStreamingAccumulator(
        MinKProbAttack, collator, 4, torch.device("cpu"), k=k
    )
    batch_f = {"index": torch.tensor([0, 1, 2, 3])}
    batch_h = {"index": torch.tensor([10, 11, 12, 13])}
    # Same sequence: full sees tail; eos prefix is first 4 positions only (truncation contract).
    forget_full = [[0.15] * 6 + [0.99] * 4] * 4
    forget_eos = [row[:4] for row in forget_full]
    hold_full = [[0.5] * 10] * 4
    hold_eos = [row[:4] for row in hold_full]

    acc_full.add_forget_batch(batch_f, per_position_scores=forget_full)
    acc_full.add_holdout_batch(batch_h, per_position_scores=hold_full)
    acc_eos.add_forget_batch(batch_f, per_position_scores=forget_eos)
    acc_eos.add_holdout_batch(batch_h, per_position_scores=hold_eos)

    sf = np.mean([v["score"] for v in acc_full.forget_value_by_index.values()])
    se = np.mean([v["score"] for v in acc_eos.forget_value_by_index.values()])
    assert abs(sf - se) > 1e-6


def test_retain_mu_components_nested_by_view():
    """Coalesced retain_mu_components_by_step nests per view when retain_agg is per-view."""
    step_key = "0"
    retain_agg = {
        step_key: {
            "full": {
                "retain_Q_A_Prob": {"agg_value": 0.8},
                "retain_Q_A_ROUGE": {"agg_value": 0.7},
            },
            "eos": {
                "retain_Q_A_Prob": {"agg_value": 0.2},
                "retain_Q_A_ROUGE": {"agg_value": 0.3},
            },
        }
    }
    components = {}
    for sk, views_dict in retain_agg.items():
        components[str(sk)] = {}
        for view in ("full", "eos"):
            if view not in views_dict:
                continue
            pre = views_dict[view]
            components[str(sk)][view] = {}
            for name in ("retain_Q_A_Prob", "retain_Q_A_ROUGE", "retain_Truth_Ratio"):
                ent = pre.get(name)
                if isinstance(ent, dict) and "agg_value" in ent:
                    components[str(sk)][view][name] = float(ent["agg_value"])
    assert "full" in components[step_key] and "eos" in components[step_key]
    assert components[step_key]["full"]["retain_Q_A_Prob"] == 0.8
    assert components[step_key]["eos"]["retain_Q_A_Prob"] == 0.2
