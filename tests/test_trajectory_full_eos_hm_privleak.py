"""Full vs eos trajectory semantics for hm_aggregate retain MU and privleak streaming."""

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

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


def test_hm_aggregate_nine_keys_uses_all():
    """hm_aggregate with 9 MU components returns harmonic mean of all 9 (full trajectory MU)."""
    import scipy.stats

    metric = METRICS_REGISTRY["hm_aggregate"]
    # 9 components: retain_*, ra_*, wf_*
    vals = [0.5 + i * 0.05 for i in range(9)]
    pre = {
        "retain_Q_A_Prob": {"agg_value": vals[0]},
        "retain_Q_A_ROUGE": {"agg_value": vals[1]},
        "retain_Truth_Ratio": {"agg_value": vals[2]},
        "ra_Q_A_Prob_normalised": {"agg_value": vals[3]},
        "ra_Q_A_ROUGE": {"agg_value": vals[4]},
        "ra_Truth_Ratio": {"agg_value": vals[5]},
        "wf_Q_A_Prob_normalised": {"agg_value": vals[6]},
        "wf_Q_A_ROUGE": {"agg_value": vals[7]},
        "wf_Truth_Ratio": {"agg_value": vals[8]},
    }
    logits = torch.zeros(1, 2, 8)
    batch_t = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "index": torch.tensor([0]),
    }
    retain_agg = {"0": {"full": pre, "eos": pre}}
    r = _call_metric_at_step(
        metric,
        logits,
        batch_t,
        metric_config={},
        sample_idx="0",
        step_index=0,
        retain_agg_by_step=retain_agg,
        trajectory_view="full",
    )
    expected = scipy.stats.hmean(vals)
    assert r["agg_value"] is not None
    assert abs(r["agg_value"] - expected) < 1e-9


def test_hm_aggregate_three_keys_unchanged():
    """hm_aggregate with 3 retain components returns hmean of 3 (backward compat)."""
    import scipy.stats

    metric = METRICS_REGISTRY["hm_aggregate"]
    pre = {
        "retain_Q_A_Prob": {"agg_value": 0.8},
        "retain_Q_A_ROUGE": {"agg_value": 0.7},
        "retain_Truth_Ratio": {"agg_value": 0.6},
    }
    logits = torch.zeros(1, 2, 8)
    batch_t = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "index": torch.tensor([0]),
    }
    retain_agg = {"0": {"full": pre, "eos": pre}}
    r = _call_metric_at_step(
        metric,
        logits,
        batch_t,
        metric_config={},
        sample_idx="0",
        step_index=0,
        retain_agg_by_step=retain_agg,
        trajectory_view="full",
    )
    assert abs(r["agg_value"] - scipy.stats.hmean([0.8, 0.7, 0.6])) < 1e-9


def test_hm_aggregate_nested_nine_keys_requires_view():
    """With 9-key nested structure, trajectory_view is required (no silent default)."""
    import pytest

    metric = METRICS_REGISTRY["hm_aggregate"]
    pre = {
        "retain_Q_A_Prob": {"agg_value": 0.5},
        "ra_Q_A_Prob_normalised": {"agg_value": 0.5},
        "wf_Q_A_Prob_normalised": {"agg_value": 0.5},
    }
    retain_agg = {"0": {"full": pre, "eos": pre}}
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


def test_hm_aggregate_returns_none_when_any_component_is_none():
    """hm_aggregate must not silently drop None; when any component is None, return agg_value=None (no hidden fallback)."""
    metric = METRICS_REGISTRY["hm_aggregate"]
    pre = {
        "retain_Q_A_Prob": {"agg_value": 0.8},
        "retain_Q_A_ROUGE": {"agg_value": None},
        "retain_Truth_Ratio": {"agg_value": 0.6},
    }
    retain_agg = {"0": {"full": pre, "eos": pre}}
    logits = torch.zeros(1, 2, 8)
    batch_t = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "index": torch.tensor([0]),
    }
    r = _call_metric_at_step(
        metric,
        logits,
        batch_t,
        metric_config={},
        sample_idx="0",
        step_index=0,
        retain_agg_by_step=retain_agg,
        trajectory_view="full",
    )
    assert r["agg_value"] is None


def test_hm_aggregate_empty_view_dict_returns_none_not_value_error():
    """Reproduces bug: step has nested views but one view has empty dict (no MU keys). Must return agg_value=None, not raise ValueError."""
    metric = METRICS_REGISTRY["hm_aggregate"]
    # Step 13: "full" is empty (no samples contributed), "eos" has valid MU components
    retain_agg = {
        "13": {
            "full": {},  # empty -> no MU keys
            "eos": {
                "retain_Q_A_Prob": {"agg_value": 0.5},
                "retain_Q_A_ROUGE": {"agg_value": 0.5},
                "retain_Truth_Ratio": {"agg_value": 0.5},
            },
        }
    }
    logits = torch.zeros(1, 2, 8)
    batch_t = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "index": torch.tensor([0]),
    }
    r = _call_metric_at_step(
        metric=metric,
        logits=logits,
        batch_template=batch_t,
        tokenizer=None,
        metric_config={},
        sample_idx="0",
        step_index=13,
        retain_agg_by_step=retain_agg,
        trajectory_view="full",
    )
    assert r["agg_value"] is None


def test_hm_aggregate_per_traj_empty_full_view_returns_none():
    """Per-traj structure: one trajectory type has step with empty 'full' view; must return None for that view."""
    metric = METRICS_REGISTRY["hm_aggregate"]
    retain_agg = {
        "steps": {"0": {"full": {"retain_Q_A_Prob": {"agg_value": 0.5}}, "eos": {"retain_Q_A_Prob": {"agg_value": 0.5}}}},
        "fixation_start": {
            "0": {"full": {}, "eos": {"retain_Q_A_Prob": {"agg_value": 0.5}}},
        },
    }
    logits = torch.zeros(1, 2, 8)
    batch_t = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "index": torch.tensor([0]),
    }
    r = _call_metric_at_step(
        metric=metric,
        logits=logits,
        batch_template=batch_t,
        tokenizer=None,
        metric_config={},
        sample_idx="0",
        step_index=0,
        retain_agg_by_step=retain_agg,
        traj_name="fixation_start",
        trajectory_view="full",
    )
    assert r["agg_value"] is None


def test_hm_aggregate_view_dict_with_non_mu_keys_returns_none():
    """View dict exists but has no MU component keys (e.g. only 'other'); must return None, not raise."""
    metric = METRICS_REGISTRY["hm_aggregate"]
    retain_agg = {
        "0": {
            "full": {"other_key": 1},
            "eos": {"retain_Q_A_Prob": {"agg_value": 0.5}, "retain_Q_A_ROUGE": {"agg_value": 0.5}},
        },
    }
    logits = torch.zeros(1, 2, 8)
    batch_t = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "index": torch.tensor([0]),
    }
    r = _call_metric_at_step(
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
    assert r["agg_value"] is None


# --- Many more scenarios: step key type, empty retain, wrong view, per-traj edge cases ---

def test_hm_aggregate_step_index_int_and_str_both_resolved():
    """Step key can be int or str; lookup uses str(step_index) or step_key."""
    import pytest

    metric = METRICS_REGISTRY["hm_aggregate"]
    pre = {"retain_Q_A_Prob": {"agg_value": 0.5}, "retain_Q_A_ROUGE": {"agg_value": 0.5}, "retain_Truth_Ratio": {"agg_value": 0.5}}
    logits = torch.zeros(1, 2, 8)
    batch_t = {"input_ids": torch.zeros(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2, dtype=torch.long), "index": torch.tensor([0])}

    for step_index, step_key in [(0, "0"), (1, "1")]:
        retain_agg = {step_key: {"full": pre, "eos": pre}}
        r = _call_metric_at_step(
            metric, logits, batch_t, metric_config={}, sample_idx="0", step_index=step_index,
            retain_agg_by_step=retain_agg, trajectory_view="full",
        )
        assert r["agg_value"] is not None


def test_hm_aggregate_retain_agg_empty_dict_returns_none():
    """When retain_agg_by_step is empty, hm_aggregate gets no pre_compute -> returns None."""
    metric = METRICS_REGISTRY["hm_aggregate"]
    logits = torch.zeros(1, 2, 8)
    batch_t = {"input_ids": torch.zeros(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2, dtype=torch.long), "index": torch.tensor([0])}
    r = _call_metric_at_step(
        metric, logits, batch_t, metric_config={}, sample_idx="0", step_index=0,
        retain_agg_by_step={}, trajectory_view="full",
    )
    assert r["agg_value"] is None


def test_hm_aggregate_retain_agg_none_treated_as_empty():
    """When retain_agg_by_step is None (caller passes None), treated as empty -> None."""
    metric = METRICS_REGISTRY["hm_aggregate"]
    logits = torch.zeros(1, 2, 8)
    batch_t = {"input_ids": torch.zeros(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2, dtype=torch.long), "index": torch.tensor([0])}
    r = _call_metric_at_step(
        metric, logits, batch_t, metric_config={}, sample_idx="0", step_index=0,
        retain_agg_by_step=None, trajectory_view="full",
    )
    assert r["agg_value"] is None


def test_hm_aggregate_trajectory_view_invalid_raises():
    """trajectory_view must be 'full' or 'eos' when structure is per-view; invalid value raises."""
    import pytest

    metric = METRICS_REGISTRY["hm_aggregate"]
    retain_agg = {"0": {"full": {"retain_Q_A_Prob": {"agg_value": 0.5}}, "eos": {"retain_Q_A_Prob": {"agg_value": 0.5}}}}
    logits = torch.zeros(1, 2, 8)
    batch_t = {"input_ids": torch.zeros(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2, dtype=torch.long), "index": torch.tensor([0])}
    with pytest.raises(ValueError, match="trajectory_view"):
        _call_metric_at_step(
            metric, logits, batch_t, metric_config={}, sample_idx="0", step_index=0,
            retain_agg_by_step=retain_agg, trajectory_view="invalid",
        )


def test_hm_aggregate_step_key_missing_returns_none():
    """When step_index has no entry in retain_agg_by_step, pre_compute is missing -> None."""
    metric = METRICS_REGISTRY["hm_aggregate"]
    retain_agg = {"0": {"full": {"retain_Q_A_Prob": {"agg_value": 0.5}}, "eos": {"retain_Q_A_Prob": {"agg_value": 0.5}}}}
    logits = torch.zeros(1, 2, 8)
    batch_t = {"input_ids": torch.zeros(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2, dtype=torch.long), "index": torch.tensor([0])}
    r = _call_metric_at_step(
        metric, logits, batch_t, metric_config={}, sample_idx="0", step_index=99,
        retain_agg_by_step=retain_agg, trajectory_view="full",
    )
    assert r["agg_value"] is None


def test_hm_aggregate_per_traj_missing_traj_name_returns_none():
    """Per-traj: traj_name not in retain_agg_by_step -> by_traj.get(traj_name) empty -> None."""
    metric = METRICS_REGISTRY["hm_aggregate"]
    retain_agg = {"steps": {"0": {"full": {"retain_Q_A_Prob": {"agg_value": 0.5}}, "eos": {"retain_Q_A_Prob": {"agg_value": 0.5}}}}}
    logits = torch.zeros(1, 2, 8)
    batch_t = {"input_ids": torch.zeros(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2, dtype=torch.long), "index": torch.tensor([0])}
    r = _call_metric_at_step(
        metric, logits, batch_t, metric_config={}, sample_idx="0", step_index=0,
        retain_agg_by_step=retain_agg, traj_name="fixation_ratio", trajectory_view="full",
    )
    assert r["agg_value"] is None


def test_hm_aggregate_eos_empty_full_populated_returns_none_for_eos():
    """Inverse of full-empty: eos view empty dict, full populated; request eos -> None."""
    metric = METRICS_REGISTRY["hm_aggregate"]
    retain_agg = {
        "0": {
            "full": {"retain_Q_A_Prob": {"agg_value": 0.5}, "retain_Q_A_ROUGE": {"agg_value": 0.5}, "retain_Truth_Ratio": {"agg_value": 0.5}},
            "eos": {},
        },
    }
    logits = torch.zeros(1, 2, 8)
    batch_t = {"input_ids": torch.zeros(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2, dtype=torch.long), "index": torch.tensor([0])}
    r_full = _call_metric_at_step(metric, logits, batch_t, metric_config={}, sample_idx="0", step_index=0, retain_agg_by_step=retain_agg, trajectory_view="full")
    r_eos = _call_metric_at_step(metric, logits, batch_t, metric_config={}, sample_idx="0", step_index=0, retain_agg_by_step=retain_agg, trajectory_view="eos")
    assert r_full["agg_value"] is not None
    assert r_eos["agg_value"] is None


def test_hm_aggregate_nine_keys_all_none_returns_none():
    """All 9 components have agg_value None -> hmean would fail; hm_aggregate returns None."""
    metric = METRICS_REGISTRY["hm_aggregate"]
    from evals.metrics.trajectory_metrics import EXPECTED_9_MU_KEYS
    pre = {k: {"agg_value": None} for k in EXPECTED_9_MU_KEYS}
    retain_agg = {"0": {"full": pre, "eos": pre}}
    logits = torch.zeros(1, 2, 8)
    batch_t = {"input_ids": torch.zeros(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2, dtype=torch.long), "index": torch.tensor([0])}
    r = _call_metric_at_step(metric, logits, batch_t, metric_config={}, sample_idx="0", step_index=0, retain_agg_by_step=retain_agg, trajectory_view="full")
    assert r["agg_value"] is None


def test_hm_aggregate_flat_structure_no_view_keys_used_as_is():
    """Legacy flat: pre_compute_step has no 'full'/'eos' keys but has MU keys -> used as single pre_compute."""
    metric = METRICS_REGISTRY["hm_aggregate"]
    pre = {"retain_Q_A_Prob": {"agg_value": 0.5}, "retain_Q_A_ROUGE": {"agg_value": 0.5}, "retain_Truth_Ratio": {"agg_value": 0.5}}
    retain_agg = {"0": pre}
    logits = torch.zeros(1, 2, 8)
    batch_t = {"input_ids": torch.zeros(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2, dtype=torch.long), "index": torch.tensor([0])}
    r = _call_metric_at_step(metric, logits, batch_t, metric_config={}, sample_idx="0", step_index=0, retain_agg_by_step=retain_agg, trajectory_view="full")
    assert r["agg_value"] is not None


def test_hm_aggregate_per_traj_all_four_trajectory_types_both_views():
    """Per-traj: all four traj types with both views; each (traj_name, view) returns valid hmean."""
    import scipy.stats

    metric = METRICS_REGISTRY["hm_aggregate"]
    pre = {"retain_Q_A_Prob": {"agg_value": 0.5}, "retain_Q_A_ROUGE": {"agg_value": 0.5}, "retain_Truth_Ratio": {"agg_value": 0.5}}
    traj_names = ("steps", "fixation_start", "fixation_end", "fixation_ratio")
    retain_agg = {t: {"0": {"full": pre, "eos": pre}} for t in traj_names}
    logits = torch.zeros(1, 2, 8)
    batch_t = {"input_ids": torch.zeros(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2, dtype=torch.long), "index": torch.tensor([0])}
    for traj_name in traj_names:
        for view in ("full", "eos"):
            r = _call_metric_at_step(
                metric, logits, batch_t, metric_config={}, sample_idx="0", step_index=0,
                retain_agg_by_step=retain_agg, traj_name=traj_name, trajectory_view=view,
            )
            assert r["agg_value"] is not None
            assert abs(r["agg_value"] - 0.5) < 1e-9


def test_hm_aggregate_per_traj_step_missing_for_one_traj_returns_none():
    """Per-traj: steps has step 0; fixation_start has no step 0 -> None for fixation_start step 0."""
    metric = METRICS_REGISTRY["hm_aggregate"]
    retain_agg = {
        "steps": {"0": {"full": {"retain_Q_A_Prob": {"agg_value": 0.5}}, "eos": {"retain_Q_A_Prob": {"agg_value": 0.5}}}},
        "fixation_start": {},
    }
    logits = torch.zeros(1, 2, 8)
    batch_t = {"input_ids": torch.zeros(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2, dtype=torch.long), "index": torch.tensor([0])}
    r = _call_metric_at_step(
        metric, logits, batch_t, metric_config={}, sample_idx="0", step_index=0,
        retain_agg_by_step=retain_agg, traj_name="fixation_start", trajectory_view="full",
    )
    assert r["agg_value"] is None
