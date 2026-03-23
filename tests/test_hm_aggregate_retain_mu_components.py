"""Non-trajectory model_utility: hm_aggregate exposes retain_mu_components (9-way MU parity with traj JSON)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.stats

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics import METRICS_REGISTRY


@pytest.fixture
def hm():
    return METRICS_REGISTRY["hm_aggregate"]


def test_hm_aggregate_nine_components_retain_mu_components_and_hmean(hm):
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
    r = hm.evaluate_metric(None, "model_utility", pre_compute=pre)
    assert r["agg_value"] is not None
    assert abs(float(r["agg_value"]) - scipy.stats.hmean(vals)) < 1e-9
    comp = r["retain_mu_components"]
    assert len(comp) == 9
    for k, v in zip(pre.keys(), vals):
        assert comp[k] == pytest.approx(v)


def test_hm_aggregate_three_retain_only_retain_mu_components(hm):
    pre = {
        "retain_Q_A_Prob": {"agg_value": 0.8},
        "retain_Q_A_ROUGE": {"agg_value": 0.7},
        "retain_Truth_Ratio": {"agg_value": 0.6},
    }
    r = hm.evaluate_metric(None, "model_utility", pre_compute=pre)
    exp = scipy.stats.hmean([0.8, 0.7, 0.6])
    assert abs(float(r["agg_value"]) - exp) < 1e-9
    assert r["retain_mu_components"] == {
        "retain_Q_A_Prob": 0.8,
        "retain_Q_A_ROUGE": 0.7,
        "retain_Truth_Ratio": 0.6,
    }


def test_hm_aggregate_one_none_agg_value_none_but_components_partial(hm):
    pre = {
        "retain_Q_A_Prob": {"agg_value": 0.5},
        "retain_Q_A_ROUGE": {"agg_value": None},
        "retain_Truth_Ratio": {"agg_value": 0.6},
    }
    r = hm.evaluate_metric(None, "model_utility", pre_compute=pre)
    assert r["agg_value"] is None
    assert r["retain_mu_components"]["retain_Q_A_Prob"] == 0.5
    assert r["retain_mu_components"]["retain_Q_A_ROUGE"] is None
    assert "retain_Truth_Ratio" not in r["retain_mu_components"]


def test_hm_aggregate_empty_pre_compute(hm):
    r = hm.evaluate_metric(None, "model_utility", pre_compute=None)
    assert r == {"agg_value": None}
    r2 = hm.evaluate_metric(None, "model_utility", pre_compute={})
    assert r2["agg_value"] is None
    assert "retain_mu_components" not in r2


def test_ra_wf_mu_prob_normalised_vs_truth_ratio_use_distinct_aggregates():
    """Regression: ra/wf *_Q_A_Prob_normalised and *_Truth_Ratio must not be conflated.

    In ``trajectory_metrics._compute_mu_for_dataset``, probability is mean normalised
    P(correct); truth ratio uses per-sample wrong/correct then ``mean(max(0, 1 - tr))``
    (TOFU ``true_better`` aggregate). Same code path applies to ``retain_mu_components_by_step``.
    """
    prob_vals = [0.25, 0.5, 0.75]
    tr_per_sample = [0.2, 0.4, 0.6]
    agg_prob = float(np.mean(prob_vals))
    agg_tr = float(np.mean(np.maximum(0, 1 - np.array(tr_per_sample, dtype=np.float64))))
    assert agg_prob == pytest.approx(0.5)
    assert agg_tr == pytest.approx((0.8 + 0.6 + 0.4) / 3)
    assert abs(agg_prob - agg_tr) > 1e-6


def test_hm_aggregate_numpy_floating_accepted(hm):
    pre = {"retain_Q_A_Prob": {"agg_value": np.float64(0.4)}, "retain_Q_A_ROUGE": {"agg_value": np.float64(0.6)}}
    r = hm.evaluate_metric(None, "model_utility", pre_compute=pre)
    assert r["agg_value"] is not None
    assert abs(float(r["agg_value"]) - scipy.stats.hmean([0.4, 0.6])) < 1e-9
    assert r["retain_mu_components"]["retain_Q_A_Prob"] == pytest.approx(0.4)


def test_hm_aggregate_ignores_non_mu_keys_for_hmean(hm):
    """Only retain_/ra_/wf_ keys contribute; other pre_compute entries are ignored."""
    pre = {
        "retain_Q_A_Prob": {"agg_value": 0.5},
        "retain_Q_A_ROUGE": {"agg_value": 0.5},
        "retain_Truth_Ratio": {"agg_value": 0.5},
        "other_metric": {"agg_value": 0.99},
    }
    r = hm.evaluate_metric(None, "model_utility", pre_compute=pre)
    assert r["agg_value"] is not None
    assert abs(float(r["agg_value"]) - 0.5) < 1e-9
    assert len(r["retain_mu_components"]) == 3
    assert "other_metric" not in r["retain_mu_components"]
