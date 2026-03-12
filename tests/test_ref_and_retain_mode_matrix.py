"""Full matrix: ref_logs (path set/not) × retain_reference_mode (on/off) × benchmark × metrics.

Covers all four ref/retain_mode cases, each with TOFU/MUSE and all/some metrics.
Uses real config shape and evaluate() path (mocked model/metric) to assert
reference_logs is passed or not as expected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from omegaconf import OmegaConf

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


def _canonical_retain_json_path(tmp_path: Path) -> Path:
    p = tmp_path / "retain_ref.json"
    data = {
        "mia_min_k": {"agg_value": 0.1},
        "forget_truth_ratio": {
            "value_by_index": {"0": {"score": 0.5}},
            "agg_value": 0.5,
        },
    }
    p.write_text(json.dumps(data))
    return p


def _make_tofu_eval_cfg(
    *,
    retain_logs_path: str | None,
    retain_reference_mode: bool,
    some_metrics: bool,
    coalesce: bool = True,
) -> dict:
    """Minimal TOFU trajectory eval_cfg for get_evaluators + evaluate."""
    ref_block = {}
    if retain_logs_path:
        ref_block = {
            "reference_logs": {
                "retain_model_logs": {
                    "path": retain_logs_path,
                    "include": {
                        "mia_min_k": {"access_key": "retain"},
                        "forget_truth_ratio": {"access_key": "retain_ftr"},
                    },
                },
            },
        }
    metrics_config = {
        "handler": "trajectory_metrics",
        "datasets": {},
        "collators": {},
        "metrics": ["privleak", "truth_ratio"] if some_metrics else ["privleak", "truth_ratio", "probability", "rouge"],
        "metric_display_names": ["trajectory_privleak", "trajectory_forget_quality"] if some_metrics else ["trajectory_privleak", "trajectory_forget_quality", "trajectory_probability", "trajectory_rouge"],
        **ref_block,
    }
    cfg = {
        "handler": "TOFUEvaluator",
        "output_dir": "/tmp/test_matrix",
        "overwrite": True,
        "coalesce_trajectory_metrics": coalesce,
        "retain_reference_mode": retain_reference_mode,
        "retain_logs_path": retain_logs_path or None,
        "metrics": {"trajectory_all": metrics_config},
    }
    return cfg


def _make_muse_eval_cfg(
    *,
    retain_logs_path: str | None,
    retain_reference_mode: bool,
    some_metrics: bool,
    coalesce: bool = True,
) -> dict:
    """Minimal MUSE trajectory eval_cfg for get_evaluators + evaluate."""
    ref_block = {}
    if retain_logs_path:
        ref_block = {
            "reference_logs": {
                "retain_model_logs": {
                    "path": retain_logs_path,
                    "include": {
                        "mia_min_k": {"access_key": "retain"},
                        "forget_truth_ratio": {"access_key": "retain_ftr"},
                    },
                },
            },
        }
    metrics_config = {
        "handler": "trajectory_metrics",
        "datasets": {},
        "collators": {},
        "metrics": ["privleak", "forget_knowmem_rouge"] if some_metrics else ["privleak", "forget_knowmem_rouge", "retain_knowmem_rouge", "extraction_strength"],
        "metric_display_names": ["trajectory_privleak", "trajectory_forget_knowmem"] if some_metrics else ["trajectory_privleak", "trajectory_forget_knowmem", "trajectory_retain_knowmem", "trajectory_extraction_strength"],
        **ref_block,
    }
    cfg = {
        "handler": "MUSEEvaluator",
        "output_dir": "/tmp/test_matrix",
        "overwrite": True,
        "coalesce_trajectory_metrics": coalesce,
        "data_split": "News",
        "retain_reference_mode": retain_reference_mode,
        "retain_logs_path": retain_logs_path or None,
        "metrics": {"trajectory_all": metrics_config},
    }
    return cfg


@pytest.mark.parametrize("ref_logs_set", [True, False], ids=["ref_logs_set", "ref_logs_not_set"])
@pytest.mark.parametrize("retain_reference_mode", [True, False], ids=["retain_mode_on", "retain_mode_off"])
@pytest.mark.parametrize("benchmark", ["tofu", "muse"], ids=["tofu", "muse"])
@pytest.mark.parametrize("metrics_kind", ["all", "some"], ids=["all_metrics", "some_metrics"])
@pytest.mark.parametrize("coalesce", [True, False], ids=["coalesce_on", "coalesce_off"])
def test_ref_and_retain_mode_matrix(
    tmp_path: Path,
    ref_logs_set: bool,
    retain_reference_mode: bool,
    benchmark: str,
    metrics_kind: str,
    coalesce: bool,
) -> None:
    """Full matrix: (ref_logs set/not) × (retain_reference_mode on/off) × (tofu/muse) × (all/some metrics) × (coalesce on/off).
    Asserts reference_logs is passed to metric only when path is set; evaluate() completes."""
    from evals import get_evaluators

    try:
        from evals.metrics import METRICS_REGISTRY
        if "trajectory_metrics" not in METRICS_REGISTRY:
            pytest.skip("trajectory_metrics not registered")
    except Exception:
        pytest.skip("evals not importable")

    path_str = str(_canonical_retain_json_path(tmp_path)) if ref_logs_set else None
    some_metrics = metrics_kind == "some"

    if benchmark == "tofu":
        cfg_dict = _make_tofu_eval_cfg(
            retain_logs_path=path_str,
            retain_reference_mode=retain_reference_mode,
            some_metrics=some_metrics,
            coalesce=coalesce,
        )
        eval_key = "tofu_trajectory"
    else:
        cfg_dict = _make_muse_eval_cfg(
            retain_logs_path=path_str,
            retain_reference_mode=retain_reference_mode,
            some_metrics=some_metrics,
            coalesce=coalesce,
        )
        eval_key = "muse_trajectory"

    eval_cfg = OmegaConf.create(cfg_dict)
    mock_metric = Mock(return_value={"trajectory_all": {"agg_value": 0.5}})
    with patch("evals.base.get_metrics") as get_metrics_mock:
        get_metrics_mock.return_value = {"trajectory_all": mock_metric}
        evaluators = get_evaluators({eval_key: eval_cfg})
    ev = evaluators[eval_key]
    mock_model = MagicMock()

    load_return = {}
    if ref_logs_set and path_str:
        load_return = {
            "mia_min_k": {"agg_value": 0.1},
            "forget_truth_ratio": {"value_by_index": {"0": {"score": 0.5}}, "agg_value": 0.5},
        }

    with (
        patch.object(ev, "load_logs_from_file", return_value=load_return),
        patch.object(ev, "save_logs"),
        patch.object(ev, "prepare_model", return_value=mock_model),
    ):
        ev.evaluate(mock_model)

    mock_metric.assert_called_once()
    call_kwargs = mock_metric.call_args[1]

    if ref_logs_set and path_str:
        assert "reference_logs" in call_kwargs, (
            f"ref_logs_set=True ({benchmark}, {metrics_kind}): metric must receive reference_logs"
        )
        ref = call_kwargs["reference_logs"]
        assert "retain_model_logs" in ref
        rml = ref["retain_model_logs"]
        assert rml.get("retain") is not None or "retain" in str(rml)
        assert rml.get("retain_ftr") is not None or "retain_ftr" in str(rml)
    else:
        assert "reference_logs" not in call_kwargs, (
            f"ref_logs_set=False or path null ({benchmark}, {metrics_kind}, retain_reference_mode={retain_reference_mode}): "
            "metric must not receive reference_logs so it does not require step-matched data."
        )

    assert ev.eval_cfg.get("retain_reference_mode") is retain_reference_mode
    if ref_logs_set and path_str:
        assert ev.eval_cfg.get("retain_logs_path") == path_str
    else:
        assert ev.eval_cfg.get("retain_logs_path") is None or str(ev.eval_cfg.get("retain_logs_path")).lower() in ("null", "none", "")
