"""Regression: evaluator-injected reference_logs (load_and_validate output, no path) reaches ks_test.

Non-trajectory Evaluator merges cached retain JSON into metrics_args; UnlearningMetric.prepare_kwargs
must not discard that shape (previously cleared to {} when no nested ``path``, breaking forget_quality).
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.base import _reference_logs_payload_is_evaluator_preloaded


class TestReferenceLogsPayloadIsEvaluatorPreloaded:
    def test_true_retain_ftr_only(self) -> None:
        assert _reference_logs_payload_is_evaluator_preloaded(
            {"retain_model_logs": {"retain_ftr": {"agg_value": 0.5}}}
        )

    def test_true_retain_only(self) -> None:
        assert _reference_logs_payload_is_evaluator_preloaded(
            {"retain_model_logs": {"retain": {"agg_value": 0.1}}}
        )

    def test_true_retain_mia_by_step(self) -> None:
        assert _reference_logs_payload_is_evaluator_preloaded(
            {
                "retain_model_logs": {
                    "retain_mia_by_step": {"0": {"agg_value": 0.2}},
                }
            }
        )

    def test_true_retain_forget_tr_by_step(self) -> None:
        assert _reference_logs_payload_is_evaluator_preloaded(
            {
                "retain_model_logs": {
                    "retain_forget_tr_by_step": {
                        "0": {"value_by_index": {"0": {"score": 0.5}}},
                    },
                }
            }
        )

    def test_false_empty_container(self) -> None:
        assert not _reference_logs_payload_is_evaluator_preloaded({})
        assert not _reference_logs_payload_is_evaluator_preloaded({"retain_model_logs": {}})

    def test_false_yaml_shell_path_null(self) -> None:
        assert not _reference_logs_payload_is_evaluator_preloaded(
            {
                "retain_model_logs": {
                    "path": None,
                    "include": {"forget_truth_ratio": {"access_key": "retain_ftr"}},
                },
            }
        )

    def test_false_missing_retain_model_logs(self) -> None:
        assert not _reference_logs_payload_is_evaluator_preloaded({"other": {}})

    def test_false_truthy_path_with_slots(self) -> None:
        """YAML config shape: must load from file (has_path), not treated as injected payload."""
        assert not _reference_logs_payload_is_evaluator_preloaded(
            {
                "retain_model_logs": {
                    "path": "/tmp/x.json",
                    "retain_ftr": {"agg_value": 0.5},
                }
            }
        )


class TestKsTestEvaluateWithPreloadedReferenceLogs:
    def test_forget_quality_agg_value_not_none(self) -> None:
        from evals.metrics import METRICS_REGISTRY

        m = METRICS_REGISTRY["ks_test"]
        forget_pre = {
            "value_by_index": {"0": {"score": 0.3}, "1": {"score": 0.35}},
            "agg_value": 0.32,
        }
        cache = {"forget_truth_ratio": forget_pre}
        preloaded = {
            "retain_model_logs": {
                "retain_ftr": {
                    "value_by_index": {"0": {"score": 0.5}, "1": {"score": 0.55}},
                    "agg_value": 0.5,
                },
            }
        }
        out = m.evaluate(
            model=None,
            metric_name="forget_quality",
            cache=cache,
            tokenizer=None,
            template_args=None,
            eval_cfg={},
            pre_compute={"forget_truth_ratio": {"access_key": "forget"}},
            reference_logs=preloaded,
        )
        assert isinstance(out, dict)
        assert out.get("agg_value") is not None
        p = float(out["agg_value"])
        assert 0.0 <= p <= 1.0
