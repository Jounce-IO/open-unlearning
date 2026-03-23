"""Guards and DEBUG lines for trajectory_audit_verbose (no full eval jobs)."""

import logging

import torch

from evals.metrics.trajectory_audit import (
    forget_trajectory_audit_runtime,
    log_metric_audit,
    log_pre_compute_probability,
)


def test_forget_audit_off_when_not_debug():
    lg = logging.getLogger("evaluator")
    old = lg.level
    lg.setLevel(logging.INFO)
    try:
        on, _, _ = forget_trajectory_audit_runtime(
            {"trajectory_audit_verbose": True}, n_samples=5
        )
        assert on is False
    finally:
        lg.setLevel(old)


def test_forget_audit_off_when_large_dataset_without_allow():
    lg = logging.getLogger("evaluator")
    old = lg.level
    lg.setLevel(logging.DEBUG)
    try:
        on, _, _ = forget_trajectory_audit_runtime(
            {"trajectory_audit_verbose": True}, n_samples=100
        )
        assert on is False
    finally:
        lg.setLevel(old)


def test_forget_audit_on_small_dataset():
    lg = logging.getLogger("evaluator")
    old = lg.level
    lg.setLevel(logging.DEBUG)
    try:
        on, path, sid = forget_trajectory_audit_runtime(
            {
                "trajectory_audit_verbose": True,
                "trajectory_audit_jsonl_path": "/tmp/x.jsonl",
                "trajectory_audit_jsonl_sample_id": "3",
            },
            n_samples=10,
        )
        assert on is True
        assert path == "/tmp/x.jsonl"
        assert sid == "3"
    finally:
        lg.setLevel(old)


def test_log_pre_compute_noop_when_audit_runtime_false(caplog):
    caplog.set_level(logging.DEBUG, logger="evaluator")
    log_pre_compute_probability(
        access_key="correct",
        handler_name="probability",
        sample_idx="0",
        step=0,
        labels_field="labels_correct",
        agg_prob=0.5,
        trajectory_audit_runtime=False,
    )
    assert "TRAJECTORY_AUDIT_PRE_COMPUTE_PROB" not in caplog.text


def test_log_pre_compute_emits_when_runtime_true(caplog):
    caplog.set_level(logging.DEBUG, logger="evaluator")
    log_pre_compute_probability(
        access_key="correct",
        handler_name="probability",
        sample_idx="0",
        step=0,
        labels_field="labels_correct",
        agg_prob=0.5,
        trajectory_audit_runtime=True,
    )
    assert "TRAJECTORY_AUDIT_PRE_COMPUTE_PROB" in caplog.text


def test_log_metric_audit_noop_without_runtime(caplog):
    caplog.set_level(logging.DEBUG, logger="evaluator")
    logits = torch.zeros(1, 4, 8)
    log_metric_audit(
        handler_metric_name="truth_ratio",
        result={"agg_value": 1.0},
        logits=logits,
        tokenizer=None,
        sample_input_ids=None,
        sample_prompt_len=0,
        sample_idx="0",
        kwargs={"trajectory_audit_runtime": False},
        pre_compute_results={
            "correct": {
                "agg_value": 0.9,
                "value_by_index": {"0": {"prob": 0.9}},
            },
            "wrong": {
                "agg_value": 0.1,
                "value_by_index": {"0": {"prob": 0.1}},
            },
        },
    )
    assert "TRAJECTORY_AUDIT metric=truth_ratio" not in caplog.text


def test_log_metric_audit_truth_pre_when_runtime(caplog):
    caplog.set_level(logging.DEBUG, logger="evaluator")
    logits = torch.zeros(1, 4, 8)
    log_metric_audit(
        handler_metric_name="truth_ratio",
        result={"agg_value": 9.0},
        logits=logits,
        tokenizer=None,
        sample_input_ids=None,
        sample_prompt_len=0,
        sample_idx="0",
        kwargs={
            "trajectory_audit_runtime": True,
            "traj_name": "steps",
            "step": 0,
            "trajectory_audit_view": "full",
            "trajectory_audit_batch_idx": 0,
            "step_index": 0,
        },
        pre_compute_results={
            "correct": {
                "agg_value": 0.9,
                "value_by_index": {"0": {"prob": 0.9}},
            },
            "wrong": {
                "agg_value": 0.1,
                "value_by_index": {"0": {"prob": 0.1}},
            },
        },
    )
    assert "TRAJECTORY_AUDIT metric=truth_ratio" in caplog.text
    assert "TRAJECTORY_AUDIT_TRUTH_PRE" in caplog.text
