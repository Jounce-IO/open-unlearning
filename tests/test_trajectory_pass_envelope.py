"""pass_envelope builder."""

from evals.metrics.trajectory_pass_envelope import build_pass_envelope, validate_pass_envelope


def test_build_forget_unguided():
    env = build_pass_envelope("forget__unguided", version_tag={"model_checkpoint": "x"})
    validate_pass_envelope(env)
    assert env["pass_id"] == "forget__unguided"
    assert env["sampling_regime"] == "guidance_free"
    assert env["guided_variant"] is None
    assert env["dataset_access_key"] == "forget"
    assert env["version_tag"] == {"model_checkpoint": "x"}


def test_build_retain_sft_unguided_dataset_access_key_is_retain_bucket():
    env = build_pass_envelope("retain_sft__unguided")
    validate_pass_envelope(env)
    assert env["pass_id"] == "retain_sft__unguided"
    assert env["dataset_access_key"] == "retain"
    assert env["metric_keys_in_pass"] == ["trajectory_retain_sft_Q_A_ROUGE"]
