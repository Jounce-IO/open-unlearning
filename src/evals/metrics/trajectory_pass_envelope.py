"""pass_envelope for per-pass evaluator JSON (spec: dllm trajectory-pass-artifact contract)."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from evals.metrics.trajectory_pass_binding import PassSpec, SamplingRegime, get_pass_spec

__all__ = ["build_pass_envelope", "validate_pass_envelope"]


def _dataset_from_pass_id(pass_id: str) -> str:
    return pass_id.split("__", 1)[0]


def _envelope_dataset_access_key(pass_id: str, spec: PassSpec) -> str:
    """Artifact ``dataset_access_key`` must match TOFU data buckets (forget/retain/ra/wf).

    Pass ids like ``retain_sft__unguided`` would mis-parse as ``retain_sft`` from the id
    alone; single-dataset passes use ``PassSpec.dataset_access_keys`` instead.
    """
    keys = spec.dataset_access_keys
    if len(keys) == 1:
        return next(iter(sorted(keys)))
    return _dataset_from_pass_id(pass_id)


def build_pass_envelope(
    pass_id: str,
    *,
    version_tag: Optional[Mapping[str, Any]] = None,
    spec: PassSpec | None = None,
) -> dict[str, Any]:
    spec = spec or get_pass_spec(pass_id)
    if spec.evaluation_mode == "unguided":
        regime = SamplingRegime.GUIDANCE_FREE.value
        guided_variant = None
    else:
        regime = SamplingRegime.GUIDED.value
        guided_variant = spec.evaluation_mode
    return {
        "schema_version": 1,
        "pass_id": pass_id,
        "sampling_regime": regime,
        "guided_variant": guided_variant,
        "dataset_access_key": _envelope_dataset_access_key(pass_id, spec),
        "version_tag": dict(version_tag or {}),
        "metric_keys_in_pass": list(spec.display_names_emitted),
    }


def validate_pass_envelope(env: Any) -> None:
    if not isinstance(env, dict):
        raise ValueError("pass_envelope must be a dict")
    required = (
        "schema_version",
        "pass_id",
        "sampling_regime",
        "guided_variant",
        "dataset_access_key",
        "version_tag",
        "metric_keys_in_pass",
    )
    for k in required:
        if k not in env:
            raise ValueError(f"pass_envelope missing {k!r}")
