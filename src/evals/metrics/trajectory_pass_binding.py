"""Canonical trajectory pass IDs and metric–pass binding (OU-aligned).

Reference: dllm OPEN_ISSUES.md (trajectory evaluation_mode vs OpenUnlearning AR).
Each pass is one inference run: one dataset access key × sampling regime × optional guided variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, FrozenSet, Mapping, Sequence

__all__ = [
    "SamplingRegime",
    "PassSpec",
    "canonical_pass_ids_eight",
    "canonical_pass_ids_twelve",
    "get_pass_spec",
    "list_all_pass_specs",
    "DISPLAY_METRIC_BINDING",
    "HM_AGGREGATE_SUBMETRIC_BINDING",
    "filter_metrics_and_data_for_pass",
]


class SamplingRegime(str, Enum):
    GUIDANCE_FREE = "guidance_free"
    GUIDED = "guided"


@dataclass(frozen=True)
class PassSpec:
    """What one trajectory inference pass may compute."""

    pass_id: str
    dataset_access_keys: FrozenSet[str]
    internal_metric_keys: FrozenSet[str]
    evaluation_mode: str  # unguided | guided_native | guided_skew
    display_names_emitted: tuple[str, ...]


def canonical_pass_ids_eight() -> tuple[str, ...]:
    """Four splits × two regimes (unguided + guided_native)."""
    bases = ("forget", "retain", "ra", "wf")
    out: list[str] = []
    for b in bases:
        out.append(f"{b}__unguided")
        out.append(f"{b}__guided_native")
    return tuple(out)


def canonical_pass_ids_twelve() -> tuple[str, ...]:
    """Four splits × (unguided + guided_native + guided_skew)."""
    bases = ("forget", "retain", "ra", "wf")
    out: list[str] = []
    for b in bases:
        out.extend(
            (
                f"{b}__unguided",
                f"{b}__guided_native",
                f"{b}__guided_skew",
            )
        )
    return tuple(out)


def _spec(
    pass_id: str,
    datasets: Sequence[str],
    metrics: Sequence[str],
    mode: str,
    displays: Sequence[str],
) -> PassSpec:
    return PassSpec(
        pass_id=pass_id,
        dataset_access_keys=frozenset(datasets),
        internal_metric_keys=frozenset(metrics),
        evaluation_mode=mode,
        display_names_emitted=tuple(displays),
    )


# Implemented pass filters (extend for retain/ra/wf and guided_skew in follow-up tasks).
_PASS_SPECS: dict[str, PassSpec] = {
    "forget__unguided": _spec(
        "forget__unguided",
        ("forget",),
        ("rouge",),
        "unguided",
        ("trajectory_forget_Q_A_ROUGE",),
    ),
    "forget__guided_native": _spec(
        "forget__guided_native",
        ("forget",),
        (
            "probability",
            "extraction_strength",
            "truth_ratio",
        ),
        "guided_native",
        (
            "trajectory_forget_Q_A_Prob",
            "trajectory_extraction_strength",
            "trajectory_forget_Truth_Ratio",
        ),
    ),
    "retain__unguided": _spec(
        "retain__unguided",
        ("retain",),
        ("rouge",),
        "unguided",
        ("trajectory_retain_Q_A_ROUGE",),
    ),
    "retain__guided_native": _spec(
        "retain__guided_native",
        ("retain",),
        ("probability", "extraction_strength", "truth_ratio"),
        "guided_native",
        (
            "trajectory_retain_Q_A_Prob",
            "trajectory_retain_extraction_strength",
            "trajectory_retain_Truth_Ratio",
        ),
    ),
    "ra__unguided": _spec(
        "ra__unguided",
        ("ra",),
        ("rouge",),
        "unguided",
        ("trajectory_ra_Q_A_ROUGE",),
    ),
    "ra__guided_native": _spec(
        "ra__guided_native",
        ("ra",),
        ("probability", "extraction_strength", "truth_ratio"),
        "guided_native",
        (
            "trajectory_ra_Q_A_Prob_normalised",
            "trajectory_ra_extraction_strength",
            "trajectory_ra_Truth_Ratio",
        ),
    ),
    "wf__unguided": _spec(
        "wf__unguided",
        ("wf",),
        ("rouge",),
        "unguided",
        ("trajectory_wf_Q_A_ROUGE",),
    ),
    "wf__guided_native": _spec(
        "wf__guided_native",
        ("wf",),
        ("probability", "extraction_strength", "truth_ratio"),
        "guided_native",
        (
            "trajectory_wf_Q_A_Prob_normalised",
            "trajectory_wf_extraction_strength",
            "trajectory_wf_Truth_Ratio",
        ),
    ),
}

_fg = _PASS_SPECS["forget__guided_native"]
_PASS_SPECS["forget__guided_skew"] = PassSpec(
    pass_id="forget__guided_skew",
    dataset_access_keys=_fg.dataset_access_keys,
    internal_metric_keys=_fg.internal_metric_keys,
    evaluation_mode="guided_skew",
    display_names_emitted=_fg.display_names_emitted,
)
_rg = _PASS_SPECS["retain__guided_native"]
_PASS_SPECS["retain__guided_skew"] = PassSpec(
    pass_id="retain__guided_skew",
    dataset_access_keys=_rg.dataset_access_keys,
    internal_metric_keys=_rg.internal_metric_keys,
    evaluation_mode="guided_skew",
    display_names_emitted=_rg.display_names_emitted,
)
_rga = _PASS_SPECS["ra__guided_native"]
_PASS_SPECS["ra__guided_skew"] = PassSpec(
    pass_id="ra__guided_skew",
    dataset_access_keys=_rga.dataset_access_keys,
    internal_metric_keys=_rga.internal_metric_keys,
    evaluation_mode="guided_skew",
    display_names_emitted=_rga.display_names_emitted,
)
_wg = _PASS_SPECS["wf__guided_native"]
_PASS_SPECS["wf__guided_skew"] = PassSpec(
    pass_id="wf__guided_skew",
    dataset_access_keys=_wg.dataset_access_keys,
    internal_metric_keys=_wg.internal_metric_keys,
    evaluation_mode="guided_skew",
    display_names_emitted=_wg.display_names_emitted,
)


def get_pass_spec(pass_id: str) -> PassSpec:
    if pass_id not in _PASS_SPECS:
        raise KeyError(
            f"Unknown trajectory_pass_id={pass_id!r}. "
            f"Known: {sorted(_PASS_SPECS.keys())}"
        )
    return _PASS_SPECS[pass_id]


# trajectory_all display name → (OU regime label, primary dataset access for that leg)
DISPLAY_METRIC_BINDING: dict[str, tuple[str, str]] = {
    "trajectory_forget_Q_A_Prob": ("guided", "forget"),
    "trajectory_forget_Q_A_ROUGE": ("guidance_free", "forget"),
    "trajectory_extraction_strength": ("guided", "forget"),
    "trajectory_forget_Truth_Ratio": ("guided", "forget"),
    "trajectory_forget_quality": ("guided", "forget"),
    "trajectory_model_utility": ("mixed", "retain_ra_wf"),
    "trajectory_privleak": ("guided", "forget"),
}

HM_AGGREGATE_SUBMETRIC_BINDING: dict[str, tuple[str, str]] = {
    "retain_Q_A_Prob": ("guided", "retain"),
    "retain_Q_A_ROUGE": ("guidance_free", "retain"),
    "retain_Truth_Ratio": ("guided", "retain"),
    "ra_Q_A_Prob_normalised": ("guided", "ra"),
    "ra_Q_A_ROUGE": ("guidance_free", "ra"),
    "ra_Truth_Ratio": ("guided", "ra"),
    "wf_Q_A_Prob_normalised": ("guided", "wf"),
    "wf_Q_A_ROUGE": ("guidance_free", "wf"),
    "wf_Truth_Ratio": ("guided", "wf"),
}


def filter_metrics_and_data_for_pass(
    pass_id: str,
    metrics_to_compute: dict[str, Any],
    data: Any,
    trajectory_config: Any,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Return filtered (metrics, data, trajectory_config) for a single pass."""
    spec = get_pass_spec(pass_id)
    filtered_metrics = {
        k: v for k, v in metrics_to_compute.items() if k in spec.internal_metric_keys
    }
    if not filtered_metrics:
        raise ValueError(
            f"trajectory_pass_id={pass_id!r} left no metrics after filter; "
            f"wanted one of {sorted(spec.internal_metric_keys)}"
        )
    if isinstance(data, dict):
        filtered_data = {
            k: v for k, v in data.items() if k in spec.dataset_access_keys
        }
    else:
        filtered_data = data
    if isinstance(trajectory_config, Mapping):
        tc = dict(trajectory_config)
    else:
        try:
            from omegaconf import OmegaConf

            tc = OmegaConf.to_container(trajectory_config, resolve=True) or {}
        except Exception:
            tc = {}
    if not isinstance(tc, dict):
        tc = {}
    tc["evaluation_mode"] = spec.evaluation_mode
    return filtered_metrics, filtered_data, tc


def list_all_pass_specs() -> Mapping[str, PassSpec]:
    return _PASS_SPECS
