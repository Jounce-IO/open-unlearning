"""Canonical trajectory pass IDs and metric–pass binding (OU-aligned).

Reference: dllm OPEN_ISSUES.md (trajectory evaluation_mode vs OpenUnlearning AR).
Each pass is one inference run: one dataset access key × sampling regime × optional guided variant.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, FrozenSet, Mapping, Sequence

logger = logging.getLogger(__name__)

DEPRECATED_PASS_IDS: frozenset[str] = frozenset(
    {
        "forget__guided_native",
        "retain__guided_native",
        "ra__guided_native",
        "wf__guided_native",
    }
)

__all__ = [
    "DEPRECATED_PASS_IDS",
    "SamplingRegime",
    "PassSpec",
    "canonical_pass_ids_eight",
    "canonical_pass_ids_twelve",
    "canonical_pass_ids_fourteen_mu",
    "canonical_pass_ids_sixteen_mu",
    "canonical_pass_ids_legacy_mu_norm",
    "trajectory_pass_ids_extended",
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


def canonical_pass_ids_fourteen_mu() -> tuple[str, ...]:
    """Four splits × (unguided + prob or two TR legs). RA/WF norm is merge-synthesized."""
    bases = ("forget", "retain", "ra", "wf")
    out: list[str] = []
    for b in bases:
        out.append(f"{b}__unguided")
        if b in ("forget", "retain"):
            out.append(f"{b}__guided_prob")
            out.extend((f"{b}__guided_tr_para", f"{b}__guided_tr_pert"))
        else:
            out.extend((f"{b}__guided_tr_correct", f"{b}__guided_tr_pert"))
    return tuple(out)


def canonical_pass_ids_legacy_mu_norm() -> tuple[str, ...]:
    """Optional GPU passes superseded by merge-time ``probability_w_options`` synthesis."""
    return ("ra__guided_prob", "wf__guided_prob")


def canonical_pass_ids_sixteen_mu() -> tuple[str, ...]:
    """Deprecated name: returns :func:`canonical_pass_ids_fourteen_mu` (was 16 passes)."""
    return canonical_pass_ids_fourteen_mu()


def canonical_pass_ids_eight() -> tuple[str, ...]:
    """Deprecated alias: use :func:`canonical_pass_ids_fourteen_mu`."""
    return canonical_pass_ids_fourteen_mu()


def canonical_pass_ids_twelve() -> tuple[str, ...]:
    """Deprecated alias: use :func:`canonical_pass_ids_fourteen_mu`."""
    return canonical_pass_ids_fourteen_mu()


def trajectory_pass_ids_extended() -> tuple[str, ...]:
    """Trajectory pass IDs implemented outside the 4×3 skew grid (e.g. SFT-parity retain)."""
    return ("retain_sft__unguided",)


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


# OU-aligned fourteen-pass MU compute bundle (+ optional legacy / parity ids).
_PASS_SPECS: dict[str, PassSpec] = {
    "forget__unguided": _spec(
        "forget__unguided",
        ("forget",),
        ("rouge",),
        "unguided",
        ("trajectory_forget_Q_A_ROUGE",),
    ),
    "forget__guided_prob": _spec(
        "forget__guided_prob",
        ("forget",),
        ("probability", "extraction_strength", "golden_token_prob_heatmap"),
        "guided_native",
        (
            "trajectory_forget_Q_A_Prob",
            "trajectory_extraction_strength",
            "trajectory_forget_golden_token_prob_heatmap",
        ),
    ),
    "forget__guided_tr_para": _spec(
        "forget__guided_tr_para",
        ("forget",),
        ("probability",),
        "guided_native",
        ("trajectory_forget_Q_A_PARA_Prob",),
    ),
    "forget__guided_tr_pert": _spec(
        "forget__guided_tr_pert",
        ("forget",),
        ("probability",),
        "guided_native",
        ("trajectory_forget_Q_A_PERT_Prob",),
    ),
    "retain__unguided": _spec(
        "retain__unguided",
        ("retain",),
        ("rouge",),
        "unguided",
        ("trajectory_retain_Q_A_ROUGE",),
    ),
    "retain_sft__unguided": _spec(
        "retain_sft__unguided",
        ("retain",),
        ("rouge",),
        "unguided",
        ("trajectory_retain_sft_Q_A_ROUGE",),
    ),
    "retain__guided_prob": _spec(
        "retain__guided_prob",
        ("retain",),
        ("probability",),
        "guided_native",
        ("trajectory_retain_Q_A_Prob",),
    ),
    "retain__guided_tr_para": _spec(
        "retain__guided_tr_para",
        ("retain",),
        ("probability",),
        "guided_native",
        ("trajectory_retain_Q_A_PARA_Prob",),
    ),
    "retain__guided_tr_pert": _spec(
        "retain__guided_tr_pert",
        ("retain",),
        ("probability",),
        "guided_native",
        ("trajectory_retain_Q_A_PERT_Prob",),
    ),
    "ra__unguided": _spec(
        "ra__unguided",
        ("ra",),
        ("rouge",),
        "unguided",
        ("trajectory_ra_Q_A_ROUGE",),
    ),
    "ra__guided_prob": _spec(
        "ra__guided_prob",
        ("ra",),
        ("probability",),
        "guided_native",
        ("trajectory_ra_Q_A_Prob_normalised",),
    ),
    "ra__guided_tr_correct": _spec(
        "ra__guided_tr_correct",
        ("ra",),
        ("probability",),
        "guided_native",
        ("trajectory_ra_Q_A_Prob",),
    ),
    "ra__guided_tr_pert": _spec(
        "ra__guided_tr_pert",
        ("ra",),
        ("probability",),
        "guided_native",
        ("trajectory_ra_Q_A_PERT_Prob",),
    ),
    "wf__unguided": _spec(
        "wf__unguided",
        ("wf",),
        ("rouge",),
        "unguided",
        ("trajectory_wf_Q_A_ROUGE",),
    ),
    "wf__guided_prob": _spec(
        "wf__guided_prob",
        ("wf",),
        ("probability",),
        "guided_native",
        ("trajectory_wf_Q_A_Prob_normalised",),
    ),
    "wf__guided_tr_correct": _spec(
        "wf__guided_tr_correct",
        ("wf",),
        ("probability",),
        "guided_native",
        ("trajectory_wf_Q_A_Prob",),
    ),
    "wf__guided_tr_pert": _spec(
        "wf__guided_tr_pert",
        ("wf",),
        ("probability",),
        "guided_native",
        ("trajectory_wf_Q_A_PERT_Prob",),
    ),
}

# Legacy pass ids (map to nearest OU-aligned spec for Hydra / skew experiments).
_fg = _PASS_SPECS["forget__guided_prob"]
_PASS_SPECS["forget__guided_native"] = PassSpec(
    pass_id="forget__guided_native",
    dataset_access_keys=_fg.dataset_access_keys,
    internal_metric_keys=_fg.internal_metric_keys | frozenset({"truth_ratio"}),
    evaluation_mode="guided_native",
    display_names_emitted=_fg.display_names_emitted
    + ("trajectory_forget_Truth_Ratio",),
)
_PASS_SPECS["forget__guided_skew"] = PassSpec(
    pass_id="forget__guided_skew",
    dataset_access_keys=_fg.dataset_access_keys,
    internal_metric_keys=_fg.internal_metric_keys,
    evaluation_mode="guided_skew",
    display_names_emitted=_fg.display_names_emitted,
)
_rp = _PASS_SPECS["retain__guided_prob"]
_PASS_SPECS["retain__guided_native"] = PassSpec(
    pass_id="retain__guided_native",
    dataset_access_keys=_rp.dataset_access_keys,
    internal_metric_keys=_rp.internal_metric_keys
    | frozenset({"extraction_strength", "truth_ratio"}),
    evaluation_mode="guided_native",
    display_names_emitted=(
        "trajectory_retain_Q_A_Prob",
        "trajectory_retain_extraction_strength",
        "trajectory_retain_Truth_Ratio",
    ),
)
_PASS_SPECS["retain__guided_skew"] = PassSpec(
    pass_id="retain__guided_skew",
    dataset_access_keys=_rp.dataset_access_keys,
    internal_metric_keys=_PASS_SPECS["retain__guided_native"].internal_metric_keys,
    evaluation_mode="guided_skew",
    display_names_emitted=_PASS_SPECS["retain__guided_native"].display_names_emitted,
)
_rap = _PASS_SPECS["ra__guided_prob"]
_PASS_SPECS["ra__guided_native"] = PassSpec(
    pass_id="ra__guided_native",
    dataset_access_keys=_rap.dataset_access_keys,
    internal_metric_keys=_rap.internal_metric_keys
    | frozenset({"extraction_strength", "truth_ratio"}),
    evaluation_mode="guided_native",
    display_names_emitted=(
        "trajectory_ra_Q_A_Prob_normalised",
        "trajectory_ra_extraction_strength",
        "trajectory_ra_Truth_Ratio",
    ),
)
_PASS_SPECS["ra__guided_skew"] = PassSpec(
    pass_id="ra__guided_skew",
    dataset_access_keys=_rap.dataset_access_keys,
    internal_metric_keys=_PASS_SPECS["ra__guided_native"].internal_metric_keys,
    evaluation_mode="guided_skew",
    display_names_emitted=_PASS_SPECS["ra__guided_native"].display_names_emitted,
)
_wfp = _PASS_SPECS["wf__guided_prob"]
_PASS_SPECS["wf__guided_native"] = PassSpec(
    pass_id="wf__guided_native",
    dataset_access_keys=_wfp.dataset_access_keys,
    internal_metric_keys=_wfp.internal_metric_keys
    | frozenset({"extraction_strength", "truth_ratio"}),
    evaluation_mode="guided_native",
    display_names_emitted=(
        "trajectory_wf_Q_A_Prob_normalised",
        "trajectory_wf_extraction_strength",
        "trajectory_wf_Truth_Ratio",
    ),
)
_PASS_SPECS["wf__guided_skew"] = PassSpec(
    pass_id="wf__guided_skew",
    dataset_access_keys=_wfp.dataset_access_keys,
    internal_metric_keys=_PASS_SPECS["wf__guided_native"].internal_metric_keys,
    evaluation_mode="guided_skew",
    display_names_emitted=_PASS_SPECS["wf__guided_native"].display_names_emitted,
)


def get_pass_spec(pass_id: str) -> PassSpec:
    if pass_id not in _PASS_SPECS:
        raise KeyError(
            f"Unknown trajectory_pass_id={pass_id!r}. "
            f"Known: {sorted(_PASS_SPECS.keys())}"
        )
    if pass_id in DEPRECATED_PASS_IDS:
        msg = (
            f"trajectory_pass_id={pass_id!r} is deprecated (dual-label / QAwithDualAnswers). "
            "Use OU-aligned split passes: *_guided_tr_para|pert or *_guided_tr_correct|pert."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        logger.warning(msg)
    return _PASS_SPECS[pass_id]


# trajectory_all display name → (OU regime label, primary dataset access for that leg)
DISPLAY_METRIC_BINDING: dict[str, tuple[str, str]] = {
    "trajectory_forget_Q_A_Prob": ("guided", "forget"),
    "trajectory_forget_Q_A_ROUGE": ("guidance_free", "forget"),
    "trajectory_extraction_strength": ("guided", "forget"),
    "trajectory_forget_Truth_Ratio": ("guided", "forget"),
    "trajectory_forget_golden_token_prob_heatmap": ("guided", "forget"),
    "trajectory_forget_quality": ("guided", "forget"),
    "trajectory_model_utility": ("mixed", "retain_ra_wf"),
    "trajectory_privleak": ("guided", "forget"),
    "trajectory_retain_sft_Q_A_ROUGE": ("guidance_free", "retain"),
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
