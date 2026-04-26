"""Effective parity snapshot + stable hash for evaluation JSON (FR-012, contract eval-parity-artifact)."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _stable_hash(obj: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def attach_effective_parity_to_cache(
    cache: dict[str, Any],
    eval_cfg: Any,
    model: Any,
) -> None:
    """Attach ``effective_parity`` block when ``eval.decoupling`` is present (idempotent)."""
    if cache.get("effective_parity") is not None:
        return
    if eval_cfg is None or not callable(getattr(eval_cfg, "get", None)):
        return
    dec = eval_cfg.get("decoupling")
    if dec is None:
        return
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(dec):
            d: dict[str, Any] = OmegaConf.to_container(dec, resolve=True)
        else:
            d = dict(dec) if isinstance(dec, dict) else {}
    except Exception:
        return
    parity_relevant: dict[str, Any] = {}
    parity_relevant["benchmark"] = d.get("benchmark")
    parity_relevant["split"] = d.get("split")
    parity_relevant["trajectory_mode"] = d.get("trajectory_mode")
    parity_relevant["feature_profile_hash"] = d.get("feature_profile_hash")
    em_req = d.get("evaluation_mode_request")
    if em_req:
        parity_relevant["evaluation_mode_request"] = em_req
    app = d.get("applicability_statuses")
    if isinstance(app, str):
        parity_relevant["applicability_statuses"] = app
    dm = getattr(model, "adapter_config", None)
    if dm is not None:
        parity_relevant["evaluation_mode"] = getattr(dm, "evaluation_mode", "unguided")
        parity_relevant["tokens_per_step"] = getattr(dm, "tokens_per_step", None)
        parity_relevant["max_new_tokens"] = getattr(dm, "max_new_tokens", None)
        parity_relevant["trajectory_sample_interval"] = getattr(
            dm, "trajectory_sample_interval", None
        )
    h = _stable_hash(parity_relevant)
    cache["effective_parity"] = {
        "schema_version": 1,
        "effective_parity_snapshot": {"parity_relevant": parity_relevant},
        "effective_parity_hash": h,
    }
