"""
DEBUG-only trajectory audit helpers (trajectory_audit_verbose in trajectory_config).

All public entry points no-op unless ``logger.isEnabledFor(logging.DEBUG)`` and the
caller passes ``trajectory_audit_runtime=True`` (after dataset-size guard).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("evaluator")


def trajectory_config_as_dict(trajectory_config: Any) -> Dict[str, Any]:
    if trajectory_config is None:
        return {}
    if OmegaConf.is_config(trajectory_config):
        c = OmegaConf.to_container(trajectory_config, resolve=True) or {}
        return dict(c) if isinstance(c, dict) else {}
    if isinstance(trajectory_config, dict):
        return dict(trajectory_config)
    return {}


def forget_trajectory_audit_runtime(
    trajectory_config: Any, n_samples: int
) -> Tuple[bool, Optional[str], str]:
    """
    Returns (active, jsonl_path_or_none, jsonl_sample_id_str).

    When verbose is on but n_samples exceeds trajectory_audit_max_dataset_samples (default 32)
    and trajectory_audit_allow_large is false, logs ERROR and returns inactive.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return False, None, "0"
    tc = trajectory_config_as_dict(trajectory_config)
    if not tc.get("trajectory_audit_verbose"):
        return False, None, "0"
    limit = int(tc.get("trajectory_audit_max_dataset_samples", 32))
    allow = bool(tc.get("trajectory_audit_allow_large", False))
    if n_samples > limit and not allow:
        logger.error(
            "trajectory_audit_verbose disabled for this run: forget dataset n_samples=%s > "
            "trajectory_audit_max_dataset_samples=%s. Set trajectory_audit_allow_large=true "
            "or raise the limit.",
            n_samples,
            limit,
        )
        return False, None, "0"
    jpath = tc.get("trajectory_audit_jsonl_path")
    jpath_s = str(jpath).strip() if jpath else None
    if jpath_s == "":
        jpath_s = None
    sid = str(tc.get("trajectory_audit_jsonl_sample_id", "0"))
    return True, jpath_s, sid


def mu_trajectory_audit_runtime(trajectory_config: Any, n_retain_samples: int) -> bool:
    """Same guard as forget, but for retain MU dataset length."""
    if not logger.isEnabledFor(logging.DEBUG):
        return False
    tc = trajectory_config_as_dict(trajectory_config)
    if not tc.get("trajectory_audit_verbose"):
        return False
    limit = int(tc.get("trajectory_audit_max_dataset_samples", 32))
    allow = bool(tc.get("trajectory_audit_allow_large", False))
    if n_retain_samples > limit and not allow:
        logger.error(
            "trajectory_audit_verbose MU lines disabled: retain dataset len=%s > limit=%s",
            n_retain_samples,
            limit,
        )
        return False
    return True


def _max_decode_chars(tc: Dict[str, Any]) -> int:
    v = tc.get("trajectory_audit_max_decode_chars", 8000)
    try:
        return max(100, int(v))
    except (TypeError, ValueError):
        return 8000


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "..."


def _scalar_summary(result: Any) -> Any:
    if result is None:
        return None
    if isinstance(result, (int, float, np.floating, np.integer)):
        return float(result)
    if isinstance(result, dict):
        if "agg_value" in result and result["agg_value"] is not None:
            return {"agg_value": result["agg_value"]}
        if "prob" in result:
            return {"prob": result["prob"]}
        if "score" in result:
            return {"score": result["score"]}
        if "value_by_index" in result and isinstance(result["value_by_index"], dict):
            vbi = result["value_by_index"]
            if not vbi:
                return {"value_by_index": {}}
            first = next(iter(vbi.values()))
            return {"value_by_index_preview": first}
        return {k: _scalar_summary(v) for k, v in list(result.items())[:12]}
    if isinstance(result, list) and result:
        if isinstance(result[0], dict):
            return {"list0": _scalar_summary(result[0])}
        return {"list_len": len(result), "list0": result[0]}
    return str(type(result).__name__)


def _prob_from_pre_entry(ent: Any, idx_key: str) -> Optional[float]:
    if not isinstance(ent, dict):
        return None
    vbi = ent.get("value_by_index") or {}
    row = vbi.get(idx_key)
    if row is None and vbi:
        row = next(iter(vbi.values()))
    if isinstance(row, dict):
        p = row.get("prob")
        if p is not None:
            try:
                return float(p)
            except (TypeError, ValueError):
                return None
    return None


def log_truth_ratio_pre_compute(
    *,
    pre_compute_results: Dict[str, Any],
    sample_idx: Optional[str],
    step: Any,
    traj_name: Any,
    view: Any,
    batch_idx: Any,
    aggregator: Any = None,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    if not pre_compute_results:
        return
    idx_key = str(sample_idx if sample_idx is not None else "0")
    pc = _prob_from_pre_entry(pre_compute_results.get("correct"), idx_key)
    pw_ent = pre_compute_results.get("wrong")
    pw_list: List[Optional[float]] = []
    if isinstance(pw_ent, list):
        for item in pw_ent:
            pw_list.append(_prob_from_pre_entry(item, idx_key))
    else:
        pw_list.append(_prob_from_pre_entry(pw_ent, idx_key))
    eps = 1e-12
    ratio = None
    if pc is not None and pw_list and pw_list[0] is not None:
        ratio = float(pc / (pw_list[0] + eps))
    logger.debug(
        "TRAJECTORY_AUDIT_TRUTH_PRE sample_idx=%s batch_idx=%s traj=%s view=%s step=%s "
        "p_correct=%s p_wrong_opts=%s ratio_pc_div_pw=%s aggregator=%s",
        idx_key,
        batch_idx,
        traj_name,
        view,
        step,
        pc,
        pw_list,
        ratio,
        aggregator,
    )


def log_hm_pre_compute_snapshot(
    *,
    pre_compute: Dict[str, Any],
    sample_idx: Optional[str],
    step: Any,
    traj_name: Any,
    view: Any,
    batch_idx: Any,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG) or not pre_compute:
        return
    snap = {}
    for k, v in pre_compute.items():
        if isinstance(v, dict) and "agg_value" in v:
            snap[k] = v.get("agg_value")
        elif isinstance(v, dict):
            snap[k] = "<dict>"
        else:
            snap[k] = str(type(v).__name__)
    logger.debug(
        "TRAJECTORY_AUDIT_MU_HM_PRE sample_idx=%s batch_idx=%s traj=%s view=%s step=%s agg_values=%s",
        sample_idx,
        batch_idx,
        traj_name,
        view,
        step,
        snap,
    )


def log_retain_reference_resolution(
    *,
    metric_name: str,
    step_val: Any,
    step_index: Any,
    step_str_by_val: Optional[str],
    step_str_by_idx: Optional[str],
    by_step_keys_sample: List[str],
    resolved: bool,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "TRAJECTORY_AUDIT_RETAIN_REF metric=%s step_val=%s step_index=%s "
        "tried=%s/%s by_step_key_count=%s by_step_keys_sample=%s resolved=%s",
        metric_name,
        step_val,
        step_index,
        step_str_by_val,
        step_str_by_idx,
        len(by_step_keys_sample),
        by_step_keys_sample[:24],
        resolved,
    )


def _logits_numeric_record(logits: torch.Tensor) -> Dict[str, Any]:
    """Small CPU summary for JSONL (audit-only callers)."""
    with torch.no_grad():
        x = logits.detach().float()
        if x.dim() == 3:
            x = x[0]
        if x.dim() != 2:
            return {"shape": list(logits.shape)}
        argmax = x.argmax(dim=-1)
        norms = x.norm(dim=-1)
        L = int(x.shape[0])
        rec: Dict[str, Any] = {
            "shape": list(logits.shape),
            "mean_logit_norm": float(norms.mean().cpu()),
            "L": L,
            "first8_argmax": argmax[: min(8, L)].cpu().tolist(),
        }
        if L > 8:
            rec["last4_argmax"] = argmax[-4:].cpu().tolist()
        return rec


def maybe_write_audit_jsonl(
    path: Optional[str],
    jsonl_sample_id: str,
    sample_idx: Optional[str],
    record: Dict[str, Any],
) -> None:
    if not path or not logger.isEnabledFor(logging.DEBUG):
        return
    if str(sample_idx) != str(jsonl_sample_id):
        return
    line = json.dumps(record, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def log_metric_audit(
    *,
    handler_metric_name: str,
    result: Any,
    logits: torch.Tensor,
    tokenizer: Any,
    sample_input_ids: Any,
    sample_prompt_len: Union[int, torch.Tensor],
    sample_idx: Optional[str],
    kwargs: Dict[str, Any],
    pre_compute_results: Optional[Dict[str, Any]] = None,
    hm_pre_compute: Optional[Dict[str, Any]] = None,
    es_branch: Optional[str] = None,
    jsonl_path: Optional[str] = None,
    jsonl_sample_id: str = "0",
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    if not kwargs.get("trajectory_audit_runtime"):
        return
    tc = trajectory_config_as_dict(kwargs.get("trajectory_config"))
    max_ch = _max_decode_chars(tc)
    traj_name = kwargs.get("traj_name")
    view = kwargs.get("trajectory_audit_view")
    step = kwargs.get("step")
    step_index = kwargs.get("step_index")
    batch_idx = kwargs.get("trajectory_audit_batch_idx")
    idx_key = str(sample_idx if sample_idx is not None else "0")

    prompt_txt = ""
    gen_txt = ""
    if tokenizer is not None and sample_input_ids is not None:
        try:
            pl = int(sample_prompt_len.item()) if hasattr(sample_prompt_len, "item") else int(sample_prompt_len)
            row = sample_input_ids[0] if sample_input_ids.dim() > 1 else sample_input_ids
            ids = row[:pl].tolist() if hasattr(row, "tolist") else []
            prompt_txt = tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            prompt_txt = "(prompt_decode_failed)"
        try:
            lo = logits
            if lo.dim() == 3:
                lo = lo[0]
            pred = torch.argmax(lo, dim=-1)
            gen_txt = tokenizer.decode(pred.tolist(), skip_special_tokens=True)
        except Exception:
            gen_txt = "(gen_decode_failed)"

    summ = _scalar_summary(result)
    logger.debug(
        "TRAJECTORY_AUDIT metric=%s sample_idx=%s batch_idx=%s traj=%s view=%s step=%s step_index=%s "
        "logits_shape=%s es_branch=%s result=%s prompt_len_chars=%s prompt=%r gen_argmax=%r",
        handler_metric_name,
        idx_key,
        batch_idx,
        traj_name,
        view,
        step,
        step_index,
        tuple(logits.shape),
        es_branch,
        summ,
        len(prompt_txt),
        _truncate(prompt_txt, max_ch),
        _truncate(gen_txt, max_ch),
    )

    if handler_metric_name == "truth_ratio" and pre_compute_results:
        log_truth_ratio_pre_compute(
            pre_compute_results=pre_compute_results,
            sample_idx=sample_idx,
            step=step,
            traj_name=traj_name,
            view=view,
            batch_idx=batch_idx,
            aggregator=kwargs.get("aggregator"),
        )

    if handler_metric_name == "hm_aggregate" and hm_pre_compute:
        log_hm_pre_compute_snapshot(
            pre_compute=hm_pre_compute,
            sample_idx=sample_idx,
            step=step,
            traj_name=traj_name,
            view=view,
            batch_idx=batch_idx,
        )

    if jsonl_path:
        rec = {
            "kind": "trajectory_audit_metric",
            "metric": handler_metric_name,
            "sample_idx": idx_key,
            "batch_idx": batch_idx,
            "traj_name": traj_name,
            "view": view,
            "step": step,
            "step_index": step_index,
            "result_summary": summ,
            **_logits_numeric_record(logits),
        }
        if es_branch:
            rec["es_branch"] = es_branch
        maybe_write_audit_jsonl(jsonl_path, jsonl_sample_id, sample_idx, rec)


def log_mu_components_snapshot(
    *,
    retain_agg_by_step: Dict[str, Any],
    trajectory_audit_mu: bool,
) -> None:
    if not trajectory_audit_mu or not logger.isEnabledFor(logging.DEBUG):
        return
    if not retain_agg_by_step:
        return
    _rk = next(iter(retain_agg_by_step.keys()), None)
    _per_traj = _rk in ("steps", "fixation_start", "fixation_end", "fixation_ratio")
    steps_dict = retain_agg_by_step.get("steps", retain_agg_by_step) if _per_traj else retain_agg_by_step
    if not steps_dict:
        return

    def _step_sort(k):
        s = str(k)
        return int(s) if s.isdigit() else 0

    sk0 = min(steps_dict.keys(), key=_step_sort)
    ent0 = steps_dict.get(sk0)
    if not isinstance(ent0, dict):
        return
    for view in ("full", "eos"):
        inner = ent0.get(view)
        if not isinstance(inner, dict):
            continue
        snap = {
            k: (v.get("agg_value") if isinstance(v, dict) else None)
            for k, v in inner.items()
            if isinstance(v, dict)
        }
        logger.debug(
            "TRAJECTORY_AUDIT_MU_STEP step_key=%s view=%s component_agg_values=%s",
            sk0,
            view,
            snap,
        )


def log_pre_compute_probability(
    *,
    access_key: str,
    handler_name: Optional[str],
    sample_idx: str,
    step: Any,
    labels_field: Any,
    agg_prob: Any,
    trajectory_audit_runtime: bool,
) -> None:
    if not trajectory_audit_runtime or not logger.isEnabledFor(logging.DEBUG):
        return
    if handler_name != "probability":
        return
    logger.debug(
        "TRAJECTORY_AUDIT_PRE_COMPUTE_PROB access_key=%s sample_idx=%s step=%s labels_field=%s agg_prob=%s",
        access_key,
        sample_idx,
        step,
        labels_field,
        agg_prob,
    )
