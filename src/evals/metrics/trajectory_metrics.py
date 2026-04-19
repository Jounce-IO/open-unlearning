"""
Trajectory-based metrics for dLLM unlearning evaluation.

This module computes metrics at each diffusion step across three trajectory types
(steps, fixation, ratio), supporting any metric from the open-unlearning framework.

Trajectory evals use interval mode only (trajectory_sample_interval, default 8).
Every-step mode (no interval) is not used.
"""

import copy
import gc
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import DictConfig, ListConfig, OmegaConf

from evals.metrics.base import unlearning_metric
from evals.metrics.samplers import LengthSortedSampler
from evals.metrics.utils import (
    evaluate_probability,
    evaluate_probability_confidence_ordered,
    eval_rouge_recall_batch,
    eval_rouge_recall_batch_worker,
    tokenwise_vocab_logprobs,
    IGNORE_INDEX,
)
from rouge_score import rouge_scorer
from evals.metrics.step_wise_score import (
    FixationStepWiseScoreProvider,
    build_effective_step_fixation_logits,
    build_effective_step_fixation_logits_from_history,
    build_fixation_logits_from_R_F,
    compute_prob_from_fixation_logits as _compute_prob_from_fixation_logits,
    compute_prob_packed_shifted_segments as _compute_prob_packed_shifted_segments,
    sequence_probability_from_scores,
    extraction_strength_from_fixation,
    trajectory_step_logits_to_prob_batch,
)
from evals.metrics.trajectory_utils import (
    trajectories_from_logits,
    effective_lengths_from_eos,
    compute_fixation_start_trajectory,
    compute_fixation_end_trajectory,
    compute_fixation_ratio_trajectory,
    compute_fixation_start_from_history,
    compute_fixation_end_from_history,
    compute_fixation_ratio_from_history,
)
from evals.metrics.trajectory_adapters import (
    LogitModelWrapper,
    DualLogitModelWrapper,
)
from evals.metrics.mia.utils import get_attacker, MIAStreamingAccumulator
from evals.gpu_phase_logger import set_phase as gpu_set_phase
from evals.guardrails import (
    load_icul_pools,
    transform_output_text,
    transform_prompts,
)

logger = logging.getLogger("evaluator")

# One-time warnings for trajectory fallback to single retain reference (see docs/evaluation-notes.md)
_trajectory_single_retain_warned: set = set()

# Trajectory evals use interval mode only; every-step mode is not used.
DEFAULT_TRAJECTORY_SAMPLE_INTERVAL = 8
# Packed shifted-CE (post-loop probability / truth_ratio legs): cap segments per GPU chunk when unset.
DEFAULT_TRAJECTORY_STEP_VIEW_BATCH_CHUNK_MAX = 32
# Max reporting steps prefetched at once for ``steps`` trajectory (limits ``index_select`` width on ``R``).
DEFAULT_TRAJECTORY_STEP_PREFETCH_MAX_STEPS = 16
DEFAULT_TRAJECTORY_LOGITS_STORAGE = "dense_r"


def _trajectory_should_empty_cuda_cache(trajectory_config: Optional[Any]) -> bool:
    """When False, skip ``torch.cuda.empty_cache()`` in hot paths (set ``aggressive_cuda_empty_cache: false``)."""
    if not torch.cuda.is_available():
        return False
    if trajectory_config is None:
        return True
    tc = (
        OmegaConf.to_container(trajectory_config, resolve=True)
        if OmegaConf.is_config(trajectory_config)
        else trajectory_config
    )
    if not isinstance(tc, dict):
        return True
    return bool(tc.get("aggressive_cuda_empty_cache", True))


def _trajectory_metric_display_name(metric_name: str, loaded_metrics: Dict[str, Any]) -> str:
    """Prefer metric config ``display_name`` when set (coalesced / Hydra); else logical key."""
    info = loaded_metrics.get(metric_name) if isinstance(loaded_metrics, dict) else None
    cfg = (info or {}).get("config") if isinstance(info, dict) else None
    if isinstance(cfg, dict):
        dn = cfg.get("display_name")
        if dn is not None and str(dn).strip():
            return str(dn)
    return metric_name


def _trajectory_submetric_generalized_applied(
    metric_name: str, trajectory_config: Dict[str, Any]
) -> str:
    """Contract metric-logging: generalized_applied for trajectory sub-metrics."""
    u = bool(trajectory_config.get("use_generalized_sequence_probability", True))
    if metric_name in (
        "probability",
        "truth_ratio",
        "probability_confidence_ordered",
        "extraction_strength",
        "privleak",
        "mia_loss",
        "mia_zlib",
        "mia_reference",
    ):
        return str(u)
    if metric_name in (
        "mia_min_k",
        "mia_min_k_plus_plus",
        "mia_gradnorm",
    ):
        return "not_applicable"
    return "not_applicable"


def _debug_log_trajectory_metric_coverage(
    agg_value_by_view: Dict[str, Any],
    loaded_metrics: Dict[str, Any],
    trajectory_names: List[str],
    include_views: List[str],
    trajectory_config: Dict[str, Any],
) -> None:
    """Emit TRAJECTORY_METRIC_COVERAGE lines (grep/parse to verify all metrics ran)."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    metric_names = sorted(loaded_metrics.keys())
    for view in include_views:
        for traj_name in trajectory_names:
            lengths: Dict[str, int] = {}
            finite: Dict[str, int] = {}
            for metric_name in metric_names:
                _handler = (
                    (loaded_metrics.get(metric_name) or {}).get("metric").name
                    if isinstance(loaded_metrics.get(metric_name), dict)
                    and (loaded_metrics.get(metric_name) or {}).get("metric") is not None
                    else metric_name
                )
                _gen = _trajectory_submetric_generalized_applied(
                    metric_name, trajectory_config
                )
                block = (agg_value_by_view.get(view) or {}).get(traj_name) or {}
                arr = block.get(metric_name)
                if arr is None:
                    logger.debug(
                        "TRAJECTORY_METRIC_COVERAGE view=%s traj=%s metric=%s array_len=0 finite_values=0 missing=1",
                        view,
                        traj_name,
                        metric_name,
                    )
                    _disp = _trajectory_metric_display_name(metric_name, loaded_metrics)
                    logger.debug(
                        "TRAJECTORY_SUBMETRIC_GENERALIZED view=%s traj=%s metric=%s display_name=%s handler=%s generalized_applied=%s",
                        view,
                        traj_name,
                        metric_name,
                        _disp,
                        _handler,
                        _gen,
                    )
                    continue
                a = np.asarray(arr, dtype=np.float64)
                n = int(a.size)
                n_fin = int(np.sum(np.isfinite(a)))
                lengths[metric_name] = n
                finite[metric_name] = n_fin
                logger.debug(
                    "TRAJECTORY_METRIC_COVERAGE view=%s traj=%s metric=%s array_len=%s finite_values=%s",
                    view,
                    traj_name,
                    metric_name,
                    n,
                    n_fin,
                )
                _disp = _trajectory_metric_display_name(metric_name, loaded_metrics)
                logger.debug(
                    "TRAJECTORY_SUBMETRIC_GENERALIZED view=%s traj=%s metric=%s display_name=%s handler=%s generalized_applied=%s",
                    view,
                    traj_name,
                    metric_name,
                    _disp,
                    _handler,
                    _gen,
                )
            pos_lens = {m: lengths[m] for m in lengths if lengths[m] > 0}
            if len(set(pos_lens.values())) > 1:
                logger.warning(
                    "TRAJECTORY_METRIC_LEN_MISMATCH view=%s traj=%s array_len_by_metric=%s",
                    view,
                    traj_name,
                    pos_lens,
                )
            if pos_lens:
                ref = max(pos_lens.values())
                thin = [m for m in metric_names if lengths.get(m, 0) not in (0, ref)]
                if thin:
                    logger.debug(
                        "TRAJECTORY_METRIC_SHORT_OR_EMPTY view=%s traj=%s expected_len=%s metrics_not_len=%s",
                        view,
                        traj_name,
                        ref,
                        thin,
                    )


def _debug_log_mu_submetric_coverage(retain_agg_by_step: Dict[str, Any]) -> None:
    """Emit TRAJECTORY_MU_SUBMETRIC_* for hm_aggregate pre_compute components per step."""
    if not logger.isEnabledFor(logging.DEBUG) or not retain_agg_by_step:
        return

    def _step_sort(k):
        s = str(k)
        return int(s) if s.isdigit() else 0

    _first = next(iter(retain_agg_by_step.keys()), None)
    _per_traj = _first in ("steps", "fixation_start", "fixation_end", "fixation_ratio")
    by_step = retain_agg_by_step.get("steps", retain_agg_by_step) if _per_traj else retain_agg_by_step
    steps = sorted(by_step.keys(), key=_step_sort)
    logger.debug(
        "TRAJECTORY_MU_SUBMETRIC_STEPS mu_aggregate_steps=%s first_step=%s last_step=%s",
        len(steps),
        steps[0] if steps else None,
        steps[-1] if steps else None,
    )
    for sk in (steps[0], steps[-1]) if steps else []:
        pre = by_step[sk]
        if not isinstance(pre, dict):
            continue
        for view in ("full", "eos"):
            inner = pre.get(view)
            if not isinstance(inner, dict):
                continue
            keys = sorted(
                k
                for k, v in inner.items()
                if isinstance(v, dict) and "agg_value" in v
            )
            logger.debug(
                "TRAJECTORY_MU_SUBMETRIC_COVERAGE step=%s view=%s submetric_count=%s submetrics=%s",
                sk,
                view,
                len(keys),
                keys,
            )


EVALUATION_MODES = ("unguided", "guided_native", "guided_skew")


def _trajectory_sampler_kwargs(trajectory_config: Union[Dict, DictConfig]) -> dict:
    """Return sampler_kwargs with trajectory_sample_interval defaulting to 8 when return_logits is used.

    Also passes evaluation_mode from trajectory_config (default "unguided"); allowed values:
    unguided, guided_native, guided_skew.
    We set right_shift_logits=False so all dLLM backends (Dream, LLaDA) return same-position
    logits; trajectory metrics then apply one AR shift before probability computation.
    """
    kwargs = dict(trajectory_config.get("sampler_kwargs", {}) or {})
    if trajectory_config.get("return_logits") and (
        kwargs.get("trajectory_sample_interval") is None
        or kwargs.get("trajectory_sample_interval", 0) < 1
    ):
        kwargs["trajectory_sample_interval"] = DEFAULT_TRAJECTORY_SAMPLE_INTERVAL
    mode = trajectory_config.get("evaluation_mode", "unguided")
    if mode not in EVALUATION_MODES:
        mode = "unguided"
    kwargs["evaluation_mode"] = mode
    kwargs["right_shift_logits"] = False
    return kwargs


def _generation_start(
    sample_idx: int,
    prompt_starts: list[int],
    prompt_lens: list[int],
    prompt_only_input_ids: bool,
) -> int:
    """Index in labels (and aligned input_ids) where the generated region starts.

    Full-convo (prompt_only_input_ids=True, e.g. TOFU): labels = [pad, prompt+response];
    generation start = content_start + prompt_len = prompt_starts[i] + prompt_lens[i].
    IGNORE-for-prompt (prompt_only_input_ids=False): labels = [IGNORE]*prompt_len + [response];
    generation start = first non-IGNORE = prompt_starts[i].
    """
    ps = prompt_starts[sample_idx]
    pl = prompt_lens[sample_idx]
    if isinstance(ps, torch.Tensor):
        ps = int(ps.item())
    if isinstance(pl, torch.Tensor):
        pl = int(pl.item())
    return (ps + pl) if prompt_only_input_ids else ps


def _build_prompts_for_sampler(
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor],
    tokenizer: Any,
    ignore_index: int = IGNORE_INDEX,
    prompt_only_input_ids: bool = False,
) -> tuple[list[list[int]], list[int], list[int]]:
    """Build prompt token lists and lengths for sampler.sample(inputs=...).

    Returns (prompts, prompt_lens, prompt_starts).
    - prompts: content-only token lists (leading pad stripped when pad_token_id is set).
    - prompt_lens: prompt_lens[i] = len(prompts[i]) (length of prompt sent to sampler).
      Use for trajectories_from_logits and effective_lengths_from_eos.
    - prompt_starts: prompt_starts[i] = first index where labels[i] != ignore_index
      (start index of generation in labels). Use for _build_target_sequences_for_sampler
      and any labels/input_ids slicing.

    Two data conventions (controlled by prompt_only_input_ids, typically from
    getattr(dataset, "predict_with_generate", False)):
    - prompt_only_input_ids=True: input_ids per sample contain only the prompt (no response).
      Prompt is the non-pad tokens of input_ids[i]. prompt_starts still from labels.
    - prompt_only_input_ids=False (training-style): prompt_end from labels, prompt = slice
      input_ids[i, :prompt_end], strip leading pad; if empty after strip and row has
      non-pad tokens, use non-pad tokens as prompt.
    """
    B = input_ids.shape[0]
    prompts: list[list[int]] = []
    prompt_lens: list[int] = []
    prompt_starts: list[int] = []
    _pad = getattr(tokenizer, "pad_token_id", None) if tokenizer else None
    pad_token_id = _pad if isinstance(_pad, (int, float)) else None

    def _strip_leading_pad(tokens: list[int], pad_id: int) -> list[int]:
        while tokens and tokens[0] == pad_id:
            tokens = tokens[1:]
        return tokens

    for i in range(B):
        if labels is not None:
            label_mask = labels[i] != ignore_index
            if label_mask.any():
                prompt_end = label_mask.nonzero()[0][0].item()
            else:
                prompt_end = input_ids.shape[1]
        else:
            prompt_end = input_ids.shape[1]
        prompt_starts.append(prompt_end)

        if prompt_only_input_ids:
            if pad_token_id is not None:
                non_pad = (input_ids[i] != pad_token_id).view(-1)
                prompt = input_ids[i][non_pad].cpu().tolist()
            else:
                prompt = input_ids[i].cpu().tolist()
        else:
            prompt = input_ids[i, :prompt_end].cpu().tolist()
            if pad_token_id is not None:
                prompt = _strip_leading_pad(prompt, pad_token_id)
                non_pad_count = (input_ids[i] != pad_token_id).sum().item()
                if len(prompt) == 0 and non_pad_count > 0:
                    non_pad = (input_ids[i] != pad_token_id).view(-1)
                    prompt = input_ids[i][non_pad].cpu().tolist()

        prompts.append(prompt)
        prompt_lens.append(len(prompt))
    return prompts, prompt_lens, prompt_starts


def _build_target_sequences_for_sampler(
    labels: torch.Tensor,
    prompt_starts: List[int],
    L: int,
    ignore_index: int = IGNORE_INDEX,
) -> List[List[int]]:
    """Build target token lists for the generated region only (one list per batch sample, length L).

    For each sample j, takes labels[j, prompt_starts[j]:prompt_starts[j]+L]. prompt_starts[j]
    is the start index of the generation region in labels. If the slice is shorter than L,
    pads with ignore_index so the sampler receives exactly L tokens per sample.
    """
    B = labels.shape[0]
    target_sequences = []
    for j in range(B):
        start = prompt_starts[j]
        end = min(start + L, labels.shape[1])
        row = labels[j, start:end].cpu().tolist()
        if len(row) < L:
            row = row + [ignore_index] * (L - len(row))
        target_sequences.append(row)
    return target_sequences


def _slice_labels_by_content_start(
    row: torch.Tensor, L: int, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """Slice the first L tokens of the content (non-ignore region) from a labels row.

    With left padding, each row's content starts at a different index. Using the
    same start (e.g. from the correct item) for labels_wrong can make the slice
    fall entirely in padding when the wrong item is shorter, causing truth_ratio
    to get no valid positions. This uses this row's own content start.
    """
    if row.dim() > 1:
        row = row.squeeze(0)
    mask = row != ignore_index
    if not mask.any():
        return torch.full((L,), ignore_index, dtype=row.dtype, device=row.device)
    content_start = mask.nonzero(as_tuple=True)[0][0].item()
    end = min(content_start + L, row.shape[0])
    gen_alt = row[content_start:end].clone()
    if gen_alt.shape[0] < L:
        padding = torch.full(
            (L - gen_alt.shape[0],), ignore_index, dtype=gen_alt.dtype, device=gen_alt.device
        )
        gen_alt = torch.cat([gen_alt, padding])
    return gen_alt


def _batch_template_dual_labels(
    batch: dict,
    sample_idx: int,
    key: str,
    L: int,
    ignore_index: int = IGNORE_INDEX,
) -> Any:
    """Build batch_template[key] for labels_correct or labels_wrong. Handles N wrong options (3D [B,N,L])."""
    tensor = batch[key]
    if key == "labels_wrong" and tensor.dim() == 3:
        N = tensor.shape[1]
        return [
            _slice_labels_by_content_start(tensor[sample_idx][k], L, ignore_index).unsqueeze(0)
            for k in range(N)
        ]
    return _slice_labels_by_content_start(tensor[sample_idx], L, ignore_index).unsqueeze(0)


def _slice_batch_template_to_length(
    batch_template: Dict[str, Any], length: int
) -> Dict[str, Any]:
    """Slice batch_template sequence dimensions to length (probability invariant: logits and labels same length)."""
    out = {}
    for k, v in batch_template.items():
        if v is None:
            out[k] = v
            continue
        if isinstance(v, list):
            out[k] = [
                x[:, :length].clone() if isinstance(x, torch.Tensor) and x.dim() >= 2 else x
                for x in v
            ]
            continue
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            seq_dim = 1 if v.dim() >= 2 else 0
            if v.shape[seq_dim] > length:
                if v.dim() == 1:
                    out[k] = v[:length].clone()
                else:
                    out[k] = v[:, :length].clone()
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def _batch_template_has_list_tensors(batch_template: Dict[str, Any]) -> bool:
    return any(
        isinstance(v, list) and bool(v) and isinstance(v[0], torch.Tensor)
        for v in batch_template.values()
    )


def _trajectory_tc_allowlist_sets(
    trajectory_config: Optional[Any],
) -> tuple[
    Dict[str, Any],
    FrozenSet[str],
    FrozenSet[str],
    FrozenSet[str],
    Optional[int],
    int,
    str,
]:
    """Parse trajectory_config for step-batch allowlist, CPU-offload sets, T×views batching, and memory layout.

    ``trajectory_step_view_batch_allowlist`` (default empty): metric keys that may use one packed
    or stacked pass per (sample, traj_name, metric) over all ``steps_to_use`` × ``include_views``.
    Segment / row order is lexicographic ``(step, view)`` — for each step in ``steps_to_use`` order,
    then each view in ``include_views`` order.

    ``trajectory_step_view_batch_chunk_max`` (optional int): max number of packed shifted-CE
    segments per GPU chunk (for ``probability`` / ``truth_ratio`` legs). When the key is absent,
    defaults to ``DEFAULT_TRAJECTORY_STEP_VIEW_BATCH_CHUNK_MAX``. Use ``0`` or a negative value to
    disable chunking (single fused pass, highest peak memory).

    ``trajectory_step_prefetch_max_steps`` (optional int): max number of reporting steps to pull
    from ``R`` at once for the ``steps`` trajectory (limits ``index_select`` width). Default 16.

    ``trajectory_logits_storage``: ``dense_r`` (stack to ``[B,V,L,S]``) or ``list_history`` (keep
    length-``S`` list of ``[B,L,V]`` tensors, no dense ``R``).
    """
    _tc_plain: Dict[str, Any] = {}
    if trajectory_config is not None:
        if OmegaConf.is_config(trajectory_config):
            _tc_plain = OmegaConf.to_container(trajectory_config, resolve=True) or {}
        elif isinstance(trajectory_config, dict):
            _tc_plain = trajectory_config
    _raw_allow = _tc_plain.get("trajectory_step_metric_batch_allowlist") or ()
    step_allow = frozenset(
        str(x) for x in (_raw_allow if isinstance(_raw_allow, (list, tuple)) else ())
    )
    _raw_cpu = _tc_plain.get("trajectory_cpu_offload_metrics") or ()
    cpu_off = frozenset(str(x) for x in (_raw_cpu if isinstance(_raw_cpu, (list, tuple)) else ()))
    _raw_tv = _tc_plain.get("trajectory_step_view_batch_allowlist") or ()
    tv_allow = frozenset(str(x) for x in (_raw_tv if isinstance(_raw_tv, (list, tuple)) else ()))
    _raw_chunk = _tc_plain.get("trajectory_step_view_batch_chunk_max")
    chunk_max: Optional[int]
    if _raw_chunk is None:
        chunk_max = DEFAULT_TRAJECTORY_STEP_VIEW_BATCH_CHUNK_MAX
    else:
        try:
            cmi = int(_raw_chunk)
            chunk_max = cmi if cmi > 0 else None
        except (TypeError, ValueError):
            chunk_max = DEFAULT_TRAJECTORY_STEP_VIEW_BATCH_CHUNK_MAX
    _raw_spf = _tc_plain.get("trajectory_step_prefetch_max_steps")
    step_prefetch_max = DEFAULT_TRAJECTORY_STEP_PREFETCH_MAX_STEPS
    if _raw_spf is not None:
        try:
            spi = int(_raw_spf)
            if spi > 0:
                step_prefetch_max = spi
        except (TypeError, ValueError):
            step_prefetch_max = DEFAULT_TRAJECTORY_STEP_PREFETCH_MAX_STEPS
    _ls_raw = _tc_plain.get("trajectory_logits_storage")
    logits_storage = DEFAULT_TRAJECTORY_LOGITS_STORAGE
    if isinstance(_ls_raw, str) and _ls_raw.lower() in ("dense_r", "list_history"):
        logits_storage = _ls_raw.lower()
    return _tc_plain, step_allow, cpu_off, tv_allow, chunk_max, step_prefetch_max, logits_storage


def _trajectory_step_view_batch_metric_enabled(
    metric_name: str, metric: Any, tv_allow: FrozenSet[str]
) -> bool:
    return bool(tv_allow) and (metric_name in tv_allow or metric.name in tv_allow)


def _packed_shifted_probs_chunked(
    seg_logits: List[torch.Tensor],
    seg_labels: List[torch.Tensor],
    device: torch.device,
    ignore_index: int,
    chunk_max: Optional[int],
) -> List[Dict[str, float]]:
    """Run :func:`_compute_prob_packed_shifted_segments` in fixed-size chunks when configured."""
    if not seg_logits:
        return []
    if chunk_max is None or chunk_max <= 0 or len(seg_logits) <= chunk_max:
        return _compute_prob_packed_shifted_segments(
            seg_logits, seg_labels, device, ignore_index
        )
    out: List[Dict[str, float]] = []
    for i in range(0, len(seg_logits), chunk_max):
        out.extend(
            _compute_prob_packed_shifted_segments(
                seg_logits[i : i + chunk_max],
                seg_labels[i : i + chunk_max],
                device,
                ignore_index,
            )
        )
    return out


def _pre_cfg_plain_tv(cfg_any: Any) -> dict[str, Any]:
    if cfg_any is None:
        return {}
    if OmegaConf.is_config(cfg_any):
        return OmegaConf.to_container(cfg_any, resolve=True) or {}
    if isinstance(cfg_any, dict):
        return dict(cfg_any)
    return {}


def _truth_ratio_tv_precompute_compatible(metric_cfg: Any) -> bool:
    """Whether truth_ratio pre_compute matches the packed shifted-CE (correct+wrong) path."""
    _pcfg = metric_cfg.get("pre_compute") if hasattr(metric_cfg, "get") else None
    if OmegaConf.is_config(_pcfg):
        _pcfg = OmegaConf.to_container(_pcfg, resolve=True) or {}
    elif _pcfg is None:
        _pcfg = {}
    if not isinstance(_pcfg, dict):
        return False
    c_raw = _pcfg.get("correct")
    w_raw = _pcfg.get("wrong")
    ccfg = _pre_cfg_plain_tv(c_raw)
    wcfg = _pre_cfg_plain_tv(w_raw)
    if not c_raw or not w_raw:
        return False
    if str(ccfg.get("handler") or "probability") != "probability":
        return False
    if str(wcfg.get("handler") or "probability") != "probability":
        return False
    lf_c = ccfg.get("labels_field") or "labels"
    lf_w = wcfg.get("labels_field") or "labels"
    return lf_c != lf_w


def _build_shifted_ce_segments_step_view_lex(
    logits_by_step: Dict[int, torch.Tensor],
    steps_to_use: List[int],
    include_views: List[str],
    labels_full_1l: torch.Tensor,
    labels_eos_1l: Optional[torch.Tensor],
    L_eff: int,
    device: torch.device,
) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Build (logits, labels) segment lists in (step, view) lex order for packed shifted CE."""
    lc_full = labels_full_1l.to(device=device, dtype=torch.long).contiguous()
    lc_eos = (
        labels_eos_1l.to(device=device, dtype=torch.long).contiguous()
        if labels_eos_1l is not None
        else None
    )
    seg_logits, seg_lc, _seg_lw = _build_shifted_ce_segments_step_view_lex_dual(
        logits_by_step,
        steps_to_use,
        include_views,
        lc_full,
        lc_eos,
        lc_full,
        lc_eos,
        L_eff,
        device,
    )
    return seg_logits, seg_lc


def _build_shifted_ce_segments_step_view_lex_dual(
    logits_by_step: Dict[int, torch.Tensor],
    steps_to_use: List[int],
    include_views: List[str],
    labels_a_full_1l: torch.Tensor,
    labels_a_eos_1l: Optional[torch.Tensor],
    labels_b_full_1l: torch.Tensor,
    labels_b_eos_1l: Optional[torch.Tensor],
    L_eff: int,
    device: torch.device,
) -> tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Shared logits segments with two label streams (e.g. truth_ratio correct vs wrong)."""
    seg_logits: List[torch.Tensor] = []
    seg_la: List[torch.Tensor] = []
    seg_lb: List[torch.Tensor] = []
    la_f = labels_a_full_1l.to(device=device, dtype=torch.long).contiguous()
    lb_f = labels_b_full_1l.to(device=device, dtype=torch.long).contiguous()
    la_e = (
        labels_a_eos_1l.to(device=device, dtype=torch.long).contiguous()
        if labels_a_eos_1l is not None
        else None
    )
    lb_e = (
        labels_b_eos_1l.to(device=device, dtype=torch.long).contiguous()
        if labels_b_eos_1l is not None
        else None
    )
    for step in steps_to_use:
        sl = logits_by_step[int(step)]
        log_b = trajectory_step_logits_to_prob_batch(sl).contiguous()
        for view in include_views:
            if view == "full" or view not in ("full", "eos"):
                seg_logits.append(log_b)
                seg_la.append(la_f.unsqueeze(0))
                seg_lb.append(lb_f.unsqueeze(0))
            elif view == "eos":
                if la_e is None or lb_e is None:
                    raise ValueError(
                        "trajectory_step_view_batch: eos view requires sliced label tensors"
                    )
                Ls = min(L_eff, int(log_b.shape[1]))
                seg_logits.append(log_b[:, :Ls, :].contiguous())
                seg_la.append(la_e[:Ls].unsqueeze(0).contiguous())
                seg_lb.append(lb_e[:Ls].unsqueeze(0).contiguous())
    return seg_logits, seg_la, seg_lb


def _exact_mem_tv_stack_logits_and_batch(
    logits_by_step: Dict[int, torch.Tensor],
    steps_to_use: List[int],
    include_views: List[str],
    batch_template_full: Dict[str, Any],
    batch_template_eos: Dict[str, Any],
    L_eff: int,
    ignore_index: int,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Stack (step, view) lexicographic rows into ``[TV, L, V]`` + expanded batch for one ``_call_metric_at_step``."""
    lab_full = batch_template_full["labels"]
    if not isinstance(lab_full, torch.Tensor):
        raise ValueError("exact_mem tv batch: batch_template requires tensor labels")
    lab_full_1 = lab_full.squeeze(0) if lab_full.dim() > 1 else lab_full
    L_full = int(lab_full_1.shape[0])
    device = lab_full_1.device
    dtype = logits_by_step[int(steps_to_use[0])].dtype
    pad_logit = torch.finfo(torch.float32).min / 4
    lab_eos_t = batch_template_eos.get("labels")
    lab_eos_1: Optional[torch.Tensor] = None
    if isinstance(lab_eos_t, torch.Tensor):
        lab_eos_1 = lab_eos_t.squeeze(0) if lab_eos_t.dim() > 1 else lab_eos_t

    rows_logits: List[torch.Tensor] = []
    rows_labels: List[torch.Tensor] = []
    for step in steps_to_use:
        sl = logits_by_step[int(step)]
        lv = trajectory_step_logits_to_prob_batch(sl).squeeze(0).to(device=device, dtype=dtype)
        for view in include_views:
            if view == "full" or view not in ("full", "eos"):
                rows_logits.append(lv)
                rows_labels.append(lab_full_1.to(device=device))
            else:
                Ls = min(L_eff, int(lv.shape[0]))
                row_l = torch.full(
                    (L_full, lv.shape[1]),
                    pad_logit,
                    device=device,
                    dtype=lv.dtype,
                )
                row_l[:Ls] = lv[:Ls]
                row_lab = torch.full(
                    (L_full,),
                    ignore_index,
                    dtype=lab_full_1.dtype,
                    device=device,
                )
                if lab_eos_1 is not None:
                    le = lab_eos_1.to(device=device)
                    row_lab[:Ls] = le[:Ls]
                rows_logits.append(row_l)
                rows_labels.append(row_lab)
    logits_tv = torch.stack(rows_logits, dim=0)
    tv = logits_tv.shape[0]
    bt_exp = dict(batch_template_full)
    bt_exp["labels"] = torch.stack(rows_labels, dim=0)
    for key, val in list(bt_exp.items()):
        if key == "labels" or not isinstance(val, torch.Tensor):
            continue
        if val.dim() >= 1 and val.shape[0] == 1:
            bt_exp[key] = val.expand(tv, *val.shape[1:]).contiguous()
    return logits_tv, bt_exp


def _trajectory_build_sample_batch_template(
    sample_idx: int,
    batch: Dict[str, Any],
    labels: Optional[torch.Tensor],
    input_ids: torch.Tensor,
    indices: Any,
    prompt_starts: List[Any],
    prompt_lens: List[Any],
    L: int,
    prompt_only_input_ids: bool,
    tokenizer: Any,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[Dict[str, Any], str, str]:
    """One-sample batch_template for trajectory logit metrics; ground_truth_str for ROUGE; idx_str."""
    idx_raw = indices[sample_idx]
    idx_str = str(idx_raw.item() if torch.is_tensor(idx_raw) else idx_raw)
    sample_labels = labels[sample_idx] if labels is not None else None
    sample_input_ids = input_ids[sample_idx]
    sample_generation_start = _generation_start(
        sample_idx, prompt_starts, prompt_lens, prompt_only_input_ids
    )
    _gs = sample_generation_start
    generated_labels: Optional[torch.Tensor] = None
    if sample_labels is not None:
        generated_labels = sample_labels[_gs : _gs + L]
        if generated_labels.shape[0] < L:
            padding = torch.full(
                (L - generated_labels.shape[0],),
                ignore_index,
                dtype=generated_labels.dtype,
                device=generated_labels.device,
            )
            generated_labels = torch.cat([generated_labels, padding])
        assert generated_labels.shape[0] == L, (
            "batch_template invariant: generated_labels length must equal L; "
            "got %s, L=%s" % (generated_labels.shape[0], L)
        )
    generated_input_ids = sample_input_ids[_gs : _gs + L]
    if generated_input_ids.shape[0] < L:
        padding = torch.zeros(
            L - generated_input_ids.shape[0],
            dtype=generated_input_ids.dtype,
            device=sample_input_ids.device,
        )
        generated_input_ids = torch.cat([generated_input_ids, padding])
    batch_template: Dict[str, Any] = {
        "input_ids": generated_input_ids.unsqueeze(0),
        "labels": generated_labels.unsqueeze(0) if generated_labels is not None else None,
        "attention_mask": torch.ones((1, L), dtype=torch.long, device=sample_input_ids.device),
        "index": torch.tensor([int(idx_str)], dtype=torch.long, device=sample_input_ids.device),
    }
    for key in ("labels_correct", "labels_wrong"):
        if key in batch:
            batch_template[key] = _batch_template_dual_labels(
                batch, sample_idx, key, L, ignore_index
            )
    if generated_labels is not None:
        valid_labels_gt = generated_labels[generated_labels != ignore_index]
        ground_truth_str = (
            tokenizer.decode(valid_labels_gt.tolist(), skip_special_tokens=True)
            if len(valid_labels_gt) > 0
            else ""
        )
    else:
        ground_truth_str = ""
    return batch_template, ground_truth_str, idx_str


def _trajectory_logits_vl_at_step(
    R_b: torch.Tensor,
    F_b: torch.Tensor,
    S: int,
    L: int,
    traj_name: str,
    step: int,
) -> torch.Tensor:
    """[V, L] logits at a single reporting step for one batch row."""
    st_b: Dict[str, Any] = {"R": R_b, "F": F_b, "S": S, "L": L}
    if traj_name == "steps":
        return R_b[:, :, step]
    return _get_logits_at_step(st_b, traj_name, int(step))


def _trajectory_logits_vl_from_history(
    lh: List[torch.Tensor],
    b: int,
    F_b: torch.Tensor,
    S: int,
    L: int,
    traj_name: str,
    step: int,
) -> torch.Tensor:
    """[V, L] logits at a single reporting step for batch row ``b`` (list-backed storage)."""
    st_b: Dict[str, Any] = {"lh": lh, "b": b, "F": F_b, "S": S, "L": L}
    return _get_logits_at_step(st_b, traj_name, int(step))


def _trajectory_decode_prediction_for_rouge(
    tokenizer: Any,
    logits_vl: torch.Tensor,
    view: str,
    L_eff_b: int,
) -> str:
    """Decode argmax prediction from [V,L] (or [1,V,L]) logits for ROUGE gen text."""
    lv = logits_vl
    if lv.dim() == 3:
        lv = lv[0]
    # Trajectory slice is [V, L] (same as ``trajectory_step_logits_to_prob_batch`` input); argmax over vocab.
    pred_ids = torch.argmax(lv, dim=0).tolist()
    if view == "eos":
        pred_ids = pred_ids[: min(int(L_eff_b), len(pred_ids))]
    return tokenizer.decode(pred_ids, skip_special_tokens=True)


def _trajectory_batch_decode_predictions_for_rouge(
    tokenizer: Any,
    logits_rows: Sequence[torch.Tensor],
    view: str,
    L_eff_list: Sequence[int],
) -> List[str]:
    """Batched argmax over ``B`` rows of ``[V,L]`` logits (shared ``L``); eos view trims on device before decode."""
    if not logits_rows:
        return []
    normed: list[torch.Tensor] = []
    for lv in logits_rows:
        if lv.dim() == 3:
            lv = lv[0]
        if lv.dim() != 2:
            raise ValueError(
                "trajectory ROUGE decode: expected [V, L] logits per row, "
                f"got dim={lv.dim()} shape={tuple(lv.shape)}"
            )
        normed.append(lv)
    lengths = [int(t.shape[1]) for t in normed]
    if len(set(lengths)) != 1:
        raise ValueError(
            "trajectory ROUGE batch decode requires a common sequence length per row; "
            f"got lengths {lengths}"
        )
    stacked = torch.stack(normed, dim=0)
    pred = stacked.argmax(dim=1)
    token_rows: list[list[int]] = []
    for b in range(pred.shape[0]):
        row = pred[b]
        if view == "eos":
            n = min(int(L_eff_list[b]), int(row.shape[0]))
            row = row[:n]
        token_rows.append(row.detach().cpu().tolist())
    batch_decode = getattr(tokenizer, "batch_decode", None)
    if callable(batch_decode):
        return list(batch_decode(token_rows, skip_special_tokens=True))
    return [
        tokenizer.decode(ids, skip_special_tokens=True) for ids in token_rows
    ]


def _trajectory_exact_mem_post_loop_metric_names(
    *,
    metrics_to_run: List[str],
    loaded_metrics: Dict[str, Any],
    step_metric_batch_allowlist: FrozenSet[str],
    view_step_batch_allowlist: FrozenSet[str],
    B: int,
    batch: Dict[str, Any],
    labels: Optional[torch.Tensor],
    input_ids: torch.Tensor,
    indices: Any,
    prompt_starts: List[Any],
    prompt_lens: List[Any],
    L: int,
    prompt_only_input_ids: bool,
    tokenizer: Any,
) -> FrozenSet[str]:
    """Metric keys whose allowlisted exact_memorization runs in the post-sample loop (all traj types)."""
    out: set[str] = set()
    for metric_name in metrics_to_run:
        mi = loaded_metrics[metric_name]
        metric = mi["metric"]
        if metric.name != "exact_memorization":
            continue
        mcfg = mi.get("config") or {}
        _pcfg = mcfg.get("pre_compute") if hasattr(mcfg, "get") else None
        if OmegaConf.is_config(_pcfg):
            _pcfg = OmegaConf.to_container(_pcfg, resolve=True) or {}
        elif _pcfg is None:
            _pcfg = {}
        if isinstance(_pcfg, dict) and len(_pcfg) > 0:
            continue
        in_step = metric_name in step_metric_batch_allowlist or metric.name in step_metric_batch_allowlist
        in_view = metric_name in view_step_batch_allowlist or metric.name in view_step_batch_allowlist
        if not in_step and not in_view:
            continue
        ok_all = True
        for b in range(B):
            bt, _, _ = _trajectory_build_sample_batch_template(
                b,
                batch,
                labels,
                input_ids,
                indices,
                prompt_starts,
                prompt_lens,
                L,
                prompt_only_input_ids,
                tokenizer,
            )
            if _batch_template_has_list_tensors(bt):
                ok_all = False
                break
        if ok_all:
            out.add(metric_name)
    return frozenset(out)


def _trajectory_append_post_loop_rouge(
    *,
    trajectory_names: List[str],
    steps_to_use: List[int],
    include_views: List[str],
    R: Optional[torch.Tensor],
    F: torch.Tensor,
    lh: Optional[List[torch.Tensor]],
    S: int,
    L: int,
    B: int,
    effective_lengths: List[int],
    labels: Optional[torch.Tensor],
    input_ids: torch.Tensor,
    batch: Dict[str, Any],
    indices: Any,
    prompt_starts: List[Any],
    prompt_lens: List[Any],
    prompt_only_input_ids: bool,
    tokenizer: Any,
    metrics_to_run: List[str],
    loaded_metrics: Dict[str, Any],
    step_values_by_view: Dict[str, Any],
    executor: Optional[ProcessPoolExecutor],
    all_rouge_futures: List[Any],
    kwargs: Dict[str, Any],
) -> None:
    """Gather gen/gt strings across ``B`` per (traj, step, view); append ROUGE like probability post-loop."""
    rouge_metrics_in_run = [
        m for m in metrics_to_run if loaded_metrics[m]["metric"].name == "rouge"
    ]
    if not rouge_metrics_in_run:
        return
    rouge_scorer_batch = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    if (R is None) == (lh is None):
        raise ValueError("post-loop ROUGE: exactly one of R or lh must be set")
    for traj_name in trajectory_names:
        for step in steps_to_use:
            for view in include_views:
                if step not in step_values_by_view[view][traj_name]:
                    step_values_by_view[view][traj_name][step] = {
                        m: [] for m in loaded_metrics.keys()
                    }
                logits_rows: list[torch.Tensor] = []
                for b in range(B):
                    if lh is not None:
                        logits_rows.append(
                            _trajectory_logits_vl_from_history(lh, b, F[b], S, L, traj_name, step)
                        )
                    else:
                        assert R is not None
                        logits_rows.append(
                            _trajectory_logits_vl_at_step(R[b], F[b], S, L, traj_name, step)
                        )
                flat_lens: set[int] = set()
                for t in logits_rows:
                    lv = t[0] if t.dim() == 3 else t
                    flat_lens.add(int(lv.shape[1]))
                if len(flat_lens) == 1:
                    gen_texts = _trajectory_batch_decode_predictions_for_rouge(
                        tokenizer, logits_rows, view, effective_lengths
                    )
                else:
                    gen_texts = [
                        _trajectory_decode_prediction_for_rouge(
                            tokenizer,
                            logits_rows[b],
                            view,
                            int(effective_lengths[b]),
                        )
                        for b in range(B)
                    ]
                ground_truths: List[str] = []
                for b in range(B):
                    _, gt_str, _ = _trajectory_build_sample_batch_template(
                        b,
                        batch,
                        labels,
                        input_ids,
                        indices,
                        prompt_starts,
                        prompt_lens,
                        L,
                        prompt_only_input_ids,
                        tokenizer,
                    )
                    ground_truths.append(gt_str)
                if executor is None:
                    scores = eval_rouge_recall_batch(
                        gen_texts,
                        ground_truths,
                        use_stemmer=True,
                        scorer=rouge_scorer_batch,
                    )
                    for metric_name in rouge_metrics_in_run:
                        metric_cfg = loaded_metrics[metric_name]["config"]
                        rouge_type = metric_cfg.get("rouge_type") or kwargs.get("rouge_type", "rougeL_f1")
                        for b_idx in range(B):
                            if b_idx < len(scores) and isinstance(scores[b_idx], dict) and rouge_type in scores[b_idx]:
                                step_values_by_view[view][traj_name][step][metric_name].append(
                                    scores[b_idx][rouge_type]
                                )
                else:
                    fut = executor.submit(
                        eval_rouge_recall_batch_worker,
                        gen_texts,
                        ground_truths,
                        True,
                    )
                    all_rouge_futures.append(
                        (fut, traj_name, rouge_metrics_in_run, step, view)
                    )


def _trajectory_append_post_loop_exact_mem_allowlist(
    *,
    trajectory_names: List[str],
    exact_mem_post_loop_metrics: FrozenSet[str],
    steps_to_use: List[int],
    include_views: List[str],
    R: Optional[torch.Tensor],
    F: torch.Tensor,
    lh: Optional[List[torch.Tensor]],
    S: int,
    L: int,
    B: int,
    effective_lengths: List[int],
    labels: Optional[torch.Tensor],
    input_ids: torch.Tensor,
    batch: Dict[str, Any],
    indices: Any,
    prompt_starts: List[Any],
    prompt_lens: List[Any],
    prompt_only_input_ids: bool,
    tokenizer: Any,
    metrics_to_run: List[str],
    loaded_metrics: Dict[str, Any],
    step_values_by_view: Dict[str, Any],
    trajectory_config: Optional[Any],
    primary_data: Any,
    kwargs: Dict[str, Any],
    batch_idx: int,
    rouge_scorer_shared: Any,
    guardrail_config_with_pools: Optional[dict],
    view_step_batch_allowlist: FrozenSet[str],
    step_prefetch_max_steps: int,
) -> None:
    """Allowlisted ``exact_memorization``: batched over T (and optionally T×views) per sample, all B in post-loop."""
    if not exact_mem_post_loop_metrics:
        return
    if (R is None) == (lh is None):
        raise ValueError("post-loop exact_mem allowlist: exactly one of R or lh must be set")
    for metric_name in metrics_to_run:
        if metric_name not in exact_mem_post_loop_metrics:
            continue
        metric_info = loaded_metrics[metric_name]
        metric = metric_info["metric"]
        metric_cfg = metric_info["config"]
        tv_on = _trajectory_step_view_batch_metric_enabled(
            metric_name, metric, view_step_batch_allowlist
        )
        gpu_set_phase(
            "trajectory_metric",
            metric=metric_name,
            batch_idx=batch_idx,
            step=-1,
        )
        for traj_name in trajectory_names:
            for b in range(B):
                batch_template, ground_truth_str, idx_str = _trajectory_build_sample_batch_template(
                    b,
                    batch,
                    labels,
                    input_ids,
                    indices,
                    prompt_starts,
                    prompt_lens,
                    L,
                    prompt_only_input_ids,
                    tokenizer,
                )
                if lh is not None:
                    sample_traj = {"lh": lh, "b": b, "F": F[b], "S": S, "L": L}
                else:
                    assert R is not None
                    sample_traj = {"R": R[b], "F": F[b], "S": S, "L": L}
                L_eff_slice = min(int(effective_lengths[b]), batch_template["input_ids"].shape[1])
                batch_template_eos = _slice_batch_template_to_length(batch_template, L_eff_slice)
                kwargs_clean = {k: v for k, v in kwargs.items() if k != "tokenizer"}
                if primary_data is not None:
                    kwargs_clean["data"] = primary_data
                kwargs_clean["ground_truth"] = ground_truth_str
                kwargs_clean["rouge_scorer"] = rouge_scorer_shared
                kwargs_clean["sample_traj"] = sample_traj
                kwargs_clean["step"] = steps_to_use[0]
                kwargs_clean["step_index"] = 0
                if trajectory_config is not None:
                    kwargs_clean["trajectory_config"] = trajectory_config
                if guardrail_config_with_pools is not None:
                    kwargs_clean["guardrail_config"] = guardrail_config_with_pools

                for step_window in _chunked_reporting_steps(steps_to_use, step_prefetch_max_steps):
                    logits_by_step = _prefetch_logits_by_step(sample_traj, traj_name, step_window)
                    if tv_on:
                        logits_tv, bt_exp = _exact_mem_tv_stack_logits_and_batch(
                            logits_by_step,
                            step_window,
                            include_views,
                            batch_template,
                            batch_template_eos,
                            L_eff_slice,
                            IGNORE_INDEX,
                        )
                        kwargs_metric = dict(kwargs_clean)
                        kwargs_metric["traj_name"] = traj_name
                        result = _call_metric_at_step(
                            metric=metric,
                            logits=logits_tv,
                            batch_template=bt_exp,
                            tokenizer=tokenizer,
                            sample_labels=labels[b] if labels is not None else None,
                            sample_input_ids=input_ids[b],
                            sample_prompt_len=prompt_lens[b],
                            metric_config=metric_cfg,
                            sample_idx=idx_str,
                            **kwargs_metric,
                        )
                        tv_expected = len(step_window) * len(include_views)
                        if not isinstance(result, list) or len(result) != tv_expected:
                            raise RuntimeError(
                                "batched exact_memorization post-loop (T×views): expected list of length "
                                f"{tv_expected}, got {type(result).__name__} len="
                                f"{len(result) if isinstance(result, list) else 'n/a'}"
                            )
                        j = 0
                        for step in step_window:
                            for view in include_views:
                                if step not in step_values_by_view[view][traj_name]:
                                    step_values_by_view[view][traj_name][step] = {
                                        m: [] for m in loaded_metrics.keys()
                                    }
                                rd = result[j]
                                j += 1
                                metric_value = None
                                if isinstance(rd, dict):
                                    if "score" in rd:
                                        metric_value = rd["score"]
                                    elif "prob" in rd:
                                        metric_value = rd["prob"]
                                if metric_value is not None:
                                    step_values_by_view[view][traj_name][step][metric_name].append(
                                        metric_value
                                    )
                        del logits_by_step
                        continue

                    for view in include_views:
                        bt = batch_template if view == "full" else batch_template_eos
                        leff = L_eff_slice if view == "eos" else None
                        logits_batched = _stack_step_logits_for_prob_batch(
                            logits_by_step, step_window, leff
                        )
                        kwargs_metric = dict(kwargs_clean)
                        kwargs_metric["traj_name"] = traj_name
                        result = _call_metric_at_step(
                            metric=metric,
                            logits=logits_batched,
                            batch_template=bt,
                            tokenizer=tokenizer,
                            sample_labels=labels[b] if labels is not None else None,
                            sample_input_ids=input_ids[b],
                            sample_prompt_len=prompt_lens[b],
                            metric_config=metric_cfg,
                            sample_idx=idx_str,
                            **kwargs_metric,
                        )
                        if not isinstance(result, list) or len(result) != len(step_window):
                            raise RuntimeError(
                                "batched exact_memorization post-loop: expected list of length "
                                f"{len(step_window)}, got {type(result).__name__} len="
                                f"{len(result) if isinstance(result, list) else 'n/a'}"
                            )
                        for j, step in enumerate(step_window):
                            if step not in step_values_by_view[view][traj_name]:
                                step_values_by_view[view][traj_name][step] = {
                                    m: [] for m in loaded_metrics.keys()
                                }
                            rd = result[j]
                            metric_value = None
                            if isinstance(rd, dict):
                                if "score" in rd:
                                    metric_value = rd["score"]
                                elif "prob" in rd:
                                    metric_value = rd["prob"]
                            if metric_value is not None:
                                step_values_by_view[view][traj_name][step][metric_name].append(
                                    metric_value
                                )
                    del logits_by_step
        if _trajectory_should_empty_cuda_cache(trajectory_config):
            torch.cuda.empty_cache()


def should_run_gc(threshold: float = 0.9) -> bool:
    """Return True if CUDA is available and VRAM usage (allocated/total) is >= threshold."""
    if not torch.cuda.is_available():
        return False
    total = torch.cuda.get_device_properties(0).total_memory
    if total <= 0:
        return False
    return (torch.cuda.memory_allocated() / total) >= threshold


def _chunked_reporting_steps(steps_to_use: List[int], max_per_chunk: int) -> List[List[int]]:
    """Split ``steps_to_use`` into consecutive windows (last may be shorter)."""
    if not steps_to_use or max_per_chunk <= 0 or len(steps_to_use) <= max_per_chunk:
        return [list(steps_to_use)]
    return [steps_to_use[i : i + max_per_chunk] for i in range(0, len(steps_to_use), max_per_chunk)]


def _per_position_scores_from_R_F_batch(
    R: Optional[torch.Tensor],
    F: torch.Tensor,
    labels: Optional[torch.Tensor],
    prompt_starts: List[int],
    L: int,
    trajectory_config: Dict[str, Any],
    report_step: Optional[int] = None,
    *,
    lh_batch: Optional[List[torch.Tensor]] = None,
) -> Optional[List[List[float]]]:
    """Build per-sample per-position probability scores from ``R``/``F`` or list-backed ``lh_batch``.

    prompt_starts[i] is the start index of the generation region in labels (use for labels[i, start:start+L]).
    If report_step is set, uses effective-step logits at that step (s_eff(ell,s)=min(s,F[ell])).
    Returns list of list of float (one list per sample), or None if labels missing.
    """
    if labels is None:
        return None
    logit_alignment = trajectory_config.get("logit_alignment", "causal")
    provider = FixationStepWiseScoreProvider(logit_alignment=logit_alignment)
    out: List[List[float]] = []
    if lh_batch is not None:
        if report_step is None:
            raise ValueError("lh_batch path requires report_step for effective-step fixation logits")
        B = int(F.shape[0])
        fl_all = build_effective_step_fixation_logits_from_history(lh_batch, F, int(report_step))
        for i in range(B):
            start = prompt_starts[i] if isinstance(prompt_starts[i], int) else int(prompt_starts[i].item())
            gen_labels = labels[i, start : start + L]
            batch_prov = {"labels": gen_labels.unsqueeze(0)}
            model_or_logits: Dict[str, Any] = {"fixation_logits": fl_all[i : i + 1]}
            results = provider.get_per_position_scores(
                model_or_logits, batch_prov, ignore_index=IGNORE_INDEX
            )
            out.append(results[0][0] if results and results[0][0] else [])
        return out
    if R is None:
        raise ValueError("trajectory per-position scores: need R or lh_batch")
    B = int(R.shape[0])
    for i in range(B):
        start = prompt_starts[i] if isinstance(prompt_starts[i], int) else int(prompt_starts[i].item())
        gen_labels = labels[i, start : start + L]
        batch_prov = {"labels": gen_labels.unsqueeze(0)}
        model_or_logits = {"R": R[i].unsqueeze(0), "F": F[i].unsqueeze(0)}
        if report_step is not None:
            model_or_logits["report_step"] = report_step
        results = provider.get_per_position_scores(
            model_or_logits, batch_prov, ignore_index=IGNORE_INDEX
        )
        out.append(results[0][0] if results and results[0][0] else [])
    return out


def _truncate_per_position_scores_eos(
    per_position_scores: List[List[float]],
    effective_lengths: List[int],
    cap_L: int,
) -> List[List[float]]:
    """Truncate each sample's per-position scores to eos-aligned length (same rule as forget eos view)."""
    out: List[List[float]] = []
    for i, scores in enumerate(per_position_scores):
        le = int(effective_lengths[i]) if i < len(effective_lengths) else cap_L
        le = min(le, cap_L, len(scores))
        out.append(list(scores[:le]))
    return out


def _derive_steps_to_use(
    S: int,
    trajectory_config: Union[Dict, DictConfig],
) -> tuple:
    """Derive which step indices to compute/store and token positions for report metadata.

    Report interval is fixed at 8 tokens. When trajectory_sample_interval is set, the
    sampler already returns subsampled steps (every 8 tokens); use all S steps and
    build metadata as token positions [interval*1, interval*2, ..., min(max_new_tokens, interval*S)].
    When trajectory_sample_interval is not set, the sampler returned every diffusion step;
    subsample to steps where token position is 0, 8, 16, ... and return those step indices
    and corresponding token positions.

    Returns:
        (steps_to_use, step_values_metadata): steps_to_use is list of step indices (into R);
        step_values_metadata is list of token positions for report (same length).
    """
    if S <= 0:
        return ([], [])
    sampler_kwargs = trajectory_config.get("sampler_kwargs", {}) or {}
    trajectory_sample_interval = sampler_kwargs.get("trajectory_sample_interval")
    max_new_tokens = sampler_kwargs.get("max_new_tokens")
    steps = sampler_kwargs.get("steps") or 50

    if trajectory_sample_interval is not None and trajectory_sample_interval > 0:
        steps_to_use = list(range(S))
        interval = int(trajectory_sample_interval)
        if max_new_tokens is not None:
            step_values_metadata = [
                min(interval * (k + 1), int(max_new_tokens)) for k in range(S)
            ]
        else:
            step_values_metadata = [interval * (k + 1) for k in range(S)]
        return (steps_to_use, step_values_metadata)

    if max_new_tokens is None or steps is None or steps <= 0:
        steps_to_use = list(range(S))
        step_values_metadata = list(range(S))
        return (steps_to_use, step_values_metadata)

    tokens_per_step_approx = float(max_new_tokens) / float(steps)
    report_interval = 8
    seen = set()
    steps_to_use = []
    step_values_metadata = []
    for t in range(0, int(max_new_tokens) + 1, report_interval):
        s = round(t / tokens_per_step_approx)
        s = max(0, min(S - 1, s))
        if s not in seen:
            seen.add(s)
            steps_to_use.append(s)
            step_values_metadata.append(min(t, int(max_new_tokens)))
    if not steps_to_use:
        steps_to_use = [0]
        step_values_metadata = [0]
    return (steps_to_use, step_values_metadata)


def _get_sampler_from_model(model) -> Optional[Any]:
    """Extract sampler from model (handles adapter wrapping)."""
    # Check if model is wrapped with DiffusionModelAdapter
    if hasattr(model, "sampler"):
        return model.sampler
    
    # Check if model has a model attribute (nested wrapping)
    if hasattr(model, "model"):
        if hasattr(model.model, "sampler"):
            return model.model.sampler
        # Check if model.model is the adapter
        if hasattr(model.model, "sampler"):
            return model.model.sampler
    
    return None


def _generate_trajectories_for_dataloader(
    sampler: Any,
    dataloader: DataLoader,
    trajectory_config: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Generate trajectories for all samples in a dataloader. Returns {idx_str: trajectories}."""
    trajectories_by_idx = {}
    tokenizer = getattr(sampler, "tokenizer", None)
    n_dataset = len(dataloader.dataset)
    n_batches = len(dataloader)
    prompt_only_input_ids = getattr(dataloader.dataset, "predict_with_generate", False)
    logger.info(
        "[trajectory_dataloader] starting: dataset_size=%s batches=%s predict_with_generate=%s",
        n_dataset, n_batches, prompt_only_input_ids,
    )
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        indices = batch.get(
            "index",
            torch.arange(
                batch_idx * input_ids.shape[0],
                (batch_idx + 1) * input_ids.shape[0],
            ),
        )
        B = input_ids.shape[0]
        if tokenizer is not None:
            prompts, prompt_lens, prompt_starts = _build_prompts_for_sampler(
                input_ids, labels, tokenizer, IGNORE_INDEX,
                prompt_only_input_ids=prompt_only_input_ids,
            )
        else:
            prompts = []
            prompt_lens = []
            prompt_starts = []
            for i in range(B):
                if labels is not None:
                    label_mask = labels[i] != IGNORE_INDEX
                    prompt_end = (
                        label_mask.nonzero()[0][0].item()
                        if label_mask.any()
                        else input_ids.shape[1]
                    )
                else:
                    prompt_end = input_ids.shape[1]
                prompt_starts.append(prompt_end)
                if prompt_only_input_ids:
                    prompts.append(input_ids[i].cpu().tolist())
                else:
                    prompts.append(input_ids[i, :prompt_end].cpu().tolist())
                prompt_lens.append(len(prompts[-1]))

        # Log when sample 0 has empty or EOS/pad-only prompt (for correlating with sampler empty_response_sample0_detected)
        if B > 0 and logger.isEnabledFor(logging.DEBUG):
            pl0 = prompt_lens[0]
            idx0 = indices[0].item() if torch.is_tensor(indices[0]) else indices[0]
            p0 = prompts[0]
            if pl0 == 0:
                logger.debug(
                    "[trajectory] batch=%s dataset_index_sample0=%s prompt_len_sample0=0 (empty prompt sent to sampler)",
                    batch_idx, idx0,
                )
            elif len(p0) > 0:
                pad_id = getattr(sampler, "tokenizer", None) and getattr(sampler.tokenizer, "pad_token_id", None)
                eos_id = getattr(sampler, "tokenizer", None) and getattr(sampler.tokenizer, "eos_token_id", None)
                if pad_id is None and eos_id is not None:
                    pad_id = eos_id
                if pad_id is not None and all(t == pad_id or t == eos_id for t in p0):
                    logger.debug(
                        "[trajectory] batch=%s dataset_index_sample0=%s prompt_len_sample0=%s prompt_first5=%s (all tokens EOS/pad)",
                        batch_idx, idx0, pl0, p0[:5],
                    )
        _sampler_kw = _trajectory_sampler_kwargs(trajectory_config)
        evaluation_mode = _sampler_kw.get("evaluation_mode", "unguided")
        sample_kw = dict(
            inputs=prompts,
            config=None,
            return_dict=True,
            return_logits=True,
            **_sampler_kw,
        )
        if evaluation_mode in ("guided_native", "guided_skew") and labels is not None:
            L_gen = _sampler_kw.get("max_new_tokens")
            if L_gen is None:
                L_gen = max(labels.shape[1] - prompt_starts[i] for i in range(B))
            else:
                L_gen = int(L_gen)
            target_sequences = _build_target_sequences_for_sampler(
                labels, prompt_starts, L_gen, IGNORE_INDEX
            )
            sample_kw["target_sequences"] = target_sequences
            sample_kw["evaluation_mode"] = evaluation_mode
        sampler_output = sampler.sample(**sample_kw)
        logits_history = sampler_output.logits_history
        fixation_steps = sampler_output.fixation_steps
        if logits_history is None or len(logits_history) == 0:
            logger.warning(
                "[trajectory_dataloader] batch_idx=%s skipped: no logits_history",
                batch_idx,
            )
            continue
        if fixation_steps is None:
            logger.warning(
                "[trajectory_dataloader] batch_idx=%s skipped: no fixation_steps",
                batch_idx,
            )
            continue

        out = trajectories_from_logits(
            logits_history, fixation_steps, prompt_lens, return_trajectory_tensors=False
        )
        R, F, S, L = out["R"], out["F"], out["S"], out["L"]
        seq_eff = getattr(sampler_output, "sequences", None)
        eos_id = getattr(sampler.tokenizer, "eos_token_id", None) if getattr(sampler, "tokenizer", None) else None
        if eos_id is None:
            eos_id = trajectory_config.get("eos_token_id") if trajectory_config else None
        if seq_eff is not None and eos_id is not None and seq_eff.dim() >= 2:
            eff_lens = effective_lengths_from_eos(seq_eff, prompt_lens, L, eos_id)
        else:
            eff_lens = [L] * R.shape[0]
        for i in range(R.shape[0]):
            idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
            le = int(eff_lens[i]) if i < len(eff_lens) else L
            trajectories_by_idx[str(idx)] = {
                "R": R[i],
                "F": F[i],
                "S": S,
                "L": L,
                "effective_length": min(le, int(L)),
            }
        del logits_history, out
    n_collected = len(trajectories_by_idx)
    logger.info(
        "[trajectory_dataloader] done: trajectories_collected=%s dataset_size=%s",
        n_collected, n_dataset,
    )
    if n_collected != n_dataset:
        logger.warning(
            "[trajectory_dataloader] trajectory count mismatch: got %s expected %s (some batches may have been skipped)",
            n_collected, n_dataset,
        )
    return trajectories_by_idx


def _get_logits_at_step(traj: Dict[str, Any], traj_name: str, step: int) -> torch.Tensor:
    """Get [V, L] logits at a trajectory step.

    ``traj`` is either dense ``{"R","F","S","L"}`` or list-backed ``{"lh","b","F","S","L"}``.
    """
    if traj.get("lh") is not None:
        lh: List[torch.Tensor] = traj["lh"]
        b = int(traj["b"])
        F_raw = traj["F"]
        F_sample = F_raw.squeeze(0) if F_raw.dim() > 1 else F_raw
        S = int(traj["S"])
        if traj_name == "steps":
            return lh[step][b].transpose(0, 1).contiguous()
        if traj_name == "fixation_start":
            return compute_fixation_start_from_history(lh, b, step, F_sample, S)
        if traj_name == "fixation_end":
            return compute_fixation_end_from_history(lh, b, step, F_sample, S)
        if traj_name == "fixation_ratio":
            return compute_fixation_ratio_from_history(lh, b, step, F_sample, S)
        raise ValueError(f"Unknown traj_name: {traj_name}")
    R_sample = traj["R"]
    F_sample = traj["F"]
    if traj_name == "steps":
        return R_sample[:, :, step]
    if traj_name == "fixation_start":
        return compute_fixation_start_trajectory(R_sample, step, F_sample)
    if traj_name == "fixation_end":
        return compute_fixation_end_trajectory(R_sample, step, F_sample)
    if traj_name == "fixation_ratio":
        return compute_fixation_ratio_trajectory(R_sample, step, F_sample)
    raise ValueError(f"Unknown traj_name: {traj_name}")


def _prefetch_logits_by_step(
    sample_traj: Dict[str, Any], traj_name: str, steps_to_use: List[int]
) -> Dict[int, torch.Tensor]:
    """Map step -> [V, L] logits for reporting steps in ``steps_to_use``.

    For dense ``R`` and ``traj_name=="steps"``, uses one ``index_select`` when multiple
    steps are requested (caller should pass windows to cap peak memory).
    List-backed ``lh`` always materializes one ``[V,L]`` tensor per step (no wide select).
    """
    if not steps_to_use:
        return {}
    if sample_traj.get("lh") is not None:
        if traj_name == "steps":
            lh: List[torch.Tensor] = sample_traj["lh"]
            b = int(sample_traj["b"])
            return {int(s): lh[int(s)][b].transpose(0, 1).contiguous() for s in steps_to_use}
        return {int(s): _get_logits_at_step(sample_traj, traj_name, int(s)) for s in steps_to_use}
    R_sample = sample_traj["R"]
    if traj_name == "steps":
        device = R_sample.device
        idx = torch.tensor(steps_to_use, dtype=torch.long, device=device)
        stacked = R_sample.index_select(2, idx)
        return {int(steps_to_use[j]): stacked[:, :, j] for j in range(len(steps_to_use))}
    return {int(s): _get_logits_at_step(sample_traj, traj_name, int(s)) for s in steps_to_use}


def _stack_step_logits_for_prob_batch(
    logits_by_step: Dict[int, torch.Tensor], steps_to_use: List[int], L_eff_slice: Optional[int]
) -> torch.Tensor:
    """Stack [V,L] per-step logits into [T,L,V] for batched ``exact_memorization`` / probability-style CE."""
    rows: List[torch.Tensor] = []
    for s in steps_to_use:
        sl = logits_by_step[int(s)]
        if L_eff_slice is not None:
            sl = sl[:, :L_eff_slice]
        rows.append(trajectory_step_logits_to_prob_batch(sl).squeeze(0))
    return torch.stack(rows, dim=0)


def _expand_batch_template_leading_dim(
    batch_template: Dict[str, Any], T: int
) -> Dict[str, Any]:
    """Expand tensors with leading batch dim 1 to ``(T, …)`` for batched logit metrics."""
    out: Dict[str, Any] = {}
    for k, v in batch_template.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == 1:
            out[k] = v.expand(T, *v.shape[1:]).contiguous()
        else:
            out[k] = v
    return out


def _worker_exact_memorization_cpu(
    logits_l_v_numpy: np.ndarray,
    labels_1l_numpy: np.ndarray,
    _ignore_index: int,
) -> Optional[float]:
    """Picklable ProcessPool worker: [L,V] float logits + [L] int labels → exact_mem score (CPU)."""
    from evals.metrics.trajectory_adapters import LogitModelWrapper
    from evals.metrics.utils import tokenwise_vocab_logprobs

    logits_lv = torch.from_numpy(np.ascontiguousarray(logits_l_v_numpy)).float()
    labels_1l = torch.from_numpy(np.ascontiguousarray(labels_1l_numpy)).long()
    logits_b = logits_lv.unsqueeze(0)
    L = labels_1l.shape[0]
    batch = {
        "labels": labels_1l.unsqueeze(0),
        "input_ids": torch.zeros((1, L), dtype=torch.long),
        "attention_mask": torch.ones((1, L), dtype=torch.long),
    }
    model = LogitModelWrapper(logits_b, logits_b.device)
    log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
        model, batch, grad=False, return_labels=True
    )
    if len(log_probs_batch) == 0 or len(labels_batch) == 0:
        return None
    log_probs = log_probs_batch[0]
    labels_use = labels_batch[0]
    if len(labels_use) == 0:
        return None
    preds = torch.argmax(log_probs, dim=-1)
    em_score = (preds == labels_use).sum().float() / float(len(labels_use))
    return float(em_score.item())


def _get_metric_from_registry(metric_name: str):
    """Get metric from registry by name."""
    # Import here to avoid circular import
    from evals.metrics import METRICS_REGISTRY

    metric = METRICS_REGISTRY.get(metric_name)
    if metric is None:
        raise ValueError(
            f"Metric '{metric_name}' not found in registry. "
            f"Available metrics: {list(METRICS_REGISTRY.keys())}"
        )
    return metric


def _extract_metric_scalar(result: Any) -> Optional[float]:
    """Extract a single float from a metric result (list of dicts or dict with agg_value/score/prob)."""
    if result is None:
        return None
    item = result[0] if isinstance(result, (list, tuple)) and len(result) > 0 else result
    if not isinstance(item, dict):
        return None
    for key in ("agg_value", "score", "prob"):
        if key in item and item[key] is not None:
            v = item[key]
            return float(v) if hasattr(v, "item") else float(v)
    return None


def _compute_retain_mu_by_step(
    model: Any,
    data_retain: Any,
    collator: Any,
    batch_size: int,
    trajectory_config: Union[Dict, DictConfig],
    tokenizer: Any,
    loaded_metrics: Dict[str, Any],
    sort_by_length: bool,
    use_distributed_sampler: bool,
    world_size: int,
    rank: int,
    **kwargs: Any,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute per-step retain-set aggregates (Prob, ROUGE, Truth Ratio) for trajectory MU.
    Returns retain_agg_by_step[step_index][view] = {retain_Q_A_Prob: {agg_value: v}, ...}
    for view in include_views subset of (full, eos). Legacy flat step dict (no view key) is not produced.
    Used so hm_aggregate at each step uses current model's retain metrics, not reference_logs.
    Returns retain_agg_by_step[traj_name][step_index][view] so MU can differ by trajectory type.
    """
    from evals.gpu_phase_logger import set_phase as gpu_set_phase

    trajectory_names = (
        list(trajectory_config.get("trajectory_names", ["steps", "fixation_start", "fixation_end", "fixation_ratio"]))
        if trajectory_config
        else ["steps"]
    )
    retain_agg_by_step = {}
    run_steps_to_use = None
    # Sub-metrics needed for MU; use from loaded_metrics or registry
    mu_sub_metrics = ("probability", "rouge", "truth_ratio")
    metric_objs = {}
    for name in mu_sub_metrics:
        if name in loaded_metrics:
            metric_objs[name] = loaded_metrics[name]["metric"]
            metric_objs[f"{name}_config"] = loaded_metrics[name].get("config") or {}
        else:
            try:
                metric_objs[name] = _get_metric_from_registry(name)
                metric_objs[f"{name}_config"] = {}
            except ValueError:
                metric_objs[name] = None
                metric_objs[f"{name}_config"] = {}
    if not all(metric_objs.get(m) for m in mu_sub_metrics):
        logger.warning("trajectory_model_utility: missing probability/rouge/truth_ratio; retain MU will be None")
        return retain_agg_by_step

    if sort_by_length:
        retain_dataloader = DataLoader(
            data_retain,
            batch_size=batch_size,
            sampler=LengthSortedSampler(data_retain),
            collate_fn=collator,
        )
    elif use_distributed_sampler:
        retain_dataloader = DataLoader(
            data_retain,
            batch_size=batch_size,
            sampler=DistributedSampler(data_retain, num_replicas=world_size, rank=rank, shuffle=False),
            collate_fn=collator,
        )
    else:
        retain_dataloader = DataLoader(
            data_retain, batch_size=batch_size, collate_fn=collator
        )

    sampler = _get_sampler_from_model(model)
    if sampler is None:
        logger.warning("trajectory_model_utility: no sampler on model; retain MU will be None")
        return retain_agg_by_step

    _sampler_kw = _trajectory_sampler_kwargs(trajectory_config)
    _ = trajectory_config.get("rouge_type") or kwargs.get("rouge_type", "rougeL_recall")  # rouge_type, reserved
    rouge_scorer_instance = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    # Use same default as main trajectory loop so retain MU has both full and eos when main loop expects both.
    _iv = trajectory_config.get("include_views", ["full", "eos"]) if trajectory_config else ["full", "eos"]
    if isinstance(_iv, str):
        _iv = [_iv]
    _mu_views = [str(v).lower() for v in _iv if str(v).lower() in ("full", "eos")]
    if not _mu_views:
        _mu_views = ["full", "eos"]
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[MU retain path] resolved _mu_views=%s (from trajectory_config.include_views)",
            _mu_views,
        )

    def _empty_mu_pl() -> Dict[str, Dict[str, List[Any]]]:
        return {v: {"prob": [], "rouge": [], "tr": []} for v in _mu_views}

    # per_step_lists[traj_name][step_idx][view] so retain MU can differ by trajectory type
    per_step_lists: Dict[str, Dict[int, Dict[str, Dict[str, List[Any]]]]] = {
        t: defaultdict(_empty_mu_pl) for t in trajectory_names
    }
    _retain_batches = len(retain_dataloader)
    _retain_log_interval = max(1, _retain_batches // 10) if _retain_batches else 1
    logger.info("Retain MU: %s batches to process", _retain_batches)

    for batch_idx, batch in enumerate(retain_dataloader):
        gpu_set_phase("retain_mu_batch", batch_idx=batch_idx)
        if batch_idx % _retain_log_interval == 0 or batch_idx == 0 or batch_idx == _retain_batches - 1:
            logger.info("Retain MU batch %s/%s", batch_idx + 1, _retain_batches)
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        _ = batch.get("attention_mask")  # reserved
        _ = batch.get("index", torch.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size))  # indices, reserved
        B = input_ids.shape[0]
        prompts, prompt_lens, prompt_starts = _build_prompts_for_sampler(
            input_ids, labels, tokenizer, IGNORE_INDEX,
            prompt_only_input_ids=getattr(data_retain, "predict_with_generate", False),
        )
        evaluation_mode = _sampler_kw.get("evaluation_mode", "unguided")
        sample_kw = dict(
            inputs=prompts,
            config=None,
            return_dict=True,
            return_logits=True,
            **_sampler_kw,
        )
        if evaluation_mode in ("guided_native", "guided_skew") and labels is not None:
            L_gen = _sampler_kw.get("max_new_tokens") or max(labels.shape[1] - prompt_starts[j] for j in range(B))
            target_sequences = _build_target_sequences_for_sampler(labels, prompt_starts, int(L_gen), IGNORE_INDEX)
            sample_kw["target_sequences"] = target_sequences
            sample_kw["evaluation_mode"] = evaluation_mode
        sampler_output = sampler.sample(**sample_kw)
        logits_history = sampler_output.logits_history
        fixation_steps = sampler_output.fixation_steps
        if logits_history is None or len(logits_history) == 0:
            logger.warning(
                "[retain_MU] batch_idx=%s skipped: no logits_history",
                batch_idx,
            )
            continue
        if fixation_steps is None:
            logger.warning(
                "[retain_MU] batch_idx=%s skipped: no fixation_steps",
                batch_idx,
            )
            continue
        out = trajectories_from_logits(
            logits_history, fixation_steps, prompt_lens, return_trajectory_tensors=False
        )
        R, F, S, L = out["R"], out["F"], out["S"], out["L"]
        del logits_history
        if _trajectory_should_empty_cuda_cache(trajectory_config):
            torch.cuda.empty_cache()
        if run_steps_to_use is None:
            run_steps_to_use, _ = _derive_steps_to_use(S, trajectory_config)
        steps_to_use = [s for s in run_steps_to_use if s < S]
        sequences = getattr(sampler_output, "sequences", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None) or (trajectory_config.get("eos_token_id") if trajectory_config else None)
        if sequences is not None and eos_token_id is not None:
            effective_lengths = effective_lengths_from_eos(sequences, prompt_lens, L, eos_token_id)
        else:
            effective_lengths = [L] * B

        for sample_idx in range(B):
            sample_traj = {"R": R[sample_idx], "F": F[sample_idx], "S": S, "L": L}
            L_eff_b = effective_lengths[sample_idx]
            sample_labels = labels[sample_idx] if labels is not None else None
            sample_prompt_start = prompt_starts[sample_idx]
            generated_labels = None
            if sample_labels is not None:
                generated_labels = sample_labels[sample_prompt_start : sample_prompt_start + L]
                if generated_labels.shape[0] < L:
                    padding = torch.full(
                        (L - generated_labels.shape[0],), IGNORE_INDEX,
                        dtype=generated_labels.dtype, device=generated_labels.device,
                    )
                    generated_labels = torch.cat([generated_labels, padding])
            generated_input_ids = input_ids[sample_idx][sample_prompt_start : sample_prompt_start + L]
            if generated_input_ids.shape[0] < L:
                padding = torch.zeros(L - generated_input_ids.shape[0], dtype=generated_input_ids.dtype, device=input_ids.device)
                generated_input_ids = torch.cat([generated_input_ids, padding])
            batch_template = {
                "input_ids": generated_input_ids.unsqueeze(0),
                "labels": generated_labels.unsqueeze(0) if generated_labels is not None else None,
                "attention_mask": torch.ones((1, L), dtype=torch.long, device=input_ids.device),
                "index": torch.tensor([0], dtype=torch.long, device=input_ids.device),
            }
            for key in ("labels_correct", "labels_wrong"):
                if key in batch:
                    batch_template[key] = _batch_template_dual_labels(
                        batch, sample_idx, key, L, IGNORE_INDEX
                    )
            ground_truth_str = ""
            if generated_labels is not None:
                valid = generated_labels[generated_labels != IGNORE_INDEX]
                ground_truth_str = tokenizer.decode(valid.tolist(), skip_special_tokens=True) if len(valid) > 0 else ""

            _rouge_prompt_dbg = tokenizer.decode(
                prompts[sample_idx], skip_special_tokens=True
            )
            kwargs_retain = {
                "ground_truth": ground_truth_str,
                "rouge_scorer": rouge_scorer_instance,
                "trajectory_config": trajectory_config,
                "rouge_debug_prompt_text": _rouge_prompt_dbg,
                **{k: v for k, v in kwargs.items() if k not in ("tokenizer", "model", "data")},
            }

            for traj_name in trajectory_names:
                for step_idx, step in enumerate(steps_to_use):
                    logits = _get_logits_at_step(sample_traj, traj_name, step)
                    if logits.dim() == 2:
                        logits = logits.transpose(0, 1).unsqueeze(0)
                    L_eff_slice = min(int(L_eff_b), L)
                    logits_eos = (
                        logits[:, :L_eff_slice] if L_eff_slice < logits.shape[1] else logits
                    )
                    batch_template_eos = (
                        _slice_batch_template_to_length(batch_template, L_eff_slice)
                        if L_eff_slice < L
                        else batch_template
                    )

                    for metric_name, key in (("probability", "prob"), ("rouge", "rouge"), ("truth_ratio", "tr")):
                        m = metric_objs.get(metric_name)
                        if m is None:
                            continue
                        cfg = metric_objs.get(f"{metric_name}_config") or {}
                        val_full: Optional[float] = None
                        try:
                            if "full" in _mu_views:
                                res_f = _call_metric_at_step(
                                    metric=m,
                                    logits=logits,
                                    batch_template=batch_template,
                                    tokenizer=tokenizer,
                                    sample_labels=generated_labels.unsqueeze(0)
                                    if generated_labels is not None
                                    else None,
                                    sample_input_ids=input_ids[sample_idx].unsqueeze(0),
                                    sample_prompt_len=sample_prompt_start,
                                    metric_config=cfg,
                                    sample_idx="0",
                                    step=step,
                                    step_index=step_idx,
                                    **kwargs_retain,
                                )
                                val_full = _extract_metric_scalar(res_f)
                                if val_full is not None:
                                    per_step_lists[traj_name][step_idx]["full"][key].append(val_full)
                            if "eos" in _mu_views:
                                if L_eff_slice >= L and val_full is not None:
                                    # Same sequence (no early EOS); reuse full result — correct, not hidden fallback
                                    per_step_lists[traj_name][step_idx]["eos"][key].append(val_full)
                                else:
                                    try:
                                        res_e = _call_metric_at_step(
                                            metric=m,
                                            logits=logits_eos,
                                            batch_template=batch_template_eos,
                                            tokenizer=tokenizer,
                                            sample_labels=generated_labels.unsqueeze(0)
                                            if generated_labels is not None
                                            else None,
                                            sample_input_ids=input_ids[sample_idx].unsqueeze(0),
                                            sample_prompt_len=sample_prompt_start,
                                            metric_config=cfg,
                                            sample_idx="0",
                                            step=step,
                                            step_index=step_idx,
                                            **kwargs_retain,
                                        )
                                        ve = _extract_metric_scalar(res_e)
                                        if ve is None:
                                            logger.warning(
                                                "retain MU eos: metric=%s diffusion_step=%s retain_step_idx=%s "
                                                "sample_idx=%s batch_idx=%s returned no scalar (not copying full view)",
                                                metric_name,
                                                step,
                                                step_idx,
                                                sample_idx,
                                                batch_idx,
                                            )
                                        else:
                                            per_step_lists[traj_name][step_idx]["eos"][key].append(ve)
                                    except Exception as ee:
                                        logger.warning(
                                            "retain MU eos: metric=%s diffusion_step=%s retain_step_idx=%s "
                                            "sample_idx=%s batch_idx=%s failed: %s",
                                            metric_name,
                                            step,
                                            step_idx,
                                            sample_idx,
                                            batch_idx,
                                            ee,
                                            exc_info=True,
                                        )
                        except Exception as e:
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug("retain MU %s at step %s: %s", metric_name, step, e)

        del R, F, out
        if _trajectory_should_empty_cuda_cache(trajectory_config):
            torch.cuda.empty_cache()

    for traj_name in trajectory_names:
        retain_agg_by_step[traj_name] = {}
        for step_idx in sorted(per_step_lists[traj_name].keys()):
            step_entry: Dict[str, Any] = {}
            for view in _mu_views:
                pl = per_step_lists[traj_name][step_idx][view]
                prob_vals = pl["prob"]
                rouge_vals = pl["rouge"]
                tr_vals = pl["tr"]
                agg_prob = float(np.mean(prob_vals)) if prob_vals else None
                agg_rouge = float(np.mean(rouge_vals)) if rouge_vals else None
                agg_tr = float(np.mean(tr_vals)) if tr_vals else None
                pre = {}
                if agg_prob is not None:
                    pre["retain_Q_A_Prob"] = {"agg_value": agg_prob}
                if agg_rouge is not None:
                    pre["retain_Q_A_ROUGE"] = {"agg_value": agg_rouge}
                if agg_tr is not None:
                    pre["retain_Truth_Ratio"] = {"agg_value": agg_tr}
                if pre:
                    step_entry[view] = pre
            if step_entry:
                retain_agg_by_step[traj_name][str(step_idx)] = step_entry

    n_steps = len(retain_agg_by_step.get("steps", {})) if retain_agg_by_step else 0
    logger.info("trajectory_model_utility: computed retain-set aggregates for %s steps x %s trajectory types (current model)", n_steps, len(trajectory_names))
    return retain_agg_by_step


# Metric key names per dataset for full 9-metric MU (paper/invariants).
_MU_DATASET_KEYS = {
    "retain": ("retain_Q_A_Prob", "retain_Q_A_ROUGE", "retain_Truth_Ratio"),
    "ra": ("ra_Q_A_Prob_normalised", "ra_Q_A_ROUGE", "ra_Truth_Ratio"),
    "wf": ("wf_Q_A_Prob_normalised", "wf_Q_A_ROUGE", "wf_Truth_Ratio"),
}
EXPECTED_9_MU_KEYS = frozenset(
    k for keys in _MU_DATASET_KEYS.values() for k in keys
)


def _validate_merged_9_mu(retain_agg_by_step: dict) -> None:
    """Raise ValueError if any step/view does not have exactly EXPECTED_9_MU_KEYS (for 9-metric path).
    Supports both flat {step_key: {view: {...}}} and per-traj {traj_name: {step_key: {view: {...}}}}.
    """
    if not retain_agg_by_step:
        return
    _traj_names = ("steps", "fixation_start", "fixation_end", "fixation_ratio")
    first_key = next(iter(retain_agg_by_step.keys()))
    if first_key in _traj_names and isinstance(retain_agg_by_step[first_key], dict):
        # Per-traj: validate each trajectory's step dict
        for traj_name in retain_agg_by_step:
            step_dict = retain_agg_by_step[traj_name]
            if not step_dict:
                continue
            first_step = next(iter(step_dict.values()))
            for view in ("full", "eos"):
                if view not in first_step:
                    continue
                keys = set(first_step[view].keys())
                if keys != EXPECTED_9_MU_KEYS:
                    missing = EXPECTED_9_MU_KEYS - keys
                    extra = keys - EXPECTED_9_MU_KEYS
                    raise ValueError(
                        "Trajectory MU 9-metric merge failed: expected exactly "
                        f"{EXPECTED_9_MU_KEYS!r}, got keys {keys!r} for traj_name={traj_name!r} view={view!r}; "
                        f"missing={missing!r} extra={extra!r}"
                    )
    else:
        # Flat: original behavior
        first_step = next(iter(retain_agg_by_step.values()))
        for view in ("full", "eos"):
            if view not in first_step:
                continue
            keys = set(first_step[view].keys())
            if keys != EXPECTED_9_MU_KEYS:
                missing = EXPECTED_9_MU_KEYS - keys
                extra = keys - EXPECTED_9_MU_KEYS
                raise ValueError(
                    "Trajectory MU 9-metric merge failed: expected exactly "
                    f"{EXPECTED_9_MU_KEYS!r}, got keys {keys!r}; "
                    f"missing={missing!r} extra={extra!r}"
                )


def _compute_mu_for_dataset(
    model: Any,
    data_dataset: Any,
    dataset_key: str,
    collator: Any,
    batch_size: int,
    trajectory_config: Union[Dict, DictConfig],
    tokenizer: Any,
    loaded_metrics: Dict[str, Any],
    sort_by_length: bool,
    use_distributed_sampler: bool,
    world_size: int,
    rank: int,
    **kwargs: Any,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Compute per-step, per-view aggregates for one MU dataset (retain, ra, or wf) for all trajectory types.
    Returns {traj_name: {step_index: {view: {metric_key: {"agg_value": v}}}}} with 3 keys per dataset.
    For retain: retain_Q_A_Prob, retain_Q_A_ROUGE, retain_Truth_Ratio.
    For ra/wf: *_Q_A_Prob_normalised, *_Q_A_ROUGE, *_Truth_Ratio.
    Probability: retain = raw P(correct); ra/wf = normalised correct/(correct+wrong+1e-10).
    Truth Ratio: TOFU definition wrong/correct per sample; aggregated with true_better → mean(max(0, 1 - tr)).
    """
    from evals.gpu_phase_logger import set_phase as gpu_set_phase

    trajectory_names = (
        list(trajectory_config.get("trajectory_names", ["steps", "fixation_start", "fixation_end", "fixation_ratio"]))
        if trajectory_config
        else ["steps"]
    )
    key_prob, key_rouge, key_tr = _MU_DATASET_KEYS[dataset_key]
    use_normalised_prob = dataset_key in ("ra", "wf")
    result_by_step: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    run_steps_to_use = None
    mu_sub_metrics = ("probability", "rouge", "truth_ratio")
    metric_objs = {}
    for name in mu_sub_metrics:
        if name in loaded_metrics:
            metric_objs[name] = loaded_metrics[name]["metric"]
            metric_objs[f"{name}_config"] = loaded_metrics[name].get("config") or {}
        else:
            try:
                metric_objs[name] = _get_metric_from_registry(name)
                metric_objs[f"{name}_config"] = {}
            except ValueError:
                metric_objs[name] = None
                metric_objs[f"{name}_config"] = {}
    if not all(metric_objs.get(m) for m in mu_sub_metrics):
        logger.warning(
            "trajectory_model_utility: missing probability/rouge/truth_ratio for dataset_key=%s; skipping",
            dataset_key,
        )
        return result_by_step

    if sort_by_length:
        dataloader = DataLoader(
            data_dataset,
            batch_size=batch_size,
            sampler=LengthSortedSampler(data_dataset),
            collate_fn=collator,
        )
    elif use_distributed_sampler:
        dataloader = DataLoader(
            data_dataset,
            batch_size=batch_size,
            sampler=DistributedSampler(data_dataset, num_replicas=world_size, rank=rank, shuffle=False),
            collate_fn=collator,
        )
    else:
        dataloader = DataLoader(data_dataset, batch_size=batch_size, collate_fn=collator)

    sampler = _get_sampler_from_model(model)
    if sampler is None:
        logger.warning(
            "trajectory_model_utility: no sampler on model for dataset_key=%s; skipping",
            dataset_key,
        )
        return result_by_step

    _sampler_kw = _trajectory_sampler_kwargs(trajectory_config)
    rouge_scorer_instance = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    # Use same default as main trajectory loop (["full", "eos"]) so MU pre-compute always produces
    # both views when the main loop will ask for both; default ["full"] alone caused full-view
    # hm_aggregate to be empty in reports when config had only eos or key was missing.
    _iv = trajectory_config.get("include_views", ["full", "eos"]) if trajectory_config else ["full", "eos"]
    if isinstance(_iv, str):
        _iv = [_iv]
    _mu_views = [str(v).lower() for v in _iv if str(v).lower() in ("full", "eos")]
    if not _mu_views:
        _mu_views = ["full", "eos"]
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[MU %s] resolved _mu_views=%s (dataset MU pre-compute; must match main trajectory include_views)",
            dataset_key,
            _mu_views,
        )

    def _empty_pl():
        return {"prob": [], "prob_wrong": [], "rouge": [], "tr": []}

    def _empty_step_view():
        return {v: _empty_pl() for v in _mu_views}

    per_step_lists: Dict[str, Dict[int, Dict[str, Dict[str, List[Any]]]]] = {
        t: defaultdict(_empty_step_view) for t in trajectory_names
    }
    n_batches = len(dataloader)
    log_interval = max(1, n_batches // 10) if n_batches else 1
    logger.info(
        "Trajectory MU: computing per-step aggregates for dataset %s (%s batches)",
        dataset_key,
        n_batches,
    )

    for batch_idx, batch in enumerate(dataloader):
        gpu_set_phase("mu_batch", metric=dataset_key, batch_idx=batch_idx)
        if batch_idx % log_interval == 0 or batch_idx == 0 or batch_idx == n_batches - 1:
            logger.info("Trajectory MU dataset %s batch %s/%s", dataset_key, batch_idx + 1, n_batches)
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        has_dual = "labels_correct" in batch and "labels_wrong" in batch
        B = input_ids.shape[0]
        prompts, prompt_lens, prompt_starts = _build_prompts_for_sampler(
            input_ids,
            labels,
            tokenizer,
            IGNORE_INDEX,
            prompt_only_input_ids=getattr(data_dataset, "predict_with_generate", False),
        )
        evaluation_mode = _sampler_kw.get("evaluation_mode", "unguided")
        sample_kw = dict(
            inputs=prompts,
            config=None,
            return_dict=True,
            return_logits=True,
            **_sampler_kw,
        )
        if evaluation_mode in ("guided_native", "guided_skew") and labels is not None:
            L_gen = _sampler_kw.get("max_new_tokens") or max(
                labels.shape[1] - prompt_starts[j] for j in range(B)
            )
            target_sequences = _build_target_sequences_for_sampler(
                labels, prompt_starts, int(L_gen), IGNORE_INDEX
            )
            sample_kw["target_sequences"] = target_sequences
            sample_kw["evaluation_mode"] = evaluation_mode
        sampler_output = sampler.sample(**sample_kw)
        logits_history = sampler_output.logits_history
        fixation_steps = sampler_output.fixation_steps
        if logits_history is None or len(logits_history) == 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[MU %s] batch_idx=%s: no logits_history", dataset_key, batch_idx)
            continue
        if fixation_steps is None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[MU %s] batch_idx=%s: no fixation_steps", dataset_key, batch_idx)
            continue
        out = trajectories_from_logits(
            logits_history, fixation_steps, prompt_lens, return_trajectory_tensors=False
        )
        R, F, S, L = out["R"], out["F"], out["S"], out["L"]
        del logits_history
        if _trajectory_should_empty_cuda_cache(trajectory_config):
            torch.cuda.empty_cache()
        if run_steps_to_use is None:
            run_steps_to_use, _ = _derive_steps_to_use(S, trajectory_config)
        steps_to_use = [s for s in run_steps_to_use if s < S]
        sequences = getattr(sampler_output, "sequences", None)
        eos_token_id = (
            getattr(tokenizer, "eos_token_id", None)
            or (trajectory_config.get("eos_token_id") if trajectory_config else None)
        )
        if sequences is not None and eos_token_id is not None:
            effective_lengths = effective_lengths_from_eos(
                sequences, prompt_lens, L, eos_token_id
            )
        else:
            effective_lengths = [L] * B

        for sample_idx in range(B):
            sample_traj = {"R": R[sample_idx], "F": F[sample_idx], "S": S, "L": L}
            L_eff_b = effective_lengths[sample_idx]
            sample_labels = labels[sample_idx] if labels is not None else None
            sample_prompt_start = prompt_starts[sample_idx]
            generated_labels = None
            if sample_labels is not None:
                generated_labels = sample_labels[
                    sample_prompt_start : sample_prompt_start + L
                ]
                if generated_labels.shape[0] < L:
                    padding = torch.full(
                        (L - generated_labels.shape[0],),
                        IGNORE_INDEX,
                        dtype=generated_labels.dtype,
                        device=generated_labels.device,
                    )
                    generated_labels = torch.cat([generated_labels, padding])
            generated_input_ids = input_ids[sample_idx][
                sample_prompt_start : sample_prompt_start + L
            ]
            if generated_input_ids.shape[0] < L:
                padding = torch.zeros(
                    L - generated_input_ids.shape[0],
                    dtype=generated_input_ids.dtype,
                    device=input_ids.device,
                )
                generated_input_ids = torch.cat([generated_input_ids, padding])
            batch_template = {
                "input_ids": generated_input_ids.unsqueeze(0),
                "labels": generated_labels.unsqueeze(0) if generated_labels is not None else None,
                "attention_mask": torch.ones((1, L), dtype=torch.long, device=input_ids.device),
                "index": torch.tensor([0], dtype=torch.long, device=input_ids.device),
            }
            for key in ("labels_correct", "labels_wrong"):
                if key in batch:
                    batch_template[key] = _batch_template_dual_labels(
                        batch, sample_idx, key, L, IGNORE_INDEX
                    )
            ground_truth_str = ""
            if generated_labels is not None:
                valid = generated_labels[generated_labels != IGNORE_INDEX]
                ground_truth_str = (
                    tokenizer.decode(valid.tolist(), skip_special_tokens=True)
                    if len(valid) > 0
                    else ""
                )
            _rouge_prompt_dbg_mu = tokenizer.decode(
                prompts[sample_idx], skip_special_tokens=True
            )
            kwargs_mu = {
                "ground_truth": ground_truth_str,
                "rouge_scorer": rouge_scorer_instance,
                "trajectory_config": trajectory_config,
                "dataset_key": dataset_key,
                "last_step_index": len(steps_to_use) - 1,
                "rouge_debug_prompt_text": _rouge_prompt_dbg_mu,
                **{k: v for k, v in kwargs.items() if k not in ("tokenizer", "model", "data")},
            }

            for step_idx, step in enumerate(steps_to_use):
                prob_obj = metric_objs.get("probability")
                rouge_obj = metric_objs.get("rouge")
                cfg_prob = metric_objs.get("probability_config") or {}
                cfg_rouge = metric_objs.get("rouge_config") or {}

                def _run_prob(bt, log, vname):
                    res = _call_metric_at_step(
                        metric=prob_obj,
                        logits=log,
                        batch_template=bt,
                        tokenizer=tokenizer,
                        sample_labels=generated_labels.unsqueeze(0)
                        if generated_labels is not None
                        else None,
                        sample_input_ids=input_ids[sample_idx].unsqueeze(0),
                        sample_prompt_len=sample_prompt_start,
                        metric_config=cfg_prob,
                        sample_idx="0",
                        step=step,
                        step_index=step_idx,
                        **kwargs_mu,
                    )
                    return _extract_metric_scalar(res)

                def _run_rouge(bt, log, vname):
                    res = _call_metric_at_step(
                        metric=rouge_obj,
                        logits=log,
                        batch_template=bt,
                        tokenizer=tokenizer,
                        sample_labels=generated_labels.unsqueeze(0)
                        if generated_labels is not None
                        else None,
                        sample_input_ids=input_ids[sample_idx].unsqueeze(0),
                        sample_prompt_len=sample_prompt_start,
                        metric_config=cfg_rouge,
                        sample_idx="0",
                        step=step,
                        step_index=step_idx,
                        **kwargs_mu,
                    )
                    return _extract_metric_scalar(res)

                for traj_name in trajectory_names:
                    logits = _get_logits_at_step(sample_traj, traj_name, step)
                    if logits.dim() == 2:
                        logits = logits.transpose(0, 1).unsqueeze(0)
                    L_eff_slice = min(int(L_eff_b), L)
                    logits_eos = (
                        logits[:, :L_eff_slice]
                        if L_eff_slice < logits.shape[1]
                        else logits
                    )
                    batch_template_eos = (
                        _slice_batch_template_to_length(batch_template, L_eff_slice)
                        if L_eff_slice < L
                        else batch_template
                    )
                    for view_name in _mu_views:
                        bt = batch_template if view_name == "full" else batch_template_eos
                        lg = logits if view_name == "full" else logits_eos
                        pl = per_step_lists[traj_name][step_idx][view_name]
                        if logger.isEnabledFor(logging.DEBUG) and (
                            batch_idx == 0 and sample_idx == 0 and step_idx == 0
                        ):
                            logger.debug(
                                "[MU %s] view_loop traj=%s step_idx=%s diffusion_step=%s view=%s "
                                "logits_shape=%s L_eff_slice=%s pl_id=%s",
                                dataset_key,
                                traj_name,
                                step_idx,
                                step,
                                view_name,
                                tuple(lg.shape),
                                L_eff_slice,
                                id(pl),
                            )
                        try:
                            if has_dual:
                                labels_correct_slice = bt.get("labels_correct")
                                labels_wrong_raw = bt.get("labels_wrong")
                                if labels_wrong_raw is not None and labels_correct_slice is not None:
                                    bt_correct = dict(bt)
                                    bt_correct["labels"] = (
                                        labels_correct_slice
                                        if not isinstance(labels_correct_slice, list)
                                        else labels_correct_slice[0]
                                    )
                                    pc = _run_prob(bt_correct, lg, view_name)
                                    if isinstance(labels_wrong_raw, list):
                                        wrong_tensors = labels_wrong_raw
                                    else:
                                        wrong_tensors = [labels_wrong_raw]
                                    if logger.isEnabledFor(logging.DEBUG):
                                        logger.debug(
                                            "[MU %s] dual-label probs: N_wrong=%s view=%s traj=%s "
                                            "step_idx=%s batch=%s sample=%s",
                                            dataset_key,
                                            len(wrong_tensors),
                                            view_name,
                                            traj_name,
                                            step_idx,
                                            batch_idx,
                                            sample_idx,
                                        )
                                    pw_list: list[float] = []
                                    for k_wrong, wt in enumerate(wrong_tensors):
                                        bt_wrong = dict(bt)
                                        lab = (
                                            wt.unsqueeze(0)
                                            if isinstance(wt, torch.Tensor) and wt.dim() == 1
                                            else wt
                                        )
                                        bt_wrong["labels"] = lab
                                        pw_k = _run_prob(bt_wrong, lg, view_name)
                                        if pw_k is not None:
                                            pw_list.append(float(pw_k))
                                        if logger.isEnabledFor(logging.DEBUG):
                                            logger.debug(
                                                "[MU %s] wrong option k=%s/%s prob_scalar=%s",
                                                dataset_key,
                                                k_wrong,
                                                len(wrong_tensors),
                                                pw_k,
                                            )
                                    if pc is not None and pw_list:
                                        mean_pw = float(
                                            np.mean(np.array(pw_list, dtype=np.float64))
                                        )
                                        if logger.isEnabledFor(logging.DEBUG):
                                            logger.debug(
                                                "[MU %s] mean over wrong probs: valid=%s/%s "
                                                "mean_pw=%s pc=%s normalised=%s",
                                                dataset_key,
                                                len(pw_list),
                                                len(wrong_tensors),
                                                mean_pw,
                                                pc,
                                                use_normalised_prob,
                                            )
                                        norm = pc / (pc + mean_pw + 1e-10)
                                        # TR: TOFU wrong/correct with wrong = mean_k P(wrong_k) (same contract as truth_ratio multi-wrong).
                                        tr_ratio = mean_pw / (pc + 1e-10)
                                        pl["tr"].append(tr_ratio)
                                        if use_normalised_prob:
                                            pl["prob"].append(norm)
                                        else:
                                            pl["prob"].append(pc)
                                    elif pc is not None and not use_normalised_prob:
                                        pl["prob"].append(pc)
                                        if logger.isEnabledFor(logging.DEBUG):
                                            logger.debug(
                                                "[MU %s] dual-label: pc present but no valid wrong "
                                                "prob (%s options tried); skip TR this cell",
                                                dataset_key,
                                                len(wrong_tensors),
                                            )
                                else:
                                    pc = _run_prob(bt, lg, view_name)
                                    if pc is not None:
                                        pl["prob"].append(pc)
                                        if not has_dual:
                                            pl["tr"].append(pc)
                            else:
                                pc = _run_prob(bt, lg, view_name)
                                if pc is not None:
                                    pl["prob"].append(pc)
                                    pl["tr"].append(pc)
                            rv = _run_rouge(bt, lg, view_name)
                            if rv is not None:
                                pl["rouge"].append(rv)
                            if logger.isEnabledFor(logging.DEBUG) and (
                                batch_idx == 0 and sample_idx == 0 and step_idx == 0
                            ):
                                logger.debug(
                                    "[MU %s] after_append traj=%s view=%s pl_lens prob=%s rouge=%s tr=%s",
                                    dataset_key,
                                    traj_name,
                                    view_name,
                                    len(pl["prob"]),
                                    len(pl["rouge"]),
                                    len(pl["tr"]),
                                )
                        except Exception as e:
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(
                                    "[MU %s] metric_fail view=%s traj=%s diffusion_step=%s sample_idx=%s: %s",
                                    dataset_key,
                                    view_name,
                                    traj_name,
                                    step,
                                    sample_idx,
                                    e,
                                )

        del R, F, out
        if _trajectory_should_empty_cuda_cache(trajectory_config):
            torch.cuda.empty_cache()

    if logger.isEnabledFor(logging.DEBUG) and per_step_lists:
        for _tn in trajectory_names:
            _steps = per_step_lists.get(_tn) or {}
            if not _steps:
                logger.debug("[MU %s] pre_aggregate traj=%s: no steps in per_step_lists", dataset_key, _tn)
                continue
            _first_si = min(_steps.keys())
            for _vn in _mu_views:
                _pl = _steps[_first_si][_vn]
                logger.debug(
                    "[MU %s] pre_aggregate traj=%s step_idx=%s view=%s n_prob=%s n_rouge=%s n_tr=%s",
                    dataset_key,
                    _tn,
                    _first_si,
                    _vn,
                    len(_pl["prob"]),
                    len(_pl["rouge"]),
                    len(_pl["tr"]),
                )

    for traj_name in trajectory_names:
        result_by_step[traj_name] = {}
        for step_idx in sorted(per_step_lists[traj_name].keys()):
            step_entry = {}
            for view in _mu_views:
                pl = per_step_lists[traj_name][step_idx][view]
                agg_prob = float(np.mean(pl["prob"])) if pl["prob"] else None
                agg_rouge = float(np.mean(pl["rouge"])) if pl["rouge"] else None
                # TR: TOFU uses ratio wrong/correct; retain/ra/wf configs use aggregator true_better → mean(max(0, 1 - tr)).
                agg_tr = (
                    float(np.mean(np.maximum(0, 1 - np.array(pl["tr"], dtype=np.float64))))
                    if pl["tr"]
                    else None
                )
                pre = {}
                if agg_prob is not None:
                    pre[key_prob] = {"agg_value": agg_prob}
                if agg_rouge is not None:
                    pre[key_rouge] = {"agg_value": agg_rouge}
                if agg_tr is not None:
                    pre[key_tr] = {"agg_value": agg_tr}
                if pre:
                    step_entry[view] = pre
            if step_entry:
                result_by_step[traj_name][str(step_idx)] = step_entry

    if logger.isEnabledFor(logging.DEBUG):
        for _tn in trajectory_names:
            _rs = result_by_step.get(_tn) or {}
            if not _rs:
                logger.debug("[MU %s] post_aggregate traj=%s: empty result", dataset_key, _tn)
                continue
            _sk0 = min(_rs.keys(), key=lambda x: int(x))
            _ent = _rs[_sk0]
            logger.debug(
                "[MU %s] post_aggregate traj=%s first_step_key=%s views=%s keys_per_view=%s",
                dataset_key,
                _tn,
                _sk0,
                list(_ent.keys()),
                {v: sorted(_ent[v].keys()) for v in _ent if isinstance(_ent.get(v), dict)},
            )

    n_steps = len(next(iter(result_by_step.values()), {})) if result_by_step else 0
    logger.info(
        "Trajectory MU: computed %s aggregates for dataset %s (%s trajectory types, %s steps)",
        sum(len(v) for v in result_by_step.values()),
        dataset_key,
        len(result_by_step),
        n_steps,
    )
    return result_by_step


def _compute_pre_compute_metrics_at_step(
    pre_compute_config: Dict[str, Any],
    logits: torch.Tensor,
    batch_template: Dict[str, torch.Tensor],
    tokenizer: Any,
    sample_labels: Optional[torch.Tensor],
    sample_input_ids: torch.Tensor,
    sample_prompt_len: int,
    sample_idx: str,
    model_wrapper_override: Optional[Any] = None,
    trajectory_config: Optional[Dict[str, Any]] = None,
    sample_traj: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute pre-compute metrics at a specific trajectory step.

    Args:
        pre_compute_config: Config dict mapping pre_compute metric names to their configs
            Example: {"forget_Q_A_PARA_Prob": {"access_key": "correct", ...}}
        logits: [V, L] logits at the step
        batch_template: Template batch dict
        tokenizer: Tokenizer for text processing
        sample_labels: Labels for the sample
        sample_input_ids: Input IDs for the sample
        sample_prompt_len: Length of prompt
        sample_idx: Index string for this sample
        **kwargs: Additional kwargs to pass to pre_compute metrics.
            When ``traj_name`` is set (forget trajectory loop: ``steps``, ``fixation_start``,
            etc.), nested ``probability`` pre_compute uses these ``logits`` (trajectory-sliced)
            via ``_call_metric_at_step``, not ``FixationStepWiseScoreProvider`` on full ``R``/``F``.
            Full ``R``/``F`` generalized scoring is used only when ``traj_name`` is absent.

    Returns:
        Dict mapping access_key (or metric name) to metric results:
        {
            "correct": {"agg_value": ..., "value_by_index": {sample_idx: {...}}},
            "wrong": {"agg_value": ..., "value_by_index": {sample_idx: {...}}}
        }
    """
    trajectory_config = trajectory_config or kwargs.get("trajectory_config")
    sample_traj = sample_traj or kwargs.get("sample_traj")
    step = kwargs.get("step")
    pre_compute_results = {}
    # Normalize so truth_ratio always sees same key type (str) for correct vs wrong value_by_index.
    idx_key = str(sample_idx)

    def _pre_cfg_plain(cfg_any: Any) -> dict[str, Any]:
        if cfg_any is None:
            return {}
        if OmegaConf.is_config(cfg_any):
            c = OmegaConf.to_container(cfg_any, resolve=True)
            return dict(c) if isinstance(c, dict) else {}
        return dict(cfg_any) if isinstance(cfg_any, dict) else {}

    fused_precompute_skip: set[str] = set()
    _pc_get = getattr(pre_compute_config, "get", None)
    if (
        callable(_pc_get)
        and kwargs.get("traj_name") is None
        and trajectory_config is not None
        and bool(trajectory_config.get("use_generalized_sequence_probability", True))
        and sample_traj is not None
    ):
        c_raw = _pc_get("correct")
        w_raw = _pc_get("wrong")

        ccfg = _pre_cfg_plain(c_raw)
        wcfg = _pre_cfg_plain(w_raw)
        if (
            c_raw is not None
            and w_raw is not None
            and str(ccfg.get("handler") or "probability") == "probability"
            and str(wcfg.get("handler") or "probability") == "probability"
        ):
            lf_c = ccfg.get("labels_field") or "labels"
            lf_w = wcfg.get("labels_field") or "labels"
            if lf_c != lf_w:
                R_tr = sample_traj["R"]
                L_gen = int(R_tr.shape[1]) if R_tr.dim() >= 2 else 0
                lab_c = batch_template.get(lf_c)
                lab_w = batch_template.get(lf_w)
                acc_c = str(ccfg.get("access_key", "correct"))
                acc_w = str(wcfg.get("access_key", "wrong"))
                empty_pack = {
                    "agg_value": None,
                    "value_by_index": {idx_key: {"prob": None, "avg_loss": None}},
                }

                def _pack_prob_pre(scores_list: list[float]) -> dict[str, Any]:
                    if scores_list:
                        prob_val = sequence_probability_from_scores(scores_list)
                        avg_loss_val = float(-np.log(prob_val + 1e-12))
                        return {
                            "agg_value": prob_val,
                            "value_by_index": {
                                idx_key: {"prob": prob_val, "avg_loss": avg_loss_val},
                            },
                        }
                    return {
                        "agg_value": None,
                        "value_by_index": {idx_key: {"prob": None, "avg_loss": None}},
                    }

                if L_gen == 0:
                    pre_compute_results[acc_c] = empty_pack
                    if isinstance(lab_w, list):
                        pre_compute_results[acc_w] = [
                            {
                                "agg_value": None,
                                "value_by_index": {
                                    idx_key: {"prob": None, "avg_loss": None},
                                },
                            }
                            for _ in lab_w
                        ]
                    else:
                        pre_compute_results[acc_w] = empty_pack
                    fused_precompute_skip.update({"correct", "wrong"})
                elif (
                    not isinstance(lab_c, list)
                    and not isinstance(lab_w, list)
                    and lab_c is not None
                    and lab_w is not None
                ):
                    lc = lab_c.squeeze(0) if isinstance(lab_c, torch.Tensor) and lab_c.dim() > 1 else lab_c
                    lw = lab_w.squeeze(0) if isinstance(lab_w, torch.Tensor) and lab_w.dim() > 1 else lab_w
                    if isinstance(lc, torch.Tensor) and isinstance(lw, torch.Tensor) and lc.shape == lw.shape:
                        try:
                            logit_alignment = trajectory_config.get("logit_alignment", "causal")
                            provider = FixationStepWiseScoreProvider(logit_alignment=logit_alignment)
                            device_rf = R_tr.device
                            labels_2d = torch.stack(
                                [
                                    lc.to(device=device_rf, dtype=torch.long),
                                    lw.to(device=device_rf, dtype=torch.long),
                                ],
                                dim=0,
                            )
                            r0 = R_tr.unsqueeze(0)
                            f0 = sample_traj["F"]
                            f0u = f0.unsqueeze(0) if f0.dim() == 1 else f0
                            r_batch = r0.expand(2, *r0.shape[1:]).contiguous()
                            f_batch = f0u.expand(2, *f0u.shape[1:]).contiguous()
                            model_or_logits_fused: dict[str, Any] = {
                                "R": r_batch,
                                "F": f_batch,
                                "report_step": step,
                            }
                            results_pair = provider.get_per_position_scores(
                                model_or_logits_fused,
                                {"labels": labels_2d},
                                ignore_index=IGNORE_INDEX,
                            )
                        except Exception as e:
                            logger.warning(
                                "pre_compute fused probability (correct+wrong): %s "
                                "(sample_idx=%s step=%s); falling back to separate pre_compute",
                                e,
                                sample_idx,
                                step,
                                exc_info=True,
                            )
                        else:
                            if len(results_pair) >= 2:
                                s0 = results_pair[0][0] if results_pair[0] else []
                                s1 = results_pair[1][0] if results_pair[1] else []
                                pre_compute_results[acc_c] = _pack_prob_pre(s0)
                                pre_compute_results[acc_w] = _pack_prob_pre(s1)
                                fused_precompute_skip.update({"correct", "wrong"})

    # Trajectory loop (traj_name set): truth_ratio pre_compute uses step logits + shifted CE
    # (``compute_prob_from_fixation_logits``), not FixationStepWiseScoreProvider. Fuse correct
    # and wrong into one ``_compute_prob_packed_shifted_segments`` call (same numerics as two
    # nested ``_call_metric_at_step(probability)``).
    if (
        callable(_pc_get)
        and kwargs.get("traj_name") is not None
        and not fused_precompute_skip.intersection({"correct", "wrong"})
    ):
        c_raw_tr = _pc_get("correct")
        w_raw_tr = _pc_get("wrong")
        ccfg_tr = _pre_cfg_plain(c_raw_tr)
        wcfg_tr = _pre_cfg_plain(w_raw_tr)
        if (
            c_raw_tr is not None
            and w_raw_tr is not None
            and str(ccfg_tr.get("handler") or "probability") == "probability"
            and str(wcfg_tr.get("handler") or "probability") == "probability"
        ):
            lf_ctr = ccfg_tr.get("labels_field") or "labels"
            lf_wtr = wcfg_tr.get("labels_field") or "labels"
            if lf_ctr != lf_wtr:
                lab_ctr = batch_template.get(lf_ctr)
                lab_wtr = batch_template.get(lf_wtr)
                acc_ctr = str(ccfg_tr.get("access_key", "correct"))
                acc_wtr = str(wcfg_tr.get("access_key", "wrong"))

                def _pack_shifted_ce_pre(seg: dict[str, Any]) -> dict[str, Any]:
                    prob_v = seg.get("prob")
                    al_v = seg.get("avg_loss")
                    return {
                        "agg_value": prob_v,
                        "value_by_index": {
                            idx_key: {"prob": prob_v, "avg_loss": al_v},
                        },
                    }

                def _labels_batch_1l(x: torch.Tensor) -> Optional[torch.Tensor]:
                    if x.dim() == 1:
                        return x.unsqueeze(0)
                    if x.dim() == 2 and x.shape[0] == 1:
                        return x
                    return None

                logit_1lv_tr: Optional[torch.Tensor] = None
                if isinstance(logits, torch.Tensor):
                    if logits.dim() == 2:
                        logit_1lv_tr = logits.transpose(0, 1).unsqueeze(0)
                    elif logits.dim() == 3 and logits.shape[0] == 1:
                        logit_1lv_tr = logits

                if (
                    logit_1lv_tr is not None
                    and not isinstance(lab_ctr, list)
                    and not isinstance(lab_wtr, list)
                    and lab_ctr is not None
                    and lab_wtr is not None
                ):
                    lc_tr = (
                        lab_ctr.squeeze(0)
                        if isinstance(lab_ctr, torch.Tensor) and lab_ctr.dim() > 1
                        else lab_ctr
                    )
                    lw_tr = (
                        lab_wtr.squeeze(0)
                        if isinstance(lab_wtr, torch.Tensor) and lab_wtr.dim() > 1
                        else lab_wtr
                    )
                    if (
                        isinstance(lc_tr, torch.Tensor)
                        and isinstance(lw_tr, torch.Tensor)
                        and lc_tr.shape == lw_tr.shape
                    ):
                        lab_c_2d = _labels_batch_1l(lc_tr)
                        lab_w_2d = _labels_batch_1l(lw_tr)
                        L_log_tr = int(logit_1lv_tr.shape[1])
                        if (
                            lab_c_2d is not None
                            and lab_w_2d is not None
                            and lab_c_2d.shape[1] == L_log_tr
                            and lab_w_2d.shape[1] == L_log_tr
                        ):
                            try:
                                device_tr = logit_1lv_tr.device
                                lab_c_b = lab_c_2d.to(
                                    device=device_tr, dtype=torch.long
                                ).contiguous()
                                lab_w_b = lab_w_2d.to(
                                    device=device_tr, dtype=torch.long
                                ).contiguous()
                                log_b = logit_1lv_tr.contiguous()
                                results_tr = _compute_prob_packed_shifted_segments(
                                    [log_b, log_b],
                                    [lab_c_b, lab_w_b],
                                    device_tr,
                                    IGNORE_INDEX,
                                )
                            except Exception as e:
                                logger.warning(
                                    "pre_compute fused probability (traj step CE, correct+wrong): %s "
                                    "(sample_idx=%s step=%s); falling back to separate pre_compute",
                                    e,
                                    sample_idx,
                                    step,
                                    exc_info=True,
                                )
                            else:
                                if len(results_tr) == 2:
                                    pre_compute_results[acc_ctr] = _pack_shifted_ce_pre(
                                        results_tr[0]
                                    )
                                    pre_compute_results[acc_wtr] = _pack_shifted_ce_pre(
                                        results_tr[1]
                                    )
                                    fused_precompute_skip.update({"correct", "wrong"})

    for pre_metric_name, pre_metric_cfg in pre_compute_config.items():
        if pre_metric_name in fused_precompute_skip:
            continue
        # Get access key (defaults to metric name)
        access_key = pre_metric_cfg.get("access_key", pre_metric_name)
        # labels_field: use this batch key instead of "labels" for probability (e.g. labels_correct, labels_wrong)
        labels_field = pre_metric_cfg.get("labels_field")
        
        # Load pre-compute metric from registry
        # Import here to avoid circular import
        from evals.metrics import METRICS_REGISTRY
        
        # Try multiple strategies:
        # 1. pre_metric_name is the handler name
        # 2. pre_metric_cfg has a "handler" field
        # 3. pre_metric_name matches a registered metric
        pre_metric = None
        handler_name = None
        
        # Strategy 1: Check if pre_metric_name is a registered metric
        if pre_metric_name in METRICS_REGISTRY:
            pre_metric = METRICS_REGISTRY[pre_metric_name]
            handler_name = pre_metric_name
        # Strategy 2: Check if pre_metric_cfg has a handler field
        elif isinstance(pre_metric_cfg, (dict, DictConfig)) and "handler" in pre_metric_cfg:
            handler_name = pre_metric_cfg.get("handler")
            if handler_name in METRICS_REGISTRY:
                pre_metric = METRICS_REGISTRY[handler_name]
        
        if pre_metric is None:
            raise ValueError(
                f"Pre-compute metric '{pre_metric_name}' not found in registry. "
                f"Tried handler: {handler_name}. "
                f"Available metrics: {list(METRICS_REGISTRY.keys())}"
            )

        use_generalized = (
            trajectory_config is not None
            and trajectory_config.get("use_generalized_sequence_probability", True)
            and sample_traj is not None
            and handler_name == "probability"
            and kwargs.get("traj_name") is None
        )
        if use_generalized:
            try:
                R = sample_traj["R"]
                F = sample_traj["F"]
                L_gen = R.shape[1] if R.dim() >= 2 else 0
                # When generated length is 0, provider returns no scores; skip call and set None explicitly.
                if L_gen == 0:
                    lab = batch_template.get(labels_field if labels_field else "labels")
                    if isinstance(lab, list):
                        pre_compute_results[access_key] = [
                            {"agg_value": None, "value_by_index": {idx_key: {"prob": None, "avg_loss": None}}}
                            for _ in lab
                        ]
                    elif lab is not None:
                        pre_compute_results[access_key] = {
                            "agg_value": None,
                            "value_by_index": {idx_key: {"prob": None, "avg_loss": None}},
                        }
                    else:
                        pre_compute_results[access_key] = {
                            "agg_value": None,
                            "value_by_index": {idx_key: {"prob": None, "avg_loss": None}},
                        }
                    continue
                logit_alignment = trajectory_config.get("logit_alignment", "causal")
                provider = FixationStepWiseScoreProvider(logit_alignment=logit_alignment)
                lab = batch_template.get(labels_field if labels_field else "labels")

                if isinstance(lab, list):
                    wrong_results = []
                    for opt_i, lab_t in enumerate(lab):
                        lab_flat = lab_t.squeeze(0) if lab_t.dim() > 1 else lab_t
                        _ = (lab_flat != IGNORE_INDEX).sum().item()  # num_non_ignore, reserved
                        batch_prov = {"labels": lab_flat.unsqueeze(0)}
                        model_or_logits = {
                            "R": R.unsqueeze(0),
                            "F": F.unsqueeze(0),
                            "report_step": step,
                        }
                        results = provider.get_per_position_scores(
                            model_or_logits, batch_prov, ignore_index=IGNORE_INDEX
                        )
                        if results and results[0][0]:
                            prob_val = sequence_probability_from_scores(results[0][0])
                            avg_loss_val = float(-np.log(prob_val + 1e-12))
                            wrong_results.append({
                                "agg_value": prob_val,
                                "value_by_index": {
                                    idx_key: {"prob": prob_val, "avg_loss": avg_loss_val},
                                },
                            })
                        else:
                            wrong_results.append({
                                "agg_value": None,
                                "value_by_index": {idx_key: {"prob": None, "avg_loss": None}},
                            })
                    pre_compute_results[access_key] = wrong_results
                elif lab is not None:
                    lab = lab.squeeze(0) if lab.dim() > 1 else lab
                    _ = (lab != IGNORE_INDEX).sum().item()  # num_non_ignore, reserved
                    batch_prov = {"labels": lab.unsqueeze(0)}
                    model_or_logits = {
                        "R": R.unsqueeze(0),
                        "F": F.unsqueeze(0),
                        "report_step": step,
                    }
                    results = provider.get_per_position_scores(
                        model_or_logits, batch_prov, ignore_index=IGNORE_INDEX
                    )
                    if results and results[0][0]:
                        prob_val = sequence_probability_from_scores(results[0][0])
                        avg_loss_val = float(-np.log(prob_val + 1e-12))
                        pre_result = {
                            "agg_value": prob_val,
                            "value_by_index": {
                                idx_key: {"prob": prob_val, "avg_loss": avg_loss_val},
                            },
                        }
                    else:
                        logger.info(
                            "pre_compute probability (generalized): no scores from fixation "
                            "provider — empty scores or L_use=0 (sample_idx=%s, step=%s, labels_field=%s)",
                            sample_idx,
                            step,
                            labels_field,
                        )
                        pre_result = {
                            "agg_value": None,
                            "value_by_index": {idx_key: {"prob": None, "avg_loss": None}},
                        }
                    pre_compute_results[access_key] = pre_result
                else:
                    logger.info(
                        "pre_compute probability (generalized): labels missing "
                        "(labels_field=%s, sample_idx=%s, step=%s)",
                        labels_field,
                        sample_idx,
                        step,
                    )
                    pre_result = {
                        "agg_value": None,
                        "value_by_index": {idx_key: {"prob": None, "avg_loss": None}},
                    }
                    pre_compute_results[access_key] = pre_result
            except Exception as e:
                logger.warning(
                    "pre_compute probability (generalized): exception — %s (sample_idx=%s, step=%s, labels_field=%s)",
                    e,
                    sample_idx,
                    step,
                    labels_field,
                    exc_info=True,
                )
                pre_compute_results[access_key] = {
                    "agg_value": None,
                    "value_by_index": {idx_key: {"prob": None, "avg_loss": None}},
                }
            continue

        # Compute pre-compute metric at this step
        # Note: Pre-compute metrics might have their own pre_compute requirements
        # We handle this recursively
        try:
            labels_val = batch_template.get(labels_field) if labels_field else batch_template.get("labels")
            if labels_field and labels_field in batch_template and isinstance(labels_val, list):
                wrong_results = []
                for lab_tensor in labels_val:
                    pre_bt = {**batch_template, "labels": lab_tensor}
                    kwargs_clean = {k: v for k, v in kwargs.items() if k not in ("tokenizer", "model_wrapper_override")}
                    pre_result_k = _call_metric_at_step(
                        metric=pre_metric,
                        logits=logits,
                        batch_template=pre_bt,
                        tokenizer=tokenizer,
                        sample_labels=sample_labels,
                        sample_input_ids=sample_input_ids,
                        sample_prompt_len=sample_prompt_len,
                        metric_config=pre_metric_cfg,
                        sample_idx=sample_idx,
                        model_wrapper_override=model_wrapper_override,
                        **kwargs_clean
                    )
                    # truth_ratio expects list of N dicts with value_by_index keyed by sample_idx (same as correct).
                    if isinstance(pre_result_k, list) and len(pre_result_k) > 0 and isinstance(pre_result_k[0], dict):
                        first = pre_result_k[0]
                        pre_result_k = {
                            "value_by_index": {idx_key: first},
                            "agg_value": first.get("prob") if first.get("prob") is not None else first.get("avg_loss"),
                        }
                    else:
                        vbi = pre_result_k.get("value_by_index", {}) if isinstance(pre_result_k, dict) else {}
                        if vbi and idx_key not in vbi:
                            first_key = next(iter(vbi))
                            pre_result_k = dict(pre_result_k) if isinstance(pre_result_k, dict) else {}
                            pre_result_k["value_by_index"] = {idx_key: vbi[first_key]}
                        elif not vbi:
                            # truth_ratio requires wrong to have same value_by_index keys as correct.
                            pre_result_k = dict(pre_result_k) if isinstance(pre_result_k, dict) else {}
                            pre_result_k["value_by_index"] = {idx_key: {"prob": None, "avg_loss": None}}
                    wrong_results.append(pre_result_k)
                pre_compute_results[access_key] = wrong_results
                continue
            pre_batch_template = batch_template
            if labels_field and labels_field in batch_template:
                pre_batch_template = {
                    **batch_template,
                    "labels": batch_template[labels_field],
                }
            # Call the pre-compute metric at this step
            # Remove tokenizer from kwargs if present to avoid duplicate argument
            kwargs_clean = {k: v for k, v in kwargs.items() if k not in ("tokenizer", "model_wrapper_override")}
            pre_result = _call_metric_at_step(
                metric=pre_metric,
                logits=logits,
                batch_template=pre_batch_template,
                tokenizer=tokenizer,
                sample_labels=sample_labels,
                sample_input_ids=sample_input_ids,
                sample_prompt_len=sample_prompt_len,
                metric_config=pre_metric_cfg,
                sample_idx=sample_idx,
                model_wrapper_override=model_wrapper_override,
                **kwargs_clean
            )

            # When probability was called with 3D labels (e.g. [1,N,L]), it returns list of N lists;
            # truth_ratio expects wrong = list of N dicts with value_by_index keyed by same index as correct.
            if (
                access_key == "wrong"
                and isinstance(pre_result, list)
                and len(pre_result) > 0
                and isinstance(pre_result[0], list)
            ):
                wrong_results = []
                for k in range(len(pre_result)):
                    opt_list = pre_result[k]
                    first = opt_list[0] if opt_list and isinstance(opt_list[0], dict) else {}
                    wrong_results.append({
                        "value_by_index": {
                            idx_key: first if isinstance(first, dict) else {"prob": None, "avg_loss": None},
                        },
                        "agg_value": first.get("prob") if isinstance(first, dict) else first.get("avg_loss"),
                    })
                pre_compute_results[access_key] = wrong_results
                continue

            # Structure result in the format expected by main metrics
            # Main metrics expect: {"agg_value": ..., "value_by_index": {idx: {...}}}
            if isinstance(pre_result, dict):
                if "value_by_index" in pre_result:
                    # Normalize to single key idx_key so correct and wrong have same key type (trajectory single-sample).
                    # Metric may return value_by_index keyed by data index (int); truth_ratio requires same indices as wrong (str).
                    value_by_index = pre_result["value_by_index"]
                    if idx_key in value_by_index:
                        val = value_by_index[idx_key]
                        pre_result["value_by_index"] = {idx_key: val}
                    elif len(value_by_index) > 0:
                        first_idx = list(value_by_index.keys())[0]
                        val = value_by_index[first_idx].copy() if isinstance(value_by_index[first_idx], dict) else {"prob": pre_result.get("agg_value"), "avg_loss": None}
                        pre_result["value_by_index"] = {idx_key: val}
                    else:
                        # Empty value_by_index: do not add a placeholder for forget_truth_ratio → ks_test.
                        # ks_test expects entries to have "score"; a placeholder with only "prob"/"avg_loss" causes KeyError.
                        if not (access_key == "forget" and handler_name == "truth_ratio"):
                            val = {"prob": pre_result.get("agg_value"), "avg_loss": None}
                            pre_result["value_by_index"] = {idx_key: val}
                        # else: leave value_by_index as {} so ks_test gets [] and returns pvalue=None without KeyError
                elif "agg_value" in pre_result:
                    # Create value_by_index with single entry
                    value_by_index = {idx_key: {"prob": pre_result["agg_value"]}}
                    pre_result["value_by_index"] = value_by_index
                else:
                    # Try to extract value from result
                    value = None
                    for key in ["prob", "score", "value"]:
                        if key in pre_result:
                            value = pre_result[key]
                            break
                    if value is None:
                        # Use first numeric value
                        for key, val in pre_result.items():
                            if isinstance(val, (int, float, np.number)):
                                value = float(val)
                                break
                    if value is not None:
                        value_by_index = {idx_key: {"prob": value}}
                        pre_result = {
                            "agg_value": value,
                            "value_by_index": value_by_index,
                        }
                    else:
                        logger.warning(
                            f"Could not extract value from pre-compute metric {pre_metric_name} result: {pre_result}"
                        )
                        pre_result = {
                            "agg_value": None,
                            "value_by_index": {idx_key: {"prob": None}},
                        }
            elif isinstance(pre_result, list) and len(pre_result) > 0:
                # List format - extract first result (e.g., from evaluate_probability)
                result_dict = pre_result[0] if isinstance(pre_result[0], dict) else {}
                # Preserve all fields from result_dict (e.g., prob, avg_loss)
                value = None
                for key in ["prob", "score", "value"]:
                    if key in result_dict:
                        value = result_dict[key]
                        break
                if value is None:
                    # Use first numeric value
                    for key, val in result_dict.items():
                        if isinstance(val, (int, float, np.number)):
                            value = float(val)
                            break
                if value is not None:
                    # Preserve all fields from result_dict in value_by_index
                    pre_result = {
                        "agg_value": value,
                        "value_by_index": {idx_key: result_dict.copy()},
                    }
                else:
                    pre_result = {
                        "agg_value": None,
                        "value_by_index": {idx_key: {"prob": None}},
                    }
            else:
                logger.warning(
                    f"Unexpected pre-compute result format for {pre_metric_name}: {type(pre_result)}"
                )
                pre_result = {
                    "agg_value": None,
                    "value_by_index": {idx_key: {"prob": None}},
                }
            
            pre_compute_results[access_key] = pre_result

        except Exception as e:
            from evals.metrics.base import RetainReferenceValidationError
            if isinstance(e, RetainReferenceValidationError):
                raise
            logger.warning(
                f"Error computing pre-compute metric {pre_metric_name} at step: {e}",
                exc_info=True
            )
            # Return None result so main metric can handle it
            pre_compute_results[access_key] = {
                "agg_value": None,
                "value_by_index": {idx_key: {"prob": None}},
            }
    
    return pre_compute_results


def _handle_text_based_metric(logits, tokenizer, sample_labels, sample_input_ids, sample_prompt_len, metric_name, metric_config, **kwargs):
    """
    Generic handler for text-based metrics that require model.generate().
    Decodes logits to text and computes text similarity metrics.

    When ground_truth and rouge_scorer are provided (trajectory path), uses ROUGE-only path:
    calls eval_rouge_recall_batch(gen_text, ground_truth) directly instead of eval_text_similarity,
    avoiding redundant decodes and fake model.generate().

    Args:
        logits: [V, L] logits tensor
        tokenizer: Tokenizer for decoding
        sample_labels: Labels for ground truth extraction (used when ground_truth not in kwargs)
        sample_input_ids: Input IDs for the sample
        sample_prompt_len: Length of prompt
        metric_name: Name of the metric (e.g., "rouge")
        metric_config: Config for this metric
        **kwargs: Additional kwargs including generation_args, ground_truth, rouge_scorer

    Returns:
        Metric result (list of dicts)
    """
    # Decode logits to text via argmax (trajectory uses [V, L]; LM-style tensors use [L, V]).
    if logits.dim() == 3:
        logits = logits[0]
    if logits.dim() != 2:
        raise ValueError(f"text metric decode expects 2D logits after squeeze, got {tuple(logits.shape)}")
    r0, r1 = int(logits.shape[0]), int(logits.shape[1])
    if r0 > r1:
        predicted_tokens = torch.argmax(logits, dim=0)
    else:
        predicted_tokens = torch.argmax(logits, dim=-1)
    gen_text = tokenizer.decode(predicted_tokens.tolist(), skip_special_tokens=True)
    guardrail_config = kwargs.get("guardrail_config")
    if guardrail_config is not None:
        gen_text = transform_output_text(
            gen_text,
            kwargs.get("batch") or {},
            kwargs.get("sample_idx"),
            {"guardrail": guardrail_config},
        )

    ground_truth = kwargs.get("ground_truth")
    rouge_scorer_instance = kwargs.get("rouge_scorer")
    use_rouge_only = (
        metric_name == "rouge"
        and ground_truth is not None
        and rouge_scorer_instance is not None
    )

    if use_rouge_only:
        from evals.metrics.utils import eval_rouge_recall_batch

        # DEBUG-only: log prompt/gen/gt for first and last step, sample 0 and 1 (once in a few samples per trajectory).
        if logger.isEnabledFor(logging.DEBUG):
            _step_index = kwargs.get("step_index")
            _last_step_index = kwargs.get("last_step_index")
            _sidx = kwargs.get("sample_idx")
            _first_step = _step_index == 0 or (_step_index is not None and _step_index == 0)
            _is_last_step = (
                _step_index is not None
                and _last_step_index is not None
                and _step_index == _last_step_index
            )
            _log_sample = _sidx in (0, "0", 1, "1")
            if (_first_step or _is_last_step) and _log_sample:
                _max_len = 100
                _gen_snippet = (gen_text or "")[:_max_len] + ("..." if len(gen_text or "") > _max_len else "")
                _gt_snippet = (ground_truth or "")[:_max_len] + ("..." if len(ground_truth or "") > _max_len else "")
                _prompt_snippet = ""
                _override = kwargs.get("rouge_debug_prompt_text")
                if isinstance(_override, str) and _override.strip():
                    _prompt_snippet = _override[:_max_len] + (
                        "..." if len(_override) > _max_len else ""
                    )
                else:
                    _sid = kwargs.get("sample_input_ids")
                    _spl = kwargs.get("sample_prompt_len")
                    if _sid is not None and _spl is not None:
                        try:
                            _pl = int(_spl) if not hasattr(_spl, "item") else int(
                                _spl.item()
                            )
                            _ids = _sid[0] if _sid.dim() > 1 else _sid
                            if hasattr(_ids, "tolist"):
                                _ids = _ids.tolist()
                            if _pl > 0:
                                _prompt_text = tokenizer.decode(
                                    _ids[:_pl], skip_special_tokens=True
                                )
                                _prompt_snippet = (
                                    (
                                        _prompt_text[:_max_len]
                                        + (
                                            "..."
                                            if len(_prompt_text) > _max_len
                                            else ""
                                        )
                                    )
                                    if _prompt_text
                                    else ""
                                )
                        except Exception:  # noqa: S110
                            _prompt_snippet = "(decode failed)"
                _ds_key = kwargs.get("dataset_key")
                _step = kwargs.get("step")
                logger.debug(
                    "ROUGE decoded sample (dataset_key=%s step_index=%s step=%s sample_idx=%s): prompt=%r gen=%r gt=%r",
                    _ds_key,
                    _step_index,
                    _step,
                    _sidx,
                    _prompt_snippet,
                    _gen_snippet,
                    _gt_snippet,
                )

        result = eval_rouge_recall_batch(
            [gen_text],
            [ground_truth],
            use_stemmer=True,
            scorer=rouge_scorer_instance,
        )
        rouge_type = metric_config.get("rouge_type") or kwargs.get("rouge_type", "rougeL_f1")
        if isinstance(result, list) and len(result) > 0:
            result_dict = result[0]
            if isinstance(result_dict, dict) and rouge_type in result_dict:
                score = result_dict[rouge_type]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "ROUGE %s (rouge-only path): gen_len=%s gt_len=%s score=%s",
                        rouge_type,
                        len(gen_text),
                        len(ground_truth),
                        score,
                    )
                return [{"score": score}]
        return result

    # Fallback: extract ground truth from labels if not provided
    if ground_truth is None:
        if sample_labels is not None:
            valid_labels = sample_labels[sample_labels != IGNORE_INDEX]
            if len(valid_labels) > 0:
                ground_truth = tokenizer.decode(valid_labels.tolist(), skip_special_tokens=True)
            else:
                ground_truth = ""
        else:
            ground_truth = ""

    # Create a model that returns our decoded text when generate() is called
    class TextFromLogitsModel:
        def __init__(self, gen_text, tokenizer):
            self.gen_text = gen_text
            self.tokenizer = tokenizer
            self.device = "cpu"

        def generate(self, input_ids, attention_mask=None, **kwargs):
            # Return tokens that decode to our generated text
            gen_tokens = self.tokenizer.encode(self.gen_text, return_tensors="pt", add_special_tokens=False)
            # Concatenate with input_ids to match expected format
            return torch.cat([input_ids, gen_tokens], dim=1)

    text_model = TextFromLogitsModel(gen_text, tokenizer)

    # Create batch in format expected by eval_text_similarity
    text_batch = {
        "input_ids": sample_input_ids.unsqueeze(0) if sample_input_ids is not None else torch.zeros(1, 1, dtype=torch.long, device=logits.device),
        "labels": sample_labels.unsqueeze(0) if sample_labels is not None else None,
        "attention_mask": torch.ones(1, sample_input_ids.shape[0] if sample_input_ids is not None else 1, dtype=torch.long, device=logits.device),
    }

    # Call eval_text_similarity which is used by most text-based metrics
    from evals.metrics.utils import eval_text_similarity
    from omegaconf import OmegaConf
    generation_args = kwargs.get("generation_args", {})
    # Convert to OmegaConf if it's a dict (eval_text_similarity expects OmegaConf)
    if isinstance(generation_args, dict) and not isinstance(generation_args, DictConfig):
        generation_args = OmegaConf.create(generation_args)
    result = eval_text_similarity(text_model, tokenizer, text_batch, generation_args)

    # Post-process result based on metric type
    if metric_name == "rouge":
        rouge_type = metric_config.get("rouge_type") or kwargs.get("rouge_type", "rougeL_f1")
        if isinstance(result, list) and len(result) > 0:
            result_dict = result[0]
            if isinstance(result_dict, dict) and rouge_type in result_dict:
                score = result_dict[rouge_type]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "ROUGE %s: gen_len=%s gt_len=%s score=%s",
                        rouge_type,
                        len(gen_text),
                        len(ground_truth),
                        score,
                    )
                return [{"score": score}]

    return result


def _call_metric_at_step(
    metric: Any,
    logits: torch.Tensor,
    batch_template: Dict[str, torch.Tensor],
    tokenizer: Any = None,
    sample_labels: Optional[torch.Tensor] = None,
    sample_input_ids: Optional[torch.Tensor] = None,
    sample_prompt_len: int = 0,
    metric_config: Optional[Dict[str, Any]] = None,
    sample_idx: Optional[str] = None,
    model_wrapper_override: Optional[Any] = None,
    **kwargs
) -> Any:
    """
    Call a metric function at a specific trajectory step.
    
    Args:
        metric: UnlearningMetric object from registry
        logits: [V, L] logits at the step
        batch_template: Template batch dict
        tokenizer: Tokenizer for text processing (can also be in kwargs)
        sample_labels: Labels for the sample
        sample_input_ids: Input IDs for the sample
        sample_prompt_len: Length of prompt
        metric_config: Config for this specific metric (may include pre_compute, etc.)
        sample_idx: Index string for this sample (used for pre_compute value_by_index)
        **kwargs: Additional kwargs to pass to metric (may contain tokenizer)
    
    Returns:
        Metric result (typically dict with metric values)
    """
    # Extract tokenizer from kwargs if not provided explicitly (handle duplicate)
    if tokenizer is None:
        tokenizer = kwargs.pop("tokenizer", None)
    else:
        # Remove tokenizer from kwargs if present to avoid issues downstream
        kwargs.pop("tokenizer", None)
    
    # Set defaults
    if metric_config is None:
        metric_config = {}
    if sample_input_ids is None:
        sample_input_ids = torch.zeros(1, dtype=torch.long)
    
    # Ensure logits are in [B, L, V] format (B may be T>1 for batched trajectory steps).
    if logits.dim() == 2:
        # [V, L] -> transpose to [L, V] then add batch dim -> [1, L, V]
        logits = logits.transpose(0, 1).unsqueeze(0)
    elif logits.dim() == 3 and logits.shape[0] == 1:
        # [1, L, V] - already correct
        pass
    elif logits.dim() == 3 and logits.shape[0] > 1:
        # [T, L, V] — batched diffusion steps (callers must not use pre_compute with T>1).
        pass
    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")

    batched_steps = logits.shape[0] > 1
    if batched_steps:
        batch_template = _expand_batch_template_leading_dim(batch_template, logits.shape[0])

    # Create model wrapper (or use override for DualLogitModelWrapper e.g. mia_min_k)
    device = logits.device if model_wrapper_override is None else model_wrapper_override.device
    model_wrapper = model_wrapper_override if model_wrapper_override is not None else LogitModelWrapper(logits, device)

    # Prepare batch
    batch = {}
    for key, value in batch_template.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        else:
            batch[key] = value

    # Handle pre_compute metrics if present (use get: DictConfig in struct mode does not support pop)
    pre_compute_config = metric_config.get("pre_compute", {})
    pre_compute_results = {}
    if pre_compute_config:
        if batched_steps and not kwargs.get("trajectory_allow_batched_pre_compute"):
            raise ValueError(
                "Batched trajectory logits (shape[0]>1) are not supported with pre_compute; "
                "call _call_metric_at_step per step for this metric, or pass "
                "trajectory_allow_batched_pre_compute=True for an explicitly allowlisted batched path."
            )
        if sample_idx is None:
            sample_idx = "0"  # Default index
        pre_compute_results = _compute_pre_compute_metrics_at_step(
            pre_compute_config=pre_compute_config,
            logits=logits,
            batch_template=batch_template,
            tokenizer=tokenizer,
            sample_labels=sample_labels,
            sample_input_ids=sample_input_ids,
            sample_prompt_len=sample_prompt_len,
            sample_idx=sample_idx,
            **kwargs
        )
    
    # Prepare kwargs for metric function
    # Remove model, tokenizer, pre_compute, and reference_logs from kwargs (we pass computed pre_compute and step-specific reference_logs below)
    kwargs_clean = {k: v for k, v in kwargs.items() if k not in ["model", "tokenizer", "pre_compute", "reference_logs"]}
    metric_config_no_precompute = {k: v for k, v in metric_config.items() if k != "pre_compute"}
    metric_kwargs = {
        "model": model_wrapper,
        "batch": batch,
        "tokenizer": tokenizer,
        **metric_config_no_precompute,  # Include metric-specific config (aggregator, etc.)
        **kwargs_clean,  # Include any additional kwargs (excluding model/tokenizer)
    }
    
    # Add pre_compute results if available
    if pre_compute_results:
        metric_kwargs["pre_compute"] = pre_compute_results
    # trajectory_model_utility: use current model's MU aggregates at this step (3 or 9 keys), not reference_logs
    elif metric.name == "hm_aggregate":
        retain_agg_by_step = kwargs.get("retain_agg_by_step") or {}
        step_index = kwargs.get("step_index")

        def _is_mu_component_key(k):
            return (
                str(k).startswith("retain_")
                or str(k).startswith("ra_")
                or str(k).startswith("wf_")
            )

        if step_index is not None and retain_agg_by_step:
            traj_name = kwargs.get("traj_name", "steps")
            _first = next(iter(retain_agg_by_step.keys()), None)
            _is_per_traj = _first in ("steps", "fixation_start", "fixation_end", "fixation_ratio") and isinstance(
                retain_agg_by_step.get(_first), dict
            )
            by_traj = retain_agg_by_step if _is_per_traj else {"steps": retain_agg_by_step}
            pre_compute_step = (by_traj.get(traj_name) or {}).get(str(step_index)) or (by_traj.get(traj_name) or {}).get(step_index)
            if pre_compute_step:
                _nested_views = False
                if isinstance(pre_compute_step, dict):
                    for _vx in ("full", "eos"):
                        _inner = pre_compute_step.get(_vx)
                        if isinstance(_inner, dict) and any(_is_mu_component_key(k) for k in _inner):
                            _nested_views = True
                            break
                if _nested_views:
                    view = kwargs.get("trajectory_view")
                    if view not in ("full", "eos"):
                        raise ValueError(
                            "hm_aggregate: trajectory_view must be 'full' or 'eos' when "
                            "retain_agg_by_step stores per-view aggregates (no default)."
                        )
                    _inner = pre_compute_step.get(view)
                    if isinstance(_inner, dict) and any(_is_mu_component_key(k) for k in _inner):
                        metric_kwargs["pre_compute"] = _inner
                    else:
                        # Empty or no MU keys for this view/step: pass no pre_compute so hm_aggregate returns None
                        metric_kwargs["pre_compute"] = {}
                else:
                    metric_kwargs["pre_compute"] = pre_compute_step

    # Step-matched retain reference: privleak uses retain (from retain_mia_by_step), ks_test uses retain_ftr (from retain_forget_tr_by_step).
    if metric.name in ("privleak", "ks_test"):
        ref_logs = kwargs.get("reference_logs") or {}
        retain_logs = ref_logs.get("retain_model_logs") or {}
        step_index = kwargs.get("step_index")
        step_val = kwargs.get("step")
        if step_index is None and step_val is not None:
            step_index = step_val
        step_str_by_val = str(step_val) if step_val is not None else None
        step_str_by_idx = str(step_index) if step_index is not None else None
        step_retain = None
        if retain_logs:
            if metric.name == "privleak" and retain_logs.get("retain_mia_by_step"):
                by_step = retain_logs["retain_mia_by_step"]
                step_retain = (by_step.get(step_str_by_val) if step_str_by_val else None) or (by_step.get(step_str_by_idx) if step_str_by_idx else None)
            elif metric.name == "ks_test" and retain_logs.get("retain_forget_tr_by_step"):
                by_step = retain_logs["retain_forget_tr_by_step"]
                step_retain = (by_step.get(step_str_by_val) if step_str_by_val else None) or (by_step.get(step_str_by_idx) if step_str_by_idx else None)
        if step_retain is not None:
            ref_logs_step = copy.deepcopy(ref_logs)
            if "retain_model_logs" not in ref_logs_step:
                ref_logs_step["retain_model_logs"] = {}
            if metric.name == "privleak":
                ref_logs_step["retain_model_logs"]["retain"] = step_retain
            else:
                ref_logs_step["retain_model_logs"]["retain_ftr"] = step_retain
            metric_kwargs["reference_logs"] = ref_logs_step
        else:
            if retain_logs or ref_logs.get("_required_but_missing"):
                from evals.metrics.base import RetainReferenceValidationError
                slot_name = "retain (retain_mia_by_step)" if metric.name == "privleak" else "retain_ftr (retain_forget_tr_by_step)"
                raise RetainReferenceValidationError(
                    f"reference_logs was provided but step-matched {slot_name} not found for step {step_val!r}/{step_index!r}. No fallback."
                )

    # Call the metric's underlying function
    # Note: We call _metric_fn directly, not evaluate(), because:
    # 1. We're computing at a single step, not iterating over data
    # 2. We've already prepared the model wrapper and batch
    # 3. Pre-compute metrics would need to be computed at each step separately
    
    # Some metrics iterate over data (like `probability`), so we need to use
    # their underlying batch functions instead. Map known metrics to their batch functions.
    metric_name = metric.name

    trajectory_config = kwargs.get("trajectory_config")
    sample_traj = kwargs.get("sample_traj")
    if (
        metric_name == "extraction_strength"
        and not batched_steps
        and trajectory_config is not None
        and trajectory_config.get("use_generalized_sequence_probability", True)
        and sample_traj is not None
    ):
        R = sample_traj["R"]
        F = sample_traj["F"]
        S_val = int(sample_traj["S"])
        report_step = kwargs.get("step")
        lab = batch.get("labels")
        if lab is not None:
            lab = lab.squeeze(0) if lab.dim() > 1 else lab
            if report_step is not None:
                fixation_logits = build_effective_step_fixation_logits(
                    R, F, int(report_step)
                ).squeeze(0)
            else:
                fixation_logits = build_fixation_logits_from_R_F(R, F).squeeze(0)
            logit_alignment = trajectory_config.get("logit_alignment", "causal")
            F_sq = F.squeeze(0) if F.dim() > 1 else F
            es_val = extraction_strength_from_fixation(
                fixation_logits, lab, F_sq, S_val, logit_alignment, IGNORE_INDEX
            )
            return [{"score": es_val}]

    def _exact_memorization_batch_fn(model, batch, **kwargs):
        """Compute exact memorization for a single batch."""
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
            model, batch, grad=False, return_labels=True
        )
        if len(log_probs_batch) == 0 or len(labels_batch) == 0:
            return [{"score": None}]
        log_probs = log_probs_batch[0]
        labels = labels_batch[0]
        if len(labels) == 0:
            return [{"score": None}]
        preds = torch.argmax(log_probs, dim=-1)
        em_score = (preds == labels).sum() / len(labels)
        return [{"score": em_score.item()}]

    def _extraction_strength_batch_fn(model, batch, **kwargs):
        """Compute extraction strength for a single batch."""
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
            model, batch, grad=False, return_labels=True
        )
        es_batch = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            valid_len = len(labels)
            preds = torch.argmax(log_probs, dim=-1)
            for k in range(valid_len):
                suff_preds = preds[k:]
                suff_labels = labels[k:]
                if torch.equal(suff_preds, suff_labels):
                    break
            if valid_len == 0:
                es_batch.append({"score": 0})
            else:
                es_score = 1 - (k / valid_len)
                es_batch.append({"score": es_score})
        return es_batch if es_batch else [{"score": None}]

    batch_function_map = {
        "probability": evaluate_probability,
        "probability_confidence_ordered": evaluate_probability_confidence_ordered,
        "exact_memorization": _exact_memorization_batch_fn,
        "extraction_strength": _extraction_strength_batch_fn,
    }

    if metric_name == "exact_memorization" and logits.shape[0] > 1:
        tn = logits.shape[0]
        batch_fn = batch_function_map["exact_memorization"]
        out_list: List[Any] = []
        for ti in range(tn):
            lw_i = LogitModelWrapper(logits[ti : ti + 1], device)
            sub_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == tn:
                    sub_batch[key] = value[ti : ti + 1]
                else:
                    sub_batch[key] = value
            out_list.extend(batch_fn(model=lw_i, batch=sub_batch))
        return out_list

    # Probability: invariant — step logits and batch labels must have the same sequence length
    # (single L from trajectories_from_logits). Callers must never pass mismatched lengths.
    if metric_name == "probability" and "labels" in batch and batch["labels"] is not None:
        labels_full = batch["labels"]
        if isinstance(labels_full, torch.Tensor):
            L_logits = logits.shape[1]
            L_labels = labels_full.shape[1]
            if L_logits != L_labels:
                raise ValueError(
                    "Probability metric requires step logits and batch labels to have the same "
                    "sequence length (invariant: single L from trajectory). "
                    "Got logits.shape[1]=%s, labels.shape[1]=%s. "
                    "Fix the caller (trajectory construction or batch_template)."
                    % (L_logits, L_labels)
                )
            result = _compute_prob_from_fixation_logits(
                logits, labels_full, device, ignore_index=IGNORE_INDEX
            )
            return result
    
    # Try using batch function if available
    if metric_name in batch_function_map:
        batch_fn = batch_function_map[metric_name]
        try:
            # Batch functions like evaluate_probability only accept (model, batch)
            # Don't pass any other kwargs
            result = batch_fn(model=model_wrapper, batch=batch)
            return result
        except Exception as e:
            from evals.metrics.base import RetainReferenceValidationError
            if isinstance(e, RetainReferenceValidationError):
                raise
            logger.warning(
                f"Error calling batch function for {metric_name}: {e}. "
                f"Falling back to metric function.",
                exc_info=True
            )
    
    # Try calling the metric function directly
    try:
        result = metric._metric_fn(**metric_kwargs)
        return result
    except (KeyError, TypeError, AttributeError) as e:
        # Check if this is a text-based metric that needs generation
        error_msg = str(e).lower()
        needs_generation = (
            "generate" in error_msg or
            "data" in error_msg or 
            "collators" in error_msg or
            "dataloader" in error_msg or
            "generation_args" in error_msg
        )
        
        if needs_generation:
            # This is likely a text-based metric that needs model.generate()
            # Use generic text-based handler. Text handler expects [1, L, V] or [L, V]; do not transpose.
            logger.info(
                "Using generic text-based handler for %s (direct call failed: %s).",
                metric_name,
                e,
            )
            try:
                # Pass logits as-is: already [1, L, V] from normalization above; _handle_text_based_metric expects sequence last.
                text_kw = {**kwargs, "batch": batch, "sample_idx": sample_idx}
                result = _handle_text_based_metric(
                    logits=logits,
                    tokenizer=tokenizer,
                    sample_labels=sample_labels,
                    sample_input_ids=sample_input_ids,
                    sample_prompt_len=sample_prompt_len,
                    metric_name=metric_name,
                    metric_config=metric_config,
                    **text_kw
                )
                return result
            except Exception as text_e:
                logger.warning(
                    f"Error in generic text-based handler for {metric_name}: {text_e}. "
                    f"Original error: {e}",
                    exc_info=True
                )
                raise
        else:
            logger.warning(
                f"Error calling metric {metric_name} at step: {e}. "
                f"This metric may require pre_compute metrics or other setup."
            )
            raise
    except Exception as e:
        # Check if it's a generation-related error
        error_msg = str(e).lower()
        if "generate" in error_msg or "generation" in error_msg:
            logger.info(
                f"Metric {metric_name} failed with generation error. "
                f"Trying generic text-based handler."
            )
            try:
                # Text handler expects [1, L, V] or [L, V]; logits already normalized above, do not transpose.
                text_kw = {**kwargs, "batch": batch, "sample_idx": sample_idx}
                result = _handle_text_based_metric(
                    logits=logits,
                    tokenizer=tokenizer,
                    sample_labels=sample_labels,
                    sample_input_ids=sample_input_ids,
                    sample_prompt_len=sample_prompt_len,
                    metric_name=metric_name,
                    metric_config=metric_config,
                    **text_kw
                )
                return result
            except Exception as text_e:
                logger.warning(
                    f"Error in generic text-based handler for {metric_name}: {text_e}",
                    exc_info=True
                )
                raise
        else:
            logger.warning(
                f"Error calling metric {metric_name} at step: {e}. "
                f"This metric may require pre_compute metrics or other setup."
            )
            raise


@unlearning_metric(name="trajectory_metrics")
def trajectory_metrics(model, **kwargs):
    with torch.inference_mode():
        gpu_set_phase("trajectory_entry")
        """
        Compute metrics along diffusion trajectories.
    
        This function:
        1. Generates text using the model's sampler (with return_logits=True)
        2. Extracts logits_history and fixation_steps from sampler output
        3. Computes three trajectory tensors (steps, fixation, ratio)
        4. For each trajectory type and step, computes specified metrics
        5. Returns results organized by trajectory, step, and metric
    
        Config structure:
        - metrics: list of metric names OR dict mapping metric names to configs
          Examples:
            - ["probability", "exact_memorization"]  # Simple list
            - {"probability": {}, "truth_ratio": {"aggregator": "closer_to_1_better"}}  # With configs
        - trajectory_config: config for trajectory computation
          - logits_source: "sampler" (default) or "external"
          - use_fixation_logits: true (default)  # If model is adapter, use fixation logits in __call__
          - return_logits: true  # Sampler config
          - return_fixation_steps: true  # Sampler config
        - data: dataset to evaluate on
        - collators: collator for batching
        - batch_size: batch size for evaluation
        - tokenizer: tokenizer for text processing
        - generation_args: args for text generation (for text-based metrics)
    
        Note: Metrics that require pre_compute (like truth_ratio) will need their
        pre_compute metrics to be computed at each step. This is handled automatically
        if pre_compute configs are provided in the metric configs.

        trajectory_config (T×views batching, optional, default off):
        - trajectory_step_view_batch_allowlist: list of metric keys (e.g. exact_memorization,
          truth_ratio, extraction_strength) allowed to use one packed or stacked pass per
          (sample, traj_name, metric) over steps × include_views. Row / segment order is
          lexicographic (step, then view).
        - trajectory_step_view_batch_chunk_max: int caps packed shifted-CE segment count per GPU
          chunk (post-loop probability and truth_ratio legs). Defaults to 32 when omitted; ``0`` or
          negative disables chunking.
        """
        # Extract config
        metrics_config = kwargs.get("metrics", [])
        trajectory_config = kwargs.get("trajectory_config", {})
        rank = kwargs.get("rank", 0)
        if rank == 0:
            tc = trajectory_config
            if OmegaConf.is_config(tc):
                _tc = OmegaConf.to_container(tc, resolve=True) or {}
            elif isinstance(tc, dict):
                _tc = tc
            else:
                _tc = {}
            if not isinstance(_tc, dict):
                _tc = {}
            _traj_u = bool(_tc.get("use_generalized_sequence_probability", True))
            logger.info(
                "trajectory_metrics: trajectory_config.use_generalized_sequence_probability=%s",
                _traj_u,
            )
        metric_worker_pool_size = trajectory_config.get("metric_worker_pool_size", 8)
        executor = (
            ProcessPoolExecutor(max_workers=metric_worker_pool_size)
            if metric_worker_pool_size > 0
            else None
        )
        _ = trajectory_config.get("logits_source", "sampler")  # logits_source, reserved
        data = kwargs.get("data")
        collator = kwargs.get("collators")
        batch_size = kwargs.get("batch_size", 1)
        sort_by_length = kwargs.get("sort_by_length", False)
        tokenizer = kwargs.get("tokenizer")
        _ = kwargs.get("generation_args", {})  # generation_args, reserved
        world_size = kwargs.get("world_size", 1)
        use_distributed_sampler = world_size > 1

        # Input-output baselines: build guardrail config and load ICUL pools once per run
        guardrail_config_with_pools: Optional[dict] = None
        if trajectory_config:
            guard_raw = trajectory_config.get("guardrail")
            guard = (
                OmegaConf.to_container(guard_raw, resolve=True) or {}
                if guard_raw is not None
                else {}
            )
            if guard:
                icul = (guard.get("icul") or {}) if isinstance(guard.get("icul"), dict) else {}
                if icul.get("enabled") and guard.get("benchmark"):
                    benchmark = (guard.get("benchmark") or "").lower()
                    if benchmark in ("tofu", "muse", "wmdp"):
                        full_cfg = OmegaConf.to_container(trajectory_config, resolve=True) or {}
                        merged_cfg = {**full_cfg, **(full_cfg.get("guardrail") or {})}
                        try:
                            forget_pool, retain_pool = load_icul_pools(
                                benchmark,
                                tokenizer,
                                kwargs.get("template_args"),
                                merged_cfg,
                            )
                            if forget_pool and retain_pool:
                                guard = dict(guard)
                                guard["icul_forget_pool"] = forget_pool
                                guard["icul_retain_pool"] = retain_pool
                                guardrail_config_with_pools = guard
                        except Exception as e:
                            logger.warning("ICUL pool load failed: %s", e)
                if guardrail_config_with_pools is None and guard:
                    guardrail_config_with_pools = dict(guard)

        if not metrics_config:
            raise ValueError("No metrics specified in config")
    
        if not tokenizer:
            raise ValueError("tokenizer is required for trajectory metrics")

        # When model is DiffusionModelAdapter, set use_fixation_logits so __call__ returns
        # fixation logits (trajectory run). Default True for trajectory metrics.
        # Note: This runs after prepare_kwargs_evaluate_metric (and pre_compute), so
        # pre_compute runs with the adapter still at default False (single forward).
        if hasattr(model, "adapter_config"):
            model.adapter_config.use_fixation_logits = trajectory_config.get(
                "use_fixation_logits", True
            )

        # Parse metrics config: support both list and dict formats
        # Handle OmegaConf ListConfig and DictConfig (from Hydra)
        if isinstance(metrics_config, (list, ListConfig)):
            # Simple list of metric names: ["probability", "exact_memorization"]
            # Convert ListConfig to list if needed
            if isinstance(metrics_config, ListConfig):
                metrics_config = list(metrics_config)
            metrics_to_compute = {name: {} for name in metrics_config}
        elif isinstance(metrics_config, (dict, DictConfig)):
            # Dict mapping metric names to configs: {"probability": {}, "truth_ratio": {"aggregator": "..."}}
            # Convert DictConfig to dict if needed
            if isinstance(metrics_config, DictConfig):
                metrics_to_compute = dict(metrics_config)
            else:
                metrics_to_compute = metrics_config
        else:
            raise ValueError(
                f"metrics must be a list or dict, got {type(metrics_config)}"
            )

        # Optional subset: only compute these (e.g. from CLI --metrics A B C)
        include_metrics = kwargs.get("include_metrics")
        if include_metrics is not None:
            include_metrics = list(include_metrics)
            if include_metrics:
                metrics_to_compute = {
                    k: metrics_to_compute[k]
                    for k in include_metrics
                    if k in metrics_to_compute
                }

        trajectory_pass_id = kwargs.get("trajectory_pass_id")
        if trajectory_pass_id is None:
            ev = kwargs.get("eval_cfg")
            if ev is not None and callable(getattr(ev, "get", None)):
                trajectory_pass_id = ev.get("trajectory_pass_id")
        if trajectory_pass_id is None:
            if isinstance(trajectory_config, dict):
                trajectory_pass_id = trajectory_config.get("trajectory_pass_id")
            elif OmegaConf.is_config(trajectory_config):
                trajectory_pass_id = OmegaConf.select(
                    trajectory_config, "trajectory_pass_id", default=None
                )
        if trajectory_pass_id:
            from evals.metrics.trajectory_pass_binding import (
                filter_metrics_and_data_for_pass,
                get_pass_spec,
            )

            _tpid = str(trajectory_pass_id)
            spec = get_pass_spec(_tpid)
            metrics_to_compute, data, tc_dict = filter_metrics_and_data_for_pass(
                _tpid,
                metrics_to_compute,
                data,
                trajectory_config,
            )
            kwargs["data"] = data
            trajectory_config = tc_dict
            kwargs["trajectory_config"] = trajectory_config
            kwargs["metric_display_names"] = list(spec.display_names_emitted)
            kwargs["trajectory_pass_id"] = _tpid

        # Full order for display-name mapping (after include_metrics / trajectory_pass filters)
        full_internal_order = list(metrics_to_compute.keys())
        full_display_order = list(kwargs.get("metric_display_names") or [])

        # Load metrics from registry (support handler= for logical names, e.g. forget_knowmem_rouge -> rouge)
        loaded_metrics = {}
        for metric_name, metric_cfg in metrics_to_compute.items():
            try:
                registry_name = metric_cfg.get("handler", metric_name) if hasattr(metric_cfg, "get") else metric_name
                metric = _get_metric_from_registry(registry_name)
                loaded_metrics[metric_name] = {
                    "metric": metric,
                    "config": metric_cfg if (isinstance(metric_cfg, dict) or hasattr(metric_cfg, "get")) else {},
                }
            except ValueError as e:
                logger.error(f"Failed to load metric '{metric_name}': {e}")
                raise

        trajectory_capture = True
        _evcfg = kwargs.get("eval_cfg")
        if _evcfg is not None and callable(getattr(_evcfg, "get", None)):
            _tcap = _evcfg.get("trajectory_capture")
            if _tcap is not None:
                trajectory_capture = bool(_tcap)

        # One-time warning if hm_aggregate (trajectory_model_utility) will have no pre_compute (no retain dataset)
        if "hm_aggregate" in loaded_metrics and isinstance(data, dict) and data.get("retain") is None:
            logger.warning(
                "trajectory_model_utility (hm_aggregate) will be None: add a retain dataset to the eval config (e.g. TOFU_QA_retain_eval with access_key: retain) so current model's retain-set metrics are computed per step."
            )

        # Handle multi-dataset only when there are keys beyond reserved (e.g. MUSE: forget_knowmem, retain_knowmem).
        # Reserved keys: forget, holdout, retain (main loop + MU), ra, wf (MU only). Adding ra/wf does NOT enable multi_dataset.
        RESERVED_DATA_KEYS = {"forget", "holdout", "retain", "ra", "wf"}
        multi_dataset = (
            isinstance(data, dict)
            and bool(set(data.keys()) - RESERVED_DATA_KEYS)
        )
        if isinstance(data, dict) and "forget" in data and "holdout" in data:
            primary_data = data["forget"]
            secondary_data = data["holdout"]
        elif isinstance(data, dict) and "forget" in data and not multi_dataset:
            primary_data = data["forget"]
            secondary_data = data.get("holdout")
        elif (
            isinstance(data, dict)
            and "forget" not in data
            and not multi_dataset
            and len(data) == 1
            and next(iter(data.keys())) in ("retain", "ra", "wf")
        ):
            # Single-dataset MU trajectory pass (no forget leg): run the main loop on that set.
            _only_key = next(iter(data.keys()))
            primary_data = data[_only_key]
            secondary_data = None
        else:
            primary_data = data if not multi_dataset else None
            secondary_data = None

        single_dataset_keys = [k for k in data if k not in RESERVED_DATA_KEYS] if multi_dataset else []

        # Create dataloader(s). When distributed (world_size > 1), use DistributedSampler per rank.
        if not multi_dataset:
            if use_distributed_sampler:
                sampler_primary = DistributedSampler(
                    primary_data, num_replicas=world_size, rank=rank, shuffle=False
                )
                dataloader = DataLoader(
                    primary_data,
                    batch_size=batch_size,
                    sampler=sampler_primary,
                    collate_fn=collator,
                )
            elif sort_by_length:
                dataloader = DataLoader(
                    primary_data,
                    batch_size=batch_size,
                    sampler=LengthSortedSampler(primary_data),
                    collate_fn=collator,
                )
            else:
                dataloader = DataLoader(
                    primary_data, batch_size=batch_size, collate_fn=collator
                )
        else:
            dataloader = None  # created per key in loop
        if secondary_data is not None:
            if use_distributed_sampler:
                sampler_holdout = DistributedSampler(
                    secondary_data, num_replicas=world_size, rank=rank, shuffle=False
                )
                holdout_dataloader = DataLoader(
                    secondary_data,
                    batch_size=batch_size,
                    sampler=sampler_holdout,
                    collate_fn=collator,
                )
            elif sort_by_length:
                holdout_dataloader = DataLoader(
                    secondary_data,
                    batch_size=batch_size,
                    sampler=LengthSortedSampler(secondary_data),
                    collate_fn=collator,
                )
            else:
                holdout_dataloader = DataLoader(
                    secondary_data, batch_size=batch_size, collate_fn=collator
                )
        else:
            holdout_dataloader = None

        # Compute MU per-step aggregates for trajectory_model_utility (hm_aggregate). When retain+ra+wf present, full 9-metric; else retain-only 3-metric.
        retain_agg_by_step = {}
        if (
            "hm_aggregate" in loaded_metrics
            and isinstance(data, dict)
            and data.get("retain") is not None
        ):
            _mu_kw = {
                k: v
                for k, v in kwargs.items()
                if k
                not in (
                    "tokenizer",
                    "model",
                    "data",
                    "collator",
                    "batch_size",
                    "trajectory_config",
                    "loaded_metrics",
                    "sort_by_length",
                    "use_distributed_sampler",
                    "world_size",
                    "rank",
                )
            }
            has_ra = data.get("ra") is not None
            has_wf = data.get("wf") is not None
            if has_ra and has_wf:
                logger.info(
                    "Trajectory MU: running full 9-metric (retain, ra, wf)."
                )
                logger.info(
                    "Trajectory MU: 9 components %s",
                    sorted(EXPECTED_9_MU_KEYS),
                )
                retain_res = _compute_mu_for_dataset(
                    model,
                    data["retain"],
                    "retain",
                    collator,
                    batch_size,
                    trajectory_config,
                    tokenizer,
                    loaded_metrics,
                    sort_by_length,
                    use_distributed_sampler,
                    world_size,
                    rank,
                    **_mu_kw,
                )
                ra_res = _compute_mu_for_dataset(
                    model,
                    data["ra"],
                    "ra",
                    collator,
                    batch_size,
                    trajectory_config,
                    tokenizer,
                    loaded_metrics,
                    sort_by_length,
                    use_distributed_sampler,
                    world_size,
                    rank,
                    **_mu_kw,
                )
                wf_res = _compute_mu_for_dataset(
                    model,
                    data["wf"],
                    "wf",
                    collator,
                    batch_size,
                    trajectory_config,
                    tokenizer,
                    loaded_metrics,
                    sort_by_length,
                    use_distributed_sampler,
                    world_size,
                    rank,
                    **_mu_kw,
                )
                for traj_name in retain_res:
                    retain_agg_by_step[traj_name] = {}
                    for step_key in retain_res[traj_name]:
                        merged = {}
                        for view in retain_res[traj_name][step_key]:
                            merged[view] = {
                                **retain_res[traj_name][step_key][view],
                                **ra_res.get(traj_name, {}).get(step_key, {}).get(view, {}),
                                **wf_res.get(traj_name, {}).get(step_key, {}).get(view, {}),
                            }
                        retain_agg_by_step[traj_name][step_key] = merged
                _validate_merged_9_mu(retain_agg_by_step)
                n_traj = len(retain_agg_by_step)
                n_steps = len(next(iter(retain_agg_by_step.values()), {})) if retain_agg_by_step else 0
                logger.info(
                    "Trajectory MU: merged 9 components for %s trajectory types, %s steps each (full and eos).",
                    n_traj,
                    n_steps,
                )
                # Log 9 values for one traj/step/view as a sanity check (steps, step 0, full).
                if retain_agg_by_step:
                    first_traj = next(iter(retain_agg_by_step.keys()))
                    first_step_dict = retain_agg_by_step[first_traj]
                    first_sk = next(iter(first_step_dict.keys()))
                    first_sv = first_step_dict[first_sk]
                    if isinstance(first_sv, dict) and "full" in first_sv:
                        vals = {
                            k: v.get("agg_value")
                            for k, v in first_sv["full"].items()
                            if isinstance(v, dict) and "agg_value" in v
                        }
                        logger.info(
                            "Trajectory MU: 9 values (traj=%s step %s, full) %s",
                            first_traj,
                            first_sk,
                            vals,
                        )
            else:
                retain_agg_by_step = _compute_retain_mu_by_step(
                    model,
                    data["retain"],
                    collator,
                    batch_size,
                    trajectory_config,
                    tokenizer,
                    loaded_metrics,
                    sort_by_length,
                    use_distributed_sampler,
                    world_size,
                    rank,
                    **_mu_kw,
                )
        if "hm_aggregate" in loaded_metrics and logger.isEnabledFor(logging.DEBUG):
            _debug_log_mu_submetric_coverage(retain_agg_by_step)
        kwargs["retain_agg_by_step"] = retain_agg_by_step

        # Log once when reference_logs was provided but step-matched data is missing (no fallback to aggregate).
        ref_logs = kwargs.get("reference_logs") or {}
        retain_logs = ref_logs.get("retain_model_logs") or {}
        if retain_logs:
            if "privleak" in loaded_metrics and not retain_logs.get("retain_mia_by_step"):
                if "privleak" not in _trajectory_single_retain_warned:
                    _trajectory_single_retain_warned.add("privleak")
                    logger.error(
                        "reference_logs was provided but step-matched retain (retain_mia_by_step) not found in file. "
                        "trajectory_privleak will not receive reference_logs. No fallback. "
                        "Run retain eval with trajectory and save mia_min_k_by_step, then use that file as reference."
                    )
            if "ks_test" in loaded_metrics and not retain_logs.get("retain_forget_tr_by_step"):
                if "ks_test" not in _trajectory_single_retain_warned:
                    _trajectory_single_retain_warned.add("ks_test")
                    logger.error(
                        "reference_logs was provided but step-matched retain (retain_forget_tr_by_step) not found in file. "
                        "trajectory_forget_quality (ks_test) will not receive reference_logs. No fallback. "
                        "Run retain eval with trajectory and save forget_truth_ratio_by_step, then use that file as reference."
                    )

        # Check if privleak needs dual trajectories (forget + holdout)
        privleak_has_dual_data = (secondary_data is not None and holdout_dataloader is not None) or (
            multi_dataset and "forget" in data and "holdout" in data
        )
        privleak_needs_dual = (
            "privleak" in loaded_metrics
            and privleak_has_dual_data
        )
        if privleak_needs_dual:
            privleak_cfg = loaded_metrics.get("privleak", {}).get("config", {})
            privleak_pre = privleak_cfg.get("pre_compute", {})
            privleak_needs_dual = "mia_min_k" in privleak_pre

        # Trajectory names
        trajectory_names = ["steps", "fixation_start", "fixation_end", "fixation_ratio"]

        # Views: full (all positions 0..L) vs eos (positions 0..L_eff-1 only). Default both.
        _include_views_raw = trajectory_config.get("include_views", ["full", "eos"])
        if isinstance(_include_views_raw, (list, ListConfig)):
            include_views = list(_include_views_raw)
        else:
            include_views = ["full", "eos"]
        include_views = [str(v).lower() for v in include_views if str(v).lower() in ("full", "eos")]
        if not include_views:
            include_views = ["full", "eos"]

        # Storage for aggregation: per view, then traj_name -> step -> metric_name -> list of values
        step_values_by_view = {
            v: {traj_name: {} for traj_name in trajectory_names}
            for v in include_views
        }

        # Get sampler from model
        sampler = _get_sampler_from_model(model)
        if sampler is None:
            raise ValueError(
                "Model does not have a sampler. Trajectory metrics require a diffusion model with sampler. "
                "Ensure model is wrapped with DiffusionModelAdapter or has accessible sampler."
            )

        # When privleak + dual dataset: use streaming MIA (batch-by-batch, only scores stored; no N trajectories in memory)
        trajectories_by_key = None
        use_streaming_privleak = False
        privleak_accumulators = None
        privleak_streaming_cfg = None
        if privleak_needs_dual and not multi_dataset:
            use_streaming_privleak = True
            logger.info("Privleak with dual dataset: using streaming MIA (batch-by-batch, scores only)")
            try:
                _device = getattr(model, "device", None) or next(model.parameters()).device
            except (StopIteration, AttributeError):
                _device = torch.device("cpu")
            privleak_cfg = loaded_metrics["privleak"]["config"]
            pre_compute = privleak_cfg.get("pre_compute", {})
            if "mia_min_k" in pre_compute:
                attack_cls = get_attacker("min_k")
                attack_kwargs = dict(pre_compute.get("mia_min_k", {}))
                attack_cls_name = "min_k"
            else:
                attack_cls = None
                attack_kwargs = {}
                attack_cls_name = None
            _pv_s = [v for v in include_views if v in ("full", "eos")]
            if not _pv_s:
                _pv_s = ["full"]
            privleak_streaming_cfg = {
                "device": _device,
                "privleak_cfg": privleak_cfg,
                "attack_cls": attack_cls,
                "attack_cls_name": attack_cls_name,
                "attack_kwargs": attack_kwargs,
                "_pv_views": _pv_s,
                "_layout": "dual" if len(_pv_s) > 1 else "single",
                "_single_view": _pv_s[0] if len(_pv_s) == 1 else None,
            }

        run_steps_to_use = None
        run_step_values_metadata = None
        _logged_capture_final_only = False

        keys_to_process = [None] if not multi_dataset else single_dataset_keys
        for _key in keys_to_process:
            if _key is not None:
                primary_data = data[_key]
                if use_distributed_sampler:
                    sampler_primary = DistributedSampler(
                        primary_data, num_replicas=world_size, rank=rank, shuffle=False
                    )
                    dataloader = DataLoader(
                        primary_data,
                        batch_size=batch_size,
                        sampler=sampler_primary,
                        collate_fn=collator,
                    )
                elif sort_by_length:
                    dataloader = DataLoader(
                        primary_data,
                        batch_size=batch_size,
                        sampler=LengthSortedSampler(primary_data),
                        collate_fn=collator,
                    )
                else:
                    dataloader = DataLoader(
                        primary_data, batch_size=batch_size, collate_fn=collator
                    )
                metrics_to_run = [
                    m for m in loaded_metrics
                    if (loaded_metrics[m].get("config") or {}).get("dataset_key") == _key
                ]
            else:
                metrics_to_run = [
                    m for m in loaded_metrics
                    if m != "privleak" or not privleak_needs_dual
                ]
            if not metrics_to_run:
                continue

            n_samples = (
                len(dataloader.sampler)
                if use_distributed_sampler and getattr(dataloader, "sampler", None) is not None
                else len(dataloader.dataset)
            )
            expected_batches = (n_samples + batch_size - 1) // batch_size
            if _key is not None and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Trajectory dataset_key=%s: %s samples, batch_size=%s, expected_batches=%s",
                    _key, n_samples, batch_size, expected_batches,
                )
            logger.info(
                f"Trajectory forget dataset: {n_samples} samples, batch_size {batch_size}, "
                f"expected batches: {expected_batches} (last batch index: {expected_batches - 1})"
            )
            if logger.isEnabledFor(logging.DEBUG):
                _sampler_kw_preview = _trajectory_sampler_kwargs(trajectory_config)
                _max_new = _sampler_kw_preview.get("max_new_tokens")
                _interval = _sampler_kw_preview.get("trajectory_sample_interval")
                _pred_gen = getattr(dataloader.dataset, "predict_with_generate", False)
                logger.debug(
                    "Trajectory config: max_new_tokens=%s trajectory_sample_interval=%s predict_with_generate=%s",
                    _max_new, _interval, _pred_gen,
                )
            if use_distributed_sampler and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Data parallel: rank {rank}/{world_size} processes {n_samples} samples (no duplication)"
                )
            all_rouge_futures: list = []
            all_cpu_metric_futures: list = []
            effective_length_by_index: dict[str, int] = {}
            prompt_len_by_index: dict[str, int] = {}
            # Progress logging: log every 5% of batches or at least every 5 batches, plus first and last
            _log_interval = max(1, expected_batches // 20) if expected_batches else 1
        # Process each batch
            for batch_idx, batch in enumerate(dataloader):
                gpu_set_phase("trajectory_batch_start", batch_idx=batch_idx)
                _batch_t0 = time.perf_counter()
                if batch_idx % _log_interval == 0 or batch_idx == 0 or batch_idx == expected_batches - 1:
                    _pct = 100 * (batch_idx + 1) / expected_batches if expected_batches else 0
                    logger.info(
                        "Trajectory batch %s/%s (%.0f%%), %s samples total",
                        batch_idx + 1, expected_batches, _pct, n_samples,
                    )
                input_ids = batch["input_ids"]
                labels = batch.get("labels")
                _ = batch.get("attention_mask")  # reserved
                indices = batch.get("index", torch.arange(batch_idx * batch_size, 
                                                          (batch_idx + 1) * batch_size))
                B = input_ids.shape[0]
        
                # Prepare inputs for sampler (list of token sequences)
                _prompt_only_input_ids = getattr(dataloader.dataset, "predict_with_generate", False)
                prompts, prompt_lens, prompt_starts = _build_prompts_for_sampler(
                    input_ids, labels, tokenizer, IGNORE_INDEX,
                    prompt_only_input_ids=_prompt_only_input_ids,
                )
                if guardrail_config_with_pools is not None:
                    prompts, prompt_lens = transform_prompts(
                        prompts,
                        prompt_lens,
                        batch,
                        tokenizer,
                        config={"guardrail": guardrail_config_with_pools},
                    )

                # Generate using sampler with logits tracking
                _sampler_kw = _trajectory_sampler_kwargs(trajectory_config)
                evaluation_mode = _sampler_kw.get("evaluation_mode", "unguided")
                sample_kw = dict(
                    inputs=prompts,
                    config=None,  # Use default config
                    return_dict=True,
                    return_logits=True,
                    **_sampler_kw,
                )
                if evaluation_mode in ("guided_native", "guided_skew") and labels is not None:
                    L_gen = _sampler_kw.get("max_new_tokens")
                    if L_gen is None:
                        L_gen = max(
                            labels.shape[1] - prompt_starts[j] for j in range(B)
                        )
                    else:
                        L_gen = int(L_gen)
                    target_sequences = _build_target_sequences_for_sampler(
                        labels, prompt_starts, L_gen, IGNORE_INDEX
                    )
                    sample_kw["target_sequences"] = target_sequences
                    sample_kw["evaluation_mode"] = evaluation_mode
                sampler_output = sampler.sample(**sample_kw)
                _t_after_sampler = time.perf_counter()
                _sampler_sec = _t_after_sampler - _batch_t0
                gpu_set_phase("trajectory_after_sampler", batch_idx=batch_idx)
                if (batch_idx % _log_interval == 0 or batch_idx == 0 or batch_idx == expected_batches - 1) and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Batch %s/%s: diffusion sampling done in %.1fs",
                        batch_idx + 1, expected_batches, _sampler_sec,
                    )

                # Extract logits and fixation steps
                logits_history = sampler_output.logits_history
                fixation_steps = sampler_output.fixation_steps

                if logits_history is None or len(logits_history) == 0:
                    logger.warning(f"Batch {batch_idx}: No logits_history returned from sampler")
                    continue
        
                if fixation_steps is None:
                    logger.warning(f"Batch {batch_idx}: No fixation_steps returned from sampler")
                    continue

                (
                    _tc_plain,
                    _step_metric_batch_allowlist,
                    _cpu_offload_metrics,
                    _view_step_batch_allowlist,
                    _tv_chunk_max,
                    _step_prefetch_max,
                    _logits_storage,
                ) = _trajectory_tc_allowlist_sets(trajectory_config)
                _log_traj_peak = bool(_tc_plain.get("trajectory_log_peak_memory_mb"))
                if _log_traj_peak and torch.cuda.is_available():
                    logger.info(
                        "trajectory_cuda_mb batch_idx=%s before_trajectories allocated=%.1f reserved=%.1f",
                        batch_idx,
                        torch.cuda.memory_allocated() / 1e6,
                        torch.cuda.memory_reserved() / 1e6,
                    )

                out = trajectories_from_logits(
                    logits_history,
                    fixation_steps,
                    prompt_lens,
                    return_trajectory_tensors=False,
                    output_layout=_logits_storage,
                )
                lh_batch: Optional[List[torch.Tensor]] = None
                R: Optional[torch.Tensor] = None
                if _logits_storage == "list_history":
                    lh_batch = out["lh"]
                    F, S, L = out["F"], out["S"], out["L"]
                else:
                    R = out["R"]
                    F, S, L = out["F"], out["S"], out["L"]
                del logits_history, out
                if _log_traj_peak and torch.cuda.is_available():
                    logger.info(
                        "trajectory_cuda_mb batch_idx=%s after_trajectories allocated=%.1f reserved=%.1f",
                        batch_idx,
                        torch.cuda.memory_allocated() / 1e6,
                        torch.cuda.memory_reserved() / 1e6,
                    )

                # Effective length per sample (for eos view): first EOS in generated region, or L
                sequences = getattr(sampler_output, "sequences", None)
                eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer else None
                if eos_token_id is None and trajectory_config:
                    eos_token_id = trajectory_config.get("eos_token_id")
                _pv_s = privleak_streaming_cfg.get("_pv_views") if privleak_streaming_cfg else []
                _need_eos_len = "eos" in include_views or (
                    privleak_needs_dual
                    and not multi_dataset
                    and isinstance(_pv_s, list)
                    and len(_pv_s) > 1
                )
                if sequences is not None and sequences.dim() >= 2 and _need_eos_len:
                    effective_lengths = effective_lengths_from_eos(
                        sequences, prompt_lens, L, eos_token_id
                    )
                else:
                    effective_lengths = [L] * B

                if run_steps_to_use is None:
                    run_steps_to_use, run_step_values_metadata = _derive_steps_to_use(
                        S, trajectory_config
                    )
                    if not trajectory_capture and run_steps_to_use:
                        run_steps_to_use = [run_steps_to_use[-1]]
                        if run_step_values_metadata:
                            run_step_values_metadata = [run_step_values_metadata[-1]]
                        if not _logged_capture_final_only:
                            logger.info(
                                "trajectory_capture=false: metrics at final captured step only "
                                "(step_index=%s)",
                                run_steps_to_use[0],
                            )
                            _logged_capture_final_only = True
                steps_to_use = [s for s in run_steps_to_use if s < S]
                if (
                    use_streaming_privleak
                    and privleak_streaming_cfg is not None
                    and privleak_streaming_cfg.get("attack_cls") is not None
                    and privleak_accumulators is None
                    and _key is None
                ):
                    cfg = privleak_streaming_cfg
                    _acc_kw = dict(
                        attack_cls=cfg["attack_cls"],
                        collator=collator,
                        batch_size=batch_size,
                        device=cfg["device"],
                        **cfg["attack_kwargs"],
                    )
                    if cfg.get("_layout") == "dual":
                        privleak_accumulators = {
                            v: {
                                step: MIAStreamingAccumulator(**_acc_kw)
                                for step in run_steps_to_use
                            }
                            for v in cfg["_pv_views"]
                        }
                    else:
                        privleak_accumulators = {
                            step: MIAStreamingAccumulator(**_acc_kw)
                            for step in run_steps_to_use
                        }
                if privleak_accumulators is not None and _key is None:
                    use_generalized_privleak = trajectory_config.get(
                        "use_generalized_sequence_probability", True
                    )
                    _plc = privleak_streaming_cfg or {}
                    _dual_pl = _plc.get("_layout") == "dual"
                    for step in steps_to_use:
                        if use_generalized_privleak:
                            per_position_scores_forget = _per_position_scores_from_R_F_batch(
                                R,
                                F,
                                labels,
                                prompt_starts,
                                L,
                                trajectory_config,
                                report_step=step,
                                lh_batch=lh_batch,
                            )
                            if per_position_scores_forget is None:
                                continue
                            if not _dual_pl:
                                sv = _plc.get("_single_view") or "full"
                                if sv == "full":
                                    privleak_accumulators[step].add_forget_batch(
                                        batch, per_position_scores=per_position_scores_forget
                                    )
                                else:
                                    privleak_accumulators[step].add_forget_batch(
                                        batch,
                                        per_position_scores=_truncate_per_position_scores_eos(
                                            per_position_scores_forget,
                                            list(effective_lengths),
                                            L,
                                        ),
                                    )
                            else:
                                privleak_accumulators["full"][step].add_forget_batch(
                                    batch, per_position_scores=per_position_scores_forget
                                )
                                privleak_accumulators["eos"][step].add_forget_batch(
                                    batch,
                                    per_position_scores=_truncate_per_position_scores_eos(
                                        per_position_scores_forget,
                                        list(effective_lengths),
                                        L,
                                    ),
                                )
                        else:
                            if lh_batch is not None:
                                logits_batch = lh_batch[int(step)]
                            else:
                                assert R is not None
                                logits_batch = R[:, :, :, step].permute(0, 2, 1)
                            if not _dual_pl:
                                sv = _plc.get("_single_view") or "full"
                                if sv == "full":
                                    privleak_accumulators[step].add_forget_batch(
                                        batch, logits_batch
                                    )
                                else:
                                    for i in range(B):
                                        Li = min(int(effective_lengths[i]), L)
                                        lb = logits_batch[i : i + 1, :Li, :]
                                        sb = {
                                            k: (
                                                v[i : i + 1]
                                                if torch.is_tensor(v)
                                                and v.dim() > 0
                                                and v.shape[0] == B
                                                else v
                                            )
                                            for k, v in batch.items()
                                        }
                                        privleak_accumulators[step].add_forget_batch(sb, lb)
                            else:
                                privleak_accumulators["full"][step].add_forget_batch(
                                    batch, logits_batch
                                )
                                for i in range(B):
                                    Li = min(int(effective_lengths[i]), L)
                                    lb = logits_batch[i : i + 1, :Li, :]
                                    sb = {
                                        k: (
                                            v[i : i + 1]
                                            if torch.is_tensor(v)
                                            and v.dim() > 0
                                            and v.shape[0] == B
                                            else v
                                        )
                                        for k, v in batch.items()
                                    }
                                    privleak_accumulators["eos"][step].add_forget_batch(
                                        sb, lb
                                    )
                gpu_set_phase("trajectory_after_trajectories", batch_idx=batch_idx)

                # Per-sample generated labels [L] for packed trajectory probability (same slicing as per-sample loop).
                _gen_labels_for_packed_prob: list[torch.Tensor] = []
                if "probability" in metrics_to_run and labels is not None:
                    for b in range(B):
                        sample_labels_b = labels[b]
                        gs_b = _generation_start(
                            b, prompt_starts, prompt_lens, _prompt_only_input_ids
                        )
                        gl = sample_labels_b[gs_b : gs_b + L]
                        if gl.shape[0] < L:
                            padding = torch.full(
                                (L - gl.shape[0],),
                                IGNORE_INDEX,
                                dtype=gl.dtype,
                                device=gl.device,
                            )
                            gl = torch.cat([gl, padding])
                        assert gl.shape[0] == L, (
                            "packed probability invariant: generated label length must equal L; "
                            "got %s, L=%s" % (gl.shape[0], L)
                        )
                        _gen_labels_for_packed_prob.append(gl)

                exact_mem_post_loop_metrics = _trajectory_exact_mem_post_loop_metric_names(
                    metrics_to_run=metrics_to_run,
                    loaded_metrics=loaded_metrics,
                    step_metric_batch_allowlist=_step_metric_batch_allowlist,
                    view_step_batch_allowlist=_view_step_batch_allowlist,
                    B=B,
                    batch=batch,
                    labels=labels,
                    input_ids=input_ids,
                    indices=indices,
                    prompt_starts=prompt_starts,
                    prompt_lens=prompt_lens,
                    L=L,
                    prompt_only_input_ids=_prompt_only_input_ids,
                    tokenizer=tokenizer,
                )
                rouge_metrics_in_run = [
                    m for m in metrics_to_run if loaded_metrics[m]["metric"].name == "rouge"
                ]
                rouge_scorer_shared = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

                # Process each sample in batch (each sample uses its own R, F; logits computed on-demand)
                for sample_idx in range(B):
                    if lh_batch is not None:
                        sample_traj = {
                            "lh": lh_batch,
                            "b": sample_idx,
                            "F": F[sample_idx],
                            "S": S,
                            "L": L,
                        }
                    else:
                        assert R is not None
                        sample_traj = {"R": R[sample_idx], "F": F[sample_idx], "S": S, "L": L}

                    # Get ground truth for this sample
                    sample_labels = labels[sample_idx] if labels is not None else None
                    sample_input_ids = input_ids[sample_idx]
                    sample_prompt_len = prompt_lens[sample_idx]
                    sample_generation_start = _generation_start(
                        sample_idx, prompt_starts, prompt_lens, _prompt_only_input_ids
                    )
                    _gs = sample_generation_start
                    _ps = prompt_starts[sample_idx]
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "trajectory slice: sample_idx=%s generation_start=%s (prompt_starts=%s, prompt_len=%s) L=%s",
                            sample_idx,
                            _gs.item() if hasattr(_gs, "item") else _gs,
                            _ps.item() if hasattr(_ps, "item") else _ps,
                            sample_prompt_len.item()
                            if hasattr(sample_prompt_len, "item")
                            else sample_prompt_len,
                            L,
                        )

                    batch_template, ground_truth_str, idx_str = _trajectory_build_sample_batch_template(
                        sample_idx,
                        batch,
                        labels,
                        input_ids,
                        indices,
                        prompt_starts,
                        prompt_lens,
                        L,
                        _prompt_only_input_ids,
                        tokenizer,
                    )
                    cpu_metric_futures_this_sample: list[Any] = []

                    # Compute metrics for each trajectory type and step (only report steps)
                    L_eff_b = effective_lengths[sample_idx]
                    effective_length_by_index[idx_str] = L_eff_b
                    prompt_len_by_index[idx_str] = sample_prompt_len
                    # Log prompt + EOS response vs full response (clear, labeled block per sample)
                    if sequences is not None and tokenizer is not None:
                        pl = prompt_lens[sample_idx]
                        seq = sequences[sample_idx]
                        prompt_text = tokenizer.decode(seq[:pl].tolist(), skip_special_tokens=True)
                        response_eos_text = tokenizer.decode(
                            seq[pl : pl + L_eff_b].tolist(), skip_special_tokens=True
                        )
                        # Full response: no skip_special_tokens so we see raw EOS/pad/junk after the answer
                        response_full_raw = tokenizer.decode(
                            seq[pl : pl + L].tolist(), skip_special_tokens=False
                        )
                        _max_prompt, _max_resp = 400, 600

                        def _trunc(s, n):
                            return s[:n] + ("..." if len(s) > n else "")

                        gen_ids = seq[pl : pl + L].tolist()
                        eos_slice_ids = gen_ids[:L_eff_b]
                        # Show first/last token IDs of EOS slice so we can verify EOS and no truncation
                        token_id_preview = (
                            f"first 5 token_ids={eos_slice_ids[:5]!r}"
                            if len(eos_slice_ids) >= 5
                            else f"token_ids={eos_slice_ids!r}"
                        )
                        if len(eos_slice_ids) > 5:
                            token_id_preview += f" ... last 3 token_ids={eos_slice_ids[-3:]!r}"
                        logger.info(
                            "[trajectory_response] sample_index=%s\n"
                            "  Prompt length: %s\n"
                            "  Total generated length (L): %s\n"
                            "  EOS index (L_eff): %s (tokens up to/incl first EOS; rest is padding/junk)\n"
                            "  Generated token IDs (EOS slice): %s\n"
                            "  Prompt: %s\n"
                            "  Response up to EOS (%s tokens, no char truncation): %s\n"
                            "  Full response raw (%s tokens, skip_special_tokens=False): %s\n"
                            "---",
                            idx_str,
                            pl,
                            L,
                            L_eff_b,
                            token_id_preview,
                            _trunc(prompt_text, _max_prompt),
                            L_eff_b,
                            response_eos_text,
                            L,
                            response_full_raw,
                        )
                    for traj_name in trajectory_names:
                        for view in include_views:
                            for step in steps_to_use:
                                if step not in step_values_by_view[view][traj_name]:
                                    step_values_by_view[view][traj_name][step] = {
                                        m: [] for m in loaded_metrics.keys()
                                    }

                        L_eff_slice = min(L_eff_b, batch_template["input_ids"].shape[1])
                        batch_template_eos = _slice_batch_template_to_length(batch_template, L_eff_slice)
                        _has_list_bt = _batch_template_has_list_tensors(batch_template)

                        for step_window in _chunked_reporting_steps(steps_to_use, _step_prefetch_max):
                            logits_by_step = _prefetch_logits_by_step(
                                sample_traj, traj_name, step_window
                            )
                            for metric_name, metric_info in [(m, loaded_metrics[m]) for m in metrics_to_run]:
                                try:
                                    metric = metric_info["metric"]
                                    metric_cfg = metric_info["config"]

                                    if metric_name == "privleak" and trajectories_by_key is not None:
                                        continue
                                    if metric_name in exact_mem_post_loop_metrics:
                                        continue
                                    if metric_name in rouge_metrics_in_run or metric_name == "probability":
                                        continue
                                    if metric_name == "ks_test":
                                        ref_logs = kwargs.get("reference_logs") or {}
                                        if not ref_logs.get("retain_model_logs"):
                                            continue

                                    _pcfg = metric_cfg.get("pre_compute") if hasattr(metric_cfg, "get") else None
                                    if OmegaConf.is_config(_pcfg):
                                        _pcfg = OmegaConf.to_container(_pcfg, resolve=True) or {}
                                    elif _pcfg is None:
                                        _pcfg = {}
                                    _pre_empty = not (isinstance(_pcfg, dict) and len(_pcfg) > 0)
                                    _tv_on = _trajectory_step_view_batch_metric_enabled(
                                        metric_name, metric, _view_step_batch_allowlist
                                    )
                                    _use_cpu_exact = (
                                        executor is not None
                                        and (
                                            metric_name in _cpu_offload_metrics
                                            or metric.name in _cpu_offload_metrics
                                        )
                                        and metric.name == "exact_memorization"
                                    )

                                    if (
                                        _tv_on
                                        and metric.name == "truth_ratio"
                                        and labels is not None
                                        and _truth_ratio_tv_precompute_compatible(metric_cfg)
                                        and not _use_cpu_exact
                                        and "labels_correct" in batch_template
                                        and "labels_wrong" in batch_template
                                    ):
                                        gpu_set_phase(
                                            "trajectory_metric",
                                            metric=metric_name,
                                            batch_idx=batch_idx,
                                            step=-1,
                                        )
                                        lc_t = batch_template["labels_correct"]
                                        lw_t = batch_template["labels_wrong"]
                                        lc_full_1 = (
                                            lc_t.squeeze(0) if isinstance(lc_t, torch.Tensor) and lc_t.dim() > 1 else lc_t
                                        )
                                        lw_full_1 = (
                                            lw_t.squeeze(0) if isinstance(lw_t, torch.Tensor) and lw_t.dim() > 1 else lw_t
                                        )
                                        lc_eos_t = batch_template_eos.get("labels_correct")
                                        lw_eos_t = batch_template_eos.get("labels_wrong")
                                        lc_eos_1 = (
                                            lc_eos_t.squeeze(0)
                                            if isinstance(lc_eos_t, torch.Tensor) and lc_eos_t.dim() > 1
                                            else lc_eos_t
                                            if isinstance(lc_eos_t, torch.Tensor)
                                            else None
                                        )
                                        lw_eos_1 = (
                                            lw_eos_t.squeeze(0)
                                            if isinstance(lw_eos_t, torch.Tensor) and lw_eos_t.dim() > 1
                                            else lw_eos_t
                                            if isinstance(lw_eos_t, torch.Tensor)
                                            else None
                                        )
                                        device_tv = logits_by_step[int(step_window[0])].device
                                        seg_log, seg_lc, seg_lw = _build_shifted_ce_segments_step_view_lex_dual(
                                            logits_by_step,
                                            step_window,
                                            include_views,
                                            lc_full_1,
                                            lc_eos_1,
                                            lw_full_1,
                                            lw_eos_1,
                                            L_eff_slice,
                                            device_tv,
                                        )
                                        res_c = _packed_shifted_probs_chunked(
                                            seg_log, seg_lc, device_tv, IGNORE_INDEX, _tv_chunk_max
                                        )
                                        res_w = _packed_shifted_probs_chunked(
                                            seg_log, seg_lw, device_tv, IGNORE_INDEX, _tv_chunk_max
                                        )
                                        tv_n = len(step_window) * len(include_views)
                                        if len(res_c) != tv_n or len(res_w) != tv_n:
                                            raise RuntimeError(
                                                f"truth_ratio T×V packed: expected {tv_n} segments, "
                                                f"got correct={len(res_c)} wrong={len(res_w)}"
                                            )
                                        agg = metric_cfg.get("aggregator", "closer_to_1_better")
                                        ji = 0
                                        for step in step_window:
                                            for view in include_views:
                                                sc = res_c[ji]
                                                sw = res_w[ji]
                                                ji += 1
                                                prob_c = sc.get("prob") if isinstance(sc, dict) else None
                                                al_c = sc.get("avg_loss") if isinstance(sc, dict) else None
                                                prob_w = sw.get("prob") if isinstance(sw, dict) else None
                                                al_w = sw.get("avg_loss") if isinstance(sw, dict) else None
                                                pre_c = {
                                                    "agg_value": prob_c,
                                                    "value_by_index": {
                                                        idx_str: {"prob": prob_c, "avg_loss": al_c},
                                                    },
                                                }
                                                pre_w = {
                                                    "agg_value": prob_w,
                                                    "value_by_index": {
                                                        idx_str: {"prob": prob_w, "avg_loss": al_w},
                                                    },
                                                }
                                                tr_out = metric._metric_fn(
                                                    None,
                                                    collators=kwargs.get("collators"),
                                                    batch_size=kwargs.get("batch_size", 1),
                                                    generation_args=kwargs.get("generation_args", {}),
                                                    aggregator=agg,
                                                    pre_compute={"correct": pre_c, "wrong": pre_w},
                                                )
                                                metric_value = None
                                                if isinstance(tr_out, dict) and tr_out.get("agg_value") is not None:
                                                    metric_value = tr_out["agg_value"]
                                                if metric_value is not None and not (
                                                    isinstance(metric_value, float) and np.isnan(metric_value)
                                                ):
                                                    step_values_by_view[view][traj_name][step][
                                                        metric_name
                                                    ].append(metric_value)
                                        if _trajectory_should_empty_cuda_cache(trajectory_config):
                                            torch.cuda.empty_cache()
                                        continue

                                    if (
                                        _tv_on
                                        and metric.name == "extraction_strength"
                                        and trajectory_config is not None
                                        and trajectory_config.get("use_generalized_sequence_probability", True)
                                        and not _use_cpu_exact
                                    ):
                                        gpu_set_phase(
                                            "trajectory_metric",
                                            metric=metric_name,
                                            batch_idx=batch_idx,
                                            step=-1,
                                        )
                                        F_loc = sample_traj["F"]
                                        S_val = int(sample_traj["S"])
                                        logit_alignment = trajectory_config.get("logit_alignment", "causal")
                                        F_sq = F_loc.squeeze(0) if F_loc.dim() > 1 else F_loc
                                        for step in step_window:
                                            if sample_traj.get("lh") is not None:
                                                lh_loc = sample_traj["lh"]
                                                F_1 = F_loc.unsqueeze(0) if F_loc.dim() == 1 else F_loc
                                                fixation_logits = build_effective_step_fixation_logits_from_history(
                                                    lh_loc, F_1, int(step)
                                                ).squeeze(0)
                                            else:
                                                R_loc = sample_traj["R"]
                                                fixation_logits = build_effective_step_fixation_logits(
                                                    R_loc, F_loc, int(step)
                                                ).squeeze(0)
                                            for view in include_views:
                                                lab_bt = batch_template if view == "full" else batch_template_eos
                                                lab = lab_bt.get("labels")
                                                if lab is None:
                                                    continue
                                                lab_1 = lab.squeeze(0) if lab.dim() > 1 else lab
                                                fl = fixation_logits
                                                if view == "eos":
                                                    Ls = min(L_eff_slice, int(fl.shape[0]))
                                                    fl = fl[:Ls, :]
                                                    lab_1 = lab_1[:Ls]
                                                es_val = extraction_strength_from_fixation(
                                                    fl.float(),
                                                    lab_1,
                                                    F_sq,
                                                    S_val,
                                                    logit_alignment,
                                                    IGNORE_INDEX,
                                                )
                                                step_values_by_view[view][traj_name][step][metric_name].append(
                                                    float(es_val)
                                                )
                                        if _trajectory_should_empty_cuda_cache(trajectory_config):
                                            torch.cuda.empty_cache()
                                        continue

                                    _can_em_tv = (
                                        _tv_on
                                        and metric.name == "exact_memorization"
                                        and _pre_empty
                                        and not _has_list_bt
                                        and not _use_cpu_exact
                                        and metric_name not in exact_mem_post_loop_metrics
                                    )
                                    if _can_em_tv:
                                        gpu_set_phase(
                                            "trajectory_metric",
                                            metric=metric_name,
                                            batch_idx=batch_idx,
                                            step=-1,
                                        )
                                        logits_tv, bt_exp = _exact_mem_tv_stack_logits_and_batch(
                                            logits_by_step,
                                            step_window,
                                            include_views,
                                            batch_template,
                                            batch_template_eos,
                                            L_eff_slice,
                                            IGNORE_INDEX,
                                        )
                                        kwargs_clean = {k: v for k, v in kwargs.items() if k != "tokenizer"}
                                        if primary_data is not None:
                                            kwargs_clean["data"] = primary_data
                                        kwargs_clean["ground_truth"] = ground_truth_str
                                        kwargs_clean["rouge_scorer"] = rouge_scorer_shared
                                        kwargs_clean["sample_traj"] = sample_traj
                                        kwargs_clean["step"] = step_window[0]
                                        kwargs_clean["step_index"] = 0
                                        if trajectory_config is not None:
                                            kwargs_clean["trajectory_config"] = trajectory_config
                                        if guardrail_config_with_pools is not None:
                                            kwargs_clean["guardrail_config"] = guardrail_config_with_pools
                                        kwargs_metric = dict(kwargs_clean)
                                        kwargs_metric["traj_name"] = traj_name
                                        result = _call_metric_at_step(
                                            metric=metric,
                                            logits=logits_tv,
                                            batch_template=bt_exp,
                                            tokenizer=tokenizer,
                                            sample_labels=sample_labels,
                                            sample_input_ids=sample_input_ids,
                                            sample_prompt_len=sample_prompt_len,
                                            metric_config=metric_cfg,
                                            sample_idx=idx_str,
                                            **kwargs_metric,
                                        )
                                        tv_expected = len(step_window) * len(include_views)
                                        if not isinstance(result, list) or len(result) != tv_expected:
                                            raise RuntimeError(
                                                "batched exact_memorization (T×views): expected list of length "
                                                f"{tv_expected}, got {type(result).__name__} len="
                                                f"{len(result) if isinstance(result, list) else 'n/a'}"
                                            )
                                        j = 0
                                        for step in step_window:
                                            for view in include_views:
                                                rd = result[j]
                                                j += 1
                                                metric_value = None
                                                if isinstance(rd, dict):
                                                    if "score" in rd:
                                                        metric_value = rd["score"]
                                                    elif "prob" in rd:
                                                        metric_value = rd["prob"]
                                                if metric_value is not None:
                                                    step_values_by_view[view][traj_name][step][
                                                        metric_name
                                                    ].append(metric_value)
                                        if _trajectory_should_empty_cuda_cache(trajectory_config):
                                            torch.cuda.empty_cache()
                                        continue

                                    can_batch_exact = (
                                        not _tv_on
                                        and (
                                            metric_name in _step_metric_batch_allowlist
                                            or metric.name in _step_metric_batch_allowlist
                                        )
                                        and metric.name == "exact_memorization"
                                        and _pre_empty
                                        and not _has_list_bt
                                        and not _use_cpu_exact
                                    )

                                    if can_batch_exact:
                                        gpu_set_phase(
                                            "trajectory_metric",
                                            metric=metric_name,
                                            batch_idx=batch_idx,
                                            step=-1,
                                        )
                                        kwargs_clean = {k: v for k, v in kwargs.items() if k != "tokenizer"}
                                        if primary_data is not None:
                                            kwargs_clean["data"] = primary_data
                                        kwargs_clean["ground_truth"] = ground_truth_str
                                        kwargs_clean["rouge_scorer"] = rouge_scorer_shared
                                        kwargs_clean["sample_traj"] = sample_traj
                                        kwargs_clean["step"] = step_window[0]
                                        kwargs_clean["step_index"] = 0
                                        if trajectory_config is not None:
                                            kwargs_clean["trajectory_config"] = trajectory_config
                                        if guardrail_config_with_pools is not None:
                                            kwargs_clean["guardrail_config"] = guardrail_config_with_pools

                                        for view in include_views:
                                            bt = batch_template if view == "full" else batch_template_eos
                                            leff = L_eff_slice if view == "eos" else None
                                            logits_batched = _stack_step_logits_for_prob_batch(
                                                logits_by_step, step_window, leff
                                            )
                                            kwargs_metric = dict(kwargs_clean)
                                            kwargs_metric["traj_name"] = traj_name
                                            if metric_name == "hm_aggregate":
                                                kwargs_metric["trajectory_view"] = view
                                            result = _call_metric_at_step(
                                                metric=metric,
                                                logits=logits_batched,
                                                batch_template=bt,
                                                tokenizer=tokenizer,
                                                sample_labels=sample_labels,
                                                sample_input_ids=sample_input_ids,
                                                sample_prompt_len=sample_prompt_len,
                                                metric_config=metric_cfg,
                                                sample_idx=idx_str,
                                                **kwargs_metric,
                                            )
                                            if not isinstance(result, list) or len(result) != len(step_window):
                                                raise RuntimeError(
                                                    "batched exact_memorization: expected list of length "
                                                    f"{len(step_window)}, got {type(result).__name__} len="
                                                    f"{len(result) if isinstance(result, list) else 'n/a'}"
                                                )
                                            for j, step in enumerate(step_window):
                                                rd = result[j]
                                                metric_value = None
                                                if isinstance(rd, dict):
                                                    if "score" in rd:
                                                        metric_value = rd["score"]
                                                    elif "prob" in rd:
                                                        metric_value = rd["prob"]
                                                if metric_value is not None:
                                                    step_values_by_view[view][traj_name][step][metric_name].append(
                                                        metric_value
                                                    )
                                        if _trajectory_should_empty_cuda_cache(trajectory_config):
                                            torch.cuda.empty_cache()
                                        continue

                                    for step in step_window:
                                        logits = logits_by_step[int(step)]

                                        gpu_set_phase(
                                            "trajectory_metric", metric=metric_name, batch_idx=batch_idx, step=step
                                        )

                                        kwargs_clean = {k: v for k, v in kwargs.items() if k != "tokenizer"}
                                        if primary_data is not None:
                                            kwargs_clean["data"] = primary_data
                                        kwargs_clean["ground_truth"] = ground_truth_str
                                        kwargs_clean["rouge_scorer"] = rouge_scorer_shared
                                        kwargs_clean["sample_traj"] = sample_traj
                                        kwargs_clean["step"] = step
                                        try:
                                            kwargs_clean["step_index"] = (
                                                steps_to_use.index(step) if step in steps_to_use else None
                                            )
                                        except (ValueError, NameError):
                                            kwargs_clean["step_index"] = None
                                        if trajectory_config is not None:
                                            kwargs_clean["trajectory_config"] = trajectory_config
                                        if guardrail_config_with_pools is not None:
                                            kwargs_clean["guardrail_config"] = guardrail_config_with_pools

                                        for view in include_views:
                                            bt = batch_template if view == "full" else batch_template_eos
                                            logits_view = logits[:, :L_eff_slice] if view == "eos" else logits
                                            kwargs_metric = dict(kwargs_clean)
                                            kwargs_metric["traj_name"] = traj_name
                                            if metric_name == "hm_aggregate":
                                                kwargs_metric["trajectory_view"] = view

                                            use_cpu_exact = (
                                                executor is not None
                                                and (
                                                    metric_name in _cpu_offload_metrics
                                                    or metric.name in _cpu_offload_metrics
                                                )
                                                and metric.name == "exact_memorization"
                                            )
                                            if use_cpu_exact:
                                                if bt.get("labels") is None:
                                                    raise ValueError(
                                                        "trajectory_cpu_offload_metrics exact_memorization requires labels in batch_template"
                                                    )
                                                logits_lv = trajectory_step_logits_to_prob_batch(
                                                    logits_view
                                                ).squeeze(0)
                                                lab_1d = bt["labels"].squeeze(0)
                                                fut = executor.submit(
                                                    _worker_exact_memorization_cpu,
                                                    np.asarray(
                                                        logits_lv.detach().cpu().numpy(), dtype=np.float32
                                                    ),
                                                    np.asarray(lab_1d.detach().cpu().numpy(), dtype=np.int64),
                                                    IGNORE_INDEX,
                                                )
                                                cpu_metric_futures_this_sample.append(
                                                    (fut, traj_name, step, view, metric_name)
                                                )
                                                continue

                                            result = _call_metric_at_step(
                                                metric=metric,
                                                logits=logits_view,
                                                batch_template=bt,
                                                tokenizer=tokenizer,
                                                sample_labels=sample_labels,
                                                sample_input_ids=sample_input_ids,
                                                sample_prompt_len=sample_prompt_len,
                                                metric_config=metric_cfg,
                                                sample_idx=idx_str,
                                                **kwargs_metric,
                                            )
                                            metric_value = None
                                            if isinstance(result, dict):
                                                if "agg_value" in result:
                                                    metric_value = result["agg_value"]
                                                elif "value_by_index" in result:
                                                    value_by_index = result["value_by_index"]
                                                    if value_by_index:
                                                        first_idx = list(value_by_index.keys())[0]
                                                        first_value = value_by_index[first_idx]
                                                        if isinstance(first_value, dict):
                                                            for key in ["prob", "score", "value"]:
                                                                if key in first_value:
                                                                    metric_value = first_value[key]
                                                                    break
                                                            if metric_value is None:
                                                                for key, value in first_value.items():
                                                                    if isinstance(value, (int, float, np.number)):
                                                                        metric_value = float(value)
                                                                        break
                                                elif "prob" in result:
                                                    metric_value = result["prob"]
                                                elif "score" in result:
                                                    metric_value = result["score"]
                                                else:
                                                    for key, value in result.items():
                                                        if isinstance(value, (int, float, np.number)):
                                                            metric_value = float(value)
                                                            break
                                            elif isinstance(result, list) and len(result) > 0:
                                                result_dict = result[0]
                                                if isinstance(result_dict, dict):
                                                    if "prob" in result_dict:
                                                        metric_value = result_dict["prob"]
                                                    elif "score" in result_dict:
                                                        metric_value = result_dict["score"]
                                                    else:
                                                        for key, value in result_dict.items():
                                                            if isinstance(value, (int, float, np.number)):
                                                                metric_value = float(value)
                                                                break
                                            elif isinstance(result, (int, float, np.number)):
                                                metric_value = float(result)
                                            if (
                                                logger.isEnabledFor(logging.DEBUG)
                                                and metric_name == "extraction_strength"
                                                and step in (steps_to_use[0], steps_to_use[-1])
                                                and metric_value is not None
                                            ):
                                                logger.debug(
                                                    "extraction_strength step=%s sample=%s value=%s",
                                                    step,
                                                    idx_str,
                                                    metric_value,
                                                )
                                            if metric_value is not None:
                                                step_values_by_view[view][traj_name][step][
                                                    metric_name
                                                ].append(metric_value)
                                        if metric_name in (
                                            "extraction_strength",
                                            "truth_ratio",
                                            "ks_test",
                                        ) and _trajectory_should_empty_cuda_cache(trajectory_config):
                                            torch.cuda.empty_cache()

                                except Exception as e:
                                    from evals.metrics.base import RetainReferenceValidationError

                                    if isinstance(e, RetainReferenceValidationError):
                                        raise
                                    logger.error(
                                        f"Error computing {metric_name} for {traj_name}: {e}",
                                        exc_info=True,
                                    )
                                    raise
                            del logits_by_step
                        try:
                            del batch_template_eos
                        except NameError:
                            pass

                    if executor is not None and cpu_metric_futures_this_sample:
                        all_cpu_metric_futures.extend(cpu_metric_futures_this_sample)

                    # Aggressive per-sample cleanup to avoid baseline memory growth across samples (GPU leak).
                    gc.collect()
                    if _trajectory_should_empty_cuda_cache(trajectory_config):
                        torch.cuda.empty_cache()

                _trajectory_append_post_loop_rouge(
                    trajectory_names=trajectory_names,
                    steps_to_use=steps_to_use,
                    include_views=include_views,
                    R=R,
                    lh=lh_batch,
                    F=F,
                    S=S,
                    L=L,
                    B=B,
                    effective_lengths=[int(effective_lengths[b]) for b in range(B)],
                    labels=labels,
                    input_ids=input_ids,
                    batch=batch,
                    indices=indices,
                    prompt_starts=prompt_starts,
                    prompt_lens=prompt_lens,
                    prompt_only_input_ids=_prompt_only_input_ids,
                    tokenizer=tokenizer,
                    metrics_to_run=metrics_to_run,
                    loaded_metrics=loaded_metrics,
                    step_values_by_view=step_values_by_view,
                    executor=executor,
                    all_rouge_futures=all_rouge_futures,
                    kwargs=kwargs,
                )
                _trajectory_append_post_loop_exact_mem_allowlist(
                    trajectory_names=trajectory_names,
                    exact_mem_post_loop_metrics=exact_mem_post_loop_metrics,
                    steps_to_use=steps_to_use,
                    include_views=include_views,
                    R=R,
                    lh=lh_batch,
                    F=F,
                    S=S,
                    L=L,
                    B=B,
                    effective_lengths=[int(effective_lengths[b]) for b in range(B)],
                    labels=labels,
                    input_ids=input_ids,
                    batch=batch,
                    indices=indices,
                    prompt_starts=prompt_starts,
                    prompt_lens=prompt_lens,
                    prompt_only_input_ids=_prompt_only_input_ids,
                    tokenizer=tokenizer,
                    metrics_to_run=metrics_to_run,
                    loaded_metrics=loaded_metrics,
                    step_values_by_view=step_values_by_view,
                    trajectory_config=trajectory_config,
                    primary_data=primary_data,
                    kwargs=kwargs,
                    batch_idx=batch_idx,
                    rouge_scorer_shared=rouge_scorer_shared,
                    guardrail_config_with_pools=guardrail_config_with_pools,
                    view_step_batch_allowlist=_view_step_batch_allowlist,
                    step_prefetch_max_steps=_step_prefetch_max,
                )

                # Packed generalized sequence probability: one packed shifted CE per traj_name over
                # ``B * len(steps_to_use) * len(include_views)`` segments (lexicographic step, view, batch).
                if "probability" in metrics_to_run and labels is not None and _gen_labels_for_packed_prob:
                    device_prob = (
                        lh_batch[0].device
                        if lh_batch is not None
                        else (R.device if R is not None else F.device)
                    )
                    for traj_name in trajectory_names:
                        seg_logits: list[torch.Tensor] = []
                        seg_labels: list[torch.Tensor] = []
                        for step in steps_to_use:
                            for view in include_views:
                                for b in range(B):
                                    gl = _gen_labels_for_packed_prob[b]
                                    L_eff_bp = int(effective_lengths[b])
                                    if traj_name == "steps":
                                        if lh_batch is not None:
                                            logits_step = trajectory_step_logits_to_prob_batch(
                                                lh_batch[int(step)][b]
                                            )
                                        else:
                                            assert R is not None
                                            logits_step = trajectory_step_logits_to_prob_batch(
                                                R[b, :, :, step]
                                            )
                                    else:
                                        if lh_batch is not None:
                                            st_b = {"lh": lh_batch, "b": b, "F": F[b], "S": S, "L": L}
                                        else:
                                            assert R is not None
                                            st_b = {"R": R[b], "F": F[b], "S": S, "L": L}
                                        logits_step = trajectory_step_logits_to_prob_batch(
                                            _get_logits_at_step(st_b, traj_name, step)
                                        )
                                    gl_dev = gl.to(device=device_prob, dtype=torch.long)
                                    if view == "full":
                                        seg_logits.append(logits_step)
                                        seg_labels.append(gl_dev.unsqueeze(0))
                                    else:
                                        Ls = min(L_eff_bp, logits_step.shape[1])
                                        seg_logits.append(
                                            logits_step[:, :Ls, :].contiguous()
                                        )
                                        seg_labels.append(gl_dev[:Ls].unsqueeze(0))
                        probs_out = _packed_shifted_probs_chunked(
                            seg_logits, seg_labels, device_prob, IGNORE_INDEX, _tv_chunk_max
                        )
                        k = 0
                        for step in steps_to_use:
                            for view in include_views:
                                for b in range(B):
                                    if k >= len(probs_out):
                                        raise RuntimeError(
                                            "post-loop packed probability: segment index out of range "
                                            f"(traj={traj_name!r})"
                                        )
                                    pr = probs_out[k]
                                    k += 1
                                    if step not in step_values_by_view[view][traj_name]:
                                        step_values_by_view[view][traj_name][step] = {
                                            m: [] for m in loaded_metrics.keys()
                                        }
                                    if pr and "prob" in pr:
                                        step_values_by_view[view][traj_name][step][
                                            "probability"
                                        ].append(pr["prob"])

                gpu_set_phase("trajectory_batch_end", batch_idx=batch_idx)
                _batch_duration = time.perf_counter() - _batch_t0
                _post_sampler_sec = _batch_duration - _sampler_sec
                logger.info(
                    "trajectory_batch_duration batch_idx=%s batch_size=%s duration_sec=%.2f "
                    "sampler_sec=%.3f post_sampler_sec=%.3f",
                    batch_idx,
                    B,
                    _batch_duration,
                    _sampler_sec,
                    _post_sampler_sec,
                )
                # Release batch-sized GPU data before next batch to avoid holding two batches in memory (OOM with many samples).
                # R and F are references to out["R"] and out["F"]; deleting only 'out' leaves R, F alive (see reports/oom-investigation-why-still-oom.md).
                # logits_history already deleted earlier in the loop after trajectories_from_logits.
                # CRITICAL: sample_traj holds views R[sample_idx], F[sample_idx]; logits (last from inner loop) is a view of R.
                # So long as these exist, R and F storage cannot be freed. Delete them first (fixes GPU memory leak across batches).
                try:
                    del sample_traj, batch_template, logits
                except NameError:
                    pass
                if lh_batch is not None:
                    del lh_batch
                else:
                    del R
                del F
                if should_run_gc(0.9):
                    gc.collect()
                    if _trajectory_should_empty_cuda_cache(trajectory_config):
                        if os.environ.get("DLLM_DEBUG_CUDA_SYNC"):
                            torch.cuda.synchronize()
                        torch.cuda.empty_cache()

            logger.info(
                "[trajectory_batch_phase] complete: expected_batches=%s dataset_key=%s",
                expected_batches, _key,
            )
            if executor is not None and all_rouge_futures:
                for item in all_rouge_futures:
                    future, traj_name, rouge_metrics_in_run, step_or_steps, view = item
                    rouge_scores = future.result()
                    if isinstance(step_or_steps, list):
                        steps_to_use_legacy = step_or_steps
                        for metric_name in rouge_metrics_in_run:
                            metric_cfg = loaded_metrics[metric_name]["config"]
                            rouge_type = metric_cfg.get("rouge_type") or kwargs.get(
                                "rouge_type", "rougeL_f1"
                            )
                            for i, st in enumerate(steps_to_use_legacy):
                                if (
                                    i < len(rouge_scores)
                                    and isinstance(rouge_scores[i], dict)
                                    and rouge_type in rouge_scores[i]
                                ):
                                    step_values_by_view[view][traj_name][st][metric_name].append(
                                        rouge_scores[i][rouge_type]
                                    )
                    else:
                        report_step = int(step_or_steps)
                        for metric_name in rouge_metrics_in_run:
                            metric_cfg = loaded_metrics[metric_name]["config"]
                            rouge_type = metric_cfg.get("rouge_type") or kwargs.get(
                                "rouge_type", "rougeL_f1"
                            )
                            for b_idx, sc in enumerate(rouge_scores):
                                if (
                                    b_idx < len(rouge_scores)
                                    and isinstance(sc, dict)
                                    and rouge_type in sc
                                ):
                                    step_values_by_view[view][traj_name][report_step][
                                        metric_name
                                    ].append(sc[rouge_type])

            if executor is not None and all_cpu_metric_futures:
                for fut, traj_name, step, view, metric_name in all_cpu_metric_futures:
                    score = fut.result()
                    if score is not None:
                        step_values_by_view[view][traj_name][step][metric_name].append(score)

            if _key is None and privleak_accumulators is not None and holdout_dataloader is not None:
                privleak_cfg = privleak_streaming_cfg["privleak_cfg"]
                for h_batch_idx, h_batch in enumerate(holdout_dataloader):
                    gpu_set_phase("privleak_dual_holdout_batch", batch_idx=h_batch_idx)
                    h_input_ids = h_batch["input_ids"]
                    h_labels = h_batch.get("labels")
                    _ = h_batch.get(
                        "index",
                        torch.arange(
                            h_batch_idx * h_input_ids.shape[0],
                            (h_batch_idx + 1) * h_input_ids.shape[0],
                        ),
                    )  # h_indices, reserved
                    h_prompts, h_prompt_lens, h_prompt_starts = _build_prompts_for_sampler(
                        h_input_ids, h_labels, tokenizer, IGNORE_INDEX,
                        prompt_only_input_ids=getattr(holdout_dataloader.dataset, "predict_with_generate", False),
                    )
                    h_sampler_kw = _trajectory_sampler_kwargs(trajectory_config)
                    h_eval_mode = h_sampler_kw.get("evaluation_mode", "unguided")
                    h_sample_kw = dict(
                        inputs=h_prompts,
                        config=None,
                        return_dict=True,
                        return_logits=True,
                        **h_sampler_kw,
                    )
                    if h_eval_mode in ("guided_native", "guided_skew") and h_labels is not None:
                        h_B = h_input_ids.shape[0]
                        h_L_gen = h_sampler_kw.get("max_new_tokens")
                        if h_L_gen is None:
                            h_L_gen = max(
                                h_labels.shape[1] - h_prompt_starts[j]
                                for j in range(h_B)
                            )
                        else:
                            h_L_gen = int(h_L_gen)
                        h_target_sequences = _build_target_sequences_for_sampler(
                            h_labels, h_prompt_starts, h_L_gen, IGNORE_INDEX
                        )
                        h_sample_kw["target_sequences"] = h_target_sequences
                        h_sample_kw["evaluation_mode"] = h_eval_mode
                    h_sampler_output = sampler.sample(**h_sample_kw)
                    h_logits_history = h_sampler_output.logits_history
                    h_fixation_steps = h_sampler_output.fixation_steps
                    if h_logits_history is None or len(h_logits_history) == 0 or h_fixation_steps is None:
                        continue
                    h_out = trajectories_from_logits(
                        h_logits_history,
                        h_fixation_steps,
                        h_prompt_lens,
                        return_trajectory_tensors=False,
                        output_layout=_logits_storage,
                    )
                    h_lh_batch: Optional[List[torch.Tensor]] = None
                    h_R: Optional[torch.Tensor] = None
                    if _logits_storage == "list_history":
                        h_lh_batch = h_out["lh"]
                        h_F, h_S, h_L = h_out["F"], h_out["S"], h_out["L"]
                    else:
                        h_R = h_out["R"]
                        h_F, h_S, h_L = h_out["F"], h_out["S"], h_out["L"]
                    del h_logits_history, h_out
                    h_B = int(h_F.shape[0])
                    h_steps_to_use = [s for s in run_steps_to_use if s < h_S]
                    h_sequences = getattr(h_sampler_output, "sequences", None)
                    h_eos_id = getattr(tokenizer, "eos_token_id", None) if tokenizer else None
                    if h_eos_id is None and trajectory_config:
                        h_eos_id = trajectory_config.get("eos_token_id")
                    _plc_h = privleak_streaming_cfg or {}
                    _dual_h = _plc_h.get("_layout") == "dual"
                    _pv_h = _plc_h.get("_pv_views") or ["full"]
                    _need_h_eff = "eos" in include_views or (
                        _dual_h and len(_pv_h) > 1
                    )
                    if h_sequences is not None and h_sequences.dim() >= 2 and _need_h_eff and h_eos_id is not None:
                        h_effective_lengths = effective_lengths_from_eos(
                            h_sequences, h_prompt_lens, h_L, h_eos_id
                        )
                    else:
                        h_effective_lengths = [h_L] * h_B
                    use_generalized_privleak = trajectory_config.get(
                        "use_generalized_sequence_probability", True
                    )
                    for step in h_steps_to_use:
                        if use_generalized_privleak:
                            per_position_scores_holdout = _per_position_scores_from_R_F_batch(
                                h_R,
                                h_F,
                                h_batch.get("labels"),
                                h_prompt_starts,
                                h_L,
                                trajectory_config,
                                report_step=step,
                                lh_batch=h_lh_batch,
                            )
                            if per_position_scores_holdout is None:
                                continue
                            if not _dual_h:
                                sv = _plc_h.get("_single_view") or "full"
                                if sv == "full":
                                    privleak_accumulators[step].add_holdout_batch(
                                        h_batch,
                                        per_position_scores=per_position_scores_holdout,
                                    )
                                else:
                                    privleak_accumulators[step].add_holdout_batch(
                                        h_batch,
                                        per_position_scores=_truncate_per_position_scores_eos(
                                            per_position_scores_holdout,
                                            list(h_effective_lengths),
                                            h_L,
                                        ),
                                    )
                            else:
                                privleak_accumulators["full"][step].add_holdout_batch(
                                    h_batch,
                                    per_position_scores=per_position_scores_holdout,
                                )
                                privleak_accumulators["eos"][step].add_holdout_batch(
                                    h_batch,
                                    per_position_scores=_truncate_per_position_scores_eos(
                                        per_position_scores_holdout,
                                        list(h_effective_lengths),
                                        h_L,
                                    ),
                                )
                        else:
                            if h_lh_batch is not None:
                                h_logits_batch = h_lh_batch[int(step)]
                            else:
                                assert h_R is not None
                                h_logits_batch = h_R[:, :, :, step].permute(0, 2, 1)
                            if not _dual_h:
                                sv = _plc_h.get("_single_view") or "full"
                                if sv == "full":
                                    privleak_accumulators[step].add_holdout_batch(
                                        h_batch, h_logits_batch
                                    )
                                else:
                                    for i in range(h_B):
                                        Li = min(int(h_effective_lengths[i]), h_L)
                                        lb = h_logits_batch[i : i + 1, :Li, :]
                                        sb = {
                                            k: (
                                                v[i : i + 1]
                                                if torch.is_tensor(v)
                                                and v.dim() > 0
                                                and v.shape[0] == h_B
                                                else v
                                            )
                                            for k, v in h_batch.items()
                                        }
                                        privleak_accumulators[step].add_holdout_batch(sb, lb)
                            else:
                                privleak_accumulators["full"][step].add_holdout_batch(
                                    h_batch, h_logits_batch
                                )
                                for i in range(h_B):
                                    Li = min(int(h_effective_lengths[i]), h_L)
                                    lb = h_logits_batch[i : i + 1, :Li, :]
                                    sb = {
                                        k: (
                                            v[i : i + 1]
                                            if torch.is_tensor(v)
                                            and v.dim() > 0
                                            and v.shape[0] == h_B
                                            else v
                                        )
                                        for k, v in h_batch.items()
                                    }
                                    privleak_accumulators["eos"][step].add_holdout_batch(
                                        sb, lb
                                    )
                    if h_lh_batch is not None:
                        del h_lh_batch
                    else:
                        del h_R
                    del h_F
                    if should_run_gc(0.9):
                        gc.collect()
                        if _trajectory_should_empty_cuda_cache(trajectory_config):
                            torch.cuda.empty_cache()
                _plc_f = privleak_streaming_cfg or {}
                _dual_f = _plc_f.get("_layout") == "dual"
                for step in run_steps_to_use:
                    gpu_set_phase("privleak_dual_step", step=step)
                    ref_logs = kwargs.get("reference_logs") or {}
                    retain_logs = ref_logs.get("retain_model_logs") or {}
                    by_step = retain_logs.get("retain_mia_by_step") or {}
                    step_retain = by_step.get(str(step)) or by_step.get(str(run_steps_to_use.index(step) if step in run_steps_to_use else step))
                    if step_retain is None and retain_logs:
                        from evals.metrics.base import RetainReferenceValidationError
                        raise RetainReferenceValidationError(
                            f"reference_logs was provided but step-matched retain (retain_mia_by_step) not found for step {step!r}. No fallback."
                        )
                    if step_retain is not None:
                        ref_logs = copy.deepcopy(ref_logs)
                        if "retain_model_logs" not in ref_logs:
                            ref_logs["retain_model_logs"] = {}
                        ref_logs["retain_model_logs"] = dict(retain_logs)
                        ref_logs["retain_model_logs"]["retain"] = step_retain
                    _kw_pl = {
                        k: v
                        for k, v in kwargs.items()
                        if k not in ("model", "tokenizer", "pre_compute", "reference_logs")
                    }
                    if not _dual_f:
                        pre_result = privleak_accumulators[step].aggregate()
                        privleak_result = _get_metric_from_registry("privleak")._metric_fn(
                            model=None,
                            pre_compute=pre_result,
                            reference_logs=ref_logs,
                            ref_value=privleak_cfg.get("ref_value", 0.5),
                            **_kw_pl,
                        )
                        pv = privleak_result.get("agg_value")
                        v_target = _plc_f.get("_single_view") or "full"
                        for view in include_views:
                            if view != v_target:
                                continue
                            for traj_name in trajectory_names:
                                if step not in step_values_by_view[view][traj_name]:
                                    step_values_by_view[view][traj_name][step] = {
                                        m: [] for m in loaded_metrics
                                    }
                                step_values_by_view[view][traj_name][step]["privleak"].append(
                                    pv
                                )
                    else:
                        for view in ("full", "eos"):
                            if view not in privleak_accumulators:
                                continue
                            pre_result = privleak_accumulators[view][step].aggregate()
                            privleak_result = _get_metric_from_registry("privleak")._metric_fn(
                                model=None,
                                pre_compute=pre_result,
                                reference_logs=ref_logs,
                                ref_value=privleak_cfg.get("ref_value", 0.5),
                                **_kw_pl,
                            )
                            pv = privleak_result.get("agg_value")
                            if view not in include_views:
                                continue
                            for traj_name in trajectory_names:
                                if step not in step_values_by_view[view][traj_name]:
                                    step_values_by_view[view][traj_name][step] = {
                                        m: [] for m in loaded_metrics
                                    }
                                step_values_by_view[view][traj_name][step]["privleak"].append(
                                    pv
                                )
                privleak_accumulators = None

        # Multi-dataset: run privleak dual trajectory after per-key loops
        if multi_dataset and "forget" in data and "holdout" in data and "privleak" in loaded_metrics and privleak_needs_dual:
            primary_data = data["forget"]
            secondary_data = data["holdout"]
            if use_distributed_sampler:
                dataloader = DataLoader(
                    primary_data,
                    batch_size=batch_size,
                    sampler=DistributedSampler(primary_data, num_replicas=world_size, rank=rank, shuffle=False),
                    collate_fn=collator,
                )
                holdout_dataloader = DataLoader(
                    secondary_data,
                    batch_size=batch_size,
                    sampler=DistributedSampler(secondary_data, num_replicas=world_size, rank=rank, shuffle=False),
                    collate_fn=collator,
                )
            elif sort_by_length:
                dataloader = DataLoader(
                    primary_data,
                    batch_size=batch_size,
                    sampler=LengthSortedSampler(primary_data),
                    collate_fn=collator,
                )
                holdout_dataloader = DataLoader(
                    secondary_data,
                    batch_size=batch_size,
                    sampler=LengthSortedSampler(secondary_data),
                    collate_fn=collator,
                )
            else:
                dataloader = DataLoader(
                    primary_data, batch_size=batch_size, collate_fn=collator
                )
                holdout_dataloader = DataLoader(
                    secondary_data, batch_size=batch_size, collate_fn=collator
                )
            logger.info("Privleak with dual dataset: generating trajectories for forget and holdout")
            gpu_set_phase("privleak_dual_forget")
            forget_traj = _generate_trajectories_for_dataloader(sampler, dataloader, trajectory_config)
            gpu_set_phase("privleak_dual_holdout")
            holdout_traj = _generate_trajectories_for_dataloader(sampler, holdout_dataloader, trajectory_config)
            if forget_traj and holdout_traj:
                trajectories_by_key = {"forget": forget_traj, "holdout": holdout_traj}
                S_dual = next(iter(forget_traj.values()))["S"]
                steps_to_use_dual, _ = _derive_steps_to_use(S_dual, trajectory_config)
                try:
                    device = getattr(model, "device", None) or next(model.parameters()).device
                except (StopIteration, AttributeError):
                    device = torch.device("cpu")
                privleak_cfg = loaded_metrics["privleak"]["config"]
                pre_pc = privleak_cfg.get("pre_compute", {}) or {}
                _md_attack = get_attacker("min_k") if "mia_min_k" in pre_pc else None
                _md_attack_kw = dict(pre_pc.get("mia_min_k", {})) if _md_attack else {}
                use_gen_md = trajectory_config.get(
                    "use_generalized_sequence_probability", True
                )

                def _ref_logs_privleak_md(st: int) -> Any:
                    ref_logs = kwargs.get("reference_logs") or {}
                    retain_logs = ref_logs.get("retain_model_logs") or {}
                    by_step = retain_logs.get("retain_mia_by_step") or {}
                    step_retain = by_step.get(str(st)) or by_step.get(
                        str(
                            steps_to_use_dual.index(st)
                            if st in steps_to_use_dual
                            else st
                        )
                    )
                    if step_retain is None and retain_logs:
                        from evals.metrics.base import RetainReferenceValidationError

                        raise RetainReferenceValidationError(
                            f"reference_logs was provided but step-matched retain (retain_mia_by_step) not found for step {st!r}. No fallback."
                        )
                    if step_retain is not None:
                        ref_logs = copy.deepcopy(ref_logs)
                        if "retain_model_logs" not in ref_logs:
                            ref_logs["retain_model_logs"] = {}
                        ref_logs["retain_model_logs"] = dict(retain_logs)
                        ref_logs["retain_model_logs"]["retain"] = step_retain
                    return ref_logs

                _kw_priv = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ("model", "tokenizer", "pre_compute", "reference_logs")
                }

                for step in steps_to_use_dual:
                    gpu_set_phase("privleak_dual_step", step=step)
                    ref_logs = _ref_logs_privleak_md(step)
                    pv_full = None
                    if "full" in include_views:
                        logits_by_key = {}
                        for key, traj_by_idx in trajectories_by_key.items():
                            logits_by_key[key] = {
                                idx: _get_logits_at_step(traj, "steps", step)
                                for idx, traj in traj_by_idx.items()
                            }
                        dual_wrapper = DualLogitModelWrapper(logits_by_key, device)
                        kwargs_priv = {
                            "data": {"forget": primary_data, "holdout": secondary_data},
                            "collators": collator,
                            "batch_size": batch_size,
                            **{
                                k: v
                                for k, v in kwargs.items()
                                if k not in ("tokenizer", "model", "data", "collators")
                            },
                        }
                        pre_result = _compute_pre_compute_metrics_at_step(
                            pre_compute_config=privleak_cfg.get("pre_compute", {}),
                            logits=next(iter(logits_by_key["forget"].values())),
                            batch_template={},
                            tokenizer=tokenizer,
                            sample_labels=None,
                            sample_input_ids=torch.zeros(1),
                            sample_prompt_len=0,
                            sample_idx="0",
                            model_wrapper_override=dual_wrapper,
                            **kwargs_priv,
                        )
                        privleak_result = _get_metric_from_registry("privleak")._metric_fn(
                            model=dual_wrapper,
                            pre_compute=pre_result,
                            reference_logs=ref_logs,
                            ref_value=privleak_cfg.get("ref_value", 0.5),
                            **_kw_priv,
                        )
                        pv_full = privleak_result.get("agg_value")
                        for traj_name in trajectory_names:
                            if step not in step_values_by_view["full"][traj_name]:
                                step_values_by_view["full"][traj_name][step] = {
                                    m: [] for m in loaded_metrics
                                }
                            step_values_by_view["full"][traj_name][step][
                                "privleak"
                            ].append(pv_full)

                    if "eos" in include_views:
                        if not (use_gen_md and _md_attack is not None):
                            raise ValueError(
                                "Trajectory privleak (multi_dataset, eos view): requires "
                                "trajectory_config.use_generalized_sequence_probability=true "
                                "and privleak pre_compute mia_min_k. "
                                "Eos privleak is not copied from full (no hidden fallback)."
                            )
                        try:
                            _dev_md = (
                                getattr(model, "device", None)
                                or next(model.parameters()).device
                            )
                        except (StopIteration, AttributeError):
                            _dev_md = torch.device("cpu")
                        acc_md = MIAStreamingAccumulator(
                            _md_attack,
                            collator,
                            batch_size,
                            _dev_md,
                            **_md_attack_kw,
                        )
                        prompt_only_f = getattr(
                            primary_data, "predict_with_generate", False
                        )
                        prompt_only_h = getattr(
                            secondary_data, "predict_with_generate", False
                        )
                        for f_batch in dataloader:
                            f_ids = f_batch["input_ids"]
                            f_labels = f_batch.get("labels")
                            f_idx = f_batch.get(
                                "index",
                                torch.arange(f_ids.shape[0]),
                            )
                            f_B = f_ids.shape[0]
                            _, _, f_ps = _build_prompts_for_sampler(
                                f_ids,
                                f_labels,
                                tokenizer,
                                IGNORE_INDEX,
                                prompt_only_input_ids=prompt_only_f,
                            )
                            f_R = torch.stack(
                                [
                                    forget_traj[
                                        str(
                                            f_idx[j].item()
                                            if torch.is_tensor(f_idx[j])
                                            else f_idx[j]
                                        )
                                    ]["R"]
                                    for j in range(f_B)
                                ]
                            )
                            f_F = torch.stack(
                                [
                                    forget_traj[
                                        str(
                                            f_idx[j].item()
                                            if torch.is_tensor(f_idx[j])
                                            else f_idx[j]
                                        )
                                    ]["F"]
                                    for j in range(f_B)
                                ]
                            )
                            f_L = int(forget_traj[str(f_idx[0].item())]["L"])
                            f_eff = [
                                forget_traj[
                                    str(
                                        f_idx[j].item()
                                        if torch.is_tensor(f_idx[j])
                                        else f_idx[j]
                                    )
                                ].get("effective_length", f_L)
                                for j in range(f_B)
                            ]
                            f_pp = _per_position_scores_from_R_F_batch(
                                f_R,
                                f_F,
                                f_labels,
                                f_ps,
                                f_L,
                                trajectory_config,
                                report_step=step,
                            )
                            if f_pp is not None:
                                acc_md.add_forget_batch(
                                    f_batch,
                                    per_position_scores=_truncate_per_position_scores_eos(
                                        f_pp, f_eff, f_L
                                    ),
                                )
                        for h_batch in holdout_dataloader:
                            h_ids = h_batch["input_ids"]
                            h_labels = h_batch.get("labels")
                            h_idx = h_batch.get(
                                "index",
                                torch.arange(h_ids.shape[0]),
                            )
                            h_B = h_ids.shape[0]
                            _, _, h_ps = _build_prompts_for_sampler(
                                h_ids,
                                h_labels,
                                tokenizer,
                                IGNORE_INDEX,
                                prompt_only_input_ids=prompt_only_h,
                            )
                            h_R = torch.stack(
                                [
                                    holdout_traj[
                                        str(
                                            h_idx[j].item()
                                            if torch.is_tensor(h_idx[j])
                                            else h_idx[j]
                                        )
                                    ]["R"]
                                    for j in range(h_B)
                                ]
                            )
                            h_F = torch.stack(
                                [
                                    holdout_traj[
                                        str(
                                            h_idx[j].item()
                                            if torch.is_tensor(h_idx[j])
                                            else h_idx[j]
                                        )
                                    ]["F"]
                                    for j in range(h_B)
                                ]
                            )
                            h_Lt = int(holdout_traj[str(h_idx[0].item())]["L"])
                            h_eff = [
                                holdout_traj[
                                    str(
                                        h_idx[j].item()
                                        if torch.is_tensor(h_idx[j])
                                        else h_idx[j]
                                    )
                                ].get("effective_length", h_Lt)
                                for j in range(h_B)
                            ]
                            h_pp = _per_position_scores_from_R_F_batch(
                                h_R,
                                h_F,
                                h_labels,
                                h_ps,
                                h_Lt,
                                trajectory_config,
                                report_step=step,
                            )
                            if h_pp is not None:
                                acc_md.add_holdout_batch(
                                    h_batch,
                                    per_position_scores=_truncate_per_position_scores_eos(
                                        h_pp, h_eff, h_Lt
                                    ),
                                )
                        pre_eos = acc_md.aggregate()
                        pr_e = _get_metric_from_registry("privleak")._metric_fn(
                            model=None,
                            pre_compute=pre_eos,
                            reference_logs=ref_logs,
                            ref_value=privleak_cfg.get("ref_value", 0.5),
                            **_kw_priv,
                        )
                        pv_eos = pr_e.get("agg_value")
                        if pv_eos is None:
                            logger.error(
                                "Trajectory privleak (multi_dataset, eos): agg_value is None at "
                                "step=%s (empty accumulators or privleak failure). "
                                "Not substituting full-view privleak.",
                                step,
                            )
                            raise ValueError(
                                f"Trajectory privleak (multi_dataset, eos): agg_value is None at step {step}"
                            )
                        for traj_name in trajectory_names:
                            if step not in step_values_by_view["eos"][traj_name]:
                                step_values_by_view["eos"][traj_name][step] = {
                                    m: [] for m in loaded_metrics
                                }
                            step_values_by_view["eos"][traj_name][step][
                                "privleak"
                            ].append(pv_eos)
            else:
                logger.warning("Privleak dual trajectories empty, skipping privleak")

        def _distribution_for_step(vals_clean: list) -> dict:
            """Per-step distribution: mean, std, median, p25, p75, min, max, 95% CI."""
            arr = np.array(vals_clean, dtype=np.float64)
            n = int(np.sum(~np.isnan(arr)))
            nan = np.nan
            if n == 0:
                return {
                    "mean": nan,
                    "std": nan,
                    "median": nan,
                    "p25": nan,
                    "p75": nan,
                    "min": nan,
                    "max": nan,
                    "ci_low": nan,
                    "ci_high": nan,
                }
            mean = float(np.nanmean(arr))
            if n == 1:
                std = 0.0
                ci_low = ci_high = mean
            else:
                std = float(np.nanstd(arr))
                se = std / np.sqrt(n)
                ci_low = mean - 1.96 * se
                ci_high = mean + 1.96 * se
            return {
                "mean": mean,
                "std": std,
                "median": float(np.nanpercentile(arr, 50)),
                "p25": float(np.nanpercentile(arr, 25)),
                "p75": float(np.nanpercentile(arr, 75)),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        # Aggregate results per view: compute mean and per-step distribution for each step
        agg_value_by_view = {}
        step_distribution_by_view = {}
        for view in include_views:
            step_values = step_values_by_view[view]
            agg_value = {}
            step_distribution = {}
            for traj_name in trajectory_names:
                agg_value[traj_name] = {}
                step_distribution[traj_name] = {}
                for metric_name in loaded_metrics.keys():
                    step_metric_values = {}
                    if traj_name in step_values:
                        for step, metrics_dict in step_values[traj_name].items():
                            if metric_name in metrics_dict and len(metrics_dict[metric_name]) > 0:
                                step_metric_values[step] = metrics_dict[metric_name]
                    if step_metric_values:
                        ordered_steps = (
                            run_steps_to_use
                            if run_steps_to_use is not None
                            else sorted(step_metric_values.keys())
                        )
                        aggregated = []
                        dist_means, dist_stds, dist_medians = [], [], []
                        dist_p25, dist_p75, dist_mins, dist_maxs = [], [], [], []
                        dist_ci_low, dist_ci_high = [], []
                        for step in ordered_steps:
                            if step in step_metric_values:
                                vals = step_metric_values[step]
                                vals_clean = [float(v) if v is not None else np.nan for v in vals]
                                aggregated.append(np.nanmean(vals_clean) if vals_clean else np.nan)
                                d = _distribution_for_step(vals_clean)
                                dist_means.append(d["mean"])
                                dist_stds.append(d["std"])
                                dist_medians.append(d["median"])
                                dist_p25.append(d["p25"])
                                dist_p75.append(d["p75"])
                                dist_mins.append(d["min"])
                                dist_maxs.append(d["max"])
                                dist_ci_low.append(d["ci_low"])
                                dist_ci_high.append(d["ci_high"])
                            else:
                                aggregated.append(np.nan)
                                dist_means.append(np.nan)
                                dist_stds.append(np.nan)
                                dist_medians.append(np.nan)
                                dist_p25.append(np.nan)
                                dist_p75.append(np.nan)
                                dist_mins.append(np.nan)
                                dist_maxs.append(np.nan)
                                dist_ci_low.append(np.nan)
                                dist_ci_high.append(np.nan)
                        agg_value[traj_name][metric_name] = np.array(aggregated)
                        step_distribution[traj_name][metric_name] = {
                            "mean": np.array(dist_means),
                            "std": np.array(dist_stds),
                            "median": np.array(dist_medians),
                            "p25": np.array(dist_p25),
                            "p75": np.array(dist_p75),
                            "min": np.array(dist_mins),
                            "max": np.array(dist_maxs),
                            "ci_low": np.array(dist_ci_low),
                            "ci_high": np.array(dist_ci_high),
                        }
                    else:
                        agg_value[traj_name][metric_name] = np.array([])
                        step_distribution[traj_name][metric_name] = {
                            "mean": np.array([]),
                            "std": np.array([]),
                            "median": np.array([]),
                            "p25": np.array([]),
                            "p75": np.array([]),
                            "min": np.array([]),
                            "max": np.array([]),
                            "ci_low": np.array([]),
                            "ci_high": np.array([]),
                        }
            agg_value_by_view[view] = agg_value
            step_distribution_by_view[view] = step_distribution

        if logger.isEnabledFor(logging.DEBUG):
            _debug_log_trajectory_metric_coverage(
                agg_value_by_view,
                loaded_metrics,
                trajectory_names,
                include_views,
                trajectory_config,
            )

        # Build trajectory step metadata so results can interpret step indices (which diffusion/unmasked-token step each index is).
        sampler_kwargs = trajectory_config.get("sampler_kwargs", {})
        num_trajectory_steps = 0
        first_view = include_views[0] if include_views else "full"
        agg_first = agg_value_by_view.get(first_view) or {}
        if agg_first and "steps" in agg_first and agg_first["steps"]:
            first_metric_arr = next(iter(agg_first["steps"].values()), None)
            if first_metric_arr is not None and len(first_metric_arr) > 0:
                num_trajectory_steps = int(len(first_metric_arr))
        trajectory_sample_interval = sampler_kwargs.get("trajectory_sample_interval")
        max_new_tokens = sampler_kwargs.get("max_new_tokens")
        step_meaning = (
            "unmasked_tokens_approx"
            if trajectory_sample_interval is not None and trajectory_sample_interval > 0
            else "diffusion_step"
        )
        # Actual step values: use run_step_values_metadata when step subsampling was used; else derive from interval
        step_values = None
        if run_step_values_metadata is not None and len(run_step_values_metadata) > 0:
            step_values = list(run_step_values_metadata)
        elif (
            step_meaning == "unmasked_tokens_approx"
            and trajectory_sample_interval is not None
            and max_new_tokens is not None
            and num_trajectory_steps > 0
        ):
            step_values = [
                min(k * trajectory_sample_interval, max_new_tokens)
                for k in range(num_trajectory_steps)
            ]
        trajectory_step_metadata = {
            "num_trajectory_steps": num_trajectory_steps,
            "trajectory_sample_interval": trajectory_sample_interval,
            "max_new_tokens": max_new_tokens,
            "step_meaning": step_meaning,
        }
        if step_values is not None:
            trajectory_step_metadata["step_values"] = step_values
        if effective_length_by_index:
            trajectory_step_metadata["effective_length_by_index"] = effective_length_by_index
        if prompt_len_by_index:
            trajectory_step_metadata["prompt_len_by_index"] = prompt_len_by_index

        if logger.isEnabledFor(logging.DEBUG) and num_trajectory_steps > 0:
            fv0 = include_views[0] if include_views else "full"
            prob_arr = (agg_value_by_view.get(fv0) or {}).get("steps", {}).get(
                "probability", np.array([])
            )
            prob_len = int(np.asarray(prob_arr).size)
            logger.debug(
                "TRAJECTORY_STEP_META num_trajectory_steps=%s step_values_count=%s "
                "probability_on_steps_traj_len=%s lengths_match=%s",
                num_trajectory_steps,
                len(step_values) if step_values is not None else 0,
                prob_len,
                prob_len == num_trajectory_steps,
            )

        # Single path: build one result dict; display part (per-name or single key), then canonical keys once.
        internal_names = list(loaded_metrics.keys())
        if full_display_order and len(full_display_order) >= len(full_internal_order):
            try:
                display_names = [
                    full_display_order[full_internal_order.index(k)]
                    for k in internal_names
                    if k in full_internal_order
                ]
            except ValueError:
                display_names = full_display_order[: len(internal_names)]
        else:
            display_names = list(full_display_order)[: len(internal_names)] if full_display_order else []
        display_key = (
            full_display_order[0]
            if full_display_order
            else kwargs.get("metric_name", "trajectory_all")
        )

        result = {}
        if len(display_names) == len(internal_names) and len(internal_names) > 0:
            for display_name, internal_name in zip(display_names, internal_names):
                result[display_name] = {
                    "agg_value": {
                        view: {
                            traj: {internal_name: agg_value_by_view[view][traj][internal_name]}
                            for traj in trajectory_names
                        }
                        for view in include_views
                    },
                    "value_by_index": {},
                    "step_distribution": {
                        view: {
                            traj: {internal_name: step_distribution_by_view[view][traj][internal_name]}
                            for traj in trajectory_names
                        }
                        for view in include_views
                    },
                }
            result["trajectory_step_metadata"] = {
                "agg_value": None,
                "trajectory_step_metadata": trajectory_step_metadata,
            }
        else:
            result[display_key] = {
                "agg_value": agg_value_by_view,
                "value_by_index": {},
                "step_distribution": step_distribution_by_view,
            }
            result["trajectory_step_metadata"] = {
                "agg_value": None,
                "trajectory_step_metadata": trajectory_step_metadata,
            }

        eval_cfg = kwargs.get("eval_cfg")
        if eval_cfg is not None and getattr(eval_cfg, "get", lambda k, d=None: d)("retain_reference_mode", False):
            first_view = include_views[0] if include_views else "full"
            step_vals = step_values_by_view.get(first_view, {}).get("steps", {})
            if "privleak" in loaded_metrics:
                arr = agg_value_by_view.get(first_view, {}).get("steps", {}).get("privleak", np.array([]))
                if hasattr(arr, "__len__") and len(arr) > 0:
                    ordered_steps = (
                        list(run_steps_to_use)
                        if run_steps_to_use is not None
                        else sorted(step_values_by_view.get(first_view, {}).get("steps", {}).keys(), key=lambda x: (x if isinstance(x, (int, float)) else 0))
                    )
                    if len(ordered_steps) < len(arr):
                        ordered_steps = list(range(len(arr)))
                    mia_by_step = {}
                    for i in range(len(arr)):
                        v = arr[i]
                        step = ordered_steps[i] if i < len(ordered_steps) else i
                        mia_by_step[str(step)] = {"agg_value": float(v) if hasattr(v, "item") else v}
                    result["mia_min_k_by_step"] = mia_by_step
                    first_mia = next(iter(mia_by_step.values()), None)
                    if first_mia is not None and isinstance(first_mia.get("agg_value"), (int, float)):
                        result["mia_min_k"] = {"agg_value": first_mia["agg_value"]}
                    else:
                        mean_val = float(np.nanmean(arr)) if hasattr(arr, "__len__") and len(arr) > 0 else 0.0
                        result["mia_min_k"] = {"agg_value": mean_val}
            if "truth_ratio" in loaded_metrics and step_vals:
                ordered_steps = (
                    list(run_steps_to_use)
                    if run_steps_to_use is not None
                    else sorted(step_vals.keys(), key=lambda x: (x if isinstance(x, (int, float)) else 0))
                )
                tr_by_step = {}
                for step in ordered_steps:
                    if step in step_vals and "truth_ratio" in step_vals[step]:
                        vals = step_vals[step]["truth_ratio"]
                        value_by_index = {
                            str(i): {"score": float(v) if v is not None else None}
                            for i, v in enumerate(vals)
                        }
                        tr_by_step[str(step)] = {"value_by_index": value_by_index}
                if tr_by_step:
                    result["forget_truth_ratio_by_step"] = tr_by_step
                    first_ftr = next(iter(tr_by_step.values()), None)
                    if first_ftr is not None and isinstance(first_ftr.get("value_by_index"), dict):
                        result["forget_truth_ratio"] = dict(first_ftr)
                    elif first_ftr is not None and isinstance(first_ftr.get("agg_value"), (int, float)):
                        result["forget_truth_ratio"] = {"agg_value": first_ftr["agg_value"]}
                    else:
                        all_scores = []
                        for st_v in tr_by_step.values():
                            vbi = st_v.get("value_by_index") or {}
                            for ent in vbi.values():
                                if isinstance(ent, dict) and "score" in ent and isinstance(ent["score"], (int, float)):
                                    all_scores.append(ent["score"])
                        agg = float(np.nanmean(all_scores)) if all_scores else 0.0
                        result["forget_truth_ratio"] = {"agg_value": agg}

        # Expose retain MU components per step so reports show which of Prob/ROUGE/Truth_Ratio is 0 when hm_aggregate is 0.
        retain_agg_by_step = kwargs.get("retain_agg_by_step") or {}
        if "hm_aggregate" in loaded_metrics and retain_agg_by_step:
            def _is_mu_key(x):
                return (
                    str(x).startswith("retain_")
                    or str(x).startswith("ra_")
                    or str(x).startswith("wf_")
                )
            _rk = next(iter(retain_agg_by_step.keys()), None)
            _per_traj = _rk in ("steps", "fixation_start", "fixation_end", "fixation_ratio")
            steps_dict = retain_agg_by_step.get("steps", retain_agg_by_step) if _per_traj else retain_agg_by_step
            components = {}
            for step_key, pre in steps_dict.items():
                sk = str(step_key)
                nested_by_view = False
                if isinstance(pre, dict):
                    for _v in ("full", "eos"):
                        pv0 = pre.get(_v)
                        if isinstance(pv0, dict) and any(_is_mu_key(x) for x in pv0):
                            nested_by_view = True
                            break
                if nested_by_view:
                    components[sk] = {}
                    for view in ("full", "eos"):
                        if view not in pre:
                            continue
                        pv = pre[view]
                        if not isinstance(pv, dict):
                            continue
                        components[sk][view] = {}
                        for name, ent in pv.items():
                            if _is_mu_key(name) and isinstance(ent, dict) and "agg_value" in ent:
                                v = ent["agg_value"]
                                components[sk][view][name] = (
                                    float(v) if isinstance(v, (int, float, np.floating)) else v
                                )
                else:
                    components[sk] = {}
                    if isinstance(pre, dict):
                        for name, ent in pre.items():
                            if _is_mu_key(name) and isinstance(ent, dict) and "agg_value" in ent:
                                v = ent["agg_value"]
                                components[sk][name] = (
                                    float(v) if isinstance(v, (int, float, np.floating)) else v
                                )
            if components:
                result["retain_mu_components_by_step"] = components

        if executor is not None:
            executor.shutdown(wait=True)
        _eval_cfg = kwargs.get("eval_cfg")
        if _eval_cfg is not None and callable(getattr(_eval_cfg, "get", None)):
            _dec = _eval_cfg.get("decoupling")
            if _dec is not None:
                _statuses = None
                if callable(getattr(_dec, "get", None)):
                    _statuses = _dec.get("applicability_statuses")
                elif isinstance(_dec, dict):
                    _statuses = _dec.get("applicability_statuses")
                if isinstance(_statuses, str) and _statuses:
                    feature_applicability = {}
                    for pair in _statuses.split(","):
                        if ":" not in pair:
                            continue
                        k, v = pair.split(":", 1)
                        feature_applicability[str(k)] = {"applicability_status": str(v)}
                    if feature_applicability:
                        result["feature_applicability"] = feature_applicability
        _tp_done = kwargs.get("trajectory_pass_id")
        if _tp_done:
            from evals.metrics.trajectory_pass_envelope import build_pass_envelope

            result["pass_envelope"] = build_pass_envelope(str(_tp_done))
        return result


