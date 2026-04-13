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
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union
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
    build_fixation_logits_from_R_F,
    compute_prob_from_fixation_logits as _compute_prob_from_fixation_logits,
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


def should_run_gc(threshold: float = 0.9) -> bool:
    """Return True if CUDA is available and VRAM usage (allocated/total) is >= threshold."""
    if not torch.cuda.is_available():
        return False
    total = torch.cuda.get_device_properties(0).total_memory
    if total <= 0:
        return False
    return (torch.cuda.memory_allocated() / total) >= threshold


def _per_position_scores_from_R_F_batch(
    R: torch.Tensor,
    F: torch.Tensor,
    labels: Optional[torch.Tensor],
    prompt_starts: List[int],
    L: int,
    trajectory_config: Dict[str, Any],
    report_step: Optional[int] = None,
) -> Optional[List[List[float]]]:
    """Build per-sample per-position probability scores from R, F for use with Min-K etc.

    prompt_starts[i] is the start index of the generation region in labels (use for labels[i, start:start+L]).
    If report_step is set, uses effective-step logits at that step (s_eff(ell,s)=min(s,F[ell])).
    Returns list of list of float (one list per sample), or None if labels missing.
    """
    if labels is None:
        return None
    B = R.shape[0]
    logit_alignment = trajectory_config.get("logit_alignment", "causal")
    provider = FixationStepWiseScoreProvider(logit_alignment=logit_alignment)
    out: List[List[float]] = []
    for i in range(B):
        start = prompt_starts[i] if isinstance(prompt_starts[i], int) else int(prompt_starts[i].item())
        gen_labels = labels[i, start : start + L]
        batch_prov = {"labels": gen_labels.unsqueeze(0)}
        model_or_logits: Dict[str, Any] = {"R": R[i].unsqueeze(0), "F": F[i].unsqueeze(0)}
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
    """Get [V, L] logits at a trajectory step. traj must have R, F, S, L (on-demand format)."""
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
        if torch.cuda.is_available():
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
        if torch.cuda.is_available():
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
        if torch.cuda.is_available():
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
        if torch.cuda.is_available():
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

    for pre_metric_name, pre_metric_cfg in pre_compute_config.items():
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
    # Decode logits to text via argmax
    if logits.dim() == 3:
        logits = logits[0]  # [L, V]
    predicted_tokens = torch.argmax(logits, dim=-1)  # [L]
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
    
    # Ensure logits are in [B, L, V] format
    if logits.dim() == 2:
        # [V, L] -> transpose to [L, V] then add batch dim -> [1, L, V]
        logits = logits.transpose(0, 1).unsqueeze(0)
    elif logits.dim() == 3 and logits.shape[0] == 1:
        # [1, L, V] - already correct
        pass
    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")
    
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
        metric_worker_pool_size = trajectory_config.get("metric_worker_pool_size", 4)
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
                _batch_elapsed = time.perf_counter() - _batch_t0
                gpu_set_phase("trajectory_after_sampler", batch_idx=batch_idx)
                if (batch_idx % _log_interval == 0 or batch_idx == 0 or batch_idx == expected_batches - 1) and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Batch %s/%s: diffusion sampling done in %.1fs",
                        batch_idx + 1, expected_batches, _batch_elapsed,
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

                out = trajectories_from_logits(
                    logits_history, fixation_steps, prompt_lens, return_trajectory_tensors=False
                )
                R, F, S, L = out["R"], out["F"], out["S"], out["L"]
                del logits_history # Release list of tensors immediately after stacking/slicing

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
                                R, F, labels, prompt_starts, L, trajectory_config,
                                report_step=step,
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
                            logits_list = [
                                _get_logits_at_step(
                                    {"R": R[i], "F": F[i], "S": S, "L": L}, "steps", step
                                ).T
                                for i in range(B)
                            ]
                            logits_batch = torch.stack(logits_list, dim=0)
                            if not _dual_pl:
                                sv = _plc.get("_single_view") or "full"
                                if sv == "full":
                                    privleak_accumulators[step].add_forget_batch(
                                        batch, logits_batch
                                    )
                                else:
                                    for i in range(B):
                                        Li = min(int(effective_lengths[i]), L)
                                        lb = logits_list[i][:Li].unsqueeze(0)
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
                                    lb = logits_list[i][:Li].unsqueeze(0)
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

                # Process each sample in batch (each sample uses its own R, F; logits computed on-demand)
                for sample_idx in range(B):
                    idx_str = str(indices[sample_idx].item() if torch.is_tensor(indices[sample_idx]) else indices[sample_idx])
                    sample_traj = {"R": R[sample_idx], "F": F[sample_idx], "S": S, "L": L}
                    use_generalized = (
                        trajectory_config.get("use_generalized_sequence_probability", True)
                        if trajectory_config else False
                    )

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

                    # Extract only the generated portion of labels to match logits shape [V, L]
                # Logits from trajectory only cover generated tokens (L), not the prompt
                # evaluate_probability does: logits[..., :-1, :] and labels[..., 1:]
                # So if logits are [1, L, V], after processing: logits [1, L-1, V], labels [1, L-1]
                    # This means we need labels of length L to get L-1 after shift
                    if sample_labels is not None:
                        # Extract generated region at generation start (not prompt_lens; wrong when left-padded)
                        generated_labels = sample_labels[sample_generation_start:sample_generation_start + L]
                        # Pad with IGNORE_INDEX if needed (shouldn't happen, but safety check)
                        if generated_labels.shape[0] < L:
                            padding = torch.full(
                                (L - generated_labels.shape[0],),
                                IGNORE_INDEX,
                                dtype=generated_labels.dtype,
                                device=generated_labels.device
                            )
                            generated_labels = torch.cat([generated_labels, padding])
                    else:
                        generated_labels = None
                    if generated_labels is not None:
                        assert generated_labels.shape[0] == L, (
                            "batch_template invariant: generated_labels length must equal L; "
                            "got %s, L=%s" % (generated_labels.shape[0], L)
                        )
            
                    # Create batch template for logit metrics
                    # Use generated token IDs so metrics that use input_ids (e.g. mia_min_k via tokenwise_logprobs)
                    # score log P(actual next token) at each position, not dummy zeros.
                    generated_input_ids = sample_input_ids[sample_generation_start : sample_generation_start + L]
                    if generated_input_ids.shape[0] < L:
                        padding = torch.zeros(
                            L - generated_input_ids.shape[0],
                            dtype=generated_input_ids.dtype,
                            device=sample_input_ids.device,
                        )
                        generated_input_ids = torch.cat([generated_input_ids, padding])
                    batch_template = {
                        "input_ids": generated_input_ids.unsqueeze(0),
                        "labels": generated_labels.unsqueeze(0) if generated_labels is not None else None,
                        "attention_mask": torch.ones((1, L), dtype=torch.long, device=sample_input_ids.device),  # All positions valid
                        "index": torch.tensor([int(idx_str)], dtype=torch.long, device=sample_input_ids.device),  # Required by run_batchwise_evals
                    }
                    # Add labels_correct/labels_wrong for truth_ratio pre_compute (dual-answer dataset).
                    # Use each row's content start; support N wrong options (batch[key] 3D [B,N,L]).
                    for key in ("labels_correct", "labels_wrong"):
                        if key in batch:
                            batch_template[key] = _batch_template_dual_labels(
                                batch, sample_idx, key, L, IGNORE_INDEX
                            )

                    # Decode ground truth once per sample for ROUGE-only path (reuse across steps and trajectory types)
                    if generated_labels is not None:
                        valid_labels_gt = generated_labels[generated_labels != IGNORE_INDEX]
                        ground_truth_str = (
                            tokenizer.decode(valid_labels_gt.tolist(), skip_special_tokens=True)
                            if len(valid_labels_gt) > 0
                            else ""
                        )
                    else:
                        ground_truth_str = ""
                    rouge_scorer_instance = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
                    rouge_futures_this_sample = []

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

                        rouge_metrics_in_run = [
                            m for m in metrics_to_run
                            if loaded_metrics[m]["metric"].name == "rouge"
                        ]
                        if rouge_metrics_in_run:
                            pred_token_lists = []
                            for step in steps_to_use:
                                logits_s = _get_logits_at_step(sample_traj, traj_name, step)
                                if logits_s.dim() == 3:
                                    logits_s = logits_s[0]
                                pred_tokens = torch.argmax(logits_s, dim=-1)
                                pred_token_lists.append(pred_tokens.tolist())
                            gen_texts_full = tokenizer.batch_decode(
                                pred_token_lists, skip_special_tokens=True
                            )
                            gen_texts_eos = [
                                tokenizer.decode(
                                    (p[:L_eff_b] if len(p) > L_eff_b else p),
                                    skip_special_tokens=True,
                                )
                                for p in pred_token_lists
                            ]
                            if executor is None:
                                for view in include_views:
                                    gen_texts = gen_texts_full if view == "full" else gen_texts_eos
                                    rouge_scores = eval_rouge_recall_batch(
                                        gen_texts,
                                        [ground_truth_str] * len(steps_to_use),
                                        use_stemmer=True,
                                        scorer=rouge_scorer_instance,
                                    )
                                    for metric_name in rouge_metrics_in_run:
                                        metric_cfg = loaded_metrics[metric_name]["config"]
                                        rouge_type = metric_cfg.get("rouge_type") or kwargs.get("rouge_type", "rougeL_f1")
                                        for i, step in enumerate(steps_to_use):
                                            if i < len(rouge_scores) and isinstance(rouge_scores[i], dict) and rouge_type in rouge_scores[i]:
                                                step_values_by_view[view][traj_name][step][metric_name].append(rouge_scores[i][rouge_type])
                            else:
                                for view in include_views:
                                    gen_texts = gen_texts_full if view == "full" else gen_texts_eos
                                    future = executor.submit(
                                        eval_rouge_recall_batch_worker,
                                        gen_texts,
                                        [ground_truth_str] * len(steps_to_use),
                                        True,
                                    )
                                    rouge_futures_this_sample.append(
                                        (future, traj_name, rouge_metrics_in_run, steps_to_use, view)
                                    )

                        if "probability" in metrics_to_run and generated_labels is not None:
                            use_generalized = trajectory_config.get(
                                "use_generalized_sequence_probability", True
                            )
                            if use_generalized:
                                # Use traj_name-specific logits so probability differs by trajectory type (steps/fixation_start/etc).
                                device = _get_logits_at_step(
                                    sample_traj, traj_name, steps_to_use[0]
                                ).device
                                labels_full = generated_labels.to(device=device, dtype=torch.long)
                                for i, step in enumerate(steps_to_use):
                                    logits_step = trajectory_step_logits_to_prob_batch(
                                        _get_logits_at_step(
                                            sample_traj, traj_name, step
                                        )
                                    )
                                    for view in include_views:
                                        if view == "full":
                                            logits_v = logits_step
                                            lab = labels_full.unsqueeze(0)
                                        else:
                                            L_eff_slice = min(
                                                L_eff_b, logits_step.shape[1]
                                            )
                                            logits_v = logits_step[
                                                :, :L_eff_slice, :
                                            ].contiguous()
                                            lab = labels_full[
                                                :L_eff_slice
                                            ].unsqueeze(0)
                                        prob_results = _compute_prob_from_fixation_logits(
                                            logits_v, lab, device, IGNORE_INDEX
                                        )
                                        if prob_results and "prob" in prob_results[0]:
                                            step_values_by_view[view][traj_name][step][
                                                "probability"
                                            ].append(prob_results[0]["prob"])
                                    if torch.cuda.is_available():
                                        del logits_step
                                        torch.cuda.synchronize()
                                        torch.cuda.empty_cache()
                            else:
                                # Process probability per step to avoid OOM: stacking all steps
                                # (num_steps x L x V) can exceed GPU memory (e.g. 25 x 200 x 126k).
                                device = _get_logits_at_step(
                                    sample_traj, traj_name, steps_to_use[0]
                                ).device
                                labels_full = generated_labels.to(device=device, dtype=torch.long)
                                for i, step in enumerate(steps_to_use):
                                    logits_step = trajectory_step_logits_to_prob_batch(
                                        _get_logits_at_step(
                                            sample_traj, traj_name, step
                                        )
                                    )
                                    for view in include_views:
                                        if view == "full":
                                            logits_v = logits_step
                                            lab = labels_full.unsqueeze(0)
                                        else:
                                            L_eff_slice = min(
                                                L_eff_b, logits_step.shape[1]
                                            )
                                            logits_v = logits_step[
                                                :, :L_eff_slice, :
                                            ].contiguous()
                                            lab = labels_full[
                                                :L_eff_slice
                                            ].unsqueeze(0)
                                        prob_results = _compute_prob_from_fixation_logits(
                                            logits_v, lab, device, IGNORE_INDEX
                                        )
                                        if prob_results and "prob" in prob_results[0]:
                                            step_values_by_view[view][traj_name][step][
                                                "probability"
                                            ].append(prob_results[0]["prob"])
                                    if torch.cuda.is_available():
                                        del logits_step
                                        torch.cuda.synchronize()
                                        torch.cuda.empty_cache()

                        for step in steps_to_use:
                            # Trajectory-sliced logits per layout; truth_ratio nested probability uses them when traj_name is passed into pre_compute.
                            logits = _get_logits_at_step(sample_traj, traj_name, step)  # [V, L]

                            # Build eos batch_template (sliced to L_eff_slice) for eos view.
                            # Use helper so labels_correct/labels_wrong (list of tensors) are sliced too (probability invariant).
                            L_eff_slice = min(L_eff_b, batch_template["input_ids"].shape[1])
                            batch_template_eos = _slice_batch_template_to_length(batch_template, L_eff_slice)

                            # Compute each requested metric (skip rouge and probability; already batched above) for each view
                            for metric_name, metric_info in [(m, loaded_metrics[m]) for m in metrics_to_run]:
                                try:
                                    metric = metric_info["metric"]
                                    metric_cfg = metric_info["config"]

                                    if metric_name == "privleak" and trajectories_by_key is not None:
                                        continue
                                    if metric_name in rouge_metrics_in_run or metric_name == "probability":
                                        continue
                                    # When producing reference (no reference provided), skip metrics that only compare to reference.
                                    if metric_name == "ks_test":
                                        ref_logs = kwargs.get("reference_logs") or {}
                                        if not ref_logs.get("retain_model_logs"):
                                            continue

                                    gpu_set_phase("trajectory_metric", metric=metric_name, batch_idx=batch_idx, step=step)

                                    kwargs_clean = {k: v for k, v in kwargs.items() if k != "tokenizer"}
                                    if primary_data is not None:
                                        kwargs_clean["data"] = primary_data
                                    kwargs_clean["ground_truth"] = ground_truth_str
                                    kwargs_clean["rouge_scorer"] = rouge_scorer_instance
                                    kwargs_clean["sample_traj"] = sample_traj
                                    kwargs_clean["step"] = step
                                    try:
                                        kwargs_clean["step_index"] = steps_to_use.index(step) if step in steps_to_use else None
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
                                            **kwargs_metric
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
                                            step_values_by_view[view][traj_name][step][metric_name].append(metric_value)
                                    # Return GPU memory after metrics that allocate large log_probs/contiguous logits
                                    # so the next metric (e.g. hm_aggregate) or next step does not OOM (see .monitor/oom-investigation-exact-cause.md).
                                    if metric_name in ("extraction_strength", "truth_ratio", "ks_test") and torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                        
                                except Exception as e:
                                    from evals.metrics.base import RetainReferenceValidationError
                                    if isinstance(e, RetainReferenceValidationError):
                                        raise
                                    logger.error(
                                        f"Error computing {metric_name} at step {step} for {traj_name}: {e}",
                                        exc_info=True
                                    )
                                    raise
                            # Release per-step tensors so baseline memory does not grow across steps (fixes GPU leak).
                            try:
                                del batch_template_eos, logits
                            except NameError:
                                pass

                    if executor is not None and rouge_futures_this_sample:
                        all_rouge_futures.extend(rouge_futures_this_sample)

                    # Aggressive per-sample cleanup to avoid baseline memory growth across samples (GPU leak).
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                gpu_set_phase("trajectory_batch_end", batch_idx=batch_idx)
                _batch_duration = time.perf_counter() - _batch_t0
                logger.info(
                    "trajectory_batch_duration batch_idx=%s batch_size=%s duration_sec=%.2f",
                    batch_idx,
                    B,
                    _batch_duration,
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
                del out, R, F
                if should_run_gc(0.9):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

            logger.info(
                "[trajectory_batch_phase] complete: expected_batches=%s dataset_key=%s",
                expected_batches, _key,
            )
            if executor is not None and all_rouge_futures:
                for item in all_rouge_futures:
                    if len(item) == 5:
                        future, traj_name, rouge_metrics_in_run, steps_to_use, view = item
                    else:
                        future, traj_name, rouge_metrics_in_run, steps_to_use = item
                        view = "full"
                    rouge_scores = future.result()
                    for metric_name in rouge_metrics_in_run:
                        metric_cfg = loaded_metrics[metric_name]["config"]
                        rouge_type = metric_cfg.get("rouge_type") or kwargs.get("rouge_type", "rougeL_f1")
                        for i, step in enumerate(steps_to_use):
                            if i < len(rouge_scores) and isinstance(rouge_scores[i], dict) and rouge_type in rouge_scores[i]:
                                step_values_by_view[view][traj_name][step][metric_name].append(rouge_scores[i][rouge_type])

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
                        h_logits_history, h_fixation_steps, h_prompt_lens, return_trajectory_tensors=False
                    )
                    h_R, h_F, h_S, h_L = h_out["R"], h_out["F"], h_out["S"], h_out["L"]
                    del h_logits_history, h_out
                    h_B = h_R.shape[0] if hasattr(h_R, "shape") else len(h_R)
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
                                h_R, h_F, h_batch.get("labels"), h_prompt_starts, h_L, trajectory_config,
                                report_step=step,
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
                            h_logits_list = [
                                _get_logits_at_step(
                                    {"R": h_R[i], "F": h_F[i], "S": h_S, "L": h_L}, "steps", step
                                ).T
                                for i in range(h_B)
                            ]
                            h_logits_batch = torch.stack(h_logits_list, dim=0)
                            if not _dual_h:
                                sv = _plc_h.get("_single_view") or "full"
                                if sv == "full":
                                    privleak_accumulators[step].add_holdout_batch(
                                        h_batch, h_logits_batch
                                    )
                                else:
                                    for i in range(h_B):
                                        Li = min(int(h_effective_lengths[i]), h_L)
                                        lb = h_logits_list[i][:Li].unsqueeze(0)
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
                                    lb = h_logits_list[i][:Li].unsqueeze(0)
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
                    del h_R, h_F
                    if should_run_gc(0.9):
                        gc.collect()
                        if torch.cuda.is_available():
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


