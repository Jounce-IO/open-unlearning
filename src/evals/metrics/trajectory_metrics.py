"""
Trajectory-based metrics for dLLM unlearning evaluation.

This module computes metrics at each diffusion step across three trajectory types
(steps, fixation, ratio), supporting any metric from the open-unlearning framework.

Trajectory evals use interval mode only (trajectory_sample_interval, default 8).
Every-step mode (no interval) is not used.
"""

import gc
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union
from torch.utils.data import DataLoader
from omegaconf import ListConfig, DictConfig

from evals.metrics.base import unlearning_metric
from evals.metrics.samplers import LengthSortedSampler
from evals.metrics.utils import (
    evaluate_probability,
    evaluate_probability_confidence_ordered,
    eval_rouge_recall_batch,
    eval_rouge_recall_batch_worker,
    tokenwise_vocab_logprobs,
    IGNORE_INDEX,
    _tensor_to_list_of_floats,
)
from rouge_score import rouge_scorer
from evals.metrics.step_wise_score import (
    FixationStepWiseScoreProvider,
    build_effective_step_fixation_logits,
    build_fixation_logits_from_R_F,
    sequence_probability_from_scores,
    extraction_strength_from_fixation,
)
from evals.metrics.trajectory_utils import (
    trajectories_from_logits,
    effective_lengths_from_eos,
    compute_trajectories,
    compute_fixation_start_trajectory,
    compute_fixation_end_trajectory,
    compute_fixation_ratio_trajectory,
    extract_logits_at_step,
    decode_logits_to_text,
)
from evals.metrics.trajectory_adapters import (
    LogitModelWrapper,
    DualLogitModelWrapper,
    compute_logit_metric_at_step,
    compute_text_metric_at_step,
)
from evals.metrics.mia.utils import get_attacker, MIAStreamingAccumulator
from evals.gpu_phase_logger import set_phase as gpu_set_phase

logger = logging.getLogger("evaluator")

# IGNORE_INDEX from data.utils
IGNORE_INDEX = -100

# Trajectory evals use interval mode only; every-step mode is not used.
DEFAULT_TRAJECTORY_SAMPLE_INTERVAL = 8


EVALUATION_MODES = ("unguided", "guided_native", "guided_skew")


def _trajectory_sampler_kwargs(trajectory_config: Union[Dict, DictConfig]) -> dict:
    """Return sampler_kwargs with trajectory_sample_interval defaulting to 8 when return_logits is used.

    Also passes evaluation_mode from trajectory_config (default "unguided"); allowed values:
    unguided, guided_native, guided_skew.
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
    return kwargs


def _build_prompts_for_sampler(
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor],
    tokenizer: Any,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[list[list[int]], list[int]]:
    """Build prompt token lists and lengths for sampler.sample(inputs=...).

    Supports two data conventions:
    - Labels use ignore_index for the prompt: prompt = input_ids[:, :prompt_end] with
      prompt_end = first index where labels != ignore_index.
    - Prompt-only input_ids (e.g. predict_with_generate): when prompt_end would be 0,
      use the non-pad tokens of input_ids as the prompt so TOFU/MUSE work correctly.
    """
    B = input_ids.shape[0]
    prompts: list[list[int]] = []
    prompt_lens: list[int] = []
    _pad = getattr(tokenizer, "pad_token_id", None) if tokenizer else None
    pad_token_id = _pad if isinstance(_pad, (int, float)) else None
    for i in range(B):
        if labels is not None:
            label_mask = labels[i] != ignore_index
            if label_mask.any():
                prompt_end = label_mask.nonzero()[0][0].item()
            else:
                prompt_end = input_ids.shape[1]
        else:
            prompt_end = input_ids.shape[1]
        if prompt_end == 0 and pad_token_id is not None:
            non_pad = (input_ids[i] != pad_token_id).view(-1)
            prompt_len_from_input = non_pad.sum().item()
            if prompt_len_from_input > 0:
                prompt = input_ids[i][non_pad].cpu().tolist()
                prompts.append(prompt)
                prompt_lens.append(len(prompt))
                continue
        prompt = input_ids[i, :prompt_end].cpu().tolist()
        prompts.append(prompt)
        prompt_lens.append(len(prompt))
    return prompts, prompt_lens


def _build_target_sequences_for_sampler(
    labels: torch.Tensor,
    prompt_lens: List[int],
    L: int,
    ignore_index: int = IGNORE_INDEX,
) -> List[List[int]]:
    """Build target token lists for the generated region only (one list per batch sample, length L).

    For each sample j, takes labels[j, prompt_lens[j]:prompt_lens[j]+L]. If the slice is shorter
    than L, pads with ignore_index so the sampler receives exactly L tokens per sample.
    """
    B = labels.shape[0]
    target_sequences = []
    for j in range(B):
        start = prompt_lens[j]
        end = min(start + L, labels.shape[1])
        row = labels[j, start:end].cpu().tolist()
        if len(row) < L:
            row = row + [ignore_index] * (L - len(row))
        target_sequences.append(row)
    return target_sequences


def should_run_gc(threshold: float = 0.9) -> bool:
    """Return True if CUDA is available and VRAM usage (allocated/total) is >= threshold."""
    if not torch.cuda.is_available():
        return False
    total = torch.cuda.get_device_properties(0).total_memory
    if total <= 0:
        return False
    return (torch.cuda.memory_allocated() / total) >= threshold


def _compute_prob_from_fixation_logits(
    fixation_logits: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    ignore_index: int = IGNORE_INDEX,
) -> List[Dict[str, float]]:
    """Compute per-sample probability (exp(-avg_loss)) from fixation logits and labels.

    Handles length mismatch (e.g. sampler returns shorter sequence than padded batch labels)
    by trimming to min length so cross_entropy does not raise. Counts num_token_gt only
    over used (non-ignore) positions. Uses no_grad to avoid retaining gradient buffers (OOM).
    """
    with torch.no_grad():
        fixation_logits = fixation_logits.to(device)
        labels = labels.to(device)
        B, T_fl, V = fixation_logits.shape
        T_lab = labels.shape[1]
        T = min(T_fl, T_lab)
        fixation_logits = fixation_logits[:, :T, :].contiguous()
        labels = labels[:, :T].contiguous()
        shifted_logits = fixation_logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        if shifted_logits.dtype in (torch.bfloat16, torch.float16):
            shifted_logits = shifted_logits.float()
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
        losses = loss_fn(shifted_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
        num_token_gt = (shifted_labels != ignore_index).sum(dim=-1).clamp(min=1)
        avg_losses = losses / num_token_gt
        normalized_probs = torch.exp(-avg_losses)
        # Single GPU→CPU transfer per tensor; release GPU tensors promptly
        avg_losses_list = _tensor_to_list_of_floats(avg_losses)
        normalized_probs_list = _tensor_to_list_of_floats(normalized_probs)
        return [
            {"prob": prob, "avg_loss": avg_loss}
            for prob, avg_loss in zip(normalized_probs_list, avg_losses_list)
        ]


def _per_position_scores_from_R_F_batch(
    R: torch.Tensor,
    F: torch.Tensor,
    labels: Optional[torch.Tensor],
    prompt_lens: List[int],
    L: int,
    trajectory_config: Dict[str, Any],
    report_step: Optional[int] = None,
) -> Optional[List[List[float]]]:
    """Build per-sample per-position probability scores from R, F for use with Min-K etc.

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
        pl = prompt_lens[i] if isinstance(prompt_lens[i], int) else int(prompt_lens[i].item())
        gen_labels = labels[i, pl : pl + L]
        batch_prov = {"labels": gen_labels.unsqueeze(0)}
        model_or_logits: Dict[str, Any] = {"R": R[i].unsqueeze(0), "F": F[i].unsqueeze(0)}
        if report_step is not None:
            model_or_logits["report_step"] = report_step
        results = provider.get_per_position_scores(
            model_or_logits, batch_prov, ignore_index=IGNORE_INDEX
        )
        out.append(results[0][0] if results and results[0][0] else [])
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
        prompts = []
        prompt_lens = []
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
            prompts.append(input_ids[i, :prompt_end].cpu().tolist())
            prompt_lens.append(len(prompts[-1]))

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
                L_gen = max(labels.shape[1] - prompt_lens[i] for i in range(B))
            else:
                L_gen = int(L_gen)
            target_sequences = _build_target_sequences_for_sampler(
                labels, prompt_lens, L_gen, IGNORE_INDEX
            )
            sample_kw["target_sequences"] = target_sequences
            sample_kw["evaluation_mode"] = evaluation_mode
        sampler_output = sampler.sample(**sample_kw)
        logits_history = sampler_output.logits_history
        fixation_steps = sampler_output.fixation_steps
        if logits_history is None or len(logits_history) == 0:
            continue
        if fixation_steps is None:
            continue

        out = trajectories_from_logits(
            logits_history, fixation_steps, prompt_lens, return_trajectory_tensors=False
        )
        R, F, S, L = out["R"], out["F"], out["S"], out["L"]
        for i in range(R.shape[0]):
            idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
            trajectories_by_idx[str(idx)] = {
                "R": R[i],
                "F": F[i],
                "S": S,
                "L": L,
            }
        del logits_history, out
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
        **kwargs: Additional kwargs to pass to pre_compute metrics
    
    Returns:
        Dict mapping access_key (or metric name) to metric results:
        {
            "correct": {"agg_value": ..., "value_by_index": {sample_idx: {...}}},
            "wrong": {"agg_value": ..., "value_by_index": {sample_idx: {...}}}
        }
    """
    trajectory_config = trajectory_config or kwargs.get("trajectory_config")
    sample_traj = sample_traj or kwargs.get("sample_traj")
    pre_compute_results = {}

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
            and trajectory_config.get("use_generalized_sequence_probability", False)
            and sample_traj is not None
            and handler_name == "probability"
        )
        if use_generalized:
            try:
                R = sample_traj["R"]
                F = sample_traj["F"]
                logit_alignment = trajectory_config.get("logit_alignment", "causal")
                provider = FixationStepWiseScoreProvider(logit_alignment=logit_alignment)
                lab = batch_template.get(labels_field if labels_field else "labels")
                if lab is not None:
                    lab = lab.squeeze(0) if lab.dim() > 1 else lab
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
                                sample_idx: {"prob": prob_val, "avg_loss": avg_loss_val},
                            },
                        }
                    else:
                        pre_result = {
                            "agg_value": None,
                            "value_by_index": {sample_idx: {"prob": None, "avg_loss": None}},
                        }
                else:
                    pre_result = {
                        "agg_value": None,
                        "value_by_index": {sample_idx: {"prob": None, "avg_loss": None}},
                    }
                pre_compute_results[access_key] = pre_result
            except Exception as e:
                logger.warning(
                    f"Error computing generalized pre-compute probability for {pre_metric_name}: {e}",
                    exc_info=True,
                )
                pre_compute_results[access_key] = {
                    "agg_value": None,
                    "value_by_index": {sample_idx: {"prob": None, "avg_loss": None}},
                }
            continue

        # Compute pre-compute metric at this step
        # Note: Pre-compute metrics might have their own pre_compute requirements
        # We handle this recursively
        try:
            # Substitute labels with labels_field if specified (for truth_ratio dual-answer)
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
            
            # Structure result in the format expected by main metrics
            # Main metrics expect: {"agg_value": ..., "value_by_index": {idx: {...}}}
            if isinstance(pre_result, dict):
                if "value_by_index" in pre_result:
                    # Already in correct format, but ensure sample_idx is present
                    value_by_index = pre_result["value_by_index"]
                    if sample_idx not in value_by_index:
                        # Extract value from result and add to value_by_index
                        if "agg_value" in pre_result:
                            value_by_index[sample_idx] = {"prob": pre_result["agg_value"]}
                        elif len(value_by_index) > 0:
                            # Use first value as template
                            first_idx = list(value_by_index.keys())[0]
                            value_by_index[sample_idx] = value_by_index[first_idx].copy()
                elif "agg_value" in pre_result:
                    # Create value_by_index with single entry
                    value_by_index = {sample_idx: {"prob": pre_result["agg_value"]}}
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
                        value_by_index = {sample_idx: {"prob": value}}
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
                            "value_by_index": {sample_idx: {"prob": None}},
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
                        "value_by_index": {sample_idx: result_dict.copy()},
                    }
                else:
                    pre_result = {
                        "agg_value": None,
                        "value_by_index": {sample_idx: {"prob": None}},
                    }
            else:
                logger.warning(
                    f"Unexpected pre-compute result format for {pre_metric_name}: {type(pre_result)}"
                )
                pre_result = {
                    "agg_value": None,
                    "value_by_index": {sample_idx: {"prob": None}},
                }
            
            pre_compute_results[access_key] = pre_result
            
        except Exception as e:
            logger.warning(
                f"Error computing pre-compute metric {pre_metric_name} at step: {e}",
                exc_info=True
            )
            # Return None result so main metric can handle it
            pre_compute_results[access_key] = {
                "agg_value": None,
                "value_by_index": {sample_idx: {"prob": None}},
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

    ground_truth = kwargs.get("ground_truth")
    rouge_scorer_instance = kwargs.get("rouge_scorer")
    use_rouge_only = (
        metric_name == "rouge"
        and ground_truth is not None
        and rouge_scorer_instance is not None
    )

    if use_rouge_only:
        from evals.metrics.utils import eval_rouge_recall_batch

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
                logger.debug(
                    f"ROUGE {rouge_type} (rouge-only path): gen_len={len(gen_text)}, gt_len={len(ground_truth)}, score={score}"
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
                logger.debug(
                    f"ROUGE {rouge_type}: gen_len={len(gen_text)}, gt_len={len(ground_truth)}, score={score}"
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
    # Remove model, tokenizer, and pre_compute from kwargs (we pass computed pre_compute below)
    kwargs_clean = {k: v for k, v in kwargs.items() if k not in ["model", "tokenizer", "pre_compute"]}
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
    # trajectory_model_utility: hm_aggregate needs retain sub-metrics from reference_logs (no per-step pre_compute)
    elif metric.name == "hm_aggregate":
        ref_logs = kwargs.get("reference_logs") or {}
        retain_logs = ref_logs.get("retain_model_logs") or {}
        model_utility_keys = ("retain_Q_A_Prob", "retain_Q_A_ROUGE", "retain_Truth_Ratio")
        pre_compute_from_ref = {}
        for key in model_utility_keys:
            if key in retain_logs and isinstance(retain_logs[key], dict) and retain_logs[key].get("agg_value") is not None:
                pre_compute_from_ref[key] = retain_logs[key]
        if pre_compute_from_ref:
            metric_kwargs["pre_compute"] = pre_compute_from_ref
    
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
        and trajectory_config.get("use_generalized_sequence_probability", False)
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
    
    # Try using batch function if available
    if metric_name in batch_function_map:
        batch_fn = batch_function_map[metric_name]
        try:
            # Batch functions like evaluate_probability only accept (model, batch)
            # Don't pass any other kwargs
            result = batch_fn(model=model_wrapper, batch=batch)
            return result
        except Exception as e:
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
            # Use generic text-based handler
            logger.debug(
                f"Metric {metric_name} appears to be text-based (requires generation). "
                f"Using generic text-based handler."
            )
            try:
                # Get original logits (before reshaping)
                original_logits = logits  # Already in [1, L, V] format
                if original_logits.dim() == 3:
                    original_logits = original_logits[0]  # [L, V]
                original_logits = original_logits.transpose(0, 1)  # [V, L] for decode function
                
                result = _handle_text_based_metric(
                    logits=original_logits,
                    tokenizer=tokenizer,
                    sample_labels=sample_labels,
                    sample_input_ids=sample_input_ids,
                    sample_prompt_len=sample_prompt_len,
                    metric_name=metric_name,
                    metric_config=metric_config,
                    **kwargs
                )
                return result
            except Exception as text_e:
                logger.warning(
                    f"Error in generic text-based handler for {metric_name}: {text_e}. "
                    f"Original error: {e}",
                    exc_info=True
                )
                return None
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
                original_logits = logits
                if original_logits.dim() == 3:
                    original_logits = original_logits[0]  # [L, V]
                original_logits = original_logits.transpose(0, 1)  # [V, L]
                
                result = _handle_text_based_metric(
                    logits=original_logits,
                    tokenizer=tokenizer,
                    sample_labels=sample_labels,
                    sample_input_ids=sample_input_ids,
                    sample_prompt_len=sample_prompt_len,
                    metric_name=metric_name,
                    metric_config=metric_config,
                    **kwargs
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
        metric_worker_pool_size = trajectory_config.get("metric_worker_pool_size", 0)
        executor = (
            ProcessPoolExecutor(max_workers=metric_worker_pool_size)
            if metric_worker_pool_size > 0
            else None
        )
        logits_source = trajectory_config.get("logits_source", "sampler")
        data = kwargs.get("data")
        collator = kwargs.get("collators")
        batch_size = kwargs.get("batch_size", 1)
        sort_by_length = kwargs.get("sort_by_length", False)
        tokenizer = kwargs.get("tokenizer")
        generation_args = kwargs.get("generation_args", {})
    
        if not metrics_config:
            raise ValueError("No metrics specified in config")
    
        if not tokenizer:
            raise ValueError("tokenizer is required for trajectory metrics")

        # When model is DiffusionModelAdapter, set use_fixation_logits so __call__ returns
        # fixation logits (trajectory run). Default True for trajectory metrics.
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

        # Full order for display-name mapping (before optional filter)
        full_internal_order = list(metrics_to_compute.keys())
        full_display_order = list(kwargs.get("metric_display_names") or [])

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

        # One-time warning if hm_aggregate (trajectory_model_utility) will have no pre_compute
        if "hm_aggregate" in loaded_metrics:
            ref_logs = kwargs.get("reference_logs") or {}
            retain_logs = ref_logs.get("retain_model_logs") or {}
            model_utility_keys = ("retain_Q_A_Prob", "retain_Q_A_ROUGE", "retain_Truth_Ratio")
            have = [k for k in model_utility_keys if retain_logs.get(k) and isinstance(retain_logs.get(k), dict) and retain_logs[k].get("agg_value") is not None]
            if len(have) < len(model_utility_keys):
                logger.warning(
                    "trajectory_model_utility (hm_aggregate) will be None: set eval.tofu_trajectory.retain_logs_path to a retain run JSON containing retain_Q_A_Prob, retain_Q_A_ROUGE, retain_Truth_Ratio. reference_logs keys: %s",
                    list(retain_logs.keys()) if retain_logs else "(retain_logs_path not set or file missing)",
                )

        # Handle multi-dataset only when there are keys beyond forget/holdout (e.g. MUSE: forget_knowmem, retain_knowmem, forget_verbmem, forget, holdout)
        multi_dataset = (
            isinstance(data, dict)
            and bool(set(data.keys()) - {"forget", "holdout"})
        )
        if isinstance(data, dict) and "forget" in data and "holdout" in data and not multi_dataset:
            primary_data = data["forget"]
            secondary_data = data["holdout"]
        else:
            primary_data = data if not multi_dataset else None
            secondary_data = None

        single_dataset_keys = [k for k in data if k not in ("forget", "holdout")] if multi_dataset else []

        # Create dataloader(s)
        if not multi_dataset:
            if sort_by_length:
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
            if sort_by_length:
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
            privleak_streaming_cfg = {
                "device": _device,
                "privleak_cfg": privleak_cfg,
                "attack_cls": attack_cls,
                "attack_cls_name": attack_cls_name,
                "attack_kwargs": attack_kwargs,
            }

        run_steps_to_use = None
        run_step_values_metadata = None

        keys_to_process = [None] if not multi_dataset else single_dataset_keys
        for _key in keys_to_process:
            if _key is not None:
                primary_data = data[_key]
                if sort_by_length:
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

            n_samples = len(dataloader.dataset)
            expected_batches = (n_samples + batch_size - 1) // batch_size
            logger.info(
                f"Trajectory forget dataset: {n_samples} samples, batch_size {batch_size}, "
                f"expected batches: {expected_batches} (last batch index: {expected_batches - 1})"
            )
            all_rouge_futures: list = []
            effective_length_by_index: dict[str, int] = {}
            prompt_len_by_index: dict[str, int] = {}
        # Process each batch
            for batch_idx, batch in enumerate(dataloader):
                gpu_set_phase("trajectory_batch_start", batch_idx=batch_idx)
                input_ids = batch["input_ids"]
                labels = batch.get("labels")
                attention_mask = batch.get("attention_mask")
                indices = batch.get("index", torch.arange(batch_idx * batch_size, 
                                                          (batch_idx + 1) * batch_size))
                B = input_ids.shape[0]
        
                # Prepare inputs for sampler (list of token sequences)
                prompts, prompt_lens = _build_prompts_for_sampler(
                    input_ids, labels, tokenizer, IGNORE_INDEX
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
                            labels.shape[1] - prompt_lens[j] for j in range(B)
                        )
                    else:
                        L_gen = int(L_gen)
                    target_sequences = _build_target_sequences_for_sampler(
                        labels, prompt_lens, L_gen, IGNORE_INDEX
                    )
                    sample_kw["target_sequences"] = target_sequences
                    sample_kw["evaluation_mode"] = evaluation_mode
                sampler_output = sampler.sample(**sample_kw)
                gpu_set_phase("trajectory_after_sampler", batch_idx=batch_idx)

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
                if sequences is not None and sequences.dim() >= 2 and "eos" in include_views:
                    effective_lengths = effective_lengths_from_eos(
                        sequences, prompt_lens, L, eos_token_id
                    )
                else:
                    effective_lengths = [L] * B

                if run_steps_to_use is None:
                    run_steps_to_use, run_step_values_metadata = _derive_steps_to_use(
                        S, trajectory_config
                    )
                steps_to_use = [s for s in run_steps_to_use if s < S]
                if (
                    use_streaming_privleak
                    and privleak_streaming_cfg is not None
                    and privleak_streaming_cfg.get("attack_cls") is not None
                    and privleak_accumulators is None
                    and _key is None
                ):
                    cfg = privleak_streaming_cfg
                    privleak_accumulators = {
                        step: MIAStreamingAccumulator(
                            cfg["attack_cls"],
                            collator,
                            batch_size,
                            cfg["device"],
                            **cfg["attack_kwargs"],
                        )
                        for step in run_steps_to_use
                    }
                if privleak_accumulators is not None and _key is None:
                    use_generalized_privleak = trajectory_config.get(
                        "use_generalized_sequence_probability", False
                    )
                    for step in steps_to_use:
                        if use_generalized_privleak:
                            per_position_scores_forget = _per_position_scores_from_R_F_batch(
                                R, F, labels, prompt_lens, L, trajectory_config,
                                report_step=step,
                            )
                            if per_position_scores_forget is not None:
                                privleak_accumulators[step].add_forget_batch(
                                    batch, per_position_scores=per_position_scores_forget
                                )
                        else:
                            logits_list = [
                                _get_logits_at_step(
                                    {"R": R[i], "F": F[i], "S": S, "L": L}, "steps", step
                                ).T
                                for i in range(B)
                            ]
                            logits_batch = torch.stack(logits_list, dim=0)
                            privleak_accumulators[step].add_forget_batch(batch, logits_batch)
                gpu_set_phase("trajectory_after_trajectories", batch_idx=batch_idx)

                # Process each sample in batch (each sample uses its own R, F; logits computed on-demand)
                for sample_idx in range(B):
                    idx_str = str(indices[sample_idx].item() if torch.is_tensor(indices[sample_idx]) else indices[sample_idx])
                    sample_traj = {"R": R[sample_idx], "F": F[sample_idx], "S": S, "L": L}
                    use_generalized = (
                        trajectory_config.get("use_generalized_sequence_probability", False)
                        if trajectory_config else False
                    )

                    # Get ground truth for this sample
                    sample_labels = labels[sample_idx] if labels is not None else None
                    sample_input_ids = input_ids[sample_idx]
                    sample_prompt_len = prompt_lens[sample_idx]

                    # Extract only the generated portion of labels to match logits shape [V, L]
                # Logits from trajectory only cover generated tokens (L), not the prompt
                # evaluate_probability does: logits[..., :-1, :] and labels[..., 1:]
                # So if logits are [1, L, V], after processing: logits [1, L-1, V], labels [1, L-1]
                    # This means we need labels of length L to get L-1 after shift
                    if sample_labels is not None:
                        # Extract generated region: from prompt_end to prompt_end + L
                        # L is now the generated length (not full sequence length)
                        generated_labels = sample_labels[sample_prompt_len:sample_prompt_len + L]
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
            
                    # Create batch template for logit metrics
                    # Use generated token IDs so metrics that use input_ids (e.g. mia_min_k via tokenwise_logprobs)
                    # score log P(actual next token) at each position, not dummy zeros.
                    generated_input_ids = sample_input_ids[sample_prompt_len : sample_prompt_len + L]
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
                    # Add labels_correct/labels_wrong for truth_ratio pre_compute (dual-answer dataset)
                    for key in ("labels_correct", "labels_wrong"):
                        if key in batch:
                            sample_labels_alt = batch[key][sample_idx]
                            gen_alt = sample_labels_alt[sample_prompt_len:sample_prompt_len + L]
                            if gen_alt.shape[0] < L:
                                padding = torch.full(
                                    (L - gen_alt.shape[0],),
                                    IGNORE_INDEX,
                                    dtype=gen_alt.dtype,
                                    device=gen_alt.device,
                                )
                                gen_alt = torch.cat([gen_alt, padding])
                            batch_template[key] = gen_alt.unsqueeze(0)
            
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
                        _trunc = lambda s, n: s[:n] + ("..." if len(s) > n else "")
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
                                "use_generalized_sequence_probability", False
                            )
                            logit_alignment = trajectory_config.get(
                                "logit_alignment", "causal"
                            )
                            if use_generalized:
                                provider = FixationStepWiseScoreProvider(
                                    logit_alignment=logit_alignment
                                )
                                for step in steps_to_use:
                                    for view in include_views:
                                        if view == "full":
                                            R_v, F_v = sample_traj["R"], sample_traj["F"]
                                            lab_v = generated_labels
                                        else:
                                            L_eff_slice = min(L_eff_b, sample_traj["R"].shape[1])
                                            R_v = sample_traj["R"][:, :L_eff_slice, :]
                                            F_v = sample_traj["F"][:L_eff_slice]
                                            lab_v = generated_labels[:L_eff_slice]
                                        batch_prov = {"labels": lab_v.unsqueeze(0)}
                                        model_or_logits = {
                                            "R": R_v.unsqueeze(0),
                                            "F": F_v.unsqueeze(0),
                                            "report_step": step,
                                        }
                                        results = provider.get_per_position_scores(
                                            model_or_logits, batch_prov, ignore_index=IGNORE_INDEX
                                        )
                                        if results and results[0][0]:
                                            prob_val = sequence_probability_from_scores(
                                                results[0][0]
                                            )
                                            step_values_by_view[view][traj_name][step][
                                                "probability"
                                            ].append(prob_val)
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                    torch.cuda.empty_cache()
                            else:
                                logits_list = [
                                    _get_logits_at_step(sample_traj, traj_name, step)
                                    for step in steps_to_use
                                ]
                                device = logits_list[0].device
                                logits_stacked = torch.stack(
                                    [l.t() for l in logits_list], dim=0
                                )
                                labels_batch_full = generated_labels.unsqueeze(0).expand(
                                    len(steps_to_use), -1
                                ).to(device=device, dtype=torch.long)
                                for view in include_views:
                                    if view == "full":
                                        prob_results = _compute_prob_from_fixation_logits(
                                            logits_stacked, labels_batch_full, device, IGNORE_INDEX
                                        )
                                    else:
                                        L_eff_slice = min(L_eff_b, logits_stacked.shape[1])
                                        prob_results = _compute_prob_from_fixation_logits(
                                            logits_stacked[:, :L_eff_slice, :],
                                            labels_batch_full[:, :L_eff_slice],
                                            device,
                                            IGNORE_INDEX,
                                        )
                                    for i, step in enumerate(steps_to_use):
                                        if i < len(prob_results) and "prob" in prob_results[i]:
                                            step_values_by_view[view][traj_name][step]["probability"].append(
                                                prob_results[i]["prob"]
                                            )
                                    if len(include_views) > 1 and torch.cuda.is_available():
                                        del prob_results
                                        torch.cuda.synchronize()
                                        torch.cuda.empty_cache()
                                if torch.cuda.is_available():
                                    del logits_stacked, labels_batch_full
                                    torch.cuda.synchronize()
                                    torch.cuda.empty_cache()

                        for step in steps_to_use:
                            # Get logits at this step (on-demand from R, F)
                            if use_generalized:
                                R_st = sample_traj["R"].unsqueeze(0)
                                F_st = sample_traj["F"].unsqueeze(0)
                                logits = build_effective_step_fixation_logits(
                                    R_st, F_st, step
                                ).squeeze(0).T
                            else:
                                logits = _get_logits_at_step(sample_traj, traj_name, step)  # [V, L]

                            # Build eos batch_template (sliced to L_eff_b) for eos view
                            L_eff_slice = min(L_eff_b, batch_template["input_ids"].shape[1])
                            batch_template_eos = {}
                            for k, v in batch_template.items():
                                if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[-1] >= L_eff_slice:
                                    if v.dim() == 1:
                                        batch_template_eos[k] = v[:L_eff_slice].unsqueeze(0)
                                    else:
                                        batch_template_eos[k] = v[:, :L_eff_slice].clone()
                                else:
                                    batch_template_eos[k] = v

                            # Compute each requested metric (skip rouge and probability; already batched above) for each view
                            for metric_name, metric_info in [(m, loaded_metrics[m]) for m in metrics_to_run]:
                                try:
                                    metric = metric_info["metric"]
                                    metric_cfg = metric_info["config"]

                                    if metric_name == "privleak" and trajectories_by_key is not None:
                                        continue
                                    if metric_name in rouge_metrics_in_run or metric_name == "probability":
                                        continue

                                    gpu_set_phase("trajectory_metric", metric=metric_name, batch_idx=batch_idx, step=step)

                                    kwargs_clean = {k: v for k, v in kwargs.items() if k != "tokenizer"}
                                    if primary_data is not None:
                                        kwargs_clean["data"] = primary_data
                                    kwargs_clean["ground_truth"] = ground_truth_str
                                    kwargs_clean["rouge_scorer"] = rouge_scorer_instance
                                    kwargs_clean["sample_traj"] = sample_traj
                                    kwargs_clean["step"] = step
                                    if trajectory_config is not None:
                                        kwargs_clean["trajectory_config"] = trajectory_config

                                    for view in include_views:
                                        bt = batch_template if view == "full" else batch_template_eos
                                        logits_view = logits[:, :L_eff_slice] if view == "eos" else logits
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
                                            **kwargs_clean
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
                                        if metric_name == "extraction_strength" and step in (steps_to_use[0], steps_to_use[-1]) and metric_value is not None:
                                            logger.debug(
                                                f"extraction_strength step={step} sample={idx_str}: value={metric_value}"
                                            )
                                        if metric_value is not None:
                                            step_values_by_view[view][traj_name][step][metric_name].append(metric_value)
                                    # Return GPU memory after metrics that allocate large log_probs/contiguous logits
                                    # so the next metric (e.g. hm_aggregate) or next step does not OOM (see .monitor/oom-investigation-exact-cause.md).
                                    if metric_name in ("extraction_strength", "truth_ratio", "ks_test") and torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                        
                                except Exception as e:
                                    logger.warning(
                                        f"Error computing {metric_name} at step {step} for {traj_name}: {e}",
                                        exc_info=True
                                    )
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
                    h_indices = h_batch.get(
                        "index",
                        torch.arange(
                            h_batch_idx * h_input_ids.shape[0],
                            (h_batch_idx + 1) * h_input_ids.shape[0],
                        ),
                    )
                    h_prompts, h_prompt_lens = _build_prompts_for_sampler(
                        h_input_ids, h_labels, tokenizer, IGNORE_INDEX
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
                                h_labels.shape[1] - h_prompt_lens[j]
                                for j in range(h_B)
                            )
                        else:
                            h_L_gen = int(h_L_gen)
                        h_target_sequences = _build_target_sequences_for_sampler(
                            h_labels, h_prompt_lens, h_L_gen, IGNORE_INDEX
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
                    use_generalized_privleak = trajectory_config.get(
                        "use_generalized_sequence_probability", False
                    )
                    for step in h_steps_to_use:
                        if use_generalized_privleak:
                            per_position_scores_holdout = _per_position_scores_from_R_F_batch(
                                h_R, h_F, h_batch.get("labels"), h_prompt_lens, h_L, trajectory_config,
                                report_step=step,
                            )
                            if per_position_scores_holdout is not None:
                                privleak_accumulators[step].add_holdout_batch(
                                    h_batch, per_position_scores=per_position_scores_holdout
                                )
                        else:
                            h_logits_list = [
                                _get_logits_at_step(
                                    {"R": h_R[i], "F": h_F[i], "S": h_S, "L": h_L}, "steps", step
                                ).T
                                for i in range(h_B)
                            ]
                            h_logits_batch = torch.stack(h_logits_list, dim=0)
                            privleak_accumulators[step].add_holdout_batch(
                                h_batch, h_logits_batch
                            )
                    del h_R, h_F
                    if should_run_gc(0.9):
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                for step in run_steps_to_use:
                    gpu_set_phase("privleak_dual_step", step=step)
                    pre_result = privleak_accumulators[step].aggregate()
                    privleak_result = _get_metric_from_registry("privleak")._metric_fn(
                        model=None,
                        pre_compute=pre_result,
                        reference_logs=kwargs.get("reference_logs"),
                        ref_value=privleak_cfg.get("ref_value", 0.5),
                        **{k: v for k, v in kwargs.items() if k not in ("model", "tokenizer", "pre_compute")},
                    )
                    pv = privleak_result.get("agg_value")
                    for view in include_views:
                        for traj_name in trajectory_names:
                            if step not in step_values_by_view[view][traj_name]:
                                step_values_by_view[view][traj_name][step] = {m: [] for m in loaded_metrics}
                            step_values_by_view[view][traj_name][step]["privleak"].append(pv)
                privleak_accumulators = None

        # Multi-dataset: run privleak dual trajectory after per-key loops
        if multi_dataset and "forget" in data and "holdout" in data and "privleak" in loaded_metrics and privleak_needs_dual:
            primary_data = data["forget"]
            secondary_data = data["holdout"]
            if sort_by_length:
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
                for step in steps_to_use_dual:
                    gpu_set_phase("privleak_dual_step", step=step)
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
                        **{k: v for k, v in kwargs.items() if k not in ("tokenizer", "model", "data", "collators")},
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
                        reference_logs=kwargs.get("reference_logs"),
                        ref_value=privleak_cfg.get("ref_value", 0.5),
                        **{k: v for k, v in kwargs.items() if k not in ("model", "tokenizer", "pre_compute")},
                    )
                    pv = privleak_result.get("agg_value")
                    for view in include_views:
                        for traj_name in trajectory_names:
                            if step not in step_values_by_view[view][traj_name]:
                                step_values_by_view[view][traj_name][step] = {m: [] for m in loaded_metrics}
                            step_values_by_view[view][traj_name][step]["privleak"].append(pv)
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

        # Single-pass: return one result per display name so evaluator merges into logs.
        internal_names = list(loaded_metrics.keys())
        if full_display_order and len(full_display_order) >= len(full_internal_order):
            # Map each (filtered) internal name to its display name by original config order
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
        if len(display_names) == len(internal_names) and len(internal_names) > 0:
            out = {}
            for display_name, internal_name in zip(display_names, internal_names):
                out[display_name] = {
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
            out["trajectory_step_metadata"] = {
                "agg_value": None,
                "trajectory_step_metadata": trajectory_step_metadata,
            }
            if executor is not None:
                executor.shutdown(wait=True)
            return out

        if executor is not None:
            executor.shutdown(wait=True)
        return {
            "agg_value": agg_value_by_view,
            "value_by_index": {},
            "step_distribution": step_distribution_by_view,
        }


