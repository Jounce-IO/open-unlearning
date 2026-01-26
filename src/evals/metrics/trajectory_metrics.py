"""
Trajectory-based metrics for dLLM unlearning evaluation.

This module computes metrics at each diffusion step across three trajectory types
(steps, fixation, ratio), supporting any metric from the open-unlearning framework.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union
from torch.utils.data import DataLoader
from omegaconf import ListConfig, DictConfig

from evals.metrics.base import unlearning_metric
from evals.metrics.utils import (
    evaluate_probability,
    tokenwise_vocab_logprobs,
    IGNORE_INDEX,
)
from rouge_score import rouge_scorer
from evals.metrics.trajectory_utils import (
    stack_logits_history,
    compute_trajectories,
    extract_logits_at_step,
    decode_logits_to_text,
)
from evals.metrics.trajectory_adapters import (
    LogitModelWrapper,
    compute_logit_metric_at_step,
    compute_text_metric_at_step,
)

logger = logging.getLogger("evaluator")

# IGNORE_INDEX from data.utils
IGNORE_INDEX = -100


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
    pre_compute_results = {}
    
    for pre_metric_name, pre_metric_cfg in pre_compute_config.items():
        # Get access key (defaults to metric name)
        access_key = pre_metric_cfg.get("access_key", pre_metric_name)
        
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
        elif isinstance(pre_metric_cfg, dict) and "handler" in pre_metric_cfg:
            handler_name = pre_metric_cfg["handler"]
            if handler_name in METRICS_REGISTRY:
                pre_metric = METRICS_REGISTRY[handler_name]
        
        if pre_metric is None:
            raise ValueError(
                f"Pre-compute metric '{pre_metric_name}' not found in registry. "
                f"Tried handler: {handler_name}. "
                f"Available metrics: {list(METRICS_REGISTRY.keys())}"
            )
        
        # Compute pre-compute metric at this step
        # Note: Pre-compute metrics might have their own pre_compute requirements
        # We handle this recursively
        try:
            # Call the pre-compute metric at this step
            # Remove tokenizer from kwargs if present to avoid duplicate argument
            kwargs_clean = {k: v for k, v in kwargs.items() if k != "tokenizer"}
            pre_result = _call_metric_at_step(
                metric=pre_metric,
                logits=logits,
                batch_template=batch_template,
                tokenizer=tokenizer,
                sample_labels=sample_labels,
                sample_input_ids=sample_input_ids,
                sample_prompt_len=sample_prompt_len,
                metric_config=pre_metric_cfg,
                sample_idx=sample_idx,
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


def _call_metric_at_step(
    metric: Any,
    logits: torch.Tensor,
    batch_template: Dict[str, torch.Tensor],
    tokenizer: Any,
    sample_labels: Optional[torch.Tensor],
    sample_input_ids: torch.Tensor,
    sample_prompt_len: int,
    metric_config: Dict[str, Any],
    sample_idx: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Call a metric function at a specific trajectory step.
    
    Args:
        metric: UnlearningMetric object from registry
        logits: [V, L] logits at the step
        batch_template: Template batch dict
        tokenizer: Tokenizer for text processing
        sample_labels: Labels for the sample
        sample_input_ids: Input IDs for the sample
        sample_prompt_len: Length of prompt
        metric_config: Config for this specific metric (may include pre_compute, etc.)
        sample_idx: Index string for this sample (used for pre_compute value_by_index)
        **kwargs: Additional kwargs to pass to metric
    
    Returns:
        Metric result (typically dict with metric values)
    """
    # Ensure logits are in [B, L, V] format
    if logits.dim() == 2:
        # [V, L] -> transpose to [L, V] then add batch dim -> [1, L, V]
        logits = logits.transpose(0, 1).unsqueeze(0)
    elif logits.dim() == 3 and logits.shape[0] == 1:
        # [1, L, V] - already correct
        pass
    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")
    
    # Create model wrapper
    device = logits.device
    model_wrapper = LogitModelWrapper(logits, device)
    
    # Prepare batch
    batch = {}
    for key, value in batch_template.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        else:
            batch[key] = value
    
    # Handle pre_compute metrics if present
    pre_compute_config = metric_config.pop("pre_compute", {})
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
    metric_kwargs = {
        "model": model_wrapper,
        "batch": batch,
        "tokenizer": tokenizer,
        **metric_config,  # Include metric-specific config (aggregator, etc., but not pre_compute)
        **kwargs,  # Include any additional kwargs
    }
    
    # Add pre_compute results if available
    if pre_compute_results:
        metric_kwargs["pre_compute"] = pre_compute_results
    
    # Call the metric's underlying function
    # Note: We call _metric_fn directly, not evaluate(), because:
    # 1. We're computing at a single step, not iterating over data
    # 2. We've already prepared the model wrapper and batch
    # 3. Pre-compute metrics would need to be computed at each step separately
    
    # Some metrics iterate over data (like `probability`), so we need to use
    # their underlying batch functions instead. Map known metrics to their batch functions.
    metric_name = metric.name
    
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
    
    def _handle_text_based_metric(logits, tokenizer, sample_labels, sample_input_ids, sample_prompt_len, metric_name, metric_config, **kwargs):
        """
        Generic handler for text-based metrics that require model.generate().
        Decodes logits to text and computes text similarity metrics.
        """
        # Decode logits to text via argmax
        if logits.dim() == 3:
            logits = logits[0]  # [L, V]
        predicted_tokens = torch.argmax(logits, dim=-1)  # [L]
        gen_text = tokenizer.decode(predicted_tokens.tolist(), skip_special_tokens=True)
        
        # Extract ground truth text from labels
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
                    return [{"score": result_dict[rouge_type]}]
        
        return result
    
    batch_function_map = {
        "probability": evaluate_probability,
        "exact_memorization": _exact_memorization_batch_fn,
    }
    
    # Try using batch function if available
    if metric_name in batch_function_map:
        batch_fn = batch_function_map[metric_name]
        try:
            result = batch_fn(model=model_wrapper, batch=batch, **metric_kwargs)
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
            logger.info(
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
    logits_source = trajectory_config.get("logits_source", "sampler")
    data = kwargs.get("data")
    collator = kwargs.get("collators")
    batch_size = kwargs.get("batch_size", 1)
    tokenizer = kwargs.get("tokenizer")
    generation_args = kwargs.get("generation_args", {})
    
    if not metrics_config:
        raise ValueError("No metrics specified in config")
    
    if not tokenizer:
        raise ValueError("tokenizer is required for trajectory metrics")
    
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
    
    # Load metrics from registry
    loaded_metrics = {}
    for metric_name, metric_cfg in metrics_to_compute.items():
        try:
            metric = _get_metric_from_registry(metric_name)
            loaded_metrics[metric_name] = {
                "metric": metric,
                "config": metric_cfg,
            }
        except ValueError as e:
            logger.error(f"Failed to load metric '{metric_name}': {e}")
            raise
    
    # Create dataloader
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    
    # Storage for results
    all_results = {}  # {sample_idx: {trajectories: {...}}}
    trajectory_names = ["steps", "fixation", "ratio"]
    
    # Get sampler from model
    sampler = _get_sampler_from_model(model)
    if sampler is None:
        raise ValueError(
            "Model does not have a sampler. Trajectory metrics require a diffusion model with sampler. "
            "Ensure model is wrapped with DiffusionModelAdapter or has accessible sampler."
        )
    
    # Process each batch
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")
        indices = batch.get("index", torch.arange(batch_idx * batch_size, 
                                                  (batch_idx + 1) * batch_size))
        
        B = input_ids.shape[0]
        
        # Prepare inputs for sampler (list of token sequences)
        prompts = []
        prompt_lens = []
        for i in range(B):
            # Extract prompt (non-ignored tokens)
            if labels is not None:
                # Find where labels start (first non-IGNORE_INDEX)
                label_mask = labels[i] != IGNORE_INDEX
                if label_mask.any():
                    prompt_end = label_mask.nonzero()[0][0].item()
                else:
                    prompt_end = input_ids.shape[1]
            else:
                prompt_end = input_ids.shape[1]
            
            prompt = input_ids[i, :prompt_end].cpu().tolist()
            prompts.append(prompt)
            prompt_lens.append(len(prompt))
        
        # Generate using sampler with logits tracking
        sampler_output = sampler.sample(
            inputs=prompts,
            config=None,  # Use default config
            return_dict=True,
            return_logits=True,
            **trajectory_config.get("sampler_kwargs", {}),
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
        
        # Stack logits into tensor R [V, T, S] where T is full sequence length (prompt + generated)
        R_full = stack_logits_history(logits_history)  # [V, T, S]
        V, T_full, S = R_full.shape
        
        # Extract only the generated portion from R_full
        # logits_history contains [B, T, V] where T includes prompt + generated
        # We need to extract only the generated portion for trajectory computation
        max_prompt_len = max(prompt_lens)
        generated_len = T_full - max_prompt_len
        
        # Extract generated portion: R[:, max_prompt_len:max_prompt_len + generated_len, :]
        # This gives us [V, L, S] where L = generated_len
        R = R_full[:, max_prompt_len:max_prompt_len + generated_len, :]  # [V, L, S]
        V, L, S = R.shape
        
        # Extract fixation steps F [L] for each sample
        # fixation_steps is [B, T], we need [L] for generated region
        # For now, use first sample or average
        if fixation_steps.dim() == 2:
            # [B, T] -> take first sample and extract generated region
            F_full = fixation_steps[0]  # [T]
            # Extract only the generated portion (after max prompt length)
            if F_full.shape[0] > max_prompt_len:
                F = F_full[max_prompt_len:max_prompt_len + L]  # [L]
            else:
                # If fixation_steps doesn't cover full length, pad or truncate
                F = F_full[:L] if F_full.shape[0] >= L else torch.cat([
                    F_full,
                    torch.full((L - F_full.shape[0],), S - 1, dtype=torch.long, device=F_full.device)
                ])
        else:
            raise ValueError(f"Unexpected fixation_steps shape: {fixation_steps.shape}")
        
        # Compute three trajectory tensors
        T_steps, T_fixation, T_ratio = compute_trajectories(R, F, S)
        trajectories = {
            "steps": T_steps,
            "fixation": T_fixation,
            "ratio": T_ratio,
        }
        
        # Process each sample in batch
        for sample_idx in range(B):
            idx_str = str(indices[sample_idx].item() if torch.is_tensor(indices[sample_idx]) else indices[sample_idx])
            
            # For batched case, we'd need per-sample trajectories
            # For now, use shared trajectories (assuming single sample or first sample)
            sample_trajectories = trajectories if sample_idx == 0 else trajectories
            
            sample_results = {
                "trajectories": {
                    "steps": {},
                    "fixation": {},
                    "ratio": {},
                }
            }
            
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
            # Use only generated portion to match logits shape
            batch_template = {
                "input_ids": torch.zeros((1, L), dtype=torch.long, device=sample_input_ids.device),  # Dummy input_ids, not used by metrics
                "labels": generated_labels.unsqueeze(0) if generated_labels is not None else None,
                "attention_mask": torch.ones((1, L), dtype=torch.long, device=sample_input_ids.device),  # All positions valid
            }
            
            # Compute metrics for each trajectory type and step
            for traj_name, trajectory in sample_trajectories.items():
                for step in range(S):
                    step_key = f"step_{step}"
                    step_results = {}
                    
                    # Extract logits at this step
                    logits = extract_logits_at_step(trajectory, step)  # [V, L]
                    
                    # Compute each requested metric
                    for metric_name, metric_info in loaded_metrics.items():
                        try:
                            metric = metric_info["metric"]
                            metric_cfg = metric_info["config"]
                            
                            # Call metric at this step
                            # Remove tokenizer from kwargs if present to avoid duplicate argument
                            kwargs_clean = {k: v for k, v in kwargs.items() if k != "tokenizer"}
                            result = _call_metric_at_step(
                                metric=metric,
                                logits=logits,
                                batch_template=batch_template,
                                tokenizer=tokenizer,
                                sample_labels=sample_labels,
                                sample_input_ids=sample_input_ids,
                                sample_prompt_len=sample_prompt_len,
                                metric_config=metric_cfg,
                                sample_idx=idx_str,
                                **kwargs_clean
                            )
                            
                            # Extract metric value from result
                            # Handle different result formats
                            if isinstance(result, dict):
                                # Try common keys
                                if "agg_value" in result:
                                    step_results[metric_name] = result["agg_value"]
                                elif "value_by_index" in result:
                                    # Extract value from first index
                                    value_by_index = result["value_by_index"]
                                    if value_by_index:
                                        first_idx = list(value_by_index.keys())[0]
                                        first_value = value_by_index[first_idx]
                                        # Extract numeric value from first_value dict
                                        if isinstance(first_value, dict):
                                            for key in ["prob", "score", "value"]:
                                                if key in first_value:
                                                    step_results[metric_name] = first_value[key]
                                                    break
                                            else:
                                                # Use first numeric value
                                                for key, value in first_value.items():
                                                    if isinstance(value, (int, float, np.number)):
                                                        step_results[metric_name] = float(value)
                                                        break
                                elif "prob" in result:
                                    step_results[metric_name] = result["prob"]
                                elif "score" in result:
                                    step_results[metric_name] = result["score"]
                                else:
                                    # Use first numeric value
                                    for key, value in result.items():
                                        if isinstance(value, (int, float, np.number)):
                                            step_results[metric_name] = float(value)
                                            break
                            elif isinstance(result, list) and len(result) > 0:
                                # List of dicts (common format)
                                result_dict = result[0]
                                if isinstance(result_dict, dict):
                                    if "prob" in result_dict:
                                        step_results[metric_name] = result_dict["prob"]
                                    elif "score" in result_dict:
                                        step_results[metric_name] = result_dict["score"]
                                    else:
                                        # Use first numeric value
                                        for key, value in result_dict.items():
                                            if isinstance(value, (int, float, np.number)):
                                                step_results[metric_name] = float(value)
                                                break
                            elif isinstance(result, (int, float, np.number)):
                                step_results[metric_name] = float(result)
                            else:
                                logger.warning(
                                    f"Unexpected result format for {metric_name}: {type(result)}. "
                                    f"Result: {result}"
                                )
                                step_results[metric_name] = None
                        
                        except Exception as e:
                            logger.warning(
                                f"Error computing {metric_name} at step {step} for {traj_name}: {e}",
                                exc_info=True
                            )
                            step_results[metric_name] = None
                    
                    sample_results["trajectories"][traj_name][step_key] = step_results
            
            all_results[idx_str] = sample_results
    
    # Aggregate results
    agg_value = {}
    for traj_name in trajectory_names:
        agg_value[traj_name] = {}
        for metric_name in loaded_metrics.keys():
            # Collect values per step across all samples
            step_values = {}  # {step: [values across samples]}
            
            for sample_idx, sample_results in all_results.items():
                traj_results = sample_results["trajectories"][traj_name]
                for step_key, step_results in traj_results.items():
                    if metric_name in step_results and step_results[metric_name] is not None:
                        # Extract step number from step_key (e.g., "step_5" -> 5)
                        step_num = int(step_key.split("_")[1])
                        if step_num not in step_values:
                            step_values[step_num] = []
                        step_values[step_num].append(step_results[metric_name])
            
            # Aggregate: mean across samples for each step
            if step_values:
                max_step = max(step_values.keys())
                aggregated = []
                for step in range(max_step + 1):
                    if step in step_values:
                        aggregated.append(np.mean(step_values[step]))
                    else:
                        aggregated.append(np.nan)
                agg_value[traj_name][metric_name] = np.array(aggregated)
            else:
                agg_value[traj_name][metric_name] = np.array([])
    
    return {
        "agg_value": agg_value,
        "value_by_index": all_results,
    }
