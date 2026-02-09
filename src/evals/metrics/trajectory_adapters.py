"""
Adapters for computing metrics on trajectory logits.

This module provides:
- LogitModelWrapper: Wraps pre-computed logits to make them callable as model(**batch)
- DualLogitModelWrapper: Returns per-sample logits by (dataset_key, index) for MIA metrics
- Adapter functions for computing logit-based and text-based metrics at trajectory steps
"""

import torch
from typing import Callable, Any, Dict, List, Optional


class LogitModelWrapper:
    """Wraps pre-computed logits to make them callable as model(**batch)"""
    
    def __init__(self, logits: torch.Tensor, device: torch.device):
        """
        Args:
            logits: [B, L, V] logits tensor
            device: Device where logits are located
        """
        self.logits = logits  # [B, L, V]
        self.device = device
    
    def __call__(self, **batch):
        """
        Return logits in expected format when called as model(**batch).
        
        Returns:
            Output object with .logits attribute
        """
        class Output:
            logits = self.logits
        
        return Output


class DualLogitModelWrapper:
    """Returns per-sample logits by (dataset_key, index) for MIA metrics (forget + holdout)."""

    def __init__(
        self,
        logits_by_key: Dict[str, Dict[str, torch.Tensor]],
        device: torch.device,
    ):
        """
        Args:
            logits_by_key: {"forget": {"0": [V,L], "1": [V,L], ...}, "holdout": {"0": [V,L], ...}}
            device: Device where logits are located
        """
        self.logits_by_key = logits_by_key
        self.device = device
        self._current_key: Optional[str] = None

    def set_dataset_key(self, key: str) -> None:
        """Set which dataset (forget/holdout) the next batches belong to."""
        self._current_key = key

    def __call__(self, **batch):
        """
        Return logits for the sample(s) in batch, using current dataset key.
        batch["index"] contains sample index (tensor [B] or int).
        """
        if self._current_key is None:
            raise RuntimeError(
                "DualLogitModelWrapper: set_dataset_key() must be called before __call__"
            )
        key = self._current_key
        if key not in self.logits_by_key:
            raise KeyError(f"DualLogitModelWrapper: no logits for key={key}")
        idx = batch.get("index")
        if idx is None:
            raise KeyError("DualLogitModelWrapper: batch must contain 'index'")
        if isinstance(idx, torch.Tensor):
            indices = idx.cpu().tolist()
            indices = [indices] if isinstance(indices, int) else indices
        else:
            indices = [idx] if not isinstance(idx, (list, tuple)) else list(idx)
        logits_dict = self.logits_by_key[key]
        logits_list = []
        for idx_val in indices:
            idx_str = str(idx_val)
            if idx_str not in logits_dict:
                raise KeyError(
                    f"DualLogitModelWrapper: no logits for key={key} index={idx_str}, "
                    f"available: {list(logits_dict.keys())}"
                )
            lg = logits_dict[idx_str]
            if lg.dim() == 2:
                lg = lg.transpose(0, 1).unsqueeze(0)
            logits_list.append(lg)
        logits = torch.cat(logits_list, dim=0).to(self.device)

        class Output:
            pass

        out = Output()
        out.logits = logits
        return out


def compute_logit_metric_at_step(
    metric_fn: Callable,
    logits: torch.Tensor,
    batch_template: Dict[str, torch.Tensor],
    **kwargs
) -> Any:
    """
    Compute a logit-based metric at a specific trajectory step.
    
    Args:
        metric_fn: Metric function (e.g., evaluate_probability)
        logits: [V, L] or [1, L, V] logits at the step
        batch_template: Template batch dict with input_ids, labels, attention_mask
        **kwargs: Additional arguments for metric function
    
    Returns:
        Metric result (typically list of dicts with metric values)
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
    
    # Create batch from template
    batch = {}
    for key, value in batch_template.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        else:
            batch[key] = value
    
    # Call metric function
    result = metric_fn(model=model_wrapper, batch=batch, **kwargs)
    
    return result


def compute_text_metric_at_step(
    metric_fn: Callable,
    texts: List[str],
    ground_truths: List[str],
    tokenizer,
    **kwargs
) -> Any:
    """
    Compute a text-based metric at a specific trajectory step.
    
    Args:
        metric_fn: Metric function (e.g., eval_text_similarity)
        texts: List of generated text strings
        ground_truths: List of ground truth text strings
        tokenizer: Tokenizer for text processing
        **kwargs: Additional arguments for metric function
    
    Returns:
        Metric result (typically list of dicts with metric values)
    """
    # Create text batch in format expected by metric
    # Most text metrics expect a batch dict with "generation" and "ground_truth"
    text_batch = {
        "generation": texts,
        "ground_truth": ground_truths,
    }
    
    # Call metric function
    # Note: Some text metrics might need the model for generation_args
    # We'll pass model=None and let the metric handle it
    result = metric_fn(
        model=None,  # Text metrics may not need model
        tokenizer=tokenizer,
        batch=text_batch,
        **kwargs
    )
    
    return result
