"""Trajectory-based metrics: one sampler pass per batch, compute probability and/or ROUGE from trajectory output."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from rouge_score import rouge_scorer
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.utils import IGNORE_INDEX
from evals.metrics.base import unlearning_metric
from evals.metrics.utils import aggregate_to_1D

logger = logging.getLogger("metrics")


def _set_adapter_fixation_logits(model, trajectory_config: dict) -> None:
    """Set use_fixation_logits on the diffusion adapter from trajectory_config."""
    adapter_config = getattr(model, "adapter_config", None)
    if adapter_config is None:
        return
    use_fixation = trajectory_config.get("use_fixation_logits", True)
    if hasattr(adapter_config, "use_fixation_logits"):
        adapter_config.use_fixation_logits = use_fixation


def _derive_prompts_from_batch(input_ids: torch.Tensor, labels: torch.Tensor, ignore_index: int):
    """Derive prompt (prefix up to first non-ignored label) per row. Returns list of prompt token lists."""
    B = input_ids.shape[0]
    prompts = []
    for i in range(B):
        label_mask = labels[i] != ignore_index
        if label_mask.any():
            prompt_end = label_mask.nonzero()[0][0].item()
        else:
            prompt_end = input_ids.shape[1]
        prompts.append(input_ids[i, :prompt_end].cpu().tolist())
    return prompts


def _fixation_logits_from_sampler_output(out) -> torch.Tensor | None:
    """Get [B, T, V] fixation logits from sampler output (fixation_logits or logits_history+fixation_steps)."""
    if getattr(out, "fixation_logits", None) is not None:
        return out.fixation_logits
    logits_history = getattr(out, "logits_history", None)
    fixation_steps = getattr(out, "fixation_steps", None)
    if logits_history and fixation_steps is not None and len(logits_history) > 0:
        R = torch.stack(logits_history, dim=0)
        S, B, T, V = R.shape
        F = fixation_steps.clamp(0, S - 1).to(R.device)
        batch_idx = torch.arange(B, device=R.device, dtype=torch.long).unsqueeze(1).expand(B, T)
        pos_idx = torch.arange(T, device=R.device, dtype=torch.long).unsqueeze(0).expand(B, T)
        return R[F, batch_idx, pos_idx, :]
    return None


def _compute_prob_from_fixation_logits(
    fixation_logits: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    ignore_index: int,
) -> list[dict]:
    """Compute per-sample probability from fixation logits and labels (same formula as evaluate_probability).
    Trims to min(fixation_logits length, labels length) so lengths always match (e.g. when sampler
    returns shorter sequence than padded batch labels).
    """
    T_fl = fixation_logits.shape[1]
    T_lab = labels.shape[1]
    L = min(T_fl, T_lab)
    if L <= 1:
        B = fixation_logits.shape[0]
        return [{"prob": 0.0, "avg_loss": float("inf")} for _ in range(B)]
    shifted_labels = labels[..., 1:L].contiguous()
    logits = fixation_logits[..., : L - 1, :].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
    losses = loss_fn(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    num_token_gt = (shifted_labels != ignore_index).sum(-1)
    avg_losses = losses / num_token_gt.clamp(min=1)
    normalized_probs = torch.exp(-avg_losses)
    probs = normalized_probs.cpu().numpy().tolist()
    avg_losses_list = avg_losses.cpu().numpy().tolist()
    return [{"prob": p, "avg_loss": l} for p, l in zip(probs, avg_losses_list)]


def _compute_rouge_from_sequences(
    sequences: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    rouge_type: str,
    ignore_index: int,
) -> list[dict]:
    """Decode generated part of sequences and compute ROUGE vs ground truth from labels."""
    input_len = input_ids.shape[1]
    gen_ids = sequences[:, input_len:]
    gen_texts = tokenizer.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    input_texts = tokenizer.batch_decode(
        input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    tokens = [label[label != ignore_index] for label in labels]
    full_texts = tokenizer.batch_decode(
        tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    ground_truths = [
        full_text.replace(input_text, "").strip()
        for input_text, full_text in zip(input_texts, full_texts)
    ]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    evals = []
    for gen, gt in zip(gen_texts, ground_truths):
        scores = scorer.score(gt, gen)
        evals.append({
            "rouge1_recall": scores["rouge1"].recall,
            "rougeL_f1": scores["rougeL"].fmeasure,
            "rougeL_recall": scores["rougeL"].recall,
        })
    return evals


def _run_sampler_once(
    model,
    batch: dict,
    trajectory_config: dict,
    ignore_index: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Run sampler once on batch; return (fixation_logits, sequences) or (None, None) if no sampler."""
    sampler = getattr(model, "sampler", None)
    if sampler is None:
        return None, None
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    prompts = _derive_prompts_from_batch(input_ids, labels, ignore_index)
    B = input_ids.shape[0]
    T_batch = input_ids.shape[1]
    min_prompt_len = min(len(p) for p in prompts)
    max_new_tokens = T_batch - min_prompt_len
    if max_new_tokens <= 0:
        return None, None
    sampler_kwargs = dict(trajectory_config.get("sampler_kwargs", {}) or {})
    sampler_kwargs.pop("max_new_tokens", None)  # use batch-derived max_new_tokens only
    with torch.no_grad():
        out = sampler.sample(
            inputs=prompts,
            config=None,
            return_dict=True,
            return_logits=True,
            max_new_tokens=max_new_tokens,
            **sampler_kwargs,
        )
    fixation_logits = _fixation_logits_from_sampler_output(out)
    sequences = getattr(out, "sequences", None)
    if fixation_logits is not None:
        device = input_ids.device
        T_full = fixation_logits.shape[1]
        if T_full >= T_batch:
            fixation_logits = fixation_logits[:, :T_batch, :].to(device=device, dtype=fixation_logits.dtype)
        else:
            fixation_logits = fixation_logits.to(device=device)
            full_logits = torch.zeros(
                fixation_logits.shape[0], T_batch, fixation_logits.shape[2],
                device=device, dtype=fixation_logits.dtype,
            )
            full_logits[:, -T_full:, :] = fixation_logits
            fixation_logits = full_logits
    return fixation_logits, sequences


def trajectory_metrics(model, metric_name, cache, **kwargs):
    """Run one sampler pass per batch and compute requested trajectory metric(s). Returns single-metric result."""
    data = kwargs["data"]
    collators = kwargs["collators"]
    tokenizer = kwargs["tokenizer"]
    batch_size = kwargs["batch_size"]
    metrics_list = kwargs.get("metrics", ["probability"])
    trajectory_config = kwargs.get("trajectory_config") or {}
    rouge_type = kwargs.get("rouge_type", "rougeL_recall")

    _set_adapter_fixation_logits(model, trajectory_config)

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collators)
    device = getattr(model, "device", None)
    if device is None and hasattr(model, "parameters"):
        p = next(model.parameters(), None)
        device = p.device if p is not None else torch.device("cpu")
    if device is None:
        device = torch.device("cpu")

    # Single-metric path: one sampler run per batch, compute one metric
    need_prob = "probability" in metrics_list
    need_rouge = "rouge" in metrics_list

    if need_prob:
        prob_by_idx = defaultdict(dict)
        for batch in tqdm(dataloader, desc="Trajectory (prob)", total=len(dataloader)):
            if "index" in batch:
                indices = batch["index"].cpu().numpy().tolist()
                batch = {k: v for k, v in batch.items() if k != "index"}
            else:
                indices = list(range(len(batch["input_ids"])))
            fixation_logits, _ = _run_sampler_once(model, batch, trajectory_config, IGNORE_INDEX)
            if fixation_logits is not None:
                probs = _compute_prob_from_fixation_logits(
                    fixation_logits, batch["labels"], device, IGNORE_INDEX
                )
                for idx, p in zip(indices, probs):
                    prob_by_idx[idx] = p
        prob_values = np.array([prob_by_idx[i]["prob"] for i in sorted(prob_by_idx) if prob_by_idx[i].get("prob") is not None])
        agg = float(np.mean(aggregate_to_1D(prob_values))) if len(prob_values) else None
        return {"agg_value": agg, "value_by_index": dict(prob_by_idx)}
    else:
        # rouge only
        rouge_by_idx = defaultdict(dict)
        for batch in tqdm(dataloader, desc="Trajectory (rouge)", total=len(dataloader)):
            if "index" in batch:
                indices = batch["index"].cpu().numpy().tolist()
                batch = {k: v for k, v in batch.items() if k != "index"}
            else:
                indices = list(range(len(batch["input_ids"])))
            _, sequences = _run_sampler_once(model, batch, trajectory_config, IGNORE_INDEX)
            if sequences is not None:
                rouge_evals = _compute_rouge_from_sequences(
                    sequences, batch["input_ids"], batch["labels"],
                    tokenizer, rouge_type, IGNORE_INDEX,
                )
                for idx, r in zip(indices, rouge_evals):
                    rouge_by_idx[idx] = r
        rouge_values = np.array([
            rouge_by_idx[i][rouge_type] for i in sorted(rouge_by_idx)
            if rouge_by_idx[i].get(rouge_type) is not None
        ])
        agg = float(np.mean(aggregate_to_1D(rouge_values))) if len(rouge_values) else None
        return {"agg_value": agg, "value_by_index": dict(rouge_by_idx)}


def run_coalesced_trajectory_metrics(
    model,
    metrics_to_run: list[tuple[str, Any]],
    eval_cfg,
    logs: dict,
    *,
    data,
    collators,
    tokenizer,
    batch_size: int,
    trajectory_config: dict,
    rouge_type: str = "rougeL_recall",
    **kwargs,
) -> dict[str, dict]:
    """One sampler pass per batch; compute all trajectory metrics from that run. Returns {metric_name: result}."""
    _set_adapter_fixation_logits(model, trajectory_config)
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collators)
    device = getattr(model, "device", None)
    if device is None and hasattr(model, "parameters"):
        p = next(model.parameters(), None)
        device = p.device if p is not None else torch.device("cpu")
    if device is None:
        device = torch.device("cpu")

    prob_by_idx = defaultdict(dict)
    rouge_by_idx = defaultdict(dict)
    for batch in tqdm(dataloader, desc="Trajectory (coalesced)", total=len(dataloader)):
        if "index" in batch:
            indices = batch["index"].cpu().numpy().tolist()
            batch = {k: v for k, v in batch.items() if k != "index"}
        else:
            indices = list(range(len(batch["input_ids"])))
        fixation_logits, sequences = _run_sampler_once(model, batch, trajectory_config, IGNORE_INDEX)
        if fixation_logits is not None:
            probs = _compute_prob_from_fixation_logits(
                fixation_logits, batch["labels"], device, IGNORE_INDEX
            )
            for idx, p in zip(indices, probs):
                prob_by_idx[idx] = p
        if sequences is not None:
            rouge_evals = _compute_rouge_from_sequences(
                sequences, batch["input_ids"], batch["labels"],
                tokenizer, rouge_type, IGNORE_INDEX,
            )
            for idx, r in zip(indices, rouge_evals):
                rouge_by_idx[idx] = r

    results = {}
    for name, cfg in metrics_to_run:
        cfg = OmegaConf.to_container(cfg, resolve=True) if hasattr(cfg, "get") else (cfg or {})
        metrics_list = cfg.get("metrics", [])
        if "probability" in metrics_list:
            prob_values = np.array([
                prob_by_idx[i]["prob"] for i in sorted(prob_by_idx)
                if prob_by_idx[i].get("prob") is not None
            ])
            agg = float(np.mean(aggregate_to_1D(prob_values))) if len(prob_values) else None
            results[name] = {"agg_value": agg, "value_by_index": dict(prob_by_idx)}
        else:
            rt = cfg.get("rouge_type", rouge_type)
            rouge_values = np.array([
                rouge_by_idx[i][rt] for i in sorted(rouge_by_idx)
                if rouge_by_idx[i].get(rt) is not None
            ])
            agg = float(np.mean(aggregate_to_1D(rouge_values))) if len(rouge_values) else None
            results[name] = {"agg_value": agg, "value_by_index": dict(rouge_by_idx)}
    return results


# Register so handler "trajectory_metrics" resolves
_trajectory_metric = unlearning_metric(name="trajectory_metrics")(trajectory_metrics)
