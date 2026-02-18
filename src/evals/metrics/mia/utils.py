from __future__ import annotations

from typing import Any, Dict

from evals.metrics.mia.all_attacks import AllAttacks
from evals.metrics.mia.loss import LOSSAttack
from evals.metrics.mia.reference import ReferenceAttack
from evals.metrics.mia.zlib import ZLIBAttack
from evals.metrics.mia.min_k import MinKProbAttack
from evals.metrics.mia.min_k_plus_plus import MinKPlusPlusAttack
from evals.metrics.mia.gradnorm import GradNormAttack

from sklearn.metrics import roc_auc_score

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def get_attacker(attack: str):
    mapping = {
        AllAttacks.LOSS: LOSSAttack,
        AllAttacks.REFERENCE_BASED: ReferenceAttack,
        AllAttacks.ZLIB: ZLIBAttack,
        AllAttacks.MIN_K: MinKProbAttack,
        AllAttacks.MIN_K_PLUS_PLUS: MinKPlusPlusAttack,
        AllAttacks.GRADNORM: GradNormAttack,
    }
    attack_cls = mapping.get(attack, None)
    if attack_cls is None:
        raise ValueError(f"Attack {attack} not found")
    return attack_cls


def mia_auc(attack_cls, model, data, collator, batch_size, **kwargs):
    """
    Compute the MIA AUC and accuracy.

    Parameters:
      - attack_cls: the attack class to use.
      - model: the target model.
      - data: a dict with keys "forget" and "holdout".
      - collator: data collator.
      - batch_size: batch size.
      - kwargs: additional optional parameters (e.g. k, p, tokenizer, reference_model).

    Returns a dict containing the attack outputs, including "acc" and "auc".

    Note on convention: auc is 1 when the forget data is much more likely than the holdout data
    """
    # Build attack arguments from common parameters and any extras.
    attack_args = {
        "model": model,
        "collator": collator,
        "batch_size": batch_size,
    }
    attack_args.update(kwargs)

    if hasattr(model, "set_dataset_key"):
        model.set_dataset_key("forget")
    output = {
        "forget": attack_cls(data=data["forget"], **attack_args).attack(),
    }
    if hasattr(model, "set_dataset_key"):
        model.set_dataset_key("holdout")
    output["holdout"] = attack_cls(data=data["holdout"], **attack_args).attack()
    forget_scores = [
        elem["score"] for elem in output["forget"]["value_by_index"].values()
    ]
    holdout_scores = [
        elem["score"] for elem in output["holdout"]["value_by_index"].values()
    ]
    scores = np.array(forget_scores + holdout_scores)
    labels = np.array(
        [0] * len(forget_scores) + [1] * len(holdout_scores)
    )  # see note above
    auc_value = roc_auc_score(labels, scores)
    output["auc"], output["agg_value"] = auc_value, auc_value
    return output


def mia_auc_from_score_dicts(forget_value_by_index, holdout_value_by_index):
    """
    Aggregate forget and holdout value_by_index (scores only) into the same dict shape as mia_auc.

    Each value_by_index is {str(idx): {"score": float}}. Returns dict with "forget", "holdout",
    "auc", "agg_value", and per-key "value_by_index" and "agg_value" (mean score).
    """
    forget_scores = [v["score"] for v in forget_value_by_index.values()]
    holdout_scores = [v["score"] for v in holdout_value_by_index.values()]
    if not forget_scores and not holdout_scores:
        auc_value = 0.0
    else:
        scores = np.array(forget_scores + holdout_scores)
        labels = np.array([0] * len(forget_scores) + [1] * len(holdout_scores))
        auc_value = float(roc_auc_score(labels, scores))
    output = {
        "forget": {
            "value_by_index": forget_value_by_index,
            "agg_value": float(np.mean(forget_scores)) if forget_scores else 0.0,
        },
        "holdout": {
            "value_by_index": holdout_value_by_index,
            "agg_value": float(np.mean(holdout_scores)) if holdout_scores else 0.0,
        },
        "auc": auc_value,
        "agg_value": auc_value,
    }
    return output


def mia_auc_streaming(
    attack_cls,
    forget_batch_logits_iter,
    holdout_batch_logits_iter,
    collator,
    batch_size,
    device,
    **kwargs,
):
    """
    Compute MIA AUC from streams of (batch, logits) for forget and holdout.
    Only scores are stored; aggregation (AUC) happens at the end.

    Each iterator yields (batch_dict, logits_tensor) where batch_dict has "index" and
    logits_tensor is [B, L, V]. Returns the same structure as mia_auc.
    """
    attack = attack_cls(model=None, data=[], collator=collator, batch_size=batch_size, **kwargs)
    forget_value_by_index = {}
    holdout_value_by_index = {}

    for batch, logits in forget_batch_logits_iter:
        batch_values = attack.compute_batch_values_from_logits(batch, logits)
        batch_scores = attack.process_batch(batch, batch_values)
        forget_value_by_index.update(batch_scores)

    for batch, logits in holdout_batch_logits_iter:
        batch_values = attack.compute_batch_values_from_logits(batch, logits)
        batch_scores = attack.process_batch(batch, batch_values)
        holdout_value_by_index.update(batch_scores)

    return mia_auc_from_score_dicts(forget_value_by_index, holdout_value_by_index)


class MIAStreamingAccumulator:
    """
    Accumulates MIA scores batch-by-batch for forget and holdout; aggregation at the end.

    Used by trajectory_metrics when privleak runs with dual dataset: only scores are stored
    per item, then AUC is computed once per step via aggregate().
    """

    def __init__(self, attack_cls, collator, batch_size, device, **attack_kwargs):
        self.attack = attack_cls(
            model=None, data=[], collator=collator, batch_size=batch_size, **attack_kwargs
        )
        self.forget_value_by_index = {}
        self.holdout_value_by_index = {}

    def add_forget_batch(self, batch, logits=None, per_position_scores=None):
        """Process one forget batch: compute scores from logits or precomputed per-position scores and accumulate."""
        if per_position_scores is not None and hasattr(
            self.attack, "compute_batch_values_from_per_position_scores"
        ):
            batch_values = self.attack.compute_batch_values_from_per_position_scores(
                batch, per_position_scores
            )
        else:
            batch_values = self.attack.compute_batch_values_from_logits(batch, logits)
        batch_scores = self.attack.process_batch(batch, batch_values)
        self.forget_value_by_index.update(batch_scores)

    def add_holdout_batch(self, batch, logits=None, per_position_scores=None):
        """Process one holdout batch: compute scores from logits or precomputed per-position scores and accumulate."""
        if per_position_scores is not None and hasattr(
            self.attack, "compute_batch_values_from_per_position_scores"
        ):
            batch_values = self.attack.compute_batch_values_from_per_position_scores(
                batch, per_position_scores
            )
        else:
            batch_values = self.attack.compute_batch_values_from_logits(batch, logits)
        batch_scores = self.attack.process_batch(batch, batch_values)
        self.holdout_value_by_index.update(batch_scores)

    def aggregate(self):
        """Return pre_result dict (same shape as mia_auc) for privleak."""
        return mia_auc_from_score_dicts(self.forget_value_by_index, self.holdout_value_by_index)


def process_mia_batch_worker(
    attack_cls_name: str,
    attack_kwargs: dict,
    batch_size: int,
    batch_cpu: Dict[str, Any],
    logits_cpu: Any,
) -> Dict[str, Dict[str, float]]:
    """Process one MIA batch (forget or holdout) in a worker process.

    Same logic as MIAStreamingAccumulator.add_forget_batch / add_holdout_batch:
    compute_batch_values_from_logits then process_batch. For use with
    ProcessPoolExecutor so streaming MIA can run in the same pool as ROUGE.

    Must be picklable: only pass attack name, plain dicts, and CPU tensors.
    Caller is responsible for moving batch and logits to CPU before submit.

    Returns:
        {str(index): {"score": float}} for each sample in the batch.
    """
    attack_cls = get_attacker(attack_cls_name)
    # Dummy collator: we never iterate the dataloader in streaming mode.
    attack = attack_cls(
        model=None,
        data=[],
        collator=lambda x: x,
        batch_size=batch_size,
        **attack_kwargs,
    )
    batch_values = attack.compute_batch_values_from_logits(batch_cpu, logits_cpu)
    return attack.process_batch(batch_cpu, batch_values)


def batch_to_cpu(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Copy batch dict with all tensors moved to CPU. For passing to process_mia_batch_worker."""
    if torch is None:
        return batch
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.cpu()
        else:
            out[k] = v
    return out
