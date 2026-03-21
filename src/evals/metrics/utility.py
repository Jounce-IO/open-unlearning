from __future__ import annotations

import torch
import numpy as np
import scipy as sc
from typing import Any
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from evals.metrics.utils import aggregate_to_1D
from evals.metrics.base import unlearning_metric


def _is_mu_component_key(name: Any) -> bool:
    """Match trajectory_metrics / TOFU model_utility: retain_, ra_, wf_ prefixes."""
    s = str(name)
    return s.startswith("retain_") or s.startswith("ra_") or s.startswith("wf_")


@unlearning_metric(name="hm_aggregate")
def hm_aggregate(model, **kwargs):
    """Harmonic mean of MU components. None is forbidden: if any component is None, return None (no silent drop).

    For non-trajectory TOFU ``model_utility``, also returns ``retain_mu_components``: a flat map of
    component name -> scalar (mirrors inner keys of trajectory ``retain_mu_components_by_step``).
    """
    pre_compute = kwargs.get("pre_compute")
    if not pre_compute:
        return {"agg_value": None}

    retain_mu_components: dict[str, Any] = {}
    values: list[float] = []
    for name, result in pre_compute.items():
        if not _is_mu_component_key(name):
            continue
        if not isinstance(result, dict):
            continue
        v = result.get("agg_value")
        key = str(name)
        if isinstance(v, (int, float, np.floating)):
            fv = float(v)
            retain_mu_components[key] = fv
            values.append(fv)
        elif v is None:
            retain_mu_components[key] = None
            return {"agg_value": None, "retain_mu_components": retain_mu_components}
        else:
            retain_mu_components[key] = v
            return {"agg_value": None, "retain_mu_components": retain_mu_components}

    if not values:
        out: dict[str, Any] = {"agg_value": None}
        if retain_mu_components:
            out["retain_mu_components"] = retain_mu_components
        return out
    out = {
        "agg_value": float(sc.stats.hmean(values)),
        "retain_mu_components": retain_mu_components,
    }
    return out


@unlearning_metric(name="classifier_prob")
def classifier_prob(model, **kwargs):
    batch_size = kwargs.get("batch_size", 32)
    max_length = kwargs.get("max_length", 512)
    class_id = kwargs.get("class_id", 0)
    text_key = kwargs.get("text_key", "generation")
    classifier_model_args = kwargs["classifier_model_args"]
    classifier_tokenization_args = kwargs["classifier_tokenization_args"]
    device = kwargs.get("device", "cuda")

    tokenizer = AutoTokenizer.from_pretrained(**classifier_tokenization_args)
    classifier = AutoModelForSequenceClassification.from_pretrained(
        **classifier_model_args
    ).to(device)

    data = kwargs["pre_compute"]["text"]["value_by_index"]
    data_list = [
        {"text": entry[text_key], "index": int(key)} for key, entry in data.items()
    ]

    # Create DataLoader
    dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    scores_by_index = {}
    for batch in tqdm(dataloader):
        batch_texts = batch["text"]
        batch_indices = batch["index"].tolist()

        # Tokenize the batch of texts
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run the classifier
        with torch.no_grad():
            outputs = classifier(**inputs)
        # Convert logits to probabilities
        scores = F.softmax(outputs.logits, dim=-1)[:, class_id].cpu().numpy().tolist()

        # Map predictions to labels
        for idx, prob, text in zip(batch_indices, scores, batch_texts):
            # Add the prediction to the original data
            scores_by_index[idx] = {"score": prob, text_key: text}
    class_scores = np.array(
        [
            evals["score"]
            for evals in scores_by_index.values()
            if evals["score"] is not None
        ]
    )
    class_scores = aggregate_to_1D(class_scores)
    return {"agg_value": np.mean(class_scores), "value_by_index": scores_by_index}
