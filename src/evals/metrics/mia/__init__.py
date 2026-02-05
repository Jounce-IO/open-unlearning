"""
Attack implementations.
"""

from transformers import AutoModelForCausalLM

from evals.metrics.base import unlearning_metric
from evals.metrics.mia.loss import LOSSAttack
from evals.metrics.mia.min_k import MinKProbAttack
from evals.metrics.mia.min_k_plus_plus import MinKPlusPlusAttack
from evals.metrics.mia.gradnorm import GradNormAttack
from evals.metrics.mia.zlib import ZLIBAttack
from evals.metrics.mia.reference import ReferenceAttack

from evals.metrics.mia.utils import mia_auc
import logging

logger = logging.getLogger("metrics")

## NOTE: all MIA attack statistics are signed as required in order to show the
# same trends as loss (higher the score on an example, less likely the membership)


@unlearning_metric(name="mia_loss")
def mia_loss(model, **kwargs):
    return mia_auc(
        LOSSAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
    )


@unlearning_metric(name="mia_min_k")
def mia_min_k(model, **kwargs):
    k = kwargs.get("k", 0.4)
    if "k" in kwargs:
        logger.info("mia_min_k: using k=%.2f from config", k)
    else:
        logger.info(
            "mia_min_k: using default k=%.2f (config had no k; open-unlearning configs use 0.4, attack class default 0.2)",
            k,
        )
    return mia_auc(
        MinKProbAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        k=k,
    )


@unlearning_metric(name="mia_min_k_plus_plus")
def mia_min_k_plus_plus(model, **kwargs):
    return mia_auc(
        MinKPlusPlusAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        k=kwargs.get("k", 0.4),
    )


@unlearning_metric(name="mia_gradnorm")
def mia_gradnorm(model, **kwargs):
    return mia_auc(
        GradNormAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        p=kwargs.get("p", 2),
    )


@unlearning_metric(name="mia_zlib")
def mia_zlib(model, **kwargs):
    return mia_auc(
        ZLIBAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        tokenizer=kwargs.get("tokenizer"),
    )


@unlearning_metric(name="mia_reference")
def mia_reference(model, **kwargs):
    if "reference_model_path" not in kwargs:
        raise ValueError("Reference model must be provided in kwargs")
    logger.info(f"Loading reference model from {kwargs['reference_model_path']}")
    reference_model = AutoModelForCausalLM.from_pretrained(
        kwargs["reference_model_path"],
        torch_dtype=model.dtype,
        device_map={"": model.device},
    )
    return mia_auc(
        ReferenceAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        reference_model=reference_model,
    )
