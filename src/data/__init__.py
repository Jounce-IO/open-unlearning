import logging
import re
from typing import Dict, Any, Union
from omegaconf import DictConfig
from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)

from data.qa import (
    QADataset,
    MMLUUtilityDataset,
    QAwithIdkDataset,
    QAwithAlternateDataset,
    QAwithDualAnswersDataset,
)
from data.collators import (
    DataCollatorForSupervisedDataset,
)
from data.unlearn import ForgetRetainDataset
from data.pretraining import PretrainingDataset, CompletionDataset

DATASET_REGISTRY: Dict[str, Any] = {}
COLLATOR_REGISTRY: Dict[str, Any] = {}


def _register_data(data_class):
    DATASET_REGISTRY[data_class.__name__] = data_class


def _register_collator(collator_class):
    COLLATOR_REGISTRY[collator_class.__name__] = collator_class


def _get_cfg(cfg, key, default=None):
    """Get key from cfg whether it is DictConfig or plain dict."""
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _load_single_dataset(dataset_name, dataset_cfg: DictConfig, **kwargs):
    dataset_handler_name = _get_cfg(dataset_cfg, "handler")
    assert dataset_handler_name is not None, ValueError(
        f"{dataset_name} handler not set"
    )
    dataset_handler = DATASET_REGISTRY.get(dataset_handler_name)
    if dataset_handler is None:
        raise NotImplementedError(
            f"{dataset_handler_name} not implemented or not registered"
        )
    args_cfg = _get_cfg(dataset_cfg, "args")
    dataset_args = dict(args_cfg) if args_cfg is not None else {}
    # QADataset expects split in hf_args (for load_hf_dataset); move top-level split if present
    split_val = dataset_args.pop("split", None) or kwargs.pop("split", None)
    # Apply samples limit: slice base split (e.g. train -> train[:2], forget_qa -> forget_qa[:2])
    # Strip existing slice from config (e.g. forget_qa[:5] -> forget_qa) to avoid double-slicing
    samples = kwargs.get("samples")
    if samples is not None and "hf_args" in dataset_args:
        base_split = dataset_args.get("hf_args", {}).get("split", "train")
        base_name = re.sub(r"\[\d*:\d*\]", "", base_split) or "train"
        split_val = f"{base_name}[:{samples}]"
    if split_val is not None and "hf_args" in dataset_args:
        dataset_args["hf_args"] = dict(dataset_args["hf_args"])
        dataset_args["hf_args"]["split"] = split_val
    # Only pass dataset-relevant kwargs; metric config keys (collators, metrics, etc.) are not valid
    dataset_relevant_keys = {"tokenizer", "template_args", "split"}
    handler_kwargs = {k: v for k, v in kwargs.items() if k in dataset_relevant_keys}
    return dataset_handler(**dataset_args, **handler_kwargs)


def get_datasets(dataset_cfgs: Union[Dict, DictConfig], **kwargs):
    dataset = {}
    for dataset_name, dataset_cfg in dataset_cfgs.items():
        access_name = dataset_cfg.get("access_key", dataset_name)
        dataset[access_name] = _load_single_dataset(dataset_name, dataset_cfg, **kwargs)
    if len(dataset) == 1:
        # return a single dataset
        return list(dataset.values())[0]
    # return mapping to multiple datasets
    return dataset


# Default max rows per named validation split (first N by order) when ``validation_cap`` is unset.
VALIDATION_MAX_SAMPLES = 100


def _cap_dataset_for_validation(
    dataset: Union[Dataset, Any], max_samples: int
) -> Union[Dataset, Any]:
    """Return ``dataset`` capped at ``max_samples`` (first N by order)."""
    cap = max(1, int(max_samples))
    try:
        n = len(dataset)
    except TypeError:
        return dataset
    if n <= cap:
        return dataset
    return Subset(dataset, range(cap))


def _cap_dataset_at_100(dataset: Union[Dataset, Any]) -> Union[Dataset, Any]:
    """Backward-compatible name: cap at :data:`VALIDATION_MAX_SAMPLES`."""
    return _cap_dataset_for_validation(dataset, VALIDATION_MAX_SAMPLES)


def get_data(data_cfg: DictConfig, mode="train", **kwargs):
    data = {}
    data_cfg = dict(data_cfg)
    anchor = data_cfg.pop("anchor", "forget")
    val_cap_raw = data_cfg.pop("validation_cap", None)
    validation_cap = (
        int(val_cap_raw) if val_cap_raw is not None else VALIDATION_MAX_SAMPLES
    )
    validation_splits_cfg = data_cfg.pop("validation_splits", None)
    if validation_splits_cfg is not None:
        validation_keys = list(validation_splits_cfg.keys())
        logger.info(
            "[eval] get_data: mode=%s validation_splits=present keys=%s validation_cap=%s",
            mode,
            validation_keys,
            validation_cap,
        )
    else:
        logger.info(
            "[eval] get_data: mode=%s validation_splits=missing (multi-eval off if mode=unlearn)",
            mode,
        )
    for split, dataset_cfgs in data_cfg.items():
        data[split] = get_datasets(dataset_cfgs, **kwargs)
    if mode == "train" and validation_splits_cfg is not None:
        eval_dict = {}
        for name, dataset_cfgs in validation_splits_cfg.items():
            ds = get_datasets(dataset_cfgs, **kwargs)
            eval_dict[name] = _cap_dataset_for_validation(ds, validation_cap)
        data["eval_dataset"] = eval_dict
        logger.info(
            "[eval] get_data: train mode built eval_dataset keys=%s lengths=%s",
            list(eval_dict.keys()),
            {k: len(v) for k, v in eval_dict.items()},
        )
    if mode == "unlearn":
        unlearn_splits = {k: v for k, v in data.items() if k not in ("eval", "test", "eval_dataset")}
        unlearn_dataset = ForgetRetainDataset(**unlearn_splits, anchor=anchor)
        data["train"] = unlearn_dataset
        for split in list(unlearn_splits.keys()):
            data.pop(split, None)
        if validation_splits_cfg is not None:
            eval_dict = {}
            for name, dataset_cfgs in validation_splits_cfg.items():
                ds = get_datasets(dataset_cfgs, **kwargs)
                eval_dict[name] = _cap_dataset_for_validation(ds, validation_cap)
            data["eval_dataset"] = eval_dict
            logger.info(
                "[eval] get_data: unlearn mode built multi-eval eval_dataset keys=%s lengths=%s",
                list(eval_dict.keys()),
                {k: len(v) for k, v in eval_dict.items()},
            )
        else:
            logger.info(
                "[eval] get_data: unlearn mode no validation_splits so eval_dataset not set"
            )
    return data


def _get_single_collator(collator_name: str, collator_cfg: DictConfig, **kwargs):
    collator_handler_name = collator_cfg.get("handler")
    assert collator_handler_name is not None, ValueError(
        f"{collator_name} handler not set"
    )
    collator_handler = COLLATOR_REGISTRY.get(collator_handler_name)
    if collator_handler is None:
        raise NotImplementedError(
            f"{collator_handler_name} not implemented or not registered"
        )
    collator_args_cfg = _get_cfg(collator_cfg, "args")
    collator_args = dict(collator_args_cfg) if collator_args_cfg is not None else {}
    return collator_handler(**collator_args, **kwargs)


def get_collators(collator_cfgs, **kwargs):
    collators = {}
    for collator_name, collator_cfg in collator_cfgs.items():
        collators[collator_name] = _get_single_collator(
            collator_name, collator_cfg, **kwargs
        )
    if len(collators) == 1:
        # return a single collator
        return list(collators.values())[0]
    # return collators in a dict
    return collators


# Register datasets
_register_data(QADataset)
_register_data(MMLUUtilityDataset)
_register_data(QAwithIdkDataset)
_register_data(PretrainingDataset)
_register_data(CompletionDataset)
_register_data(QAwithAlternateDataset)
_register_data(QAwithDualAnswersDataset)

# Register composite datasets used in unlearning
# groups: unlearn
_register_data(ForgetRetainDataset)

# Register collators
_register_collator(DataCollatorForSupervisedDataset)
