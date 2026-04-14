import torch
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf
from transformers import Trainer, TrainingArguments

from trainer.base import FinetuneTrainer, _DummyEvalDataset
from trainer.unlearn.grad_ascent import GradAscent
from trainer.unlearn.grad_diff import GradDiff
from trainer.unlearn.npo import NPO
from trainer.unlearn.dpo import DPO
from trainer.unlearn.simnpo import SimNPO
from trainer.unlearn.rmu import RMU
from trainer.unlearn.undial import UNDIAL
from trainer.unlearn.ceu import CEU
from trainer.unlearn.satimp import SatImp
from trainer.unlearn.wga import WGA
from trainer.unlearn.pdu import PDU


import logging

logger = logging.getLogger(__name__)

TRAINER_REGISTRY: Dict[str, Any] = {}


def _register_trainer(trainer_class):
    TRAINER_REGISTRY[trainer_class.__name__] = trainer_class


def load_trainer_args(trainer_args: DictConfig, dataset):
    trainer_args = dict(trainer_args)
    warmup_epochs = trainer_args.pop("warmup_epochs", None)
    if warmup_epochs:
        batch_size = trainer_args["per_device_train_batch_size"]
        grad_accum_steps = trainer_args["gradient_accumulation_steps"]
        num_devices = torch.cuda.device_count()
        dataset_len = len(dataset)
        trainer_args["warmup_steps"] = int(
            (warmup_epochs * dataset_len)
            // (batch_size * grad_accum_steps * num_devices)
        )

    trainer_args = TrainingArguments(**trainer_args)
    return trainer_args


def _get_cfg(cfg, key, default=None):
    """Get key from cfg whether it is DictConfig or plain dict."""
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_trainer(
    trainer_cfg: DictConfig,
    model,
    train_dataset=None,
    eval_dataset=None,
    tokenizer=None,
    data_collator=None,
    evaluators=None,
    template_args=None,
):
    # Support both DictConfig and plain dict (e.g. from Hydra composition).
    trainer_args = _get_cfg(trainer_cfg, "args")
    assert trainer_args is not None, "trainer.args is required"
    _raw_ma = _get_cfg(trainer_cfg, "method_args") or {}
    if OmegaConf.is_config(_raw_ma):
        method_args = dict(OmegaConf.to_container(_raw_ma, resolve=True))
    else:
        method_args = dict(_raw_ma)
    # TOFU multi-eval / legacy four-way ROUGE keys are dllm-only; HF Trainer rejects unknown kwargs.
    try:
        from dllm.utils.tofu_multi_eval_config import FINETUNE_TOFU_MULTI_EVAL_METHOD_ARG_KEYS

        for _k in list(method_args.keys()):
            if _k in FINETUNE_TOFU_MULTI_EVAL_METHOD_ARG_KEYS:
                method_args.pop(_k, None)
    except ImportError:
        pass
    # When eval_dataset is None but eval_strategy is not "no", Trainer.__init__ in
    # transformers >= 4.57 raises. We pass a dummy so init succeeds; FinetuneTrainer
    # runs custom evaluators every epoch and never uses the dummy. See _DummyEvalDataset.
    eval_dataset_to_pass = eval_dataset
    dummy_substituted = False
    if eval_dataset_to_pass is None:
        args_dict = dict(trainer_args)
        eval_strategy = args_dict.get("eval_strategy", None)
        if eval_strategy not in (None, "no"):
            eval_dataset_to_pass = _DummyEvalDataset()
            dummy_substituted = True
            logger.info(
                "[eval] load_trainer: eval_dataset=None eval_strategy=%s -> passing dummy (multi-eval will not run)",
                eval_strategy,
            )
    if eval_dataset_to_pass is not None and not dummy_substituted:
        if isinstance(eval_dataset_to_pass, dict):
            logger.info(
                "[eval] load_trainer: eval_dataset=dict keys=%s lengths=%s",
                list(eval_dataset_to_pass.keys()),
                {k: len(v) for k, v in eval_dataset_to_pass.items()},
            )
        else:
            logger.info("[eval] load_trainer: eval_dataset=single_dataset len=%s", len(eval_dataset_to_pass))
    trainer_args = load_trainer_args(trainer_args, train_dataset)
    trainer_handler_name = _get_cfg(trainer_cfg, "handler")
    assert trainer_handler_name is not None, ValueError(
        f"{trainer_handler_name} handler not set"
    )
    trainer_cls = TRAINER_REGISTRY.get(trainer_handler_name, None)
    assert trainer_cls is not None, NotImplementedError(
        f"{trainer_handler_name} not implemented or not registered"
    )
    trainer = trainer_cls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_to_pass,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=trainer_args,
        evaluators=evaluators,
        template_args=template_args,
        **method_args,
    )
    logger.info(
        f"{trainer_handler_name} Trainer loaded, output_dir: {trainer_args.output_dir}"
    )
    return trainer, trainer_args


# Register Finetuning Trainer
_register_trainer(Trainer)
_register_trainer(FinetuneTrainer)

# Register Unlearning Trainer
_register_trainer(GradAscent)
_register_trainer(GradDiff)
_register_trainer(NPO)
_register_trainer(DPO)
_register_trainer(SimNPO)
_register_trainer(RMU)
_register_trainer(UNDIAL)
_register_trainer(CEU)
_register_trainer(SatImp)
_register_trainer(WGA)
_register_trainer(PDU)
