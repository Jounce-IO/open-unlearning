import os

import hydra
from omegaconf import DictConfig
from transformers import TrainerCallback
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything


class _WandbRunIdCallback(TrainerCallback):
    """Writes wandb.run.id to output_dir/wandb_run_id.txt so the parent (e.g. dllm unlearn) can resume the same W&B run after preemption."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self._written = False

    def _write_run_id(self):
        if self._written:
            return
        try:
            import wandb
            if wandb.run is not None:
                os.makedirs(self.output_dir, exist_ok=True)
                path = os.path.join(self.output_dir, "wandb_run_id.txt")
                with open(path, "w") as f:
                    f.write(wandb.run.id)
                self._written = True
        except Exception:
            pass

    def on_train_begin(self, args, state, control, **kwargs):
        self._write_run_id()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self._write_run_id()
        return control


def _save_inner_model_for_eval(model, save_dir: str) -> bool:
    """
    If model is a wrapper with an inner PreTrainedModel (e.g. DiffusionModelAdapter),
    save the inner model with save_pretrained so the checkpoint is loadable for eval.
    Unwraps DDP (model.module) if present. Returns True if saved, False if not applicable.
    """
    # Unwrap DDP / FSDP if present
    base = getattr(model, "module", model)
    inner = getattr(base, "model", base)
    if inner is base:
        return False
    if not hasattr(inner, "save_pretrained"):
        return False
    try:
        inner.save_pretrained(save_dir)
        print(f"Saved inner model for eval (config + weights) to {save_dir}", flush=True)
        return True
    except Exception as e:
        print(f"Warning: could not save inner model for eval to {save_dir}: {e}", flush=True)
        return False


class _SaveInnerModelForEvalCallback(TrainerCallback):
    """
    After each checkpoint save, saves the inner PreTrainedModel (e.g. under
    DiffusionModelAdapter) with save_pretrained so the checkpoint dir is loadable
    for evaluation (config.json + weights in HF format).
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(checkpoint_dir):
            _save_inner_model_for_eval(self.trainer.model, checkpoint_dir)
        return control


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    if trainer_args.do_train:
        trainer.add_callback(_WandbRunIdCallback(trainer_args.output_dir))
        trainer.add_callback(_SaveInnerModelForEvalCallback(trainer))
        resume_from_checkpoint = cfg.get("resume_from_checkpoint", None)
        if resume_from_checkpoint:
            print(f"Resuming training from checkpoint: {resume_from_checkpoint}", flush=True)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)
        # Save inner model in HF format so final output dir is loadable for eval
        _save_inner_model_for_eval(trainer.model, trainer_args.output_dir)
        # Write W&B run URL so parent (e.g. dllm unlearn) can include it in the report
        try:
            import wandb
            if wandb.run is not None:
                url_path = os.path.join(trainer_args.output_dir, "wandb_run_url.txt")
                os.makedirs(trainer_args.output_dir, exist_ok=True)
                with open(url_path, "w") as f:
                    f.write(wandb.run.get_url())
        except Exception:
            pass

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
