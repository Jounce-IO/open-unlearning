import json
import os

import hydra
from omegaconf import DictConfig
from transformers import TrainerCallback
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything


class _WandbDllmConfigCallback(TrainerCallback):
    """If DLLM_UNLEARN_CONFIG_PATH is set (by dllm unlearn), load the JSON and update wandb.config under key 'dllm' so the run has the full dllm config YAML contents in W&B."""

    def on_train_begin(self, args, state, control, **kwargs):
        config_path = os.environ.get("DLLM_UNLEARN_CONFIG_PATH", "").strip()
        if not config_path or not os.path.isfile(config_path):
            return control
        try:
            import wandb
            if wandb.run is None:
                return control
            with open(config_path, "r") as f:
                dllm_config = json.load(f)
            wandb.config.update({"dllm": dllm_config}, allow_val_change=True)
        except Exception:
            pass
        return control


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


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    # Support both DictConfig and plain dict for trainer (e.g. from Hydra composition).
    trainer_cfg = cfg.trainer
    try:
        trainer_args_cfg = getattr(trainer_cfg, "args", None) or (trainer_cfg.get("args") if isinstance(trainer_cfg, dict) else None)
    except Exception:
        trainer_args_cfg = None
    seed = 42
    if trainer_args_cfg is not None:
        seed = getattr(trainer_args_cfg, "seed", None) or (trainer_args_cfg.get("seed") if isinstance(trainer_args_cfg, dict) else None) or seed
    seed_everything(seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    resume_from_checkpoint = cfg.get("resume_from_checkpoint", None)
    model, tokenizer = get_model(model_cfg, resume_from_checkpoint=resume_from_checkpoint)

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

    # Get Evaluators (skip when eval is disabled so we never run TOFU metrics that require retain_ftr)
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    trainer_args_cfg = trainer_cfg.get("args", None) if hasattr(trainer_cfg, "get") else getattr(trainer_cfg, "args", None)
    eval_strategy = None
    if trainer_args_cfg is not None:
        eval_strategy = trainer_args_cfg.get("eval_strategy", None) if hasattr(trainer_args_cfg, "get") else getattr(trainer_args_cfg, "eval_strategy", None)
    if eval_cfgs and eval_strategy != "no":
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    # Four-way validation: use eval_dataset dict when present (forget, retain, holdout, utility)
    eval_dataset = data.get("eval_dataset", data.get("eval", None))
    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    if trainer_args.do_train:
        trainer.add_callback(_WandbRunIdCallback(trainer_args.output_dir))
        trainer.add_callback(_WandbDllmConfigCallback())
        make_checkpoint_loadable_fn = None
        try:
            from dllm.utils.checkpoint import MakeCheckpointLoadableCallback, make_checkpoint_loadable
            trainer.add_callback(MakeCheckpointLoadableCallback())
            make_checkpoint_loadable_fn = make_checkpoint_loadable
        except ImportError:
            pass
        if resume_from_checkpoint:
            print(f"Resuming training from checkpoint: {resume_from_checkpoint}", flush=True)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)
        if make_checkpoint_loadable_fn is not None:
            try:
                make_checkpoint_loadable_fn(trainer.model, trainer_args.output_dir)
            except Exception:
                pass
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

    # Skip post-training evaluate when eval is disabled to avoid requiring retain reference (retain_ftr).
    if trainer_args.do_eval and getattr(trainer_args, "eval_strategy", None) != "no":
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
