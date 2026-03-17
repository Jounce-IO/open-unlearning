import hydra
import logging
import os
from omegaconf import DictConfig

from trainer.utils import seed_everything
from model import get_model
from evals import get_evaluators
from evals.metrics.privacy import log_retain_logs_path_none_if_needed
from evals.gpu_phase_logger import set_phase as gpu_set_phase
from evals.distributed import (
    get_rank,
    get_world_size,
    _total_samples_from_merged_logs,
)

# Set up logging (level from LOGLEVEL env: DEBUG, INFO, WARNING, ERROR; default INFO)
_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
_log_level_name = os.environ.get("LOGLEVEL", "INFO").upper()
_log_level = _level_map.get(_log_level_name, logging.INFO)
logging.basicConfig(
    level=_log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Enable verbose logging for HuggingFace
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")
os.environ.setdefault("HF_HUB_VERBOSITY", "1")  # 1=info, 2=debug


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    rank = get_rank()
    world_size = get_world_size()
    if world_size > 1:
        logger.info("=== Starting evaluation (distributed: rank %s / %s) ===", rank, world_size)
    else:
        logger.info("=== Starting evaluation ===")
    gpu_set_phase("eval_start")
    seed_everything(cfg.seed)
    model_cfg = cfg.model
    template_args = model_cfg.get("template_args", None)
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    logger.info(f"Loading model: {model_cfg.get('model_args', {}).get('pretrained_model_name_or_path', 'unknown')}")
    model, tokenizer = get_model(model_cfg)
    logger.info("Model and tokenizer loaded successfully")
    gpu_set_phase("eval_model_loaded")

    eval_cfgs = cfg.eval
    # When using eval=trajectory_test, Hydra loads the config and cfg.eval
    # should be a DictConfig. Handle both dict of configs and single config
    from omegaconf import OmegaConf, open_dict
    # Check if it's a single config (has handler) vs dict of configs
    with open_dict(eval_cfgs):
        has_handler = eval_cfgs.get('handler') is not None
    # If it's a single config, wrap it in a dict
    if has_handler:
        eval_name = 'trajectory_test'  # Default
        eval_cfgs = {eval_name: eval_cfgs}
    # Ensure it's a DictConfig for get_evaluators
    if not isinstance(eval_cfgs, DictConfig):
        eval_cfgs = OmegaConf.create(eval_cfgs)
    logger.info("Getting evaluators...")
    evaluators = get_evaluators(eval_cfgs)
    logger.info(f"Found {len(evaluators)} evaluator(s): {list(evaluators.keys())}")
    for evaluator_name, evaluator in evaluators.items():
        if hasattr(evaluator, "metrics"):
            log_retain_logs_path_none_if_needed(
                "start of run",
                evaluator.metrics,
                evaluator.eval_cfg.get("retain_logs_path"),
            )
        logger.info(f"Running evaluator: {evaluator_name}")
        gpu_set_phase("evaluator_start", metric=evaluator_name)
        eval_args = {
            "template_args": template_args,
            "model": model,
            "tokenizer": tokenizer,
            "rank": rank,
            "world_size": world_size,
        }
        logs = evaluator.evaluate(**eval_args)
        output_dir = getattr(evaluator.eval_cfg, "output_dir", None) or evaluator.eval_cfg.get("output_dir")
        if world_size > 1 and isinstance(logs, dict) and "config" in logs and output_dir:
            # Data parallel: each rank writes its own file so the report can merge and show true run_info.
            samples_this_rank = _total_samples_from_merged_logs(logs) or 0
            if samples_this_rank == 0:
                # Trajectory metrics don't fill value_by_index; infer from config (DistributedSampler split).
                total = getattr(evaluator.eval_cfg, "samples", None)
                if total is not None:
                    total = int(total)
                    base, rem = divmod(total, world_size)
                    samples_this_rank = base + (1 if rank < rem else 0)
            logs["run_info"] = {
                "rank": rank,
                "world_size": world_size,
                "samples_this_rank": samples_this_rank,
                "data_parallel": True,
            }
            logger.info(
                "rank %s/%s saving %s samples to per-rank file",
                rank,
                world_size,
                samples_this_rank,
            )
            logs_file_path = evaluator.get_logs_file_path(output_dir, suffix=f"EVAL_rank{rank}")
            evaluator.save_logs(logs, logs_file_path, keep_value_by_index=True)
        # Single process: base evaluator already saves in evaluate()
        gpu_set_phase("evaluator_end", metric=evaluator_name)
        logger.info(f"Evaluator {evaluator_name} completed")
    gpu_set_phase("eval_complete")
    if rank == 0:
        logger.info("=== Evaluation complete ===")


if __name__ == "__main__":
    main()
