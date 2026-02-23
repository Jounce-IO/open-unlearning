import hydra
import logging
import os
from omegaconf import DictConfig

from trainer.utils import seed_everything
from model import get_model
from evals import get_evaluators
from evals.metrics.privacy import log_retain_logs_path_none_if_needed
from evals.gpu_phase_logger import set_phase as gpu_set_phase
from evals.distributed import get_rank, get_world_size, gather_logs_to_rank0

# Set up logging
logging.basicConfig(
    level=logging.INFO,
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
        # When distributed, gather partial logs to rank 0 and save once (base Evaluator with trajectory)
        if world_size > 1 and isinstance(logs, dict) and "config" in logs:
            merged_logs = gather_logs_to_rank0(logs, rank, world_size)
            if rank == 0 and merged_logs is not None:
                # Always set run_info before save so report PRs can verify data-parallel (no duplication).
                total_indices = None
                for key, value in merged_logs.items():
                    if key == "config" or not isinstance(value, dict):
                        continue
                    vbi = value.get("value_by_index")
                    if isinstance(vbi, dict) and vbi:
                        total_indices = len(vbi)
                        break
                if total_indices is not None:
                    merged_logs["run_info"] = {
                        "world_size": world_size,
                        "total_samples": total_indices,
                        "data_parallel": True,
                    }
                    logger.info(
                        "run_info set for distributed save: world_size=%s, total_samples=%s",
                        world_size,
                        total_indices,
                    )
                output_dir = getattr(evaluator.eval_cfg, "output_dir", None) or evaluator.eval_cfg.get("output_dir")
                if output_dir:
                    logs_file_path = evaluator.get_logs_file_path(output_dir)
                    evaluator.save_logs(merged_logs, logs_file_path)
        gpu_set_phase("evaluator_end", metric=evaluator_name)
        logger.info(f"Evaluator {evaluator_name} completed")
    gpu_set_phase("eval_complete")
    if rank == 0:
        logger.info("=== Evaluation complete ===")


if __name__ == "__main__":
    main()
