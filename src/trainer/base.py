# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

from typing import Dict, List, Optional, Union

import os
import numpy as np
import logging
from transformers import Trainer
from torch.utils.data import Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Any

logger = logging.getLogger(__name__)

# Keys that are never sent to W&B (metadata / retain-reference only).
_WANDB_SKIP_KEYS = frozenset({
    "config",
    "run_info",
    "trajectory_step_metadata",
    "mia_min_k_by_step",
    "forget_truth_ratio_by_step",
})


def _scalar_metrics_for_wandb(eval_metrics: dict) -> Dict[str, float]:
    """Reduce evaluator output to scalar-only dict for W&B logging.

    When training with report_to=wandb, only scalar aggregate evaluation
    metrics are sent to W&B. Full results (value_by_index, step_distribution,
    etc.) remain in the evaluator JSON files (e.g. TOFU_EVAL.json,
    MUSE_EVAL.json); this function only affects what is passed to self.log().

    Input: raw eval_metrics as returned by evaluators (TOFU, MUSE, LMEval).
    Output: flat dict[str, float] suitable for Trainer.log().

    Keys always skipped (never sent to W&B): config, run_info,
    trajectory_step_metadata, mia_min_k_by_step, forget_truth_ratio_by_step.

    Handling:
    (1) Top-level scalar values (e.g. LMEval): log key -> float(value).
    (2) Dict with scalar agg_value: log key -> float(agg_value).
    (3) Dict with nested agg_value (view -> traj -> metric -> array): one
        scalar per key via mean over steps (first view, first traj, first
        inner metric); numpy arrays are reduced with np.nanmean(leaf).

    Per-sample and per-step data (value_by_index, step_distribution) are
    never sent to W&B and remain only in the evaluator's JSON files.
    """
    out: Dict[str, float] = {}
    for key, value in eval_metrics.items():
        if key in _WANDB_SKIP_KEYS:
            continue
        if isinstance(value, (int, float)):
            out[key] = float(value)
            continue
        try:
            if hasattr(value, "item"):
                out[key] = float(value.item())
                continue
        except (ValueError, AttributeError):
            pass
        if isinstance(value, dict):
            agg = value.get("agg_value")
            if agg is None:
                continue
            if isinstance(agg, (int, float)):
                out[key] = float(agg)
                continue
            try:
                if hasattr(agg, "item"):
                    out[key] = float(agg.item())
                    continue
            except (ValueError, AttributeError):
                pass
            if isinstance(agg, dict):
                # Nested trajectory: view -> traj -> metric -> array
                for view_val in agg.values():
                    if not isinstance(view_val, dict):
                        continue
                    for traj_val in view_val.values():
                        if not isinstance(traj_val, dict):
                            continue
                        for arr in traj_val.values():
                            if arr is not None and hasattr(arr, "__len__") and len(arr) > 0:
                                try:
                                    out[key] = float(np.nanmean(np.asarray(arr, dtype=np.float64)))
                                except (TypeError, ValueError):
                                    pass
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
    return out


class _DummyEvalDataset(Dataset):
    """
    Minimal placeholder dataset used only to satisfy Trainer.__init__ when
    eval_strategy is not "no" but no eval_dataset is provided (e.g. data/unlearn
    has datasets@eval: null).

    Why this exists:
    - Open-unlearning defaults: GradAscent inherits finetune config with
      eval_strategy=epoch and do_eval=True, while data/unlearn.yaml has
      datasets@eval: null, so get_data() never returns an "eval" key and
      eval_dataset is always None.
    - In the original locuslab repo (transformers==4.45.1), Trainer.__init__
      does not check for eval_dataset when eval_strategy != "no", so the
      Trainer is created and at the end of each epoch FinetuneTrainer.evaluate()
      runs the custom evaluators (TOFU/MUSE metrics) and returns without ever
      needing an eval_dataset.
    - In transformers >= 4.57 (used by the dllm repo), Trainer.__init__ raises
      ValueError if eval_strategy is not "no" and eval_dataset is None. So we
      would fail at Trainer creation and never run the real evaluators.
    - This dummy is passed only to satisfy that init check. FinetuneTrainer.evaluate()
      always runs the custom evaluators first and returns; it never calls
      get_eval_dataloader() or super().evaluate() when evaluators are set, so
      this dataset is never used for any forward pass or metric. Evaluation
      remains the real TOFU/MUSE evaluators every epoch, matching open-unlearning.
    """

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int):
        raise IndexError(index)


class FinetuneTrainer(Trainer):
    def __init__(self, evaluators=None, template_args=None, *args, **kwargs):
        self.evaluators = evaluators
        self.template_args = template_args
        super().__init__(*args, **kwargs)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        # Run a custom evaluator and save results
        if self.evaluators:
            if self.accelerator.is_local_main_process:
                eval_metrics = {}
                if self.accelerator.num_processes == 1:
                    run_dir = self._get_output_dir(trial=trial)
                    checkpoint_folder = (
                        f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                    )
                    output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
                    os.makedirs(output_dir, exist_ok=True)
                    eval_metrics = {}
                    for _, evaluator in self.evaluators.items():
                        eval_args = {
                            "output_dir": output_dir,
                            "template_args": self.template_args,
                            "model": self.model,
                            "tokenizer": self.tokenizer,
                        }
                        eval_metrics.update(evaluator.evaluate(**eval_args))
                    self.log(_scalar_metrics_for_wandb(eval_metrics))
                else:
                    logger.warning(
                        "Custom evaluator can be run with this Trainer only when a single accelerator process is running."
                    )
                return eval_metrics

        if eval_dataset is None:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
