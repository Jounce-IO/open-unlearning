# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

from typing import Dict, List, Optional, Union

import os
import logging
import torch
from transformers import Trainer
from torch.utils.data import Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Any

logger = logging.getLogger(__name__)


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
                # Phase 2: trainer baseline memory when eval runs (OOM investigation)
                if os.environ.get("OOM_INVESTIGATION", "").lower() in ("1", "true", "yes") and torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / (1024**2)
                    res = torch.cuda.memory_reserved() / (1024**2)
                    logger.info(
                        f"[OOM_INVESTIGATION] evaluate_entry: memory_allocated_MiB={alloc:.0f} memory_reserved_MiB={res:.0f}"
                    )
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
                    self.log(eval_metrics)
                else:
                    logger.warning(
                        "Custom evaluator can be run with this Trainer only when a single accelerator process is running."
                    )
                return eval_metrics

        if eval_dataset is None:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
