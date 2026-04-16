# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

from typing import Dict, List, Optional, Union

import os
import numpy as np
import logging
import torch
from transformers import Trainer
from torch.utils.data import Dataset
from transformers.trainer_utils import EvalLoopOutput, PREFIX_CHECKPOINT_DIR
from typing import Any

logger = logging.getLogger(__name__)

# Four-way validation: max samples per validation set (fixed slice for determinism).
FOUR_WAY_VALIDATION_CAP = 100

# Keys that are never sent to W&B (metadata / retain-reference only).
_WANDB_SKIP_KEYS = frozenset(
    {
        "config",
        "run_info",
        "trajectory_step_metadata",
        "mia_min_k_by_step",
        "forget_truth_ratio_by_step",
    }
)


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
                            if (
                                arr is not None
                                and hasattr(arr, "__len__")
                                and len(arr) > 0
                            ):
                                try:
                                    out[key] = float(
                                        np.nanmean(np.asarray(arr, dtype=np.float64))
                                    )
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
        # Set by ``load_trainer`` when dllm is available (TOFU multi-eval Phase 2 + schedule).
        self._dllm_tofu_multi_phase2_runtime = None

    def attach_tofu_multi_eval_phase2_runtime(self, runtime) -> None:
        """Store normalized TOFU multi-eval runtime from ``trainer.method_args`` (dllm)."""
        self._dllm_tofu_multi_phase2_runtime = runtime

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        # Diagnostic log for four-way debugging (caller often passes no args; we use self.eval_dataset then)
        caller_passed = eval_dataset is not None
        self_eval = getattr(self, "eval_dataset", None)
        self_eval_type = "None"
        if self_eval is not None:
            self_eval_type = (
                "dict_keys=%s" % list(self_eval.keys())
                if isinstance(self_eval, dict)
                else "single_len=%s" % len(self_eval)
            )
        logger.info(
            "[eval] evaluate: caller_passed_dataset=%s self_eval_dataset=%s (step=%s)",
            caller_passed,
            self_eval_type,
            self.state.global_step
            if hasattr(self, "state") and self.state is not None
            else "?",
        )
        # Run a custom evaluator and save results
        if self.evaluators:
            logger.info(
                "[eval] evaluate: using evaluators path (four-way skipped) step=%s",
                self.state.global_step
                if hasattr(self, "state") and self.state is not None
                else "?",
            )
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
                    ed = getattr(self, "eval_dataset", None)
                    rt = getattr(self, "_dllm_tofu_multi_phase2_runtime", None)
                    if (
                        isinstance(ed, dict)
                        and rt is not None
                        and getattr(rt, "tofu_multi_eval", True)
                    ):
                        try:
                            from dllm.core.trainers.tofu_multi_eval_phase2 import (
                                run_tofu_multi_eval_phase2,
                                tofu_multi_eval_step_schedule,
                            )

                            run_general, include_rouge = tofu_multi_eval_step_schedule(
                                int(getattr(self.state, "global_step", 0) or 0),
                                rt.tofu_multi_eval_steps,
                                rt.tofu_multi_eval_rouge_steps,
                                tofu_multi_eval_rouge_enabled=rt.tofu_multi_eval_rouge,
                            )
                            if run_general:
                                p2 = run_tofu_multi_eval_phase2(
                                    self,
                                    ed,
                                    runtime=rt,
                                    ignore_keys=None,
                                    metric_key_prefix=metric_key_prefix,
                                    include_rouge=include_rouge,
                                    log_metrics=False,
                                )
                                for k, v in (p2.metrics or {}).items():
                                    if isinstance(v, (int, float)):
                                        eval_metrics[k] = float(v)
                        except Exception as exc:
                            logger.warning(
                                "TOFU multi-eval Phase 2 merge alongside evaluators failed: %s",
                                exc,
                                exc_info=True,
                            )
                self.log(_scalar_metrics_for_wandb(eval_metrics))
            else:
                logger.warning(
                    "Custom evaluator can be run with this Trainer only when a single accelerator process is running."
                )
            return eval_metrics

        # When the training loop calls evaluate() it does not pass eval_dataset; use the one from __init__.
        if eval_dataset is None:
            eval_dataset = getattr(self, "eval_dataset", None)
        if eval_dataset is None:
            logger.info(
                "[eval] evaluate: no eval_dataset -> returning {} (four-way skipped)"
            )
            return {}
        # Four-way validation: eval_dataset is a dict of named datasets (forget, retain, holdout, utility).
        # Compute both method loss and constant CE loss per set, then merge and log.
        if isinstance(eval_dataset, dict):
            logger.info(
                "[eval] evaluate: running four-way validation keys=%s",
                list(eval_dataset.keys()),
            )
            return self._evaluate_four_way(
                eval_dataset, ignore_keys, metric_key_prefix, trial
            )
        # Single dataset: default HF Trainer evaluate
        logger.info("[eval] evaluate: running single-dataset evaluation")
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def _evaluate_tofu_multi_eval_mdlm_style(
        self,
        eval_dataset: Dict[str, Dataset],
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ) -> EvalLoopOutput:
        """TOFU multi-split eval matching SFT: schedule gate, Phase 1 ``Trainer.evaluate`` per split, Phase 2 shared dllm pass.

        Phase 1 must call the base :class:`~transformers.Trainer` implementation (not
        :meth:`FinetuneTrainer.evaluate`) so ``retain_logs_path`` / evaluator short-circuit
        does not skip per-split HF eval.
        """
        from dllm.core.trainers.tofu_multi_eval_phase2 import (
            run_tofu_multi_eval_phase2,
            tofu_multi_eval_step_schedule,
        )

        rt = self._dllm_tofu_multi_phase2_runtime
        run_general, include_rouge = tofu_multi_eval_step_schedule(
            int(getattr(self.state, "global_step", 0) or 0),
            rt.tofu_multi_eval_steps,
            rt.tofu_multi_eval_rouge_steps,
            tofu_multi_eval_rouge_enabled=rt.tofu_multi_eval_rouge,
        )
        if not run_general:
            if self.is_world_process_zero():
                logger.info(
                    "[tofu-multi-eval eval] Skipped (global_step=%d not on "
                    "tofu_multi_eval_steps=%d boundary; align eval_steps or use "
                    "eval_on_start for step 0).",
                    int(getattr(self.state, "global_step", 0) or 0),
                    rt.tofu_multi_eval_steps,
                )
            return EvalLoopOutput(
                predictions=None, label_ids=None, metrics={}, num_samples=None
            )

        combined_metrics: Dict[str, float] = {}
        if self.is_world_process_zero():
            logger.info(
                "[tofu-multi-eval eval] Phase 1 (base Trainer.evaluate per split): "
                "method loss — no ROUGE in this pass."
            )
        for name, dataset in eval_dataset.items():
            if dataset is None or len(dataset) == 0:
                continue
            if self.is_world_process_zero():
                logger.info(
                    "[tofu-multi-eval eval] Phase 1 split=%r n_examples=%d",
                    name,
                    len(dataset),
                )
            metrics = super(FinetuneTrainer, self).evaluate(
                dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=f"{metric_key_prefix}_{name}",
            )
            if hasattr(metrics, "metrics") and getattr(metrics, "metrics", None):
                combined_metrics.update(metrics.metrics)
            elif isinstance(metrics, dict) and metrics:
                combined_metrics.update(metrics)
        if self.is_world_process_zero():
            logger.info(
                "[tofu-multi-eval eval] Phase 2 (dllm shared): include_rouge=%s step=%s",
                include_rouge,
                int(getattr(self.state, "global_step", 0) or 0),
            )
        four_way_output = run_tofu_multi_eval_phase2(
            self,
            eval_dataset,
            runtime=rt,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            include_rouge=include_rouge,
            log_metrics=False,
        )
        if four_way_output.metrics:
            combined_metrics.update(four_way_output.metrics)
        if combined_metrics and self.is_world_process_zero():
            self.log(_scalar_metrics_for_wandb(combined_metrics))
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=combined_metrics,
            num_samples=four_way_output.num_samples,
        )

    def _evaluate_four_way(
        self,
        eval_dataset: Dict[str, Dataset],
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ):
        """Evaluate on each named dataset; report method loss and constant CE loss per set.

        Both method loss and CE loss are sample-weighted averages (sum of loss*bs / sum of bs).
        Returns EvalLoopOutput to match parent Trainer.evaluate() return type.
        Under multi-GPU, sums and counts are all-reduced so metrics are global; logging on rank 0 only.
        num_samples is the total eval set size across all four splits.
        """
        rt = getattr(self, "_dllm_tofu_multi_phase2_runtime", None)
        if rt is not None and getattr(rt, "tofu_multi_eval", True):
            return self._evaluate_tofu_multi_eval_mdlm_style(
                eval_dataset, ignore_keys, metric_key_prefix, trial
            )

        eval_metrics: Dict[str, float] = {}
        device = getattr(self.args, "device", torch.device("cpu"))
        num_samples = 0
        ce_available = False
        try:
            from dllm.core.schedulers import LinearAlphaScheduler
            from dllm.core.trainers.tofu_multi_eval_phase2 import (
                compute_masked_ce_eval_loss,
            )

            adapter = getattr(self.model, "adapter_config", None)
            tokenizer = getattr(self.model, "tokenizer", None)
            if adapter is not None and tokenizer is not None:
                ce_available = True
        except Exception:
            pass

        for name, dataset in eval_dataset.items():
            if dataset is None or len(dataset) == 0:
                continue
            # get_eval_dataloader may shard per rank; all-reduce below makes metrics global.
            dataloader = self.get_eval_dataloader(dataset)
            method_loss_sum = 0.0
            method_n = 0
            ce_loss_sum = 0.0
            ce_n = 0
            self.model.eval()
            for batch in dataloader:
                batch = self._prepare_inputs(batch)
                with torch.no_grad():
                    loss, _, _ = self.prediction_step(
                        self.model,
                        batch,
                        prediction_loss_only=False,
                        ignore_keys=ignore_keys or [],
                    )
                    if loss is not None:
                        method_loss_sum += loss.item() * batch["input_ids"].size(0)
                        method_n += batch["input_ids"].size(0)
                    if ce_available:
                        try:
                            inner = getattr(self.model, "model", self.model)
                            sched = (
                                getattr(
                                    getattr(self.model, "adapter_config", None),
                                    "scheduler",
                                    None,
                                )
                                or LinearAlphaScheduler()
                            )
                            proc = getattr(self.model, "tokenizer", None)
                            if (
                                inner is not None
                                and proc is not None
                                and getattr(proc, "mask_token_id", None) is not None
                            ):
                                ce = compute_masked_ce_eval_loss(
                                    inner,
                                    batch,
                                    proc,
                                    sched,
                                    fixed_t=0.5,
                                    time_epsilon=float(
                                        getattr(adapter, "time_epsilon", 1e-3)
                                    ),
                                    loss_normalization_type=str(
                                        getattr(
                                            adapter,
                                            "loss_normalization_type",
                                            "sequence",
                                        )
                                    ),
                                    loss_weight_type=str(
                                        getattr(
                                            adapter, "loss_weight_type", "scheduler"
                                        )
                                    ),
                                )
                                ce_loss_sum += ce.item() * batch["input_ids"].size(0)
                                ce_n += batch["input_ids"].size(0)
                        except Exception:
                            pass
            # All-reduce so multi-GPU runs get correct global averages and a single log.
            stats = torch.tensor(
                [method_loss_sum, float(method_n), ce_loss_sum, float(ce_n)],
                device=device,
                dtype=torch.float64,
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
            method_loss_sum = stats[0].item()
            method_n = int(round(stats[1].item()))
            ce_loss_sum = stats[2].item()
            ce_n = int(round(stats[3].item()))
            if method_n > 0:
                eval_metrics[f"{metric_key_prefix}_{name}_loss"] = (
                    method_loss_sum / method_n
                )
            if ce_n > 0:
                eval_metrics[f"{metric_key_prefix}_{name}_loss_ce"] = ce_loss_sum / ce_n
            elif method_n > 0:
                logger.warning(
                    "Four-way eval: CE loss not computed for %s (ce_n=0); only method loss logged for this split.",
                    name,
                )
            num_samples += len(dataset)
        if eval_metrics and self.is_world_process_zero():
            self.log(_scalar_metrics_for_wandb(eval_metrics))
            logger.info(
                "Four-way eval metrics: %s", ", ".join(sorted(eval_metrics.keys()))
            )
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=eval_metrics,
            num_samples=num_samples if num_samples > 0 else None,
        )
