import os
import json
import logging
from typing import Callable, Any, Dict

import torch
from data import get_datasets, get_collators

logger = logging.getLogger("metrics")


class UnlearningMetric:
    def __init__(
        self,
        name: str,
        metric_fn: Callable[..., Any],
    ):
        self.name = name
        self._metric_fn = metric_fn
        self.data = None
        self.collators = None
        self.pre_compute_metrics: Dict[str, Callable] = {}

    def get_datasets(self, dataset_cfgs=None, **kwargs):
        """Load the datasets from config"""
        if self.data:
            return self.data
        data = get_datasets(dataset_cfgs=dataset_cfgs, **kwargs)
        return data

    def get_collators(self, collator_cfgs=None, **kwargs):
        """Load the collators from config"""
        if self.collators:
            return self.collators
        # If collator_cfgs is already a collator instance (callable), use it
        # DictConfig from Hydra is not a dict, so don't treat it as "instance"
        if collator_cfgs is not None and callable(collator_cfgs):
            return collator_cfgs
        collators = get_collators(
            tokenizer=kwargs.get("tokenizer", None), collator_cfgs=collator_cfgs
        )
        return collators

    def set_pre_compute_metrics(self, metrics: Dict[str, Callable]):
        self.pre_compute_metrics.update(metrics)

    def evaluate_metric(self, model, metric_name, **kwargs):
        logger.info(f"Evaluating {metric_name}")
        results = self._metric_fn(model, **kwargs)
        return results

    def load_logs_from_file(self, file):
        """Load a logs file, assumes json"""
        logs = {}
        if os.path.exists(file):
            logger.info(f"Loading evaluations from {file}")
            with open(file, "r") as f:
                logs = json.load(f)
        else:
            raise ValueError(f"{file} doesn't exist!")
        return logs

    def prepare_kwargs_evaluate_metric(self, model, metric_name, cache={}, **kwargs):
        """Prepare the kwargs required to call the metric_fn defined by user.

        - Loads datasets, collators, and results for pre_compute metrics.
        - Loads reference_logs from JSON when reference_logs config has a path: for each
          entry, reads the file and populates requested include keys plus (when present)
          retain_mia_by_step and retain_forget_tr_by_step for step-matched privleak/ks_test.
          Logs a summary so you can distinguish "no usable keys found" (WARNING) from
          "loaded: found [X]; missing [Y]" (INFO, or INFO + per-key WARNING for missing).
        Returns:
            Dict: Updated kwargs with data, collators, pre_compute, and reference_logs.
        """
        # Load datasets
        dataset_cfgs = kwargs.pop("datasets", None)
        if dataset_cfgs is not None:
            data = self.get_datasets(dataset_cfgs=dataset_cfgs, **kwargs)
            kwargs.update({"data": data})

        # Load collators
        collator_cfgs = kwargs.pop("collators", None)
        if collator_cfgs is not None:
            collators = self.get_collators(collator_cfgs=collator_cfgs, **kwargs)
            kwargs.update({"collators": collators})

        # Evaluate precompute and load results
        pre_compute_cfgs = kwargs.pop("pre_compute", {})
        pre_metric_results = {}
        for pre_metric_name, pre_metric_cfg in pre_compute_cfgs.items():
            access_name = pre_metric_cfg.get("access_key", pre_metric_name)
            _results = {}
            if pre_metric_name in cache:
                logger.info(
                    f"Skipping {metric_name}'s precompute {pre_metric_name}, already evaluated."
                )
                _results = cache[pre_metric_name]
            else:
                pre_metric = self.pre_compute_metrics.get(pre_metric_name, None)
                assert pre_metric is not None, ValueError(
                    f"No pre_compute metric of name {pre_metric_name}"
                )
                pre_metric_kwargs = kwargs.copy()
                pre_metric_kwargs.update(**pre_metric_cfg)
                _results = pre_metric.evaluate(
                    model, pre_metric_name, cache=cache, **pre_metric_kwargs
                )
            pre_metric_results.update({access_name: _results})
        if pre_metric_results:
            kwargs.update({"pre_compute": pre_metric_results})

        # Load reference logs from JSON (retain-model eval results for privleak / forget_quality).
        # Only keys listed in reference_log_cfg.include are requested; the loader also injects
        # retain_mia_by_step and retain_forget_tr_by_step from the file when present (for
        # step-matched privleak/ks_test). Metrics that require reference_logs: privleak,
        # forget_quality (ks_test), trajectory_privleak, trajectory_forget_quality (see
        # evals.metrics.privacy.RETAIN_LOGS_METRICS). After loading we log a summary so you
        # can distinguish "no keys found" (wrong path or empty file) from "partial" (some
        # keys found, some missing).
        reference_logs_cfgs = kwargs.pop("reference_logs", {})
        reference_logs = {}
        for reference_log_name, reference_log_cfg in reference_logs_cfgs.items():
            path = reference_log_cfg.get("path", None)
            if path is None:
                continue
            include_cfgs = reference_log_cfg.get("include", None) or {}
            assert path is not None, ValueError(
                "path not specified for {reference_log_name} in {metric_name}"
            )
            _logs = self.load_logs_from_file(path)
            _traj = _logs.get("trajectory_all") or {}
            reference_logs[reference_log_name] = {}
            found_include = []
            missing_include = []
            for key, include_cfg in include_cfgs.items():
                access_name = include_cfg.get("access_key", key)
                _results = _logs.get(key, None)
                if _results is None:
                    _results = _traj.get(key, None)
                if _results is None and key == "forget_truth_ratio":
                    _results = (
                        _logs.get("forget_Truth_Ratio")
                        or _traj.get("forget_Truth_Ratio")
                        or _logs.get("trajectory_forget_Truth_Ratio")
                        or _traj.get("trajectory_forget_Truth_Ratio")
                    )
                if _results is None and key == "retain_Truth_Ratio":
                    _results = (
                        _logs.get("retain_Truth_Ratio")
                        or _traj.get("retain_Truth_Ratio")
                        or _logs.get("trajectory_retain_Truth_Ratio")
                        or _traj.get("trajectory_retain_Truth_Ratio")
                    )
                if _results is None and key == "retain_Q_A_Prob":
                    _results = (
                        _logs.get("retain_Q_A_Prob")
                        or _traj.get("retain_Q_A_Prob")
                        or _logs.get("trajectory_retain_Q_A_Prob")
                        or _traj.get("trajectory_retain_Q_A_Prob")
                    )
                if _results is None and key == "retain_Q_A_ROUGE":
                    _results = (
                        _logs.get("retain_Q_A_ROUGE")
                        or _traj.get("retain_Q_A_ROUGE")
                        or _logs.get("trajectory_retain_Q_A_ROUGE")
                        or _traj.get("trajectory_retain_Q_A_ROUGE")
                    )
                # Do not overwrite an existing value with None (e.g. mia_min_k -> retain then forget_truth_ratio -> retain; second key missing would otherwise clear retain)
                if _results is not None or access_name not in reference_logs[reference_log_name]:
                    reference_logs[reference_log_name][access_name] = _results
                if _results is not None:
                    found_include.append(key)
                else:
                    missing_include.append(key)
                    logger.warning(
                        f"reference_logs: key '{key}' not present in {path}, set to None; may cause errors if a metric accesses it."
                    )
            # Retain trajectory: per-step refs for step-matched privleak/FQ (top-level or under trajectory_all)
            _mia = _logs.get("mia_min_k_by_step") or _traj.get("mia_min_k_by_step")
            _tr = _logs.get("forget_truth_ratio_by_step") or _traj.get("forget_truth_ratio_by_step")
            if _mia is not None:
                reference_logs[reference_log_name]["retain_mia_by_step"] = _mia
            if _tr is not None:
                reference_logs[reference_log_name]["retain_forget_tr_by_step"] = _tr
                # If "retain" (for forget_truth_ratio) was never set, use first step as aggregate so ks_test has a retain ref
                if reference_logs[reference_log_name].get("retain") is None and _tr:
                    first_step = next(iter(_tr.values()))
                    if isinstance(first_step, dict) and first_step.get("value_by_index"):
                        reference_logs[reference_log_name]["retain"] = first_step
            # Summary logging: distinguish "no usable keys" from "partial" or "full" load
            ref = reference_logs[reference_log_name]
            has_retain = ref.get("retain") is not None
            has_mia_by_step = ref.get("retain_mia_by_step") is not None
            has_tr_by_step = ref.get("retain_forget_tr_by_step") is not None
            has_any_useful = has_retain or has_mia_by_step or has_tr_by_step or len(found_include) > 0
            if not has_any_useful:
                logger.warning(
                    "reference_logs: no usable keys found for %s (path=%s). privleak and forget_quality may fail or use fallbacks.",
                    reference_log_name,
                    path,
                )
            else:
                found_list = list(found_include)
                if has_mia_by_step:
                    found_list.append("retain_mia_by_step")
                if has_tr_by_step:
                    found_list.append("retain_forget_tr_by_step")
                if has_retain:
                    found_list.append("retain")
                logger.info(
                    "reference_logs: loaded %s from %s: found [%s]%s",
                    reference_log_name,
                    path,
                    ", ".join(sorted(set(found_list))),
                    f"; missing requested: [{', '.join(missing_include)}]" if missing_include else ".",
                )
        if reference_logs:
            kwargs.update({"reference_logs": reference_logs})

        return kwargs

    def evaluate(self, model, metric_name, cache, **kwargs):
        """Evaluates a metric including its pre_compute metrics"""
        if metric_name in cache:
            logger.info(f"Skipping {metric_name}, already evaluated.")

        metric_kwargs = self.prepare_kwargs_evaluate_metric(
            model, metric_name, cache, **kwargs
        )
        # Free GPU cache after pre_compute so the main metric (e.g. trajectory_metrics)
        # has maximal contiguous memory. Pre_compute (e.g. mia_min_k) runs over the
        # full dataset; many forwards can fragment memory and cause OOM in the
        # subsequent trajectory loop even when batch_size=1 and step count match
        # a previously successful run (see docs/oom-investigation-llada-jobs-2026-02-10.md).
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results = self.evaluate_metric(model, metric_name, **metric_kwargs)
        # Single-pass trajectory: result can be dict of sub-results keyed by display name.
        if (
            isinstance(results, dict)
            and len(results) > 0
            and all(
                isinstance(v, dict) and "agg_value" in v for v in results.values()
            )
        ):
            cache.update(results)
        else:
            cache.update({metric_name: results})
        return results

    def __call__(self, model, **kwargs):
        return self.evaluate(model, **kwargs)

    def __repr__(self) -> str:
        """Represents class object as string

        Returns:
            str: string representation of the class object
        """
        return f"{type(self).__name__} {self.name}"


# decorator that wraps simple user-defined metric python functions into callable UnlearningMetric objects
class unlearning_metric:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, metric_fn: Callable[..., Any]) -> UnlearningMetric:
        name = self.name or metric_fn.__name__
        return UnlearningMetric(name=name, metric_fn=metric_fn)
