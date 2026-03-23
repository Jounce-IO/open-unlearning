import gc
import os
import json
import logging
import numpy as np
import torch
from evals.metrics import get_metrics
from evals.metrics.base import (
    RetainReferenceValidationError,
    load_and_validate_reference,
    log_generalized_sequence_probability_for_eval_metric,
)
from evals.metrics.privacy import log_retain_logs_path_none_if_needed

logger = logging.getLogger("evaluator")


def reference_logs_has_usable_retain_path(reference_logs_container) -> bool:
    """Return True if ``retain_model_logs.path`` is set to a real filesystem path.

    When Hydra resolves ``path: ${eval.tofu.retain_logs_path}`` and retain_logs_path is
    null, the metric config still contains a ``reference_logs`` *shell* (path + include).
    Passing that dict to ``ks_test`` / ``privleak`` makes them treat reference as
    "provided" and validate ``retain_ftr`` / ``retain`` slots — and fail.

    Trajectory coalesced evaluation already omits ``reference_logs`` unless the path is
    usable; non-trajectory metrics use the same rule here and in
    ``UnlearningMetric.prepare_kwargs_evaluate_metric``.
    """
    if reference_logs_container is None or not isinstance(reference_logs_container, dict):
        return False
    rml = reference_logs_container.get("retain_model_logs")
    if not isinstance(rml, dict):
        return False
    path_val = rml.get("path")
    if path_val is None:
        return False
    s = str(path_val).strip()
    if not s or s.lower() in ("null", "none", ""):
        return False
    return True


class Evaluator:
    def __init__(self, name, eval_cfg, **kwargs):
        self.name = name
        self.eval_cfg = eval_cfg
        self.metrics_cfg = self.eval_cfg.metrics
        self.metrics = self.load_metrics(self.metrics_cfg)
        logger.info(
            f"Evaluations stored in the experiment directory: {self.eval_cfg.output_dir}"
        )
        logger.info(f"Loaded {len(self.metrics)} metrics: {list(self.metrics.keys())}")

    def get_logs_file_path(self, output_dir, suffix="EVAL"):
        """Returns the path to json file to store results"""
        logs_filename = os.path.join(output_dir, f"{self.name}_{suffix}.json")
        return logs_filename

    def load_logs_from_file(self, file):
        """Returns the cache of existing results"""
        logs = {}
        if os.path.exists(file):
            logger.info(f"Loading existing evaluations from {file}")
            with open(file, "r") as f:
                logs = json.load(f)
        return logs

    def _convert_numpy_to_list(self, obj):
        """Recursively convert numpy arrays and scalars to Python native types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_to_list(item) for item in obj]
        else:
            return obj

    def _remove_value_by_index(self, obj):
        """Recursively remove value_by_index from dicts before saving to JSON"""
        if isinstance(obj, dict):
            # Create a new dict without value_by_index
            cleaned = {}
            for key, value in obj.items():
                if key == "value_by_index":
                    # Skip value_by_index - it's only for internal calculations
                    continue
                # Recursively clean nested dicts
                cleaned[key] = self._remove_value_by_index(value)
            return cleaned
        elif isinstance(obj, (list, tuple)):
            return [self._remove_value_by_index(item) for item in obj]
        else:
            return obj

    def _total_samples_from_logs(self, logs):
        """Find total sample count from first non-empty value_by_index in logs."""
        for key, value in logs.items():
            if key == "config" or not isinstance(value, dict):
                continue
            vbi = value.get("value_by_index")
            if isinstance(vbi, dict) and len(vbi) > 0:
                return len(vbi)
        return None

    def _ensure_retain_reference_keys(self, logs):
        """When in retain_reference_mode, ensure forget_truth_ratio key exists for FQ include."""
        if not self.eval_cfg.get("retain_reference_mode", False):
            return
        if "forget_truth_ratio" not in logs and "forget_Truth_Ratio" in logs:
            logs["forget_truth_ratio"] = logs["forget_Truth_Ratio"]

    def save_logs(self, logs, file, keep_value_by_index=False):
        """Save the logs in a json file.
        When keep_value_by_index is True (e.g. per-rank files for later merge), value_by_index is kept."""
        logs = dict(sorted(logs.items()))
        # Ensure run_info is present so report PRs can show data-parallel info (eval.py sets it for distributed).
        if "run_info" not in logs:
            total = self._total_samples_from_logs(logs)
            if total is not None:
                logs["run_info"] = {
                    "world_size": 1,
                    "total_samples": total,
                    "data_parallel": False,
                }
        if not keep_value_by_index:
            # Remove value_by_index (used for calculations but not needed in final JSON)
            logs = self._remove_value_by_index(logs)
        # Convert numpy arrays to lists for JSON serialization
        logs = self._convert_numpy_to_list(logs)
        os.makedirs(os.path.dirname(file), exist_ok=True)
        try:
            with open(file, "w") as f:
                json.dump(logs, f, indent=4)
        except Exception as e:
            raise RuntimeError(f"Failed to save {file}: {e}")

    def prepare_model(self, model):
        """Prepare model for evaluation"""
        model.eval()
        # If model is already wrapped (e.g., DiffusionModelAdapter), it handles eval()
        if hasattr(model, 'model'):
            model.model.eval()
        return model

    def load_metrics(self, metrics_cfg):
        """Load metrics for evaluation"""
        metrics = get_metrics(metrics_cfg)
        return metrics

    def summarize(self, logs):
        """Summarize the metrics results"""
        metric_summary = {}
        for metric_name, metric_results in logs.items():
            if metric_name == "config":
                continue
            agg_value = metric_results.get("agg_value", None)
            if agg_value is not None:
                metric_summary[metric_name] = agg_value
        return metric_summary

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        # set flag to overwrite metrics
        overwrite = self.eval_cfg.overwrite if overwrite is None else overwrite
        rank = kwargs.get("rank", 0)
        world_size = kwargs.get("world_size", 1)
        is_distributed = world_size > 1

        # Prepare model for evaluation
        model = self.prepare_model(model)

        # Set output_dir and file to store results
        output_dir = output_dir if output_dir else self.eval_cfg.output_dir
        logs_file_path = self.get_logs_file_path(output_dir)

        # Load existing results from file if any (only rank 0 when distributed, to avoid conflict)
        if not is_distributed or rank == 0:
            logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}
        else:
            logs = {}

        # Save config information (only once, at the start); only rank 0 saves when distributed
        if "config" not in logs or overwrite:
            from omegaconf import OmegaConf
            config_dict = {
                "evaluator_name": self.name,
                "eval_config": OmegaConf.to_container(self.eval_cfg, resolve=True),
            }
            # Add model info if available
            if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
                config_dict["model_name"] = model.config._name_or_path
            elif hasattr(model, "model") and hasattr(model.model, "config") and hasattr(model.model.config, "_name_or_path"):
                config_dict["model_name"] = model.model.config._name_or_path
            logs["config"] = config_dict
            if not is_distributed:
                self._ensure_retain_reference_keys(logs)
                self.save_logs(
                    logs,
                    logs_file_path,
                    keep_value_by_index=self.eval_cfg.get("retain_reference_mode", False),
                )

        logger.info(f"***** Running {self.name} evaluation suite *****")
        log_retain_logs_path_none_if_needed(
            "start of evaluation",
            self.metrics,
            self.eval_cfg.get("retain_logs_path"),
        )
        logger.info(f"Evaluations will be saved to: {logs_file_path}")
        logger.info(f"Evaluating {len(self.metrics)} metrics: {list(self.metrics.keys())}")

        _metrics_cfg_lo = self.eval_cfg.metrics
        _handlers_lo = []
        for _mn in self.metrics:
            _mc = _metrics_cfg_lo.get(_mn) if hasattr(_metrics_cfg_lo, "get") else None
            _handlers_lo.append(
                _mc.get("handler") if _mc is not None and hasattr(_mc, "get") else None
            )
        _all_traj_lo = len(self.metrics) >= 1 and all(
            h == "trajectory_metrics" for h in _handlers_lo if h is not None
        )
        _path_lo = "trajectory" if _all_traj_lo else "non_trajectory"
        _ugg = self.eval_cfg.get("use_generalized_sequence_probability", True)
        logger.info(
            "eval_start OpenUnlearning_stack=%s evaluator=%s use_generalized_sequence_probability=%s",
            "TRAJECTORY (trajectory_metrics)" if _path_lo == "trajectory" else "NON_TRAJECTORY (standard metrics)",
            self.name,
            _ugg,
        )

        # Load and validate reference once at start when any metric has reference_logs path (spec: validate before any evaluation).
        # Merge retain_model_logs.include across metrics that share the same path: forget_quality loads
        # forget_truth_ratio → retain_ftr; privleak loads mia_min_k → retain. Using only the first metric's
        # include left privleak without the retain slot (RetainReferenceValidationError).
        cached_reference_logs = None
        try:
            from omegaconf import OmegaConf as _OmegaConf
        except ImportError:
            _OmegaConf = None
        merged_ref_blocks: dict[str, dict] = {}

        for _mname in self.metrics:
            _mcfg = self.eval_cfg.metrics.get(_mname) if hasattr(self.eval_cfg.metrics, "get") else None
            if _mcfg is None:
                continue
            try:
                _ref_cfg = _mcfg.get("reference_logs") if hasattr(_mcfg, "get") else None
            except Exception:
                _ref_cfg = None
            if not _ref_cfg:
                continue
            _ref_dict = None
            if isinstance(_ref_cfg, dict):
                _ref_dict = _ref_cfg
            elif _OmegaConf is not None and hasattr(_ref_cfg, "items"):
                try:
                    _ref_dict = _OmegaConf.to_container(_ref_cfg, resolve=True)
                except Exception:
                    _ref_dict = dict(_ref_cfg)
            if not _ref_dict or not reference_logs_has_usable_retain_path(_ref_dict):
                continue
            for _rn, _rc in _ref_dict.items():
                if not isinstance(_rc, dict) or not _rc.get("path"):
                    continue
                path = _rc.get("path")
                inc = dict(_rc.get("include") or {})
                existing = merged_ref_blocks.get(_rn)
                if existing is None:
                    merged_ref_blocks[_rn] = {"path": path, "include": inc}
                else:
                    if existing["path"] != path:
                        raise RetainReferenceValidationError(
                            f"Metrics disagree on retain reference path for {_rn!r}: "
                            f"{existing['path']!r} vs {path!r}"
                        )
                    m_inc = existing["include"]
                    for ik, iv in inc.items():
                        if ik in m_inc and m_inc[ik] != iv:
                            raise RetainReferenceValidationError(
                                f"Metrics disagree on reference_logs {_rn}.include[{ik!r}]"
                            )
                        m_inc[ik] = iv

        if merged_ref_blocks:
            cached_reference_logs = load_and_validate_reference(
                merged_ref_blocks, self.load_logs_from_file
            )

        # Coalesced trajectory: one call to trajectory_metrics with all metrics (one sampler pass).
        coalesce = self.eval_cfg.get("coalesce_trajectory_metrics", False)
        metrics_cfg = self.eval_cfg.metrics
        handlers = []
        for m in self.metrics:
            cfg = metrics_cfg.get(m)
            handlers.append(cfg.get("handler") if cfg is not None and hasattr(cfg, "get") else None)
        # Use coalesced path for 2+ metrics when coalesce=True, or for 1 trajectory_metrics metric
        # (so trajectory_all uses same path and avoids per-metric to_container → dict.args bug).
        all_trajectory = (
            len(self.metrics) >= 1
            and all(h == "trajectory_metrics" for h in handlers if h is not None)
            and (coalesce or len(self.metrics) == 1)
        )
        if all_trajectory:
            already = [m for m in self.metrics if not overwrite and m in logs and logs[m]]
            if len(already) == len(self.metrics):
                for metric_name in self.metrics:
                    logger.info(f"Skipping {metric_name}, already evaluated.")
                    if logs.get(metric_name, {}).get("agg_value") is not None:
                        logger.info(
                            f"Result for metric {metric_name}:\t{logs[metric_name]['agg_value']}"
                        )
                return logs
            for m in self.metrics:
                _ = logs.pop(m, None)
            first_name = next(iter(self.metrics))
            first_metric = self.metrics[first_name]
            first_cfg = metrics_cfg[first_name]
            base_kwargs = {
                "tokenizer": kwargs.get("tokenizer"),
                "template_args": kwargs.get("template_args"),
            }
            if self.eval_cfg.get("samples") is not None:
                base_kwargs["samples"] = self.eval_cfg.samples
            data = first_metric.get_datasets(
                dataset_cfgs=first_cfg.get("datasets"), **base_kwargs
            )
            collators = first_metric.get_collators(
                collator_cfgs=first_cfg.get("collators"), **base_kwargs
            )
            trajectory_config = first_cfg.get("trajectory_config") or {}
            batch_size = first_cfg.get("batch_size", 1)
            # Merge metrics config: trajectory_metrics expects metrics (list or dict) and metric_display_names.
            # Use OmegaConf.to_container so we get plain Python types (OmegaConf ListConfig/DictConfig can behave differently in cluster).
            try:
                from omegaconf import OmegaConf
                _metrics_cfg = OmegaConf.to_container(metrics_cfg, resolve=True) or {}
            except Exception:
                _metrics_cfg = dict(metrics_cfg) if hasattr(metrics_cfg, "items") else {}
            merged_metrics = {}
            for m in self.metrics:
                cfg = _metrics_cfg.get(m) if isinstance(_metrics_cfg, dict) else None
                mc = cfg.get("metrics") if cfg is not None and isinstance(cfg, dict) else None
                if mc is None:
                    mc = []
                if isinstance(mc, (list, tuple)):
                    for name in mc:
                        merged_metrics[name] = {}
                elif isinstance(mc, dict):
                    merged_metrics.update(mc)
            for m in self.metrics:
                cfg = _metrics_cfg.get(m) if isinstance(_metrics_cfg, dict) else None
                if isinstance(cfg, dict) and cfg.get("rouge_type") is not None:
                    if "rouge" in merged_metrics:
                        merged_metrics["rouge"] = merged_metrics.get("rouge") or {}
                        merged_metrics["rouge"]["rouge_type"] = cfg.get("rouge_type", "rougeL_recall")
                    break
            if "rouge" in merged_metrics and not merged_metrics["rouge"]:
                merged_metrics["rouge"] = {"rouge_type": "rougeL_recall"}
            if not merged_metrics and len(self.metrics) >= 2:
                # Fallback when config structure differs (e.g. in cluster): use Prob + ROUGE.
                merged_metrics = {"probability": {}, "rouge": {"rouge_type": "rougeL_recall"}}
            metric_display_names = list(self.metrics.keys())
            merged_args = {
                "data": data,
                "collators": collators,
                "trajectory_config": trajectory_config,
                "batch_size": batch_size,
                "metrics": merged_metrics,
                "metric_display_names": metric_display_names,
                "eval_cfg": self.eval_cfg,
                **base_kwargs,
            }
            # Pass reference_logs: use pre-loaded validated ref when we loaded at start, else config (path).
            if cached_reference_logs is not None:
                merged_args["reference_logs"] = cached_reference_logs
            else:
                first_reference_logs = first_cfg.get("reference_logs")
                if first_reference_logs is not None:
                    try:
                        from omegaconf import OmegaConf
                        ref_container = OmegaConf.to_container(
                            first_reference_logs, resolve=True
                        )
                        # Only pass reference_logs when path is set (non-null). When path is null
                        # (e.g. retain_reference_mode run producing the reference), do not pass
                        # so the metric does not require step-matched reference data.
                        if reference_logs_has_usable_retain_path(ref_container):
                            merged_args["reference_logs"] = ref_container
                    except Exception:
                        ref_container = (
                            dict(first_reference_logs)
                            if hasattr(first_reference_logs, "items")
                            else first_reference_logs
                        )
                        # dict(OmegaConf) may not match nested shape expected by the helper; keep
                        # legacy path-string check so coalesced eval still passes reference_logs.
                        rml_fb = (ref_container or {}).get("retain_model_logs") or {}
                        path_get = getattr(rml_fb, "get", None)
                        path_fb = path_get("path") if path_get else None
                        if path_fb and str(path_fb).strip().lower() not in (
                            "null",
                            "none",
                            "",
                        ):
                            merged_args["reference_logs"] = ref_container
                        elif reference_logs_has_usable_retain_path(ref_container):
                            merged_args["reference_logs"] = ref_container
            result = first_metric(
                model,
                metric_name=first_name,
                cache=logs,
                **merged_args,
            )
            if isinstance(result, dict) and all(
                isinstance(v, dict) and "agg_value" in v for v in result.values()
            ):
                for k, v in result.items():
                    logs[k] = v
            else:
                logs[first_name] = result
            for metric_name in self.metrics:
                if logs.get(metric_name, {}).get("agg_value") is not None:
                    logger.info(f"Result for metric {metric_name}:\t{logs[metric_name]['agg_value']}")
            if not is_distributed:
                self._ensure_retain_reference_keys(logs)
                self.save_logs(
                    logs,
                    logs_file_path,
                    keep_value_by_index=self.eval_cfg.get("retain_reference_mode", False),
                )
            return logs

        for metric_name, metric_fn in self.metrics.items():
            metric_cfg = self.eval_cfg.metrics[metric_name]
            metric_display_names = metric_cfg.get("metric_display_names", None)
            if metric_display_names is not None:
                if isinstance(metric_display_names, (list, tuple)):
                    display_names = list(metric_display_names)
                else:
                    display_names = list(metric_display_names)
                skip = (
                    not overwrite
                    and display_names
                    and all(k in logs and logs.get(k) for k in display_names)
                )
                if skip:
                    logger.info(
                        f"Skipping {metric_name}, already evaluated (all sub-results present)."
                    )
                    continue
                if overwrite:
                    for k in display_names:
                        _ = logs.pop(k, None)
            else:
                if not overwrite and metric_name in logs and logs[metric_name]:
                    logger.info(f"Skipping {metric_name}, already evaluated.")
                    if "agg_value" in logs[metric_name]:
                        logger.info(
                            f"Result for metric {metric_name}:\t{logs[metric_name]['agg_value']}"
                        )
                    continue
                _ = logs.pop(metric_name, None)  # overwriting existing evals if present
            log_retain_logs_path_none_if_needed(
                f"start of metric {metric_name}",
                {metric_name: self.eval_cfg.metrics[metric_name]},
                self.eval_cfg.get("retain_logs_path"),
            )
            kwargs = {
                "tokenizer": kwargs.get("tokenizer", None),
                "template_args": kwargs.get("template_args", None),
                "eval_cfg": self.eval_cfg,
            }
            if self.eval_cfg.get("samples") is not None:
                kwargs["samples"] = self.eval_cfg.samples
            _mc = self.eval_cfg.metrics[metric_name]
            try:
                from omegaconf import OmegaConf
                metrics_args = OmegaConf.to_container(_mc, resolve=True) if _mc is not None else {}
            except Exception:
                metrics_args = dict(_mc) if _mc is not None and hasattr(_mc, "items") else {}
            if metrics_args is None:
                metrics_args = {}
            if cached_reference_logs is not None and metrics_args.get("reference_logs"):
                metrics_args["reference_logs"] = cached_reference_logs
            elif metrics_args.get("reference_logs") is not None:
                if not reference_logs_has_usable_retain_path(
                    metrics_args.get("reference_logs")
                ):
                    metrics_args.pop("reference_logs", None)
            _mh = metrics_args.get("handler")
            log_generalized_sequence_probability_for_eval_metric(
                metric_name,
                _mh,
                self.eval_cfg,
                metrics_args,
            )
            result = metric_fn(
                model,
                metric_name=metric_name,
                cache=logs,
                **kwargs,
                **metrics_args,
            )
            if "agg_value" in result:
                logger.info(f"Result for metric {metric_name}:\t{result['agg_value']}")
            if not is_distributed:
                self._ensure_retain_reference_keys(logs)
                self.save_logs(
                    logs,
                    logs_file_path,
                    keep_value_by_index=self.eval_cfg.get("retain_reference_mode", False),
                )
            # Free GPU memory between metrics to avoid OOM with large models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        return logs
