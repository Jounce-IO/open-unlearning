import gc
import os
import json
import logging
import numpy as np
import torch
from evals.metrics import get_metrics
from evals.metrics.privacy import log_retain_logs_path_none_if_needed

logger = logging.getLogger("evaluator")


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

    def save_logs(self, logs, file):
        """Save the logs in a json file"""
        logs = dict(sorted(logs.items()))
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

        # Prepare model for evaluation
        model = self.prepare_model(model)

        # Set output_dir and file to store results
        output_dir = output_dir if output_dir else self.eval_cfg.output_dir
        logs_file_path = self.get_logs_file_path(output_dir)

        # Load existing results from file if any.
        logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}
        
        # Save config information (only once, at the start)
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
            self.save_logs(logs, logs_file_path)

        logger.info(f"***** Running {self.name} evaluation suite *****")
        log_retain_logs_path_none_if_needed(
            "start of evaluation",
            self.metrics,
            self.eval_cfg.get("retain_logs_path"),
        )
        logger.info(f"Evaluations will be saved to: {logs_file_path}")
        logger.info(f"Evaluating {len(self.metrics)} metrics: {list(self.metrics.keys())}")

        # Coalesced trajectory: one call to trajectory_metrics with all metrics (one sampler pass).
        coalesce = self.eval_cfg.get("coalesce_trajectory_metrics", False)
        metrics_cfg = self.eval_cfg.metrics
        handlers = []
        for m in self.metrics:
            cfg = metrics_cfg.get(m)
            handlers.append(cfg.get("handler") if cfg is not None and hasattr(cfg, "get") else None)
        all_trajectory = (
            coalesce
            and len(self.metrics) >= 2
            and all(h == "trajectory_metrics" for h in handlers if h is not None)
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
                return self.summarize(logs)
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
                **base_kwargs,
            }
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
            self.save_logs(logs, logs_file_path)
            return self.summarize(logs)

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
            }
            if self.eval_cfg.get("samples") is not None:
                kwargs["samples"] = self.eval_cfg.samples
            metrics_args = self.eval_cfg.metrics[metric_name]
            result = metric_fn(
                model,
                metric_name=metric_name,
                cache=logs,
                **kwargs,
                **metrics_args,
            )
            if "agg_value" in result:
                logger.info(f"Result for metric {metric_name}:\t{result['agg_value']}")
            self.save_logs(logs, logs_file_path)
            # Free GPU memory between metrics to avoid OOM with large models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        return self.summarize(logs)
