import os
import json
import logging
import numpy as np
from evals.metrics import get_metrics
from evals.metrics.privacy import log_retain_logs_path_none_if_needed
from data import get_datasets, get_collators

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
        logs = {str(k): v for k, v in logs.items()}
        logs = dict(sorted(logs.items(), key=lambda x: x[0]))
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

        # Coalesce multiple trajectory_metrics into one call (one model pass, all sub-metrics)
        from omegaconf import OmegaConf
        trajectory_names = [
            n for n, m in self.metrics.items()
            if getattr(m, "name", None) == "trajectory_metrics"
        ]
        coalesced_done = set()
        if len(trajectory_names) >= 2:
            first_cfg = self.eval_cfg.metrics[trajectory_names[0]]
            merged_metrics = {}
            merged_display_names = []
            for name in trajectory_names:
                cfg = self.eval_cfg.metrics[name]
                m = cfg.get("metrics")
                d = cfg.get("metric_display_names")
                if d is None:
                    d = [name]
                elif not isinstance(d, (list, tuple)):
                    d = [d]
                d = list(d)
                if m is None:
                    continue
                try:
                    from omegaconf import ListConfig, DictConfig
                except ImportError:
                    ListConfig = ()
                    DictConfig = ()
                if isinstance(m, (list, tuple, ListConfig)):
                    m_list = list(m)
                    m_cfg = cfg.get("metrics")
                    m_cfg = OmegaConf.to_container(m_cfg, resolve=True) if m_cfg is not None else {}
                    m_cfg = dict(m_cfg) if isinstance(m_cfg, dict) else {}
                    for i, subm in enumerate(m_list):
                        merged_metrics[subm] = m_cfg.get(subm, {})
                        merged_display_names.append(d[i] if i < len(d) else name)
                else:
                    m_container = OmegaConf.to_container(m, resolve=True) if hasattr(m, "items") else m
                    m_dict = dict(m_container) if isinstance(m_container, dict) else {}
                    for i, (subm, subcfg) in enumerate(m_dict.items()):
                        merged_metrics[subm] = dict(subcfg) if isinstance(subcfg, dict) else subcfg
                        merged_display_names.append(d[i] if i < len(d) else name)

            skip_coalesced = (
                not overwrite
                and merged_display_names
                and all(disp in logs and logs.get(disp) for disp in merged_display_names)
            )
            if not skip_coalesced:
                base_kwargs = {
                    "tokenizer": kwargs.get("tokenizer", None),
                    "template_args": kwargs.get("template_args", None),
                }
                if self.eval_cfg.get("samples") is not None:
                    base_kwargs["samples"] = self.eval_cfg.samples

                shared_data = None
                shared_collators = None
                if first_cfg.get("datasets") is not None and first_cfg.get("collators") is not None:
                    try:
                        shared_data = get_datasets(
                            first_cfg.get("datasets"), **base_kwargs
                        )
                        shared_collators = get_collators(
                            collator_cfgs=first_cfg.get("collators"),
                            tokenizer=base_kwargs.get("tokenizer"),
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not pre-load shared data for coalesced trajectory metrics: {e}. "
                            "Each metric will load its own data."
                        )

                merged_kwargs = {
                    k: v for k, v in first_cfg.items()
                    if k not in ("datasets", "collators")
                }
                merged_kwargs["metrics"] = merged_metrics
                merged_kwargs["metric_display_names"] = merged_display_names
                if shared_data is not None:
                    merged_kwargs["data"] = shared_data
                if shared_collators is not None:
                    merged_kwargs["collators"] = shared_collators

                if overwrite:
                    for disp in merged_display_names:
                        _ = logs.pop(disp, None)

                log_retain_logs_path_none_if_needed(
                    "start of coalesced trajectory metrics",
                    {trajectory_names[0]: first_cfg},
                    self.eval_cfg.get("retain_logs_path"),
                )
                first_metric_fn = self.metrics[trajectory_names[0]]
                result = first_metric_fn(
                    model,
                    metric_name=trajectory_names[0],
                    cache=logs,
                    **base_kwargs,
                    **merged_kwargs,
                )
                if isinstance(result, dict) and result and not any(
                    k in result for k in ("agg_value", "value_by_index")
                ):
                    for disp in result:
                        if isinstance(result.get(disp), dict) and "agg_value" in result[disp]:
                            logger.info(
                                f"Result for {disp}:\t{result[disp]['agg_value']}"
                            )
                self.save_logs(logs, logs_file_path)
            coalesced_done = set(trajectory_names)

        for metric_name, metric_fn in self.metrics.items():
            if metric_name in coalesced_done:
                continue
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

        return self.summarize(logs)
