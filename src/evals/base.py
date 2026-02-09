import os
import json
import logging
from evals.metrics import get_metrics
from evals.metrics.trajectory_metrics import run_coalesced_trajectory_metrics

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

    def save_logs(self, logs, file):
        """Save the logs in a json file"""
        logs = dict(sorted(logs.items()))
        os.makedirs(os.path.dirname(file), exist_ok=True)
        try:
            with open(file, "w") as f:
                json.dump(logs, f, indent=4)
        except Exception as e:
            raise RuntimeError(f"Failed to save {file}: {e}")

    def prepare_model(self, model):
        """Prepare model for evaluation"""
        model.eval()
        return model

    def load_metrics(self, metrics_cfg):
        """Load metrics for evaluation"""
        metrics = get_metrics(metrics_cfg)
        return metrics

    def summarize(self, logs):
        """Summarize the metrics results"""
        metric_summary = {}
        for metric_name, metric_results in logs.items():
            if metric_name not in self.metrics:
                continue
            agg_value = metric_results.get("agg_value", None)
            if agg_value is not None:
                metric_summary[metric_name] = agg_value
        return metric_summary

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        # set flag to overwrite metrics
        overwrite = self.eval_cfg.overwrite if overwrite is None else overwrite
        # When False: run each trajectory_metrics entry separately (baseline). When True: coalesce into one pass (if implemented).
        coalesce = getattr(self.eval_cfg, "coalesce_trajectory_metrics", True)

        # Prepare model for evaluation
        model = self.prepare_model(model)

        # Set output_dir and file to store results
        output_dir = output_dir if output_dir else self.eval_cfg.output_dir
        logs_file_path = self.get_logs_file_path(output_dir)
        summary_file_path = self.get_logs_file_path(output_dir, suffix="SUMMARY")

        # Load existing results from file if any.
        logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}

        logger.info(f"***** Running {self.name} evaluation suite *****")
        logger.info(f"Fine-grained evaluations will be saved to: {logs_file_path}")
        logger.info(
            f"Aggregated evaluations will be summarised in: {summary_file_path}"
        )

        # When coalesce=True and all metrics use handler "trajectory_metrics", one sampler pass per item for all.
        metrics_cfg = self.eval_cfg.metrics
        handlers = [metrics_cfg.get(m) and metrics_cfg[m].get("handler") for m in self.metrics]
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
                    if logs[metric_name].get("agg_value") is not None:
                        logger.info(f"Result for metric {metric_name}:\t{logs[metric_name]['agg_value']}")
                self.save_logs(self.summarize(logs), summary_file_path)
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
            data = first_metric.get_datasets(dataset_cfgs=first_cfg.get("datasets"), **base_kwargs)
            collators = first_metric.get_collators(collator_cfgs=first_cfg.get("collators"), **base_kwargs)
            trajectory_config = first_cfg.get("trajectory_config") or {}
            batch_size = first_cfg.get("batch_size", 1)
            rouge_type = "rougeL_recall"
            for m in self.metrics:
                cfg = metrics_cfg.get(m)
                if cfg is not None and getattr(cfg, "get", None) and cfg.get("rouge_type"):
                    rouge_type = cfg.get("rouge_type")
                    break
            metrics_to_run = [(m, metrics_cfg[m]) for m in self.metrics]
            coalesced_results = run_coalesced_trajectory_metrics(
                model,
                metrics_to_run,
                self.eval_cfg,
                logs,
                data=data,
                collators=collators,
                tokenizer=base_kwargs["tokenizer"],
                batch_size=batch_size,
                trajectory_config=trajectory_config,
                rouge_type=rouge_type,
                **base_kwargs,
            )
            for metric_name, result in coalesced_results.items():
                logs[metric_name] = result
                if result.get("agg_value") is not None:
                    logger.info(f"Result for metric {metric_name}:\t{result['agg_value']}")
            self.save_logs(logs, logs_file_path)
            self.save_logs(self.summarize(logs), summary_file_path)
            return self.summarize(logs)

        for metric_name, metric_fn in self.metrics.items():
            if not overwrite and metric_name in logs and logs[metric_name]:
                logger.info(f"Skipping {metric_name}, already evaluated.")
                if "agg_value" in logs[metric_name]:
                    logger.info(
                        f"Result for metric {metric_name}:\t{logs[metric_name]['agg_value']}"
                    )
                self.save_logs(self.summarize(logs), summary_file_path)
                continue
            _ = logs.pop(metric_name, None)  # overwriting existing evals if present
            kwargs = {
                "tokenizer": kwargs.get("tokenizer", None),
                "template_args": kwargs.get("template_args", None),
            }
            metrics_args = self.eval_cfg.metrics[metric_name]
            _
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
            self.save_logs(self.summarize(logs), summary_file_path)

        return self.summarize(logs)
