import os
import json
import logging
import time
from typing import Callable, Any, Dict

import torch
from data import get_datasets, get_collators

logger = logging.getLogger("metrics")


def resolve_use_generalized_sequence_probability(
    eval_cfg: Any, metric_kwargs_flat: Dict[str, Any]
) -> bool:
    """Metric YAML overrides eval package; default True (matches trajectory defaults)."""
    if metric_kwargs_flat.get("use_generalized_sequence_probability") is not None:
        return bool(metric_kwargs_flat["use_generalized_sequence_probability"])
    if eval_cfg is not None and callable(getattr(eval_cfg, "get", None)):
        v = eval_cfg.get("use_generalized_sequence_probability")
        if v is not None:
            return bool(v)
    return True


class RetainReferenceValidationError(ValueError):
    """Raised when a retain reference file is provided but invalid or non-canonical."""

    pass


def _is_canonical_mia(val: Any) -> bool:
    """Accept only mia_min_k shape: {"agg_value": <int or float>}."""
    if val is None or not isinstance(val, dict):
        return False
    agg = val.get("agg_value")
    if agg is None:
        return False
    if not isinstance(agg, (int, float)) or isinstance(agg, bool):
        return False
    return True


def _is_canonical_ftr(val: Any) -> bool:
    """Accept only forget_truth_ratio canonical: value_by_index and/or agg_value (number)."""
    if val is None or not isinstance(val, dict):
        return False
    vbi = val.get("value_by_index")
    if vbi is not None:
        if not isinstance(vbi, dict):
            return False
        for k, v in vbi.items():
            if not isinstance(v, dict) or not isinstance(v.get("score"), (int, float)):
                return False
            if isinstance(v.get("score"), bool):
                return False
    agg = val.get("agg_value")
    if agg is not None:
        if not isinstance(agg, (int, float)) or isinstance(agg, bool):
            return False
    if vbi is None and agg is None:
        return False
    return True


def _validate_canonical_mia(val: Any, path: str, key: str) -> None:
    if not _is_canonical_mia(val):
        raise RetainReferenceValidationError(
            f"Retain reference at {path!r}: {key} must be canonical "
            f'{{"agg_value": number}}, got {type(val).__name__}'
        )


def _validate_canonical_ftr(val: Any, path: str, key: str) -> None:
    if not _is_canonical_ftr(val):
        raise RetainReferenceValidationError(
            f"Retain reference at {path!r}: {key} must be canonical "
            f'(value_by_index and/or agg_value number), got {type(val).__name__}'
        )


def _validate_by_step_mia(by_step: Any, path: str) -> None:
    if not isinstance(by_step, dict):
        raise RetainReferenceValidationError(
            f"Retain reference at {path!r}: mia_min_k_by_step must be a dict of step -> {{agg_value: number}}"
        )
    for step_k, step_v in by_step.items():
        if not _is_canonical_mia(step_v):
            raise RetainReferenceValidationError(
                f"Retain reference at {path!r}: mia_min_k_by_step[{step_k!r}] must be "
                f'{{"agg_value": number}}, got {type(step_v).__name__}'
            )


def _validate_by_step_ftr(by_step: Any, path: str) -> None:
    if not isinstance(by_step, dict):
        raise RetainReferenceValidationError(
            f"Retain reference at {path!r}: forget_truth_ratio_by_step must be a dict of step -> value_by_index/agg_value"
        )
    for step_k, step_v in by_step.items():
        if not _is_canonical_ftr(step_v):
            raise RetainReferenceValidationError(
                f"Retain reference at {path!r}: forget_truth_ratio_by_step[{step_k!r}] must be canonical"
            )


def load_and_validate_reference(
    reference_logs_cfgs: Dict[str, Any],
    load_fn: Callable[[str], Dict],
) -> Dict[str, Any]:
    """Load and validate reference file(s) once. Raises RetainReferenceValidationError if invalid.
    Returns the reference_logs dict (no path); for use when reference path is provided."""
    result = {}
    for reference_log_name, reference_log_cfg in reference_logs_cfgs.items():
        path = reference_log_cfg.get("path") if isinstance(reference_log_cfg, dict) else None
        if not path:
            continue
        include_cfgs = (reference_log_cfg.get("include") or {}) if isinstance(reference_log_cfg, dict) else {}
        try:
            _logs = load_fn(path)
        except Exception as e:
            raise RetainReferenceValidationError(
                f"Retain reference path {path!r} could not be loaded: {e}"
            ) from e
        _traj = _logs.get("trajectory_all") or {}
        ref = {}
        access_names_seen = {}
        for key, include_cfg in include_cfgs.items():
            access_name = include_cfg.get("access_key", key) if isinstance(include_cfg, dict) else key
            if access_name in access_names_seen:
                raise RetainReferenceValidationError(
                    f"Retain reference at {path!r}: distinct access_key required for each include key; "
                    f"{key!r} and {access_names_seen[access_name]!r} both use access_key {access_name!r}."
                )
            access_names_seen[access_name] = key
            _results = _logs.get(key)
            if _results is None:
                _results = _traj.get(key)
            if _results is None and key == "forget_truth_ratio":
                _results = (
                    _logs.get("forget_Truth_Ratio")
                    or _traj.get("forget_Truth_Ratio")
                    or _logs.get("trajectory_forget_Truth_Ratio")
                    or _traj.get("trajectory_forget_Truth_Ratio")
                )
            if _results is None:
                by_step = _logs.get("mia_min_k_by_step") or _traj.get("mia_min_k_by_step") if key == "mia_min_k" else _logs.get("forget_truth_ratio_by_step") or _traj.get("forget_truth_ratio_by_step")
                if by_step and isinstance(by_step, dict):
                    first_step = next(iter(by_step.values()), None)
                    if key == "mia_min_k" and first_step is not None and _is_canonical_mia(first_step):
                        _results = first_step
                    elif key == "forget_truth_ratio" and first_step is not None and _is_canonical_ftr(first_step):
                        _results = first_step
                if _results is None:
                    raise RetainReferenceValidationError(
                        f"Retain reference at {path!r}: required key {key!r} not found in file "
                        "(and no canonical first step in by_step)."
                    )
            if key == "mia_min_k":
                _validate_canonical_mia(_results, path, key)
                ref[access_name] = _results
            elif key == "forget_truth_ratio":
                _validate_canonical_ftr(_results, path, key)
                ref[access_name] = _results
            else:
                ref[access_name] = _results
        _mia = _logs.get("mia_min_k_by_step") or _traj.get("mia_min_k_by_step")
        _tr = _logs.get("forget_truth_ratio_by_step") or _traj.get("forget_truth_ratio_by_step")
        if _mia is not None:
            _validate_by_step_mia(_mia, path)
            ref["retain_mia_by_step"] = _mia
        if _tr is not None:
            _validate_by_step_ftr(_tr, path)
            ref["retain_forget_tr_by_step"] = _tr
        if ref:
            result[reference_log_name] = ref
            logger.info(
                "reference_logs: loaded and validated %s from %s.",
                reference_log_name,
                path,
            )
    return result


def _reference_logs_payload_is_evaluator_preloaded(container: Any) -> bool:
    """True when ``reference_logs`` is already ``load_and_validate_reference`` output.

    Evaluator merges cached retain JSON into ``metrics_args``; that shape has slot keys
    (e.g. retain_ftr) under retain_model_logs and no ``path``. YAML shells without a
    usable path must not match here (they have ``include`` / null path only).
    """
    if not isinstance(container, dict) or not container:
        return False
    rml = container.get("retain_model_logs")
    if not isinstance(rml, dict) or not rml:
        return False
    if rml.get("path"):
        return False
    slot_keys = (
        "retain_ftr",
        "retain",
        "retain_mia_by_step",
        "retain_forget_tr_by_step",
    )
    return any(k in rml for k in slot_keys)


def _extract_retain_agg_scalar(val: Any) -> Any:
    """Extract a single number from a retain metric value for privleak/rel_diff.

    Handles: (1) scalar agg_value, (2) auc when agg_value missing (e.g. mia_min_k),
    (3) nested agg_value (view -> traj -> metric) by taking first numeric found.
    Returns None if no number can be extracted.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    if not isinstance(val, dict):
        return None
    agg = val.get("agg_value")
    if isinstance(agg, (int, float)) and not isinstance(agg, bool):
        return float(agg)
    if isinstance(agg, dict):
        for v in agg.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
            if isinstance(v, dict):
                for w in v.values():
                    if isinstance(w, (int, float)) and not isinstance(w, bool):
                        return float(w)
                    if isinstance(w, dict) and "privleak" in w:
                        p = w["privleak"]
                        if isinstance(p, (int, float)) and not isinstance(p, bool):
                            return float(p)
    if "auc" in val and isinstance(val.get("auc"), (int, float)):
        return float(val["auc"])
    return None


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
          Strict: if any requested key is not found, log ERROR and do not use that ref (no fallback).
          Logs ERROR when no usable data or when step-matched data is missing (trajectory); INFO on success.
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

        resolved_gen = resolve_use_generalized_sequence_probability(
            kwargs.get("eval_cfg"), kwargs
        )
        kwargs["use_generalized_sequence_probability"] = resolved_gen
        _evc = kwargs.get("eval_cfg")
        if _evc is not None and callable(getattr(_evc, "get", None)):
            _la = _evc.get("logit_alignment")
            if _la is not None and kwargs.get("logit_alignment") is None:
                kwargs["logit_alignment"] = _la

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
                from omegaconf import OmegaConf

                cfg_d = (
                    OmegaConf.to_container(pre_metric_cfg, resolve=True)
                    if OmegaConf.is_config(pre_metric_cfg)
                    else dict(pre_metric_cfg)
                )
                if "use_generalized_sequence_probability" not in cfg_d:
                    cfg_d["use_generalized_sequence_probability"] = resolved_gen
                if (
                    "logit_alignment" not in cfg_d
                    and kwargs.get("logit_alignment") is not None
                ):
                    cfg_d["logit_alignment"] = kwargs["logit_alignment"]
                pre_metric_kwargs.update(cfg_d)
                _t0 = time.perf_counter()
                _results = pre_metric.evaluate(
                    model, pre_metric_name, cache=cache, **pre_metric_kwargs
                )
                _elapsed = time.perf_counter() - _t0
                logger.info(
                    "pre_compute %s for %s took %.1fs (one-time before trajectory; result unused in trajectory path)",
                    pre_metric_name, metric_name, _elapsed,
                )
            pre_metric_results.update({access_name: _results})
        if pre_metric_results:
            kwargs.update({"pre_compute": pre_metric_results})

        # Load reference logs from JSON when config has path. Validate completely; raise on invalid.
        # If reference_logs is already loaded (no path), use as is.
        reference_logs_cfgs = kwargs.pop("reference_logs", {})
        reference_logs = {}
        has_path = False
        if reference_logs_cfgs:
            for _rn, _rc in reference_logs_cfgs.items():
                if isinstance(_rc, dict) and _rc.get("path"):
                    has_path = True
                    break
        if has_path:
            reference_logs = load_and_validate_reference(
                reference_logs_cfgs,
                self.load_logs_from_file,
            )
        elif _reference_logs_payload_is_evaluator_preloaded(reference_logs_cfgs):
            reference_logs = reference_logs_cfgs
        else:
            # Do not pass YAML shells (path null / missing) into metric fns; same as
            # trajectory + Evaluator: ks_test/privleak treat any truthy reference_logs
            # as "loaded reference" and validate retain slots.
            reference_logs = {}
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
