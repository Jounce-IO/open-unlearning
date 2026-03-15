import numpy as np
from scipy.stats import ks_2samp
from evals.metrics.base import unlearning_metric, logger, RetainReferenceValidationError

# Metrics that require retain_model_logs (retain_logs_path)
RETAIN_LOGS_METRICS = frozenset(
    {"privleak", "forget_quality", "trajectory_privleak", "trajectory_forget_quality"}
)

# Message for retain_logs_path=None when privleak/forget_quality need it
RETAIN_LOGS_PATH_NONE_MSG = (
    "retain_logs_path is None. privleak and forget_quality compare the unlearned model "
    "to a retain (baseline) model. Without retain evals, privleak falls back to ref_value (0.5) "
    "and forget_quality returns None. To fix: (1) Evaluate your retain model (finetuned on retain split) "
    "and save results to JSON; (2) Pass retain_logs_path=/path/to/TOFU_EVAL.json via Hydra override "
    "(e.g. eval.tofu_trajectory.retain_logs_path=saves/eval/tofu_retain95/TOFU_EVAL.json for TOFU, "
    "eval.muse_trajectory.retain_logs_path=... for MUSE)."
)


def log_retain_logs_path_none_if_needed(
    context: str, metrics: dict, retain_logs_path
) -> None:
    """Log when retain-dependent metrics are chosen but retain_logs_path is None."""
    retain_metrics = RETAIN_LOGS_METRICS & set(metrics.keys())
    if not retain_metrics or retain_logs_path is not None:
        return
    logger.warning(
        "[%s] retain_logs_path is None but metrics %s require retain_model_logs. %s",
        context,
        sorted(retain_metrics),
        RETAIN_LOGS_PATH_NONE_MSG,
    )


@unlearning_metric(name="ks_test")
def ks_test(model, **kwargs):
    """Compare two forget and retain model distributions with a 2-sample KS-test and report the p-value.
    Used in the TOFU benchmark as forget_quality when computed over the truth_ratio statistic."""
    # Forget side: does not require retain; invalid input is a bug → fail (same as upstream).
    forget_vbi = kwargs["pre_compute"]["forget"].get("value_by_index") or {}
    forget_tr_stats = np.array(
        [
            evals["score"]
            for evals in forget_vbi.values()
            if isinstance(evals, dict) and evals.get("score") is not None
        ],
        dtype=np.float64,
    )
    if len(forget_tr_stats) == 0:
        logger.warning(
            "ks_test: forget pre_compute has no valid value_by_index entries with 'score'; returning agg_value=None."
        )
        return {"agg_value": None}
    reference_logs = kwargs.get("reference_logs", None)
    if reference_logs:
        rml = reference_logs.get("retain_model_logs") or {}
        if rml.get("_required_but_missing"):
            logger.error(
                "reference_logs was required but data was missing. ks_test requires retain_ftr. No fallback."
            )
            raise RetainReferenceValidationError(
                "reference_logs retain_model_logs is marked _required_but_missing; "
                "ks_test requires retain_ftr slot (forget_truth_ratio)."
            )
        # Read retain_ftr only (no fallback to retain).
        retain_ftr = rml.get("retain_ftr")
        if retain_ftr is None:
            logger.error(
                "reference_logs was provided but retain_ftr slot is missing; "
                "ks_test reads retain_ftr only. No fallback."
            )
            raise RetainReferenceValidationError(
                "reference_logs retain_model_logs has no retain_ftr slot; "
                "ks_test requires forget_truth_ratio in retain_ftr."
            )
        retain_vbi = (retain_ftr.get("value_by_index") or {}) if isinstance(retain_ftr, dict) else {}
        retain_tr_stats = np.array(
            [
                evals["score"]
                for evals in retain_vbi.values()
                if isinstance(evals, dict) and evals.get("score") is not None
            ],
            dtype=np.float64,
        )
        if len(retain_tr_stats) > 0:
            fq = ks_2samp(forget_tr_stats, retain_tr_stats)
            pvalue = fq.pvalue
        else:
            logger.warning(
                "retain_model_logs retain_ftr has no valid value_by_index scores; setting forget_quality to None"
            )
            pvalue = None
    else:
        logger.warning(
            "retain_model_logs not provided in reference_logs, setting forget_quality to None"
        )
        pvalue = None
    return {"agg_value": pvalue}


@unlearning_metric(name="privleak")
def privleak(model, **kwargs):
    """Compare two forget and retain model scores using a relative comparison of a single statistic.
    To be used for MIA AUC scores in ensuring consistency and reproducibility of the MUSE benchmark.
    This function is similar to the rel_diff function below, but due to the MUSE benchmark reporting AUC
    scores as (1-x) when the more conventional way is x, we do adjustments here to our MIA AUC scores.
    calculations in the reverse way,"""
    score = kwargs["pre_compute"]["forget"]["agg_value"]
    if score is None:
        return {"agg_value": None}
    ref_logs = kwargs.get("reference_logs") or {}
    retain_logs = ref_logs.get("retain_model_logs") or {}
    if not ref_logs or "retain_model_logs" not in ref_logs:
        logger.warning(
            "privleak: retain_model_logs not provided, using default retain auc of %s. %s",
            kwargs.get("ref_value", 0.5),
            RETAIN_LOGS_PATH_NONE_MSG,
        )
        ref = kwargs.get("ref_value", 0.5)
    elif retain_logs.get("_required_but_missing"):
        logger.error(
            "reference_logs was required but data was missing. privleak requires retain slot. No fallback."
        )
        raise RetainReferenceValidationError(
            "reference_logs retain_model_logs is marked _required_but_missing; privleak requires retain slot."
        )
    else:
        retain_slot = retain_logs.get("retain")
        if retain_slot is None:
            logger.error(
                "reference_logs was provided but retain slot is missing; privleak requires retain (mia_min_k)."
            )
            raise RetainReferenceValidationError(
                "reference_logs retain_model_logs has no retain slot; privleak requires mia_min_k in retain."
            )
        ref = (retain_slot.get("agg_value") if isinstance(retain_slot, dict) else None)
        if ref is None or not isinstance(ref, (int, float)) or isinstance(ref, bool):
            logger.error(
                "reference_logs was provided but retain agg_value is not a number (got %s). No fallback.",
                type(ref).__name__,
            )
            raise RetainReferenceValidationError(
                f"reference_logs retain slot agg_value must be a number, got {type(ref).__name__}"
            )
    score = 1 - score
    ref = 1 - ref
    return {"agg_value": (score - ref) / (ref + 1e-10) * 100}


@unlearning_metric(name="rel_diff")
def rel_diff(model, **kwargs):
    """Compare two forget and retain model scores using a relative comparison of a single statistic."""
    score = kwargs["pre_compute"]["forget"]["agg_value"]
    ref_logs = kwargs.get("reference_logs") or {}
    retain_logs = ref_logs.get("retain_model_logs") or {}
    if not ref_logs or "retain_model_logs" not in ref_logs:
        logger.warning(
            "rel_diff: retain_model_logs not provided, using default retain auc of %s. %s",
            kwargs.get("ref_value", 0.5),
            RETAIN_LOGS_PATH_NONE_MSG,
        )
        ref = kwargs.get("ref_value", 0.5)
    elif retain_logs.get("_required_but_missing"):
        logger.error(
            "reference_logs was required but data was missing. rel_diff requires retain slot. No fallback."
        )
        raise RetainReferenceValidationError(
            "reference_logs retain_model_logs is marked _required_but_missing; rel_diff requires retain slot."
        )
    else:
        retain_slot = retain_logs.get("retain")
        if retain_slot is None:
            logger.error(
                "reference_logs was provided but retain slot is missing; rel_diff requires retain."
            )
            raise RetainReferenceValidationError(
                "reference_logs retain_model_logs has no retain slot; rel_diff requires retain."
            )
        ref = (retain_slot.get("agg_value") if isinstance(retain_slot, dict) else None)
        if ref is None or not isinstance(ref, (int, float)) or isinstance(ref, bool):
            logger.error(
                "reference_logs was provided but retain agg_value is not a number (got %s). No fallback.",
                type(ref).__name__,
            )
            raise RetainReferenceValidationError(
                f"reference_logs retain slot agg_value must be a number, got {type(ref).__name__}"
            )
    return {"agg_value": (score - ref) / (ref + 1e-10) * 100}
