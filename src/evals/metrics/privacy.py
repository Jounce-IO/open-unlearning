import numpy as np
from scipy.stats import ks_2samp
from evals.metrics.base import unlearning_metric, logger

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
    forget_tr_stats = np.array(
        [
            evals["score"]
            for evals in kwargs["pre_compute"]["forget"]["value_by_index"].values()
        ]
    )
    reference_logs = kwargs.get("reference_logs", None)
    if reference_logs:
        reference_logs = reference_logs["retain_model_logs"]
        retain_tr_stats = np.array(
            [
                evals["score"]
                for evals in reference_logs["retain"]["value_by_index"].values()
            ]
        )
        fq = ks_2samp(forget_tr_stats, retain_tr_stats)
        pvalue = fq.pvalue
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
    try:
        ref = kwargs["reference_logs"]["retain_model_logs"]["retain"]["agg_value"]
    except Exception as _:
        logger.warning(
            "privleak: retain_model_logs evals not provided, using default retain auc of %s. %s",
            kwargs.get("ref_value", 0.5),
            RETAIN_LOGS_PATH_NONE_MSG,
        )
        ref = kwargs["ref_value"]
    score = 1 - score
    ref = 1 - ref
    return {"agg_value": (score - ref) / (ref + 1e-10) * 100}


@unlearning_metric(name="rel_diff")
def rel_diff(model, **kwargs):
    """Compare two forget and retain model scores using a relative comparison of a single statistic."""
    score = kwargs["pre_compute"]["forget"]["agg_value"]
    try:
        ref = kwargs["reference_logs"]["retain_model_logs"]["retain"]["agg_value"]
    except Exception as _:
        logger.warning(
            "rel_diff: retain_model_logs evals not provided, using default retain auc of %s. %s",
            kwargs.get("ref_value", 0.5),
            RETAIN_LOGS_PATH_NONE_MSG,
        )
        ref = kwargs["ref_value"]
    return {"agg_value": (score - ref) / (ref + 1e-10) * 100}
