"""
Optional GPU phase logger for correlation with the GPU monitor script.

Writes current phase, metric, batch_idx, and step to a file so the monitor script
(scripts/k8s/monitor_gpu_during_eval.sh) can correlate nvidia-smi samples with
the exact eval phase. Set GPU_PHASE_FILE in the environment to enable
(e.g. /tmp/gpu_phase.txt). When unset, set_phase is a no-op.
"""

import os


def _phase_file_path():
    return os.environ.get("GPU_PHASE_FILE")


def set_phase(
    phase: str,
    metric: str | None = None,
    batch_idx: int | None = None,
    step: int | None = None,
) -> None:
    """Write current phase to GPU_PHASE_FILE for correlation with the GPU monitor log."""
    path = _phase_file_path()
    if not path:
        return
    metric_s = metric if metric is not None else ""
    batch_s = str(batch_idx) if batch_idx is not None else ""
    step_s = str(step) if step is not None else ""
    line = f"{phase},{metric_s},{batch_s},{step_s}\n"
    try:
        with open(path, "w") as f:
            f.write(line)
    except OSError:
        pass
