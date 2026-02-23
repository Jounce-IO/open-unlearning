"""Distributed evaluation helpers. Used when eval is launched with accelerate launch --num_processes N."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def get_rank() -> int:
    """Return global rank (0..world_size-1). Returns 0 if not in a distributed run."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    """Return world size. Returns 1 if not in a distributed run."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _merge_value_by_index(all_logs: list[dict]) -> dict:
    """Merge per-rank logs: combine value_by_index (union by index), recompute agg_value as mean of values."""
    merged: dict[str, Any] = {}
    for rank_logs in all_logs:
        for key, value in rank_logs.items():
            if key == "config":
                if "config" not in merged:
                    merged["config"] = value
                continue
            if not isinstance(value, dict):
                merged[key] = value
                continue
            if key not in merged:
                merged[key] = {"value_by_index": {}, "agg_value": None}
            m = merged[key]
            vbi = value.get("value_by_index")
            if isinstance(vbi, dict):
                m["value_by_index"].update(vbi)
            agg = value.get("agg_value")
            if agg is not None and isinstance(agg, (int, float)):
                if m["agg_value"] is None:
                    m["agg_value"] = agg
                else:
                    m["agg_value"] = (m["agg_value"] + agg) / 2
            for k, v in value.items():
                if k not in ("value_by_index", "agg_value") and k not in m:
                    m[k] = v
    for key in list(merged.keys()):
        if key == "config":
            continue
        m = merged.get(key)
        if isinstance(m, dict) and "value_by_index" in m and m["value_by_index"]:
            vals = list(m["value_by_index"].values())
            if vals and all(isinstance(x, (int, float)) for x in vals):
                m["agg_value"] = sum(vals) / len(vals)
    return merged


def gather_logs_to_rank0(logs: dict, rank: int, world_size: int) -> dict | None:
    """Gather logs from all ranks to rank 0 and merge. Returns merged logs on rank 0, None on others."""
    if world_size <= 1:
        return logs
    try:
        import torch.distributed as dist

        obj_list = [None] * world_size
        dist.all_gather_object(obj_list, logs)
        if rank == 0:
            return _merge_value_by_index(obj_list)
    except Exception as e:
        logger.warning("Failed to gather logs: %s", e)
        return logs if rank == 0 else None
    return None
