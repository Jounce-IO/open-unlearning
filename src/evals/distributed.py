"""Distributed evaluation helpers. Used when eval is launched with accelerate launch --num_processes N."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _total_samples_from_merged_logs(merged: dict) -> int | None:
    """Find total unique sample count from merged logs (first non-empty value_by_index)."""
    for key, value in merged.items():
        if key == "config" or not isinstance(value, dict):
            continue
        vbi = value.get("value_by_index")
        if isinstance(vbi, dict) and len(vbi) > 0:
            return len(vbi)
    return None


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


def _samples_per_rank_from_gathered(all_logs: list[dict]) -> list[int]:
    """Return list of sample counts per rank from gathered logs (length of value_by_index per rank)."""
    counts: list[int] = []
    for rank_logs in all_logs:
        if not isinstance(rank_logs, dict):
            counts.append(0)
            continue
        n = 0
        for key, value in rank_logs.items():
            if key == "config" or not isinstance(value, dict):
                continue
            vbi = value.get("value_by_index")
            if isinstance(vbi, dict) and vbi:
                n = len(vbi)
                break
        counts.append(n)
    return counts


def gather_logs_to_rank0(logs: dict, rank: int, world_size: int) -> dict | None:
    """Gather logs from all ranks to rank 0 and merge. Returns merged logs on rank 0, None on others."""
    if world_size <= 1:
        return logs
    try:
        import torch.distributed as dist

        obj_list = [None] * world_size
        dist.all_gather_object(obj_list, logs)
        if rank == 0:
            merged = _merge_value_by_index(obj_list)
            # Set run_info for distributed run (total_samples = unique indices across ranks).
            total_indices = None
            for key, value in (merged or {}).items():
                if key == "config" or not isinstance(value, dict):
                    continue
                vbi = value.get("value_by_index")
                if isinstance(vbi, dict) and vbi:
                    total_indices = len(vbi)
                    break
            samples_per_rank = _samples_per_rank_from_gathered(obj_list)
            if merged is not None and total_indices is not None:
                run_info: dict[str, Any] = {
                    "world_size": world_size,
                    "total_samples": total_indices,
                    "data_parallel": True,
                }
                if samples_per_rank and sum(samples_per_rank) == total_indices:
                    run_info["samples_per_rank"] = samples_per_rank
                merged["run_info"] = run_info
            return merged
    except Exception as e:
        logger.warning("Failed to gather logs: %s", e)
        return logs if rank == 0 else None
    return None
