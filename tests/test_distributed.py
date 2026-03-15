"""
Tests for distributed evaluation: data-parallel split and merge.

Asserts that when running with world_size=2:
- Each rank sees len(dataset)//2 samples (no duplication).
- Merged logs have total_samples == requested dataset size (union of indices).
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))


def _make_rank_log(metric_name: str, indices: list[int], values: list[float] | None = None):
    """Build a single-rank log dict with value_by_index for the given indices."""
    if values is None:
        values = [0.5] * len(indices)
    return {
        "config": {"samples": len(indices)},
        metric_name: {
            "value_by_index": {str(i): v for i, v in zip(indices, values)},
            "agg_value": sum(values) / len(values) if values else None,
        },
    }


class TestMergeValueByIndex:
    """Merge of per-rank logs yields union of indices (no duplication)."""

    def test_merge_two_ranks_disjoint_indices(self):
        from evals.distributed import _merge_value_by_index

        rank0 = _make_rank_log("trajectory_forget_Q_A_Prob", list(range(50)))
        rank1 = _make_rank_log("trajectory_forget_Q_A_Prob", list(range(50, 100)))
        merged = _merge_value_by_index([rank0, rank1])

        vbi = merged["trajectory_forget_Q_A_Prob"]["value_by_index"]
        assert len(vbi) == 100
        assert set(vbi.keys()) == {str(i) for i in range(100)}
        assert merged["trajectory_forget_Q_A_Prob"]["agg_value"] == 0.5

    def test_merge_no_duplicate_indices(self):
        from evals.distributed import _merge_value_by_index

        # If both ranks had the same indices (bug), update() would keep last; we assert disjoint.
        rank0 = _make_rank_log("rouge", [0, 2, 4], [0.1, 0.2, 0.3])
        rank1 = _make_rank_log("rouge", [1, 3, 5], [0.4, 0.5, 0.6])
        merged = _merge_value_by_index([rank0, rank1])

        vbi = merged["rouge"]["value_by_index"]
        assert len(vbi) == 6
        assert vbi["0"] == 0.1 and vbi["5"] == 0.6
        assert merged["rouge"]["agg_value"] == pytest.approx(0.35)  # mean of all 6


class TestGatherLogsToRank0:
    """gather_logs_to_rank0 returns merged logs with run_info on rank 0."""

    def test_run_info_total_samples_equals_dataset_size(self):
        from evals.distributed import gather_logs_to_rank0

        rank0_logs = _make_rank_log("metric_a", list(range(50)))
        rank1_logs = _make_rank_log("metric_a", list(range(50, 100)))
        gathered = [rank0_logs, rank1_logs]

        def fake_all_gather_object(obj_list, _local_obj):
            for i, log in enumerate(gathered):
                obj_list[i] = log

        with patch("torch.distributed.all_gather_object", side_effect=fake_all_gather_object):
            result = gather_logs_to_rank0(rank0_logs, rank=0, world_size=2)

        assert result is not None
        run_info = result.get("run_info")
        assert run_info is not None
        assert run_info["world_size"] == 2
        assert run_info["total_samples"] == 100
        assert run_info["data_parallel"] is True
        assert len(result["metric_a"]["value_by_index"]) == 100

    def test_rank_nonzero_returns_none(self):
        from evals.distributed import gather_logs_to_rank0

        def fake_all_gather_object(obj_list, _local_obj):
            obj_list[0] = {"config": {}}
            obj_list[1] = {"config": {}}

        with patch("torch.distributed.all_gather_object", side_effect=fake_all_gather_object):
            result = gather_logs_to_rank0({"config": {}}, rank=1, world_size=2)
        assert result is None


class TestDistributedSamplerSplit:
    """DistributedSampler gives each rank len(dataset)//world_size indices (no overlap)."""

    def test_sampler_length_per_rank(self):
        class SmallDataset(Dataset):
            def __init__(self, n):
                self.n = n

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                return idx

        dataset = SmallDataset(100)
        world_size = 2

        sampler0 = DistributedSampler(
            dataset, num_replicas=world_size, rank=0, shuffle=False
        )
        sampler1 = DistributedSampler(
            dataset, num_replicas=world_size, rank=1, shuffle=False
        )

        n0 = len(sampler0)
        n1 = len(sampler1)
        assert n0 + n1 == 100
        assert n0 == 50 and n1 == 50

        indices0 = list(sampler0)
        indices1 = list(sampler1)
        assert len(set(indices0) & set(indices1)) == 0
        assert set(indices0) | set(indices1) == set(range(100))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
