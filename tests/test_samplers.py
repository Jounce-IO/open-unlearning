"""
Unit tests for evals.metrics.samplers (LengthSortedSampler).
"""

import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.samplers import LengthSortedSampler


class TestLengthSortedSampler:
    """Tests for LengthSortedSampler."""

    def test_yields_each_index_exactly_once(self):
        """Sampler yields each index exactly once."""
        # Dataset of 5 items with arbitrary lengths
        class FakeDataset:
            def __init__(self):
                self.data = [
                    {"input_ids": torch.zeros(i)} for i in [3, 7, 5, 1, 4]
                ]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = FakeDataset()
        sampler = LengthSortedSampler(dataset)
        indices = list(sampler)
        assert set(indices) == set(range(len(dataset)))
        assert len(indices) == len(dataset)

    def test_order_is_descending_by_length(self):
        """Indices are yielded in descending order by len(dataset[i]['input_ids'])."""
        # lengths: index 0 -> 3, index 1 -> 7, index 2 -> 5, index 3 -> 1
        # descending: 7, 5, 3, 1 -> indices [1, 2, 0, 3]
        class FakeDataset:
            def __init__(self):
                self.data = [
                    {"input_ids": torch.zeros(3)},
                    {"input_ids": torch.zeros(7)},
                    {"input_ids": torch.zeros(5)},
                    {"input_ids": torch.zeros(1)},
                ]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = FakeDataset()
        sampler = LengthSortedSampler(dataset)
        indices = list(sampler)
        lengths = [7, 5, 3, 1]
        expected_order = [1, 2, 0, 3]  # by length descending
        assert indices == expected_order
        for i, idx in enumerate(indices):
            assert len(dataset[idx]["input_ids"]) == lengths[i]

    def test_len_sampler_equals_len_dataset(self):
        """len(sampler) == len(dataset)."""
        class FakeDataset:
            def __init__(self, n):
                self.data = [{"input_ids": torch.zeros(1)} for _ in range(n)]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        for n in [0, 1, 5, 100]:
            dataset = FakeDataset(n)
            sampler = LengthSortedSampler(dataset)
            assert len(sampler) == len(dataset) == n
