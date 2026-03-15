"""Tests for validation_splits 100-sample cap and same samples (determinism)."""

import sys
from pathlib import Path

from torch.utils.data import Dataset

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from data import get_data, _cap_dataset_at_100, VALIDATION_MAX_SAMPLES


class _ListDataset(Dataset):
    def __init__(self, length):
        self.length = length
        self.items = list(range(length))

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.items[i]


def test_cap_dataset_at_100_under_cap():
    """Dataset with <= 100 samples is returned unchanged."""
    ds = _ListDataset(50)
    out = _cap_dataset_at_100(ds)
    assert out is ds
    assert len(out) == 50


def test_cap_dataset_at_100_over_cap():
    """Dataset with > 100 samples is capped to first 100."""
    ds = _ListDataset(200)
    out = _cap_dataset_at_100(ds)
    assert len(out) == VALIDATION_MAX_SAMPLES
    assert out[0] == 0
    assert out[99] == 99


def test_cap_dataset_same_samples_deterministic():
    """Capping twice yields the same indices (same 100 samples)."""
    ds = _ListDataset(200)
    out1 = _cap_dataset_at_100(ds)
    out2 = _cap_dataset_at_100(ds)
    assert len(out1) == len(out2) == VALIDATION_MAX_SAMPLES
    for i in range(VALIDATION_MAX_SAMPLES):
        assert out1[i] == out2[i] == i
