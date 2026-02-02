"""Unit tests for data layer split slicing.

Catches double-slicing bug: when config has pre-sliced split (e.g. forget_qa[:5])
and samples=N is passed, the result must be split[:N] (e.g. forget_qa[:2]),
not split[:5][:2] which HuggingFace datasets rejects.
"""

import sys
from pathlib import Path

from omegaconf import OmegaConf

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from data import get_datasets, DATASET_REGISTRY


class TestSamplesLimitStripExistingSlice:
    """samples limit must strip existing slice from config before appending [:N]."""

    def test_plain_split_gets_slice(self):
        """split: train, samples=2 -> train[:2]."""
        class RecordArgsHandler:
            def __init__(self, **kwargs):
                self.init_kwargs = kwargs

        try:
            DATASET_REGISTRY["RecordArgsHandler"] = RecordArgsHandler
            dataset_cfg = OmegaConf.create({
                "handler": "RecordArgsHandler",
                "args": {
                    "hf_args": {
                        "path": "locuslab/TOFU",
                        "name": "forget10",
                        "split": "train",
                    },
                },
            })
            result = get_datasets({"x": dataset_cfg}, samples=2)
            hf_args = result.init_kwargs.get("hf_args", {})
            assert hf_args.get("split") == "train[:2]"
        finally:
            DATASET_REGISTRY.pop("RecordArgsHandler", None)

    def test_presliced_split_stripped_then_sliced(self):
        """split: forget_qa[:5], samples=2 -> forget_qa[:2] (not forget_qa[:5][:2])."""
        class RecordArgsHandler:
            def __init__(self, **kwargs):
                self.init_kwargs = kwargs

        try:
            DATASET_REGISTRY["RecordArgsHandler"] = RecordArgsHandler
            dataset_cfg = OmegaConf.create({
                "handler": "RecordArgsHandler",
                "args": {
                    "hf_args": {
                        "path": "muse-bench/MUSE-News",
                        "name": "knowmem",
                        "split": "forget_qa[:5]",
                    },
                },
            })
            result = get_datasets({"x": dataset_cfg}, samples=2)
            hf_args = result.init_kwargs.get("hf_args", {})
            split_val = hf_args.get("split")
            assert split_val == "forget_qa[:2]", (
                f"Expected forget_qa[:2], got {split_val}. "
                "Double-slicing (forget_qa[:5][:2]) would fail in HuggingFace datasets."
            )
        finally:
            DATASET_REGISTRY.pop("RecordArgsHandler", None)
