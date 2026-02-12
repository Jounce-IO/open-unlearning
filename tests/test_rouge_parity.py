"""
TDD parity tests for ROUGE backends.

All drop-in backends must produce the same three values (rouge1_recall, rougeL_f1,
rougeL_recall) as the baseline within 0.001% relative tolerance.
no_stemmer is benchmark-only (different scores by design); tested against baseline
with use_stemmer=False.
"""

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from evals.metrics.utils import eval_rouge_recall_batch
from evals.metrics.rouge_impl.golden_data import (
    ROUGE_GOLDEN_PAIRS,
    get_golden_gen_gt_lists,
)

# Relative tolerance for parity: 0.001% = 1e-5
ROUGE_RTOL = 1e-5


def assert_rouge_close(actual: dict, expected: dict, rtol: float = ROUGE_RTOL) -> None:
    """Assert that actual ROUGE dict matches expected within relative tolerance."""
    keys = ("rouge1_recall", "rougeL_f1", "rougeL_recall")
    for k in keys:
        a, b = actual[k], expected[k]
        denom = max(abs(b), 1e-10)
        assert abs(a - b) / denom <= rtol, (
            f"{k}: actual={a}, expected={b}, rel_diff={abs(a - b) / denom}"
        )


def get_baseline_reference(
    gen_outputs: list[str],
    ground_truths: list[str],
    use_stemmer: bool = True,
) -> list[dict]:
    """Compute reference ROUGE values using current baseline (rouge_score)."""
    return eval_rouge_recall_batch(gen_outputs, ground_truths, use_stemmer=use_stemmer)


def _golden_gen_gt_lists():
    """Return (gen_list, gt_list) from golden pairs (prediction, target)."""
    return get_golden_gen_gt_lists()


class TestRougeParityBaseline:
    """Baseline reference: ensure we have a single source of truth."""

    def test_baseline_reference_values_stored_or_computed(self):
        """Baseline produces expected structure and deterministic values."""
        gen_list, gt_list = _golden_gen_gt_lists()
        result = get_baseline_reference(gen_list, gt_list, use_stemmer=True)
        assert len(result) == len(ROUGE_GOLDEN_PAIRS)
        for item in result:
            assert "rouge1_recall" in item
            assert "rougeL_f1" in item
            assert "rougeL_recall" in item
            assert isinstance(item["rouge1_recall"], (int, float))
            assert isinstance(item["rougeL_f1"], (int, float))
            assert isinstance(item["rougeL_recall"], (int, float))
            assert 0 <= item["rouge1_recall"] <= 1
            assert 0 <= item["rougeL_f1"] <= 1
            assert 0 <= item["rougeL_recall"] <= 1
        # Determinism: run again, same result
        result2 = get_baseline_reference(gen_list, gt_list, use_stemmer=True)
        for a, b in zip(result, result2):
            assert a["rouge1_recall"] == b["rouge1_recall"]
            assert a["rougeL_f1"] == b["rougeL_f1"]
            assert a["rougeL_recall"] == b["rougeL_recall"]


class TestRougeBackendMinimalPython:
    """Parity test for minimal_python backend (TDD: will pass once backend exists)."""

    def test_rouge_backend_minimal_python_matches_reference(self):
        """minimal_python output matches baseline within 0.001%."""
        try:
            from evals.metrics.rouge_impl import rouge_backends
            backend_fn = rouge_backends.get_backend("minimal_python")
        except ImportError:
            pytest.skip("rouge_impl not yet implemented")
        gen_list, gt_list = _golden_gen_gt_lists()
        reference = get_baseline_reference(gen_list, gt_list, use_stemmer=True)
        actual = backend_fn(gen_list, gt_list, use_stemmer=True, scorer=None)
        assert len(actual) == len(reference)
        for a, r in zip(actual, reference):
            assert_rouge_close(a, r)


class TestRougeBackendTwoRowLcs:
    """Parity test for two_row_lcs backend."""

    def test_rouge_backend_two_row_lcs_matches_reference(self):
        """two_row_lcs output matches baseline within 0.001%."""
        try:
            from evals.metrics.rouge_impl import rouge_backends
            backend_fn = rouge_backends.get_backend("two_row_lcs")
        except ImportError:
            pytest.skip("rouge_impl not yet implemented")
        gen_list, gt_list = _golden_gen_gt_lists()
        reference = get_baseline_reference(gen_list, gt_list, use_stemmer=True)
        actual = backend_fn(gen_list, gt_list, use_stemmer=True, scorer=None)
        assert len(actual) == len(reference)
        for a, r in zip(actual, reference):
            assert_rouge_close(a, r)


class TestRougeBackendBatchCached:
    """Parity test for batch_cached backend."""

    def test_rouge_backend_batch_cached_matches_reference(self):
        """batch_cached output matches baseline within 0.001%."""
        try:
            from evals.metrics.rouge_impl import rouge_backends
            backend_fn = rouge_backends.get_backend("batch_cached")
        except ImportError:
            pytest.skip("rouge_impl not yet implemented")
        gen_list, gt_list = _golden_gen_gt_lists()
        reference = get_baseline_reference(gen_list, gt_list, use_stemmer=True)
        actual = backend_fn(gen_list, gt_list, use_stemmer=True, scorer=None)
        assert len(actual) == len(reference)
        for a, r in zip(actual, reference):
            assert_rouge_close(a, r)


class TestRougeBackendNumpyLcs:
    """Parity test for numpy_lcs backend."""

    def test_rouge_backend_numpy_lcs_matches_reference(self):
        """numpy_lcs output matches baseline within 0.001%."""
        try:
            from evals.metrics.rouge_impl import rouge_backends
            backend_fn = rouge_backends.get_backend("numpy_lcs")
        except ImportError:
            pytest.skip("rouge_impl not yet implemented")
        gen_list, gt_list = _golden_gen_gt_lists()
        reference = get_baseline_reference(gen_list, gt_list, use_stemmer=True)
        actual = backend_fn(gen_list, gt_list, use_stemmer=True, scorer=None)
        assert len(actual) == len(reference)
        for a, r in zip(actual, reference):
            assert_rouge_close(a, r)


class TestRougeBackendNoStemmer:
    """no_stemmer: parity only against baseline with use_stemmer=False (not a drop-in)."""

    def test_rouge_backend_no_stemmer_matches_baseline_no_stemmer(self):
        """no_stemmer output matches baseline(use_stemmer=False) within 0.001%."""
        try:
            from evals.metrics.rouge_impl import rouge_backends
            backend_fn = rouge_backends.get_backend("no_stemmer")
        except ImportError:
            pytest.skip("rouge_impl not yet implemented")
        gen_list, gt_list = _golden_gen_gt_lists()
        reference = get_baseline_reference(gen_list, gt_list, use_stemmer=False)
        actual = backend_fn(gen_list, gt_list, use_stemmer=False, scorer=None)
        assert len(actual) == len(reference)
        for a, r in zip(actual, reference):
            assert_rouge_close(a, r)


class TestRougeBackendGpuTorchRouge1LcsCpu:
    """Parity test for gpu_torch_rouge1_lcs_cpu (hybrid). Skips when no CUDA."""

    def test_rouge_backend_gpu_torch_rouge1_lcs_cpu_matches_reference(self):
        """gpu_torch_rouge1_lcs_cpu output matches baseline within 0.001%."""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
            from evals.metrics.rouge_impl import rouge_backends
            backend_fn = rouge_backends.get_backend("gpu_torch_rouge1_lcs_cpu")
        except (ImportError, RuntimeError):
            pytest.skip("PyTorch/CUDA or rouge_impl not available")
        gen_list, gt_list = _golden_gen_gt_lists()
        reference = get_baseline_reference(gen_list, gt_list, use_stemmer=True)
        actual = backend_fn(gen_list, gt_list, use_stemmer=True, scorer=None)
        assert len(actual) == len(reference)
        for a, r in zip(actual, reference):
            assert_rouge_close(a, r)


class TestRougeBackendGpuTorchBatch:
    """Parity test for gpu_torch_batch (full GPU). Skips when no CUDA."""

    def test_rouge_backend_gpu_torch_batch_matches_reference(self):
        """gpu_torch_batch output matches baseline within 0.001%."""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
            from evals.metrics.rouge_impl import rouge_backends
            backend_fn = rouge_backends.get_backend("gpu_torch_batch")
        except (ImportError, RuntimeError):
            pytest.skip("PyTorch/CUDA or rouge_impl not available")
        gen_list, gt_list = _golden_gen_gt_lists()
        reference = get_baseline_reference(gen_list, gt_list, use_stemmer=True)
        actual = backend_fn(gen_list, gt_list, use_stemmer=True, scorer=None)
        assert len(actual) == len(reference)
        for a, r in zip(actual, reference):
            assert_rouge_close(a, r)


class TestRougeBackendMultiprocessBatch:
    """Parity test for multiprocess_batch."""

    def test_rouge_backend_multiprocess_batch_matches_reference(self):
        """multiprocess_batch output matches baseline within 0.001%."""
        try:
            from evals.metrics.rouge_impl import rouge_backends
            backend_fn = rouge_backends.get_backend("multiprocess_batch")
        except ImportError:
            pytest.skip("rouge_impl not yet implemented")
        gen_list, gt_list = _golden_gen_gt_lists()
        reference = get_baseline_reference(gen_list, gt_list, use_stemmer=True)
        actual = backend_fn(gen_list, gt_list, use_stemmer=True, scorer=None)
        assert len(actual) == len(reference)
        for a, r in zip(actual, reference):
            assert_rouge_close(a, r)


class TestRougeBackendFusedCpu:
    """Parity test for fused_cpu."""

    def test_rouge_backend_fused_cpu_matches_reference(self):
        """fused_cpu output matches baseline within 0.001%."""
        try:
            from evals.metrics.rouge_impl import rouge_backends
            backend_fn = rouge_backends.get_backend("fused_cpu")
        except ImportError:
            pytest.skip("rouge_impl not yet implemented")
        gen_list, gt_list = _golden_gen_gt_lists()
        reference = get_baseline_reference(gen_list, gt_list, use_stemmer=True)
        actual = backend_fn(gen_list, gt_list, use_stemmer=True, scorer=None)
        assert len(actual) == len(reference)
        for a, r in zip(actual, reference):
            assert_rouge_close(a, r)
