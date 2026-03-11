"""Tests for GCS model path resolution in model/__init__.py (multi-GPU eval with gs:// path)."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from model import _resolve_gcs_model_path


def test_resolve_gcs_model_path_non_gcs_returns_unchanged():
    """Non-gs:// paths are returned unchanged."""
    assert _resolve_gcs_model_path("/local/path") == "/local/path"
    assert _resolve_gcs_model_path("org/repo-name") == "org/repo-name"
    assert _resolve_gcs_model_path("") == ""


def test_resolve_gcs_model_path_uses_dllm_when_available():
    """When dllm.eval.resolve_model_path is available, gs:// is resolved via it."""
    fake_resolved = "/tmp/resolved_model"
    resolve_mock = MagicMock(return_value=fake_resolved)
    dllm_eval = MagicMock()
    dllm_eval.resolve_model_path = resolve_mock
    with patch.dict(sys.modules, {"dllm": MagicMock(), "dllm.eval": dllm_eval}):
        import importlib
        import model
        importlib.reload(model)
        result = model._resolve_gcs_model_path("gs://bucket/path/to/model")
        assert result == fake_resolved
        resolve_mock.assert_called_once_with("gs://bucket/path/to/model")
