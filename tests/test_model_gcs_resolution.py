"""Tests for GCS model path resolution in model/__init__.py (multi-GPU eval with gs:// path).

When model and tokenizer configs use a gs:// path, both must receive the resolved local path
so HuggingFace from_pretrained/AutoTokenizer does not see gs:// (HFValidationError otherwise).
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from omegaconf import OmegaConf

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from model import _resolve_gcs_model_path, get_model, get_tokenizer


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


def test_get_model_passes_resolved_path_to_tokenizer_for_gcs():
    """When config has gs:// for model and tokenizer, get_tokenizer is called with resolved path (not gs://)."""
    resolved = "/tmp/resolved_checkpoint"
    model_cfg = OmegaConf.create({
        "model_args": {"pretrained_model_name_or_path": "gs://bucket/model/checkpoint-final"},
        "tokenizer_args": {"pretrained_model_name_or_path": "gs://bucket/model/checkpoint-final"},
        "model_handler": "AutoModelForCausalLM",
    })
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return fake_model

    with patch("model._resolve_gcs_model_path", return_value=resolved):
        with patch.dict("model.MODEL_REGISTRY", {"AutoModelForCausalLM": FakeModelClass}):
            with patch("model.get_tokenizer", return_value=fake_tokenizer) as get_tokenizer_mock:
                with patch("model.wrap_model_if_diffusion", side_effect=lambda m, t, **kw: m):
                    with patch("model._DIFFUSION_ADAPTER_AVAILABLE", False):
                        get_model(model_cfg)
    get_tokenizer_mock.assert_called_once()
    call_args = get_tokenizer_mock.call_args[0][0]
    path = OmegaConf.to_container(call_args, resolve=True).get("pretrained_model_name_or_path")
    assert path == resolved, "Tokenizer must receive resolved path, not gs://"
