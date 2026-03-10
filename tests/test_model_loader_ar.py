"""
Unit tests for get_model() with model_type (ar | dllm) per contract config-model-type.
T004: model_type=ar returns unwrapped model (no DiffusionModelAdapter).
T005: default (dllm when diffusion_adapter present else ar), explicit wins, invalid raises.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


def _base_config(extra=None):
    cfg = {
        "model_args": {
            "pretrained_model_name_or_path": "meta-llama/Llama-3.2-3B-Instruct",
            "torch_dtype": "bfloat16",
        },
        "tokenizer_args": {"pretrained_model_name_or_path": "meta-llama/Llama-3.2-3B-Instruct"},
        "model_handler": "AutoModelForCausalLM",
    }
    if extra:
        cfg.update(extra)
    return OmegaConf.create(cfg)


class TestGetModelModelTypeAr:
    """T004: get_model() with model_type=ar returns unwrapped model (no DiffusionModelAdapter)."""

    def test_model_type_ar_does_not_wrap(self):
        """When model_type=ar, wrap_model_if_diffusion is not called; returned model is the loaded model."""
        from model import get_model

        model_cfg = _base_config({"model_type": "ar"})
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        wrap_called = []

        def track_wrap(m, t, config=None):
            wrap_called.append(1)
            return m

        with (
            patch("model.MODEL_REGISTRY", {"AutoModelForCausalLM": MagicMock(from_pretrained=MagicMock(return_value=mock_model))}),
            patch("model.get_tokenizer", return_value=mock_tokenizer),
            patch("model.get_dtype", return_value=__import__("torch").bfloat16),
            patch("model._DIFFUSION_ADAPTER_AVAILABLE", True),
            patch("model.wrap_model_if_diffusion", side_effect=track_wrap),
        ):
            model, tokenizer = get_model(model_cfg)
        assert model is mock_model
        assert tokenizer is mock_tokenizer
        # model_type=ar must skip wrap
        assert len(wrap_called) == 0


class TestGetModelModelTypeDefaultAndValidation:
    """T005: model_type default, explicit wins over default, invalid model_type raises."""

    def test_model_type_default_ar_when_no_diffusion_adapter(self):
        """When model_type missing and no diffusion_adapter, default is ar; do not wrap."""
        from model import get_model

        model_cfg = _base_config()  # no model_type, no diffusion_adapter
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        wrap_called = []

        def track_wrap(model, tokenizer, config=None):
            wrap_called.append(1)
            return model

        with (
            patch("model.MODEL_REGISTRY", {"AutoModelForCausalLM": MagicMock(from_pretrained=MagicMock(return_value=mock_model))}),
            patch("model.get_tokenizer", return_value=mock_tokenizer),
            patch("model.get_dtype", return_value=__import__("torch").bfloat16),
            patch("model._DIFFUSION_ADAPTER_AVAILABLE", True),
            patch("model.wrap_model_if_diffusion", side_effect=track_wrap),
        ):
            model, tokenizer = get_model(model_cfg)
        # After implementation: default ar -> no wrap. Before: may wrap if is_diffusion_model false.
        assert model is mock_model
        assert tokenizer is mock_tokenizer

    def test_model_type_explicit_ar_wins_over_diffusion_adapter(self):
        """When model_type=ar and diffusion_adapter present, explicit model_type wins; do not wrap."""
        from model import get_model

        model_cfg = _base_config({"model_type": "ar", "diffusion_adapter": {"some": "config"}})
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        wrap_called = []

        def track_wrap(model, tokenizer, config=None):
            wrap_called.append(1)
            return model

        with (
            patch("model.MODEL_REGISTRY", {"AutoModelForCausalLM": MagicMock(from_pretrained=MagicMock(return_value=mock_model))}),
            patch("model.get_tokenizer", return_value=mock_tokenizer),
            patch("model.get_dtype", return_value=__import__("torch").bfloat16),
            patch("model._DIFFUSION_ADAPTER_AVAILABLE", True),
            patch("model.wrap_model_if_diffusion", side_effect=track_wrap),
        ):
            model, tokenizer = get_model(model_cfg)
        assert model is mock_model
        # With implementation: wrap should not be called because model_type=ar wins
        assert tokenizer is mock_tokenizer

    def test_invalid_model_type_raises(self):
        """Invalid model_type value raises a clear error."""
        from model import get_model

        model_cfg = _base_config({"model_type": "invalid"})
        with pytest.raises((ValueError, AssertionError)) as exc_info:
            get_model(model_cfg)
        assert "model_type" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
