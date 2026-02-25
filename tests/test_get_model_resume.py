"""
Test that get_model(model_cfg, resume_from_checkpoint=path) loads model and tokenizer
from the checkpoint path (no HuggingFace download).
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

import sys
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


class TestGetModelResume:
    """When resume_from_checkpoint is set, model and tokenizer are loaded from that path."""

    def test_get_model_resume_from_checkpoint_uses_path(self):
        """get_model(cfg, resume_from_checkpoint=path) calls AutoModel and AutoTokenizer from_pretrained with path."""
        from model import get_model

        model_cfg = OmegaConf.create({
            "model_args": {
                "pretrained_model_name_or_path": "GSAI-ML/LLaDA-8B-Instruct",
                "torch_dtype": "bfloat16",
            },
            "tokenizer_args": {"pretrained_model_name_or_path": "GSAI-ML/LLaDA-8B-Instruct"},
            "model_handler": "AutoModelForCausalLM",
        })
        resume_path = "/tmp/checkpoint-100"

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with (
            patch("model.AutoModel.from_pretrained", return_value=mock_model) as mock_model_load,
            patch("model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer) as mock_tok_load,
            patch("model._DIFFUSION_ADAPTER_AVAILABLE", False),
        ):
            model, tokenizer = get_model(model_cfg, resume_from_checkpoint=resume_path)

        assert model is mock_model
        assert tokenizer is mock_tokenizer
        mock_model_load.assert_called_once()
        call_kw = mock_model_load.call_args
        assert call_kw[0][0] == resume_path
        mock_tok_load.assert_called_once()
        assert mock_tok_load.call_args[0][0] == resume_path
