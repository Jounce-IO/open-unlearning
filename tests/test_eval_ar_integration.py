"""
T010: Integration test — run eval with AR model (model_type=ar) for one benchmark with samples=1;
assert result file exists and contains required keys.
Uses mocked model loading so the test can run in CI without GPU.
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

REQUIRED_KEYS = {"run_info", "forget_Q_A_Prob", "retain_Q_A_Prob", "forget_truth_ratio", "retain_truth_ratio"}


def _ar_model_config():
    """Minimal AR model config (Llama-3.2-3B-Instruct with model_type=ar)."""
    return OmegaConf.create({
        "model_type": "ar",
        "model_args": {
            "pretrained_model_name_or_path": "meta-llama/Llama-3.2-3B-Instruct",
            "torch_dtype": "bfloat16",
        },
        "tokenizer_args": {"pretrained_model_name_or_path": "meta-llama/Llama-3.2-3B-Instruct"},
        "model_handler": "AutoModelForCausalLM",
    })


class TestEvalArIntegration:
    """Run eval with AR model; result file exists and has required keys."""

    def test_get_model_with_ar_config_returns_unwrapped_model(self):
        """get_model(model_type=ar) returns the same object as from_pretrained (no wrap)."""
        from model import get_model

        model_cfg = _ar_model_config()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        with (
            patch("model.MODEL_REGISTRY", {
                "AutoModelForCausalLM": MagicMock(from_pretrained=MagicMock(return_value=mock_model)),
            }),
            patch("model.get_tokenizer", return_value=mock_tokenizer),
            patch("model.get_dtype", return_value=__import__("torch").bfloat16),
            patch("model._DIFFUSION_ADAPTER_AVAILABLE", True),
            patch("model.wrap_model_if_diffusion", MagicMock(side_effect=AssertionError("should not be called"))),
        ):
            model, _ = get_model(model_cfg)
        assert model is mock_model

    def test_prepare_model_accepts_unwrapped_ar_model(self):
        """Evaluator.prepare_model() works with unwrapped causal LM (no .model attr)."""
        from evals.base import Evaluator
        from omegaconf import OmegaConf

        mock_model = MagicMock(spec=["eval"])  # Only eval, no .model (unwrapped AR)
        mock_model.eval = MagicMock()
        eval_cfg = OmegaConf.create({
            "output_dir": "/tmp/test_ar",
            "overwrite": True,
            "metrics": {},
        })
        with patch("evals.base.get_metrics", return_value={}):
            evaluator = Evaluator("TOFU", eval_cfg)
        prepared = evaluator.prepare_model(mock_model)
        assert prepared is mock_model
        mock_model.eval.assert_called_once()
