"""Contract tests: AR unlearn fails with clear message for invalid config (T020).

- Invalid model_type value must raise with clear message.
- Default model_type (ar when no diffusion_adapter) is resolvable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def test_invalid_model_type_raises_clear_error() -> None:
    """model_type not in (ar, dllm) must raise ValueError with allowed values."""
    from model import get_model

    model_cfg = OmegaConf.create({
        "model_type": "invalid",
        "model_args": {"pretrained_model_name_or_path": "meta-llama/Llama-3.2-3B-Instruct"},
        "tokenizer_args": {"pretrained_model_name_or_path": "meta-llama/Llama-3.2-3B-Instruct"},
    })
    with pytest.raises(ValueError) as exc_info:
        get_model(model_cfg)
    msg = str(exc_info.value).lower()
    assert "model_type" in msg or "ar" in msg or "dllm" in msg
    assert "invalid" in msg or "unsupported" in msg or "allowed" in msg


def test_empty_model_type_in_config_uses_default() -> None:
    """Config without model_type and without diffusion_adapter defaults to ar (causal LM)."""
    from model import _resolve_model_type
    cfg_ar = OmegaConf.create({"model_args": {}, "tokenizer_args": {}})
    assert _resolve_model_type(cfg_ar) == "ar"
    cfg_dllm = OmegaConf.create({
        "model_args": {},
        "tokenizer_args": {},
        "diffusion_adapter": {"steps": 128},
    })
    assert _resolve_model_type(cfg_dllm) == "dllm"
