from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, open_dict, OmegaConf
from typing import Dict, Any
import os
import torch
import logging
from model.probe import ProbedLlamaForCausalLM

# Enable verbose logging for HuggingFace libraries
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")
os.environ.setdefault("HF_HUB_VERBOSITY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")  # Show progress bars

# Set up logging for huggingface_hub
logging.getLogger("huggingface_hub").setLevel(logging.INFO)
logging.getLogger("transformers").setLevel(logging.INFO)

# Try to import diffusion adapter from main repo (if available)
try:
    from dllm.integrations.open_unlearning_adapter import wrap_model_if_diffusion
    _DIFFUSION_ADAPTER_AVAILABLE = True
except ImportError:
    _DIFFUSION_ADAPTER_AVAILABLE = False

    def wrap_model_if_diffusion(model, tokenizer, config=None):
        """Fallback if adapter not available."""
        return model

hf_home = os.getenv("HF_HOME", default=None)

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Any] = {}


def _register_model(model_class):
    MODEL_REGISTRY[model_class.__name__] = model_class


def get_dtype(model_args):
    with open_dict(model_args):
        torch_dtype = model_args.pop("torch_dtype", None)
    if model_args.get("attn_implementation", None) == "flash_attention_2":
        # This check handles https://github.com/Dao-AILab/flash-attention/blob/7153673c1a3c7753c38e4c10ef2c98a02be5f778/flash_attn/flash_attn_triton.py#L820
        # If you want to run at other precisions consider running "training or inference using
        # Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):`
        # decorator" or using an attn_implementation compatible with the precision in the model
        # config.
        assert torch_dtype in ["float16", "bfloat16"], ValueError(
            f"Invalid torch_dtype '{torch_dtype}' for the requested attention "
            f"implementation: 'flash_attention_2'. Supported types are 'float16' "
            f"and 'bfloat16'."
        )
    if torch_dtype == "float16":
        return torch.float16
    elif torch_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_model(model_cfg: DictConfig, resume_from_checkpoint: str | None = None):
    assert model_cfg is not None, ValueError("Model config not found.")
    logger.info("=== Starting model loading ===")
    with open_dict(model_cfg):
        model_args_dict = model_cfg.get("model_args", None)
        assert model_args_dict is not None, ValueError("model_args absent in configs/model.")
        tokenizer_args = model_cfg.get("tokenizer_args", None)
        model_handler = model_cfg.get("model_handler", "AutoModelForCausalLM")

    if resume_from_checkpoint:
        # Load model and tokenizer from checkpoint (no HuggingFace download).
        logger.info("Resuming: loading model and tokenizer from checkpoint (no HF download): %s", resume_from_checkpoint)
        model_args_copy = OmegaConf.create(OmegaConf.to_container(model_args_dict, resolve=True)) if isinstance(model_args_dict, DictConfig) else OmegaConf.create(dict(model_args_dict))
        torch_dtype = get_dtype(model_args_copy)
        import time
        start_time = time.time()
        model = AutoModel.from_pretrained(
            resume_from_checkpoint,
            torch_dtype=torch_dtype,
        )
        elapsed = time.time() - start_time
        logger.info("Model loaded from checkpoint in %.2f seconds", elapsed)
        tokenizer = AutoTokenizer.from_pretrained(resume_from_checkpoint)
        logger.info("Tokenizer loaded from checkpoint")
    else:
        # Load from config (HuggingFace or local path).
        model_args = model_args_dict
        with open_dict(model_args):
            try:
                model_path = model_args.pretrained_model_name_or_path
                del model_args["pretrained_model_name_or_path"]
            except (AttributeError, KeyError):
                model_path = model_args.get("pretrained_model_name_or_path", None)
                if model_path is not None:
                    del model_args["pretrained_model_name_or_path"]
        logger.info(f"Model path: {model_path}")
        logger.info(f"Model handler: {model_handler}")
        torch_dtype = get_dtype(model_args)
        logger.info(f"Torch dtype: {torch_dtype}")
        model_cls = MODEL_REGISTRY[model_handler]
        model_args_dict_final = OmegaConf.to_container(model_args, resolve=True) if isinstance(model_args, DictConfig) else model_args
        logger.info(f"Calling {model_handler}.from_pretrained()...")
        logger.info(f"Model args: {list(model_args_dict_final.keys())}")
        logger.info(f"Cache dir: {hf_home if hf_home else 'default (~/.cache/huggingface)'}")
        try:
            import time
            start_time = time.time()
            logger.info("Starting model download/loading (this may take several minutes)...")
            logger.info("HuggingFace will download model weights if not cached")
            from pathlib import Path
            if hf_home:
                cache_path = Path(hf_home) / "hub" / f"models--{model_path.replace('/', '--')}"
                if cache_path.exists():
                    logger.info(f"Model cache directory exists: {cache_path}")
                else:
                    logger.info(f"Model cache directory does not exist, will download")
            model = model_cls.from_pretrained(
                pretrained_model_name_or_path=model_path,
                dtype=torch_dtype,
                **model_args_dict_final,
                cache_dir=hf_home,
            )
            elapsed = time.time() - start_time
            logger.info(f"Model loaded successfully in {elapsed:.2f} seconds")
        except Exception as e:
            logger.error(f"Model {model_path} requested with {model_cfg.model_args}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise ValueError(
                f"Error {e} while fetching model using {model_handler}.from_pretrained()."
            )
        logger.info("Loading tokenizer...")
        tokenizer = get_tokenizer(tokenizer_args)
        logger.info("Tokenizer loaded successfully")
    
    # Auto-wrap diffusion models to be compatible with AR-based metrics
    # (only if adapter is available from main dllm repo)
    if _DIFFUSION_ADAPTER_AVAILABLE:
        # Add mask token for diffusion models (required by samplers)
        from dllm.integrations.open_unlearning_adapter import is_diffusion_model
        if is_diffusion_model(model):
            if tokenizer.mask_token_id is None:
                # Detect model type to use correct mask token
                model_name = type(model).__name__.lower()
                if 'llada' in model_name:
                    tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
                    logger.info("Added mask_token '<|mdm_mask|>' for LLaDA model")
                else:
                    # Default mask token for other diffusion models
                    tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
                    logger.info("Added default mask_token '<|mask|>' for diffusion model")
        diffusion_config = model_cfg.get("diffusion_adapter", None)
        model = wrap_model_if_diffusion(model, tokenizer, config=diffusion_config)
    else:
        # Adapter not available - check if this looks like a diffusion model (will fail trajectory metrics)
        model_type = type(model).__name__.lower()
        if any(x in model_type for x in ("llada", "dream", "diffusion", "mdlm", "bd3lm")):
            logger.warning(
                "DiffusionModelAdapter not available (dllm.integrations import failed). "
                "Trajectory metrics will fail. Ensure PYTHONPATH includes the repo root (e.g. export PYTHONPATH=/app)."
            )

    return model, tokenizer


def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")


def get_tokenizer(tokenizer_cfg: DictConfig):
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg, cache_dir=hf_home)
    except Exception as e:
        error_message = (
            f"{'--' * 40}\n"
            f"Error {e} fetching tokenizer using AutoTokenizer.\n"
            f"Tokenizer requested from path: {tokenizer_cfg.get('pretrained_model_name_or_path', None)}\n"
            f"Full tokenizer config: {tokenizer_cfg}\n"
            f"{'--' * 40}"
        )
        raise RuntimeError(error_message)

    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))

    return tokenizer


# register models
_register_model(AutoModelForCausalLM)
_register_model(ProbedLlamaForCausalLM)

# Dream (dLLM): load via dllm's DreamModel since Hub config has no AutoModelForCausalLM auto_map
try:
    from dllm.pipelines.dream import DreamModel
    _register_model(DreamModel)
except ImportError:
    pass  # dllm not available; DreamModel simply won't be in registry
