# Diffusion LLM Support in Open-Unlearning

This document describes how to use diffusion language models (dLLMs) with open-unlearning's evaluation framework.

**Note:** The diffusion adapter code lives in the main `dllm` repo (`dllm/integrations/open_unlearning_adapter.py`) to keep this submodule clean. The adapter is automatically imported when available.

## Overview

Open-unlearning's metrics are designed for autoregressive (AR) language models, which predict tokens sequentially. Diffusion LLMs work differently:
- They predict masked tokens bidirectionally
- They generate through iterative denoising steps
- Tokens are "fixed" when they reach high confidence

To support diffusion models **without modifying existing metrics**, we use an **adapter pattern** that makes diffusion models look like AR models.

## How It Works

The `DiffusionModelAdapter` wrapper:
1. **Intercepts `model(**batch)` calls**: Converts diffusion forward passes to AR-compatible logits
2. **Intercepts `model.generate()` calls**: Uses diffusion samplers instead of AR generation
3. **Auto-detection**: Automatically wraps diffusion models when loaded

## Usage

### Basic Usage

Simply load your diffusion model as usual. The adapter will automatically detect and wrap it:

```python
from model import get_model

# Load model (will be auto-wrapped if it's a diffusion model)
model, tokenizer = get_model(model_cfg)

# Use model normally - adapter handles conversion
evaluator.evaluate(model=model, tokenizer=tokenizer, ...)
```

### Configuration

You can configure the adapter via the model config:

```yaml
model:
  model_args:
    pretrained_model_name_or_path: "path/to/diffusion/model"
  diffusion_adapter:
    steps: 128              # Diffusion steps
    block_size: 128        # Block size for generation
    temperature: 0.0        # Sampling temperature
    remasking: "low_confidence"  # Remasking strategy
    max_new_tokens: 128     # Max tokens to generate
    cache_fixation_logits: true  # Cache logits for reuse
```

### Example Configs

**LLaDA** (`configs/model/LLaDA-8B-Instruct.yaml`):

```yaml
model:
  model_handler: "AutoModelForCausalLM"
  model_args:
    pretrained_model_name_or_path: "path/to/llada"
    torch_dtype: "bfloat16"
  tokenizer_args:
    pretrained_model_name_or_path: "path/to/llada"
  diffusion_adapter:
    steps: 128
    block_size: 128
    temperature: 0.0
    remasking: "low_confidence"
    max_new_tokens: 128
```

**Dream** (`configs/model/Dream-Instruct-7B.yaml`): Use a model path containing `dream` (e.g. `Dream-org/Dream-v0-Instruct-7B`). The adapter uses the Dream sampler and trajectory capture; same trajectory metrics as LLaDA.

```bash
dllm eval dream-tofu-forget10 Dream-org/Dream-v0-Instruct-7B tofu --samples 2
```

## Training mode and unlearning

The adapter supports **training mode** for adapter-based unlearning (GradAscent, GradDiff, WGA). When `diffusion_adapter.training: true` is set in the model config, the adapter's `__call__` performs an MDLM-style forward (single random t per batch, stochastic masking, NLL on masked positions) and returns an output with `.loss` and `.logits` so OpenUnlearning's trainers work unchanged.

### Unlearn CLI (dllm repo)

From the main **dllm** repo, run:

```bash
dllm unlearn <task_name> [--benchmark tofu] [--model PATH] [--method GradAscent|GradDiff|WGA] [--output-dir DIR] ...
```

- **Model:** Local path, HuggingFace ID, or **gs://** (downloaded before training).
- **Output dir:** Local path or **gs://**. When gs://, checkpoints and final model are written locally then uploaded to GCS; a background watcher uploads new checkpoints periodically so runs can resume after spot preemption.
- **Checkpoint and resume:** Use `--save-steps` and `--save-total-limit` to control checkpointing. If `output_dir` already contains `checkpoint-*` directories, the next run **auto-resumes** from the latest unless you pass `--no-resume`. Use `--resume-from <path>` to resume from a specific checkpoint (gs:// or local).
- **W&B:** When `WANDB_API_KEY` is set, the run logs to Weights & Biases.

See `dllm unlearn --help` for all flags. When running in K8s with `results.capture=true` and `results.createPr=true`, the job creates a PR that includes the unlearn report (config, model save path, W&B run URL).

### Experiment config

Use the diffusion unlearn experiment so the model is wrapped with training enabled:

```yaml
# experiment=unlearn/tofu/diffusion (model override + diffusion_adapter)
model:
  diffusion_adapter:
    training: true
    time_epsilon: 0.001
    loss_weight_type: "scheduler"
    loss_normalization_type: "sequence"
```

The dllm CLI passes `experiment=unlearn/tofu/diffusion` and the resolved model path when you run `dllm unlearn`.

### Resume from checkpoint (train.py)

OpenUnlearning's `train.py` accepts a top-level config key **`resume_from_checkpoint`** (e.g. set via Hydra override `+resume_from_checkpoint=/path/to/checkpoint-500`). When set, it calls `trainer.train(resume_from_checkpoint=path)` instead of `trainer.train()`. The dllm CLI discovers the latest checkpoint (or uses `--resume-from`), downloads it from GCS if needed, and passes the path into the config.

## Current Limitations

### 1. Fixation Logits

The current implementation uses the model's forward pass on the input sequence, which doesn't capture the true "fixation logits" (logits at the step where each token was fixed during generation).

**To get true fixation logits**, the samplers need to be enhanced to:
- Track which tokens are fixed at each step
- Store logits at fixation points
- Return `fixation_logits` in `SamplerOutput`

See the implementation discussion for details on how to add this.

### 2. Generation Parameters

Some AR generation parameters (e.g., `top_p`, `top_k`) are mapped to diffusion equivalents, but not all parameters are supported yet.

## How Metrics Work with Diffusion Models

### Probability/Loss Metrics

When `evaluate_probability()` calls `model(**batch)`:
1. Adapter intercepts the call
2. Model forward pass returns logits for all positions
3. Adapter shifts logits to AR format: `logits[:, :-1, :]` for next-token prediction
4. Existing metric code works unchanged

### Generation Metrics (ROUGE, etc.)

When `eval_text_similarity()` calls `model.generate()`:
1. Adapter intercepts the call
2. Uses diffusion sampler instead of AR generation
3. Returns generated sequences in same format
4. Existing metric code works unchanged

## Future Enhancements

### True Fixation Logits

To properly capture fixation logits:

1. **Enhance samplers** to track fixation:
   ```python
   # In sampler loop, when tokens are fixed:
   fixation_logits[transfer_index] = logits[transfer_index].clone()
   fixation_steps[transfer_index] = step
   ```

2. **Update SamplerOutput**:
   ```python
   @dataclass
   class SamplerOutput:
       sequences: torch.Tensor
       histories: list[torch.Tensor] | None = None
       fixation_logits: torch.Tensor | None = None  # NEW
       fixation_steps: torch.Tensor | None = None   # NEW
   ```

3. **Update adapter** to use fixation logits:
   ```python
   output = sampler.sample(..., return_fixation_logits=True)
   fixation_logits = output.fixation_logits
   # Use fixation_logits instead of forward pass
   ```

### Diffusion-Specific Metrics

**Confidence-ordered forget probability.** The **confidence-ordered forget probability** metric is the forget-probability analogue for diffusion LLMs. Instead of aggregating in causal order, it orders labeled positions by the model’s **confidence in the true label** at each position (highest first), then takes the geometric mean in that order — mirroring “set the top confidence labels first.” Use the `probability_confidence_ordered` handler with the forget split; see [evaluation.md – Confidence-ordered forget probability](evaluation.md#confidence-ordered-forget-probability-diffusion-llms) for the formal definition.

Other metrics that leverage diffusion-specific properties:
- **Denoising trajectory analysis**: Track how tokens evolve during denoising
- **Masking ratio effects**: Analyze how different masking ratios affect unlearning
- **Bidirectional attention patterns**: Probe attention patterns in diffusion models

## Troubleshooting

### Model Not Detected as Diffusion

If your diffusion model isn't auto-detected, you can manually wrap it:

```python
from model.diffusion_adapter import DiffusionModelAdapter, DiffusionAdapterConfig

adapter_config = DiffusionAdapterConfig(steps=128, ...)
wrapped_model = DiffusionModelAdapter(model, tokenizer, config=adapter_config)
```

### Performance Issues

- **Caching**: Enable `cache_fixation_logits: true` to avoid regenerating logits
- **Batch size**: Diffusion generation can be slower than AR - consider smaller batches
- **Steps**: Reduce `steps` for faster generation (may affect quality)

## References

- [Diffusion LLM Architectures](../../knowledge/summaries/dllm-architectures.md)
- [Open-Unlearning Framework](../../knowledge/pdfs/frameworks/unlearning/open-unlearning/summary.md)
