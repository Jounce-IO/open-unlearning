# Unlearn methods and AR support

**Purpose**: List which forget (unlearn) methods are **dLLM-only** (require diffusion sampling or adapter-only APIs) and which work for **both AR and dLLM**. Used by AR unlearning pipeline (specs/002-ar-llm-unlearning-pipeline) for contract tests and method availability.

## dLLM-only methods

**Currently: none.**

All registered forget methods in `src/trainer/unlearn/` use `model(**inputs)` (and optionally `ref_model`) and either:

- Use only `.loss` / `.logits` from the forward (GradAscent, GradDiff, DPO, UNDIAL, RMU, NPO, SimNPO, SatImp, CEU, PDU), which both HuggingFace causal LMs and the diffusion adapter provide, or  
- Branch on adapter presence (WGA): when `adapter_config` and `per_token_nll` / `masked_indices` are present they use `compute_wga_loss_dllm`; otherwise they use `compute_wga_loss` (AR path).

So every method has an AR-compatible path when `model_type: ar` (unwrapped causal LM).

## If you add a dLLM-only method

If you implement a forget method that **requires**:

- Diffusion sampling (e.g. multiple steps, sampler API), or  
- Adapter-only outputs (e.g. no equivalent for causal LM forward),

then add it to a **dLLM-only list** in this file (e.g. a subsection "Future dLLM-only methods") and ensure configs that select that method for an AR model fail with a clear message (see specs/002-ar-llm-unlearning-pipeline and contract test T020).

## Reference

- Trainer registry: `open-unlearning/src/trainer/__init__.py`
- Trainer configs: `open-unlearning/configs/trainer/*.yaml` (GradAscent, GradDiff, WGA, DPO, UNDIAL, RMU, NPO, SimNPO, SatImp, CEU, PDU)
- Diffusion adapter training: `docs/diffusion_support.md` (§ Training mode and unlearning)
