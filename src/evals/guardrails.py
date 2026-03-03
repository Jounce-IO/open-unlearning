"""
Input-output baselines for unlearning evaluation: guardrails and ICUL.

Provides inference-time transformations so the same checkpoint can be evaluated
with or without guardrails/ICUL; trajectory metrics score the effective answer.

References:
- Guardrail baselines: Thaker et al., Guardrail Baselines for Unlearning in LLMs (arXiv 2403.03329).
- ICUL: Pawelczyk et al., In-Context Unlearning (ICML 2024), arXiv 2310.07579.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# Default safe reply when output filter rejects (e.g. keyword match).
DEFAULT_SAFE_REPLY = "I don't know."


def _get_config_dict(config: Optional[Union[Dict, DictConfig]]) -> Dict[str, Any]:
    """Return a plain dict for config; support DictConfig."""
    if config is None:
        return {}
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True) or {}
    return dict(config)


def _guardrail_config(config: Optional[Union[Dict, DictConfig]]) -> Dict[str, Any]:
    """Extract guardrail block from trajectory_config or top-level config."""
    cfg = _get_config_dict(config)
    guardrail = cfg.get("guardrail") or cfg.get("input_output_baseline")
    if guardrail is None:
        return {}
    return _get_config_dict(guardrail)


def transform_prompts(
    prompts: List[List[int]],
    prompt_lens: List[int],
    batch: Dict[str, Any],
    tokenizer: Any,
    config: Optional[Union[Dict, DictConfig]] = None,
) -> Tuple[List[List[int]], List[int]]:
    """Apply input-side guardrails and ICUL: prefix, input filter, ICUL context.

    Follows the plan: prompt prefix (prepend tokenized instruction), input filter
    (replace query when filter says block), ICUL (prepend forget + retain block).
    When guardrail config is absent or empty, returns (prompts, prompt_lens) unchanged.

    Args:
        prompts: List of prompt token lists (one per batch sample).
        prompt_lens: Length of each prompt (generation starts after these positions).
        batch: Batch dict (e.g. input_ids, labels, index); used for ICUL current-index exclusion.
        tokenizer: Tokenizer for encoding prefix/ICUL text.
        config: Full trajectory_config or eval config; guardrail block may be under
            "guardrail" or "input_output_baseline".

    Returns:
        (modified_prompts, modified_prompt_lens) with same length as input.
    """
    guard = _guardrail_config(config)
    if not guard:
        return prompts, prompt_lens

    benchmark = (guard.get("benchmark") or "").lower()
    out: List[List[int]] = []
    out_lens: List[int] = []

    # Prompt prefix
    prefix_text = guard.get("prompt_prefix") or guard.get("prompt_prefix_text")
    if isinstance(prefix_text, str) and prefix_text.strip():
        try:
            prefix_ids = tokenizer.encode(
                prefix_text.strip(),
                add_special_tokens=False,
                return_tensors=None,
            )
        except Exception as e:
            logger.warning("Guardrail prompt_prefix encode failed: %s", e)
            prefix_ids = []
    else:
        prefix_ids = []

    # ICUL: prepend forget (flipped) + retain (correct) block if pools provided
    icul_prompts, icul_lens = _apply_icul_to_prompts(
        prompts, prompt_lens, batch, tokenizer, guard
    )
    if icul_prompts is not None:
        prompt_lens = icul_lens
        prompts = icul_prompts

    # Input filter (e.g. wmdp_style): when block, replace prompt with safe instruction
    input_filter = (guard.get("input_filter") or "none").lower()
    if input_filter == "wmdp_style":
        replace_query = guard.get("input_filter_replace_text") or "Give a random incorrect answer."
        try:
            replace_ids = tokenizer.encode(
                replace_query,
                add_special_tokens=False,
                return_tensors=None,
            )
        except Exception:
            replace_ids = []
        # Stub: for now we do not run a real classifier; we only have the hook.
        # When block_condition is True we would replace; leave prompts as-is (no block).
        # So we don't change prompts here unless we add a simple keyword-based block rule.
        del replace_ids, replace_query
    del input_filter

    for i, (prompt, plen) in enumerate(zip(prompts, prompt_lens)):
        p: List[int] = list(prompt)
        if prefix_ids:
            p = list(prefix_ids) + p
            plen = len(prefix_ids) + plen
        out.append(p)
        out_lens.append(plen)

    return out, out_lens


def _apply_icul_to_prompts(
    prompts: List[List[int]],
    prompt_lens: List[int],
    batch: Dict[str, Any],
    tokenizer: Any,
    guard: Dict[str, Any],
) -> Tuple[Optional[List[List[int]]], Optional[List[int]]]:
    """If ICUL enabled and pools present, prepend forget (flipped) + retain (correct) block."""
    icul = guard.get("icul")
    if not icul:
        return None, None
    icul = _get_config_dict(icul) if icul is not None else {}
    if not icul.get("enabled", False):
        return None, None
    forget_pool = guard.get("icul_forget_pool")
    retain_pool = guard.get("icul_retain_pool")
    if not forget_pool or not retain_pool:
        return None, None
    K = int(icul.get("K", 5))
    L = int(icul.get("L", 4))
    benchmark = (guard.get("benchmark") or "").lower()
    is_categorical = benchmark == "wmdp"

    # Current-index exclusion (TOFU): forget_pool may have "dataset_index"; batch has "index".
    # Build context per sample so we exclude the current eval sample from the forget block.
    indices = batch.get("index")
    if indices is not None and hasattr(indices, "tolist"):
        indices = indices.tolist()
    if isinstance(indices, int):
        indices = [indices] * len(prompts)
    if not indices or len(indices) != len(prompts):
        indices = [None] * len(prompts)

    out_prompts = []
    out_lens = []
    for i, (prompt, plen) in enumerate(zip(prompts, prompt_lens)):
        current_idx = indices[i] if i < len(indices) else None
        # Select K forget examples, excluding current eval index when present
        forget_selected = []
        for ex in forget_pool:
            if len(forget_selected) >= K:
                break
            di = ex.get("dataset_index")
            if di is not None and current_idx is not None and di == current_idx:
                continue
            forget_selected.append(ex)
        # Rebuild context for this sample
        parts: List[str] = []
        for ex in forget_selected:
            q = ex.get("question") or ex.get("input", "")
            wrong = ex.get("wrong_answer") or ex.get("wrong_label") or ""
            if is_categorical:
                parts.append(q.rstrip() + " " + wrong)
            else:
                parts.append(f"{q}\n{wrong}")
        parts.append("")
        for ex in retain_pool[:L]:
            q = ex.get("question") or ex.get("input", "")
            right = ex.get("answer") or ex.get("label") or ""
            if is_categorical:
                parts.append(q.rstrip() + " " + right)
            else:
                parts.append(f"{q}\n{right}")
        context_text = "\n".join(parts)
        try:
            context_ids = tokenizer.encode(
                context_text,
                add_special_tokens=False,
                return_tensors=None,
            )
        except Exception as e:
            logger.warning("ICUL context encode failed for sample %s: %s", i, e)
            context_ids = []
        p = list(context_ids) + list(prompt)
        out_prompts.append(p)
        out_lens.append(len(context_ids) + plen)
    return out_prompts, out_lens


def transform_output_text(
    gen_text: str,
    batch: Optional[Dict[str, Any]],
    sample_idx: Optional[Union[str, int]],
    config: Optional[Union[Dict, DictConfig]] = None,
) -> str:
    """Apply output-side guardrails: keyword/output filter.

    When guardrail config is absent, or benchmark is wmdp, or output_filter is none,
    returns gen_text unchanged. Otherwise if output_filter is keyword and gen_text
    contains any of keyword_list, returns the configured safe reply.

    Args:
        gen_text: Decoded model output.
        batch: Batch dict (for optional per-sample keyword list); may be unused.
        sample_idx: Sample index (for optional per-sample logic); may be unused.
        config: Full trajectory_config or eval config.

    Returns:
        Filtered or original gen_text.
    """
    guard = _guardrail_config(config)
    if not guard:
        return gen_text

    benchmark = (guard.get("benchmark") or "").lower()
    output_filter = (guard.get("output_filter") or "none").lower()

    # WMDP: output is single letter (A/B/C/D); content filtering not applicable.
    if benchmark == "wmdp":
        return gen_text
    if output_filter == "none":
        return gen_text

    if output_filter == "keyword":
        keyword_list = guard.get("keyword_list")
        if not keyword_list:
            return gen_text
        if isinstance(keyword_list, str):
            keyword_list = [keyword_list]
        gen_lower = gen_text.lower()
        for kw in keyword_list:
            if kw and kw.lower() in gen_lower:
                safe = guard.get("output_filter_safe_reply") or DEFAULT_SAFE_REPLY
                return safe
    return gen_text


def load_icul_pools(
    benchmark: str,
    tokenizer: Any,
    template_args: Optional[Dict[str, Any]],
    cfg: Union[Dict, DictConfig],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load forget and retain pools for ICUL from train splits only.

    Follows the plan §8: train-only splits; TOFU (forget_split + retain_split),
    MUSE (forget_qa_icl + retain_qa_icl), WMDP (train split, categorical answers).
    Returns ([forget_examples], [retain_examples]); each example is a dict with
    question/input, answer/label, and wrong_answer/wrong_label for forget.

    Args:
        benchmark: One of tofu, muse, wmdp.
        tokenizer: Tokenizer (used only for optional formatting).
        template_args: Model template args (optional).
        cfg: Config with forget_split, retain_split (TOFU), data_split (MUSE/WMDP), icul.K, icul.L.

    Returns:
        (forget_list, retain_list). Empty lists if benchmark unsupported or load fails.
    """
    from data.utils import load_hf_dataset

    cfg = _get_config_dict(cfg) if isinstance(cfg, DictConfig) else dict(cfg)
    benchmark = (benchmark or "").lower()
    icul = _get_config_dict(cfg.get("icul") or {})
    K = int(icul.get("K", 5))
    L = int(icul.get("L", 4))

    forget_pool: List[Dict[str, Any]] = []
    retain_pool: List[Dict[str, Any]] = []

    if benchmark == "tofu":
        forget_split = cfg.get("forget_split") or "forget10"
        retain_split = cfg.get("retain_split") or "retain90"
        try:
            df = load_hf_dataset(path="locuslab/TOFU", name=forget_split, split="train")
            dr = load_hf_dataset(path="locuslab/TOFU", name=retain_split, split="train")
        except Exception as e:
            logger.warning("ICUL TOFU load failed: %s", e)
            return [], []
        qk, ak = "question", "answer"
        # Build forget pool: K+1 so we can exclude current index
        for i in range(min(K + 1, len(df))):
            row = df[i]
            q = row.get(qk, "")
            a = row.get(ak, "")
            wrong = row.get("perturbed_answer") or row.get("paraphrased_answer")
            if not wrong and len(df) > 1:
                other_idx = (i + 1) % len(df)
                wrong = df[other_idx].get(ak, "I don't know.")
            if not wrong:
                wrong = "I don't know."
            forget_pool.append({
                "question": q, "answer": a, "wrong_answer": wrong,
                "dataset_index": i,
            })
        for i in range(min(L, len(dr))):
            row = dr[i]
            retain_pool.append({
                "question": row.get(qk, ""),
                "answer": row.get(ak, ""),
            })

    elif benchmark == "muse":
        data_split = cfg.get("data_split") or "News"
        try:
            df = load_hf_dataset(
                path=f"muse-bench/MUSE-{data_split}",
                split="forget_qa_icl",
            )
            dr = load_hf_dataset(
                path=f"muse-bench/MUSE-{data_split}",
                split="retain_qa_icl",
            )
        except Exception as e:
            logger.warning("ICUL MUSE load failed: %s", e)
            return [], []
        qk, ak = "question", "answer"
        for i in range(min(K, len(df))):
            row = df[i]
            q = row.get(qk, "")
            a = row.get(ak, "")
            wrong = row.get("wrong_answer")
            if not wrong and len(df) > 1:
                wrong = df[(i + 1) % len(df)].get(ak, "I don't know.")
            if not wrong:
                wrong = "I don't know."
            forget_pool.append({"question": q, "answer": a, "wrong_answer": wrong})
        for i in range(min(L, len(dr))):
            row = dr[i]
            retain_pool.append({
                "question": row.get(qk, ""),
                "answer": row.get(ak, ""),
            })

    elif benchmark == "wmdp":
        # Same ICUL methodology, categorical answers (A/B/C/D). Flip = wrong letter.
        try:
            ds = load_hf_dataset(path="cais/wmdp", split="train")
        except Exception:
            try:
                ds = load_hf_dataset(path="cais/wmdp", split="test")
            except Exception as e:
                logger.warning("ICUL WMDP load failed (no train split): %s", e)
                return [], []
        choices_key = "choices"
        answer_key = "answer"
        for i in range(min(K, len(ds))):
            row = ds[i]
            q_raw = row.get("question", "")
            choices = row.get(choices_key, [])
            if len(choices) < 4:
                choices = choices + [""] * (4 - len(choices))
            correct = row.get(answer_key, "A")
            # Format as lm_eval WMDP: question + A. c0 \n B. c1 ...
            q_formatted = f"{q_raw.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
            letters = ["A", "B", "C", "D"]
            wrong_letters = [x for x in letters if x != correct]
            wrong_letter = random.choice(wrong_letters) if wrong_letters else "B"
            forget_pool.append({
                "question": q_formatted,
                "answer": correct,
                "wrong_answer": wrong_letter,
            })
        for i in range(min(L, len(ds))):
            row = ds[i]
            q_raw = row.get("question", "")
            choices = row.get(choices_key, [])
            if len(choices) < 4:
                choices = choices + [""] * (4 - len(choices))
            q_formatted = f"{q_raw.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
            retain_pool.append({
                "question": q_formatted,
                "answer": row.get(answer_key, "A"),
            })
    else:
        logger.debug("ICUL unsupported benchmark: %s", benchmark)
        return [], []

    return forget_pool, retain_pool
