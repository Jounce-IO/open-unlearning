"""
ROUGE backends: each implements (gen_outputs, ground_truths, use_stemmer, scorer) -> List[Dict].
"""

from typing import Any, Callable, List, Optional

from evals.metrics.rouge_impl import tokenizer as tok
from evals.metrics.rouge_impl import rouge1
from evals.metrics.rouge_impl import rouge_l

# Optional numpy for numpy_lcs backend
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Optional torch for GPU backends
try:
    import torch
    from evals.metrics.rouge_impl import rouge_gpu
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _one_pair(
    gen: str,
    gt: str,
    use_stemmer: bool,
    tokenize_fn: Callable[[str, bool], list],
    lcs_fn: Callable[[list, list], int],
) -> dict:
    """Compute ROUGE for one (gen, gt) pair using given tokenize and LCS functions."""
    ref_tokens = tokenize_fn(gt, use_stemmer)
    pred_tokens = tokenize_fn(gen, use_stemmer)
    r1_recall, _r1_prec, _r1_f1 = rouge1.rouge1_from_tokens(ref_tokens, pred_tokens)
    lcs_len = lcs_fn(ref_tokens, pred_tokens)
    _rl_recall, _rl_prec, rl_f1 = rouge_l.rouge_l_from_tokens(
        ref_tokens, pred_tokens, lcs_len
    )
    rl_recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
    return {
        "rouge1_recall": r1_recall,
        "rougeL_f1": rl_f1,
        "rougeL_recall": rl_recall,
    }


def backend_baseline(
    gen_outputs: List[str],
    ground_truths: List[str],
    use_stemmer: bool = True,
    scorer: Any = None,
) -> List[dict]:
    """Current implementation (rouge_score)."""
    from evals.metrics.utils import eval_rouge_recall_batch
    return eval_rouge_recall_batch(
        gen_outputs, ground_truths, use_stemmer=use_stemmer, scorer=scorer
    )


def backend_minimal_python(
    gen_outputs: List[str],
    ground_truths: List[str],
    use_stemmer: bool = True,
    scorer: Any = None,
) -> List[dict]:
    """Pure Python clone: shared tokenizer + full 2D LCS."""
    results = []
    for gen, gt in zip(gen_outputs, ground_truths):
        out = _one_pair(
            gen, gt, use_stemmer,
            tokenize_fn=tok.tokenize,
            lcs_fn=rouge_l.lcs_length_full_table,
        )
        results.append(out)
    return results


def backend_two_row_lcs(
    gen_outputs: List[str],
    ground_truths: List[str],
    use_stemmer: bool = True,
    scorer: Any = None,
) -> List[dict]:
    """Same as minimal_python but LCS with two-row DP."""
    results = []
    for gen, gt in zip(gen_outputs, ground_truths):
        out = _one_pair(
            gen, gt, use_stemmer,
            tokenize_fn=tok.tokenize,
            lcs_fn=rouge_l.lcs_length_two_row,
        )
        results.append(out)
    return results


def backend_batch_cached(
    gen_outputs: List[str],
    ground_truths: List[str],
    use_stemmer: bool = True,
    scorer: Any = None,
) -> List[dict]:
    """Tokenize each unique string once; same math as minimal_python."""
    cache = {}

    def cached_tokenize(text: str, use_stem: bool) -> list:
        key = (text, use_stem)
        if key not in cache:
            cache[key] = tok.tokenize(text, use_stem)
        return cache[key]

    results = []
    for gen, gt in zip(gen_outputs, ground_truths):
        ref_tokens = cached_tokenize(gt, use_stemmer)
        pred_tokens = cached_tokenize(gen, use_stemmer)
        r1_recall, _, _ = rouge1.rouge1_from_tokens(ref_tokens, pred_tokens)
        lcs_len = rouge_l.lcs_length_full_table(ref_tokens, pred_tokens)
        _, _, rl_f1 = rouge_l.rouge_l_from_tokens(
            ref_tokens, pred_tokens, lcs_len
        )
        rl_recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
        results.append({
            "rouge1_recall": r1_recall,
            "rougeL_f1": rl_f1,
            "rougeL_recall": rl_recall,
        })
    return results


def _lcs_length_numpy(ref: list, pred: list) -> int:
    """LCS length via NumPy 2D array (same recurrence)."""
    if not ref or not pred:
        return 0
    rows, cols = len(ref) + 1, len(pred) + 1
    table = np.zeros((rows, cols), dtype=np.int64)
    for i in range(1, rows):
        for j in range(1, cols):
            if ref[i - 1] == pred[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return int(table[rows - 1, cols - 1])


def backend_numpy_lcs(
    gen_outputs: List[str],
    ground_truths: List[str],
    use_stemmer: bool = True,
    scorer: Any = None,
) -> List[dict]:
    """Same tokenization; ROUGE-L LCS via NumPy."""
    if not HAS_NUMPY:
        raise RuntimeError("numpy required for numpy_lcs backend")
    results = []
    for gen, gt in zip(gen_outputs, ground_truths):
        out = _one_pair(
            gen, gt, use_stemmer,
            tokenize_fn=tok.tokenize,
            lcs_fn=_lcs_length_numpy,
        )
        results.append(out)
    return results


def backend_no_stemmer(
    gen_outputs: List[str],
    ground_truths: List[str],
    use_stemmer: bool = True,
    scorer: Any = None,
) -> List[dict]:
    """Same pipeline with use_stemmer=False (benchmark-only; not drop-in)."""
    # Ignore use_stemmer and always use False for this backend
    results = []
    for gen, gt in zip(gen_outputs, ground_truths):
        out = _one_pair(
            gen, gt, use_stemmer=False,
            tokenize_fn=tok.tokenize,
            lcs_fn=rouge_l.lcs_length_full_table,
        )
        results.append(out)
    return results


def _tokenize_and_encode_batch(
    gen_outputs: List[str],
    ground_truths: List[str],
    use_stemmer: bool,
) -> tuple:
    """Tokenize with cache; return (ref_token_lists, pred_token_lists, token_to_id, vocab_size)."""
    cache = {}
    def cached_tokenize(text: str, use_stem: bool) -> list:
        key = (text, use_stem)
        if key not in cache:
            cache[key] = tok.tokenize(text, use_stem)
        return cache[key]
    ref_token_lists = [cached_tokenize(gt, use_stemmer) for gt in ground_truths]
    pred_token_lists = [cached_tokenize(gen, use_stemmer) for gen in gen_outputs]
    token_to_id, vocab_size = rouge_gpu.build_vocab_from_token_lists(ref_token_lists, pred_token_lists)
    return ref_token_lists, pred_token_lists, token_to_id, vocab_size


def backend_gpu_torch_rouge1_lcs_cpu(
    gen_outputs: List[str],
    ground_truths: List[str],
    use_stemmer: bool = True,
    scorer: Any = None,
) -> List[dict]:
    """Hybrid: ROUGE-1 on GPU, LCS on CPU. Requires CUDA."""
    if not HAS_TORCH or not torch.cuda.is_available():
        raise RuntimeError("gpu_torch_rouge1_lcs_cpu requires PyTorch with CUDA")
    device = torch.device("cuda")
    ref_token_lists, pred_token_lists, token_to_id, vocab_size = _tokenize_and_encode_batch(
        gen_outputs, ground_truths, use_stemmer
    )
    pad_id = vocab_size
    ref_ids, len_ref = rouge_gpu.token_lists_to_padded_ids(
        ref_token_lists, token_to_id, pad_id, device
    )
    pred_ids, len_pred = rouge_gpu.token_lists_to_padded_ids(
        pred_token_lists, token_to_id, pad_id, device
    )
    results = []
    for b in range(len(gen_outputs)):
        r1_recall, _, _ = rouge_gpu.rouge1_from_token_ids_gpu(
            ref_ids[b], pred_ids[b],
            len_ref[b].item(), len_pred[b].item(),
            vocab_size, pad_id, device,
        )
        lcs_len = rouge_l.lcs_length_full_table(
            ref_token_lists[b], pred_token_lists[b]
        )
        _, _, rl_f1 = rouge_l.rouge_l_from_tokens(
            ref_token_lists[b], pred_token_lists[b], lcs_len
        )
        rl_recall = lcs_len / len(ref_token_lists[b]) if ref_token_lists[b] else 0.0
        results.append({
            "rouge1_recall": r1_recall,
            "rougeL_f1": rl_f1,
            "rougeL_recall": rl_recall,
        })
    return results


def backend_gpu_torch_batch(
    gen_outputs: List[str],
    ground_truths: List[str],
    use_stemmer: bool = True,
    scorer: Any = None,
) -> List[dict]:
    """Full GPU: batched ROUGE-1 and batched LCS on PyTorch. Requires CUDA."""
    if not HAS_TORCH or not torch.cuda.is_available():
        raise RuntimeError("gpu_torch_batch requires PyTorch with CUDA")
    device = torch.device("cuda")
    ref_token_lists, pred_token_lists, token_to_id, vocab_size = _tokenize_and_encode_batch(
        gen_outputs, ground_truths, use_stemmer
    )
    pad_id = vocab_size
    ref_ids, len_ref = rouge_gpu.token_lists_to_padded_ids(
        ref_token_lists, token_to_id, pad_id, device
    )
    pred_ids, len_pred = rouge_gpu.token_lists_to_padded_ids(
        pred_token_lists, token_to_id, pad_id, device
    )
    r1_results = rouge_gpu.rouge1_batch_gpu(
        ref_ids, pred_ids, len_ref, len_pred, vocab_size, device
    )
    lcs_lengths = rouge_gpu.lcs_lengths_batched_gpu(
        ref_ids, pred_ids, len_ref, len_pred, device
    )
    results = []
    for b in range(len(gen_outputs)):
        r1_recall, _, _ = r1_results[b]
        lcs_len = lcs_lengths[b].item()
        ref_len = len(ref_token_lists[b])
        pred_len = len(pred_token_lists[b])
        if ref_len == 0 or pred_len == 0:
            rl_recall, rl_f1 = 0.0, 0.0
        else:
            rl_recall = lcs_len / ref_len
            rl_precision = lcs_len / pred_len
            rl_f1 = rouge_gpu._fmeasure(rl_precision, rl_recall)
        results.append({
            "rouge1_recall": r1_recall,
            "rougeL_f1": rl_f1,
            "rougeL_recall": rl_recall,
        })
    return results


def _batch_cached_chunk(args: tuple) -> List[dict]:
    """Run batch_cached logic on a chunk (for multiprocess)."""
    gen_chunk, gt_chunk, use_stemmer = args
    return backend_batch_cached(gen_chunk, gt_chunk, use_stemmer=use_stemmer, scorer=None)


def backend_multiprocess_batch(
    gen_outputs: List[str],
    ground_truths: List[str],
    use_stemmer: bool = True,
    scorer: Any = None,
) -> List[dict]:
    """Same math as batch_cached but chunks run in parallel (multiprocessing.Pool)."""
    import multiprocessing
    K = min(multiprocessing.cpu_count(), 8, max(1, len(gen_outputs) // 4))
    if K <= 1 or len(gen_outputs) < 4:
        return backend_batch_cached(gen_outputs, ground_truths, use_stemmer=use_stemmer, scorer=scorer)
    chunk_size = (len(gen_outputs) + K - 1) // K
    chunks = []
    for i in range(K):
        start = i * chunk_size
        end = min(start + chunk_size, len(gen_outputs))
        if start >= end:
            continue
        chunks.append((
            gen_outputs[start:end],
            ground_truths[start:end],
            use_stemmer,
        ))
    with multiprocessing.Pool(processes=K) as pool:
        chunk_results = pool.map(_batch_cached_chunk, chunks)
    results = []
    for cr in chunk_results:
        results.extend(cr)
    return results


def backend_fused_cpu(
    gen_outputs: List[str],
    ground_truths: List[str],
    use_stemmer: bool = True,
    scorer: Any = None,
) -> List[dict]:
    """Tokenize with cache; single tight loop inlining ROUGE-1 and two-row LCS (no per-pair function calls)."""
    cache = {}
    def cached_tokenize(text: str, use_stem: bool) -> list:
        key = (text, use_stem)
        if key not in cache:
            cache[key] = tok.tokenize(text, use_stem)
        return cache[key]
    results = []
    for gen, gt in zip(gen_outputs, ground_truths):
        ref_tokens = cached_tokenize(gt, use_stemmer)
        pred_tokens = cached_tokenize(gen, use_stemmer)
        ref_ngrams = __import__("collections").Counter(
            tuple(ref_tokens[i : i + 1]) for i in range(len(ref_tokens))
        )
        pred_ngrams = __import__("collections").Counter(
            tuple(pred_tokens[i : i + 1]) for i in range(len(pred_tokens))
        )
        intersection = sum(
            min(ref_ngrams[n], pred_ngrams.get(n, 0)) for n in ref_ngrams
        )
        ref_total = sum(ref_ngrams.values()) or 1
        pred_total = sum(pred_ngrams.values()) or 1
        r1_recall = intersection / ref_total
        r1_precision = intersection / pred_total
        r1_f1 = 2 * r1_precision * r1_recall / (r1_precision + r1_recall) if (r1_precision + r1_recall) > 0 else 0.0
        rows, cols = len(ref_tokens), len(pred_tokens)
        if rows == 0 or cols == 0:
            lcs_len = 0
        else:
            prev = [0] * (cols + 1)
            curr = [0] * (cols + 1)
            for i in range(1, rows + 1):
                curr[0] = 0
                for j in range(1, cols + 1):
                    if ref_tokens[i - 1] == pred_tokens[j - 1]:
                        curr[j] = prev[j - 1] + 1
                    else:
                        curr[j] = max(prev[j], curr[j - 1])
                prev, curr = curr, prev
            lcs_len = prev[cols]
        rl_recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
        rl_precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
        rl_f1 = 2 * rl_precision * rl_recall / (rl_precision + rl_recall) if (rl_precision + rl_recall) > 0 else 0.0
        results.append({
            "rouge1_recall": r1_recall,
            "rougeL_f1": rl_f1,
            "rougeL_recall": rl_recall,
        })
    return results


# Registry for benchmark and tests
ROUGE_BACKENDS: List[tuple[str, Callable]] = [
    ("baseline", backend_baseline),
    ("minimal_python", backend_minimal_python),
    ("two_row_lcs", backend_two_row_lcs),
    ("batch_cached", backend_batch_cached),
    ("numpy_lcs", backend_numpy_lcs),
    ("no_stemmer", backend_no_stemmer),
    ("gpu_torch_rouge1_lcs_cpu", backend_gpu_torch_rouge1_lcs_cpu),
    ("gpu_torch_batch", backend_gpu_torch_batch),
    ("multiprocess_batch", backend_multiprocess_batch),
    ("fused_cpu", backend_fused_cpu),
]


def get_backend(name: str) -> Optional[Callable]:
    """Return backend function by name, or None if not found."""
    for n, fn in ROUGE_BACKENDS:
        if n == name:
            return fn
    return None


def get_all_backends() -> List[tuple[str, Callable]]:
    """Return list of (name, backend_fn)."""
    return list(ROUGE_BACKENDS)
