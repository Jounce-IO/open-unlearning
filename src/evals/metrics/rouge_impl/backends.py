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


# Registry for benchmark and tests
ROUGE_BACKENDS: List[tuple[str, Callable]] = [
    ("baseline", backend_baseline),
    ("minimal_python", backend_minimal_python),
    ("two_row_lcs", backend_two_row_lcs),
    ("batch_cached", backend_batch_cached),
    ("numpy_lcs", backend_numpy_lcs),
    ("no_stemmer", backend_no_stemmer),
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
