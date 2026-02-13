"""
ROUGE-1: unigram overlap. Recall, precision, F1.
"""

from collections import Counter


def _fmeasure(precision: float, recall: float) -> float:
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def rouge1_from_tokens(
    ref_tokens: list[str],
    pred_tokens: list[str],
) -> tuple[float, float, float]:
    """
    Compute ROUGE-1 recall, precision, F1 from token lists.

    Returns:
        (recall, precision, fmeasure)
    """
    ref_ngrams = Counter(tuple(ref_tokens[i : i + 1]) for i in range(len(ref_tokens)))
    pred_ngrams = Counter(tuple(pred_tokens[i : i + 1]) for i in range(len(pred_tokens)))
    intersection = 0
    for ngram, count in ref_ngrams.items():
        intersection += min(count, pred_ngrams.get(ngram, 0))
    ref_total = sum(ref_ngrams.values()) or 1
    pred_total = sum(pred_ngrams.values()) or 1
    recall = intersection / ref_total
    precision = intersection / pred_total
    f1 = _fmeasure(precision, recall)
    return (recall, precision, f1)
