"""
ROUGE-L: LCS-based recall, precision, F1.
"""


def _fmeasure(precision: float, recall: float) -> float:
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def lcs_length_full_table(ref: list, pred: list) -> int:
    """LCS length via full 2D DP table (matches rouge_score)."""
    rows, cols = len(ref), len(pred)
    if rows == 0 or cols == 0:
        return 0
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if ref[i - 1] == pred[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[rows][cols]


def lcs_length_two_row(ref: list, pred: list) -> int:
    """LCS length via two-row DP (O(len(pred)) space)."""
    rows, cols = len(ref), len(pred)
    if rows == 0 or cols == 0:
        return 0
    prev = [0] * (cols + 1)
    curr = [0] * (cols + 1)
    for i in range(1, rows + 1):
        curr[0] = 0
        for j in range(1, cols + 1):
            if ref[i - 1] == pred[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[cols]


def rouge_l_from_tokens(
    ref_tokens: list,
    pred_tokens: list,
    lcs_len: int,
) -> tuple:
    """
    Compute ROUGE-L recall, precision, F1 given LCS length.

    Returns:
        (recall, precision, fmeasure)
    """
    if not ref_tokens or not pred_tokens:
        return (0.0, 0.0, 0.0)
    recall = lcs_len / len(ref_tokens)
    precision = lcs_len / len(pred_tokens)
    f1 = _fmeasure(precision, recall)
    return (recall, precision, f1)
