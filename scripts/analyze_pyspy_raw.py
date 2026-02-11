#!/usr/bin/env python3
"""
Analyze py-spy raw profile output for eval hotspots.

Reads a py-spy raw file (format: one stack per line; trailing integer = sample
count for that stack). Sums total samples and per-hotspot sample counts for:
- avg_losses / evaluate_probability (utils.py, trajectory_metrics.py)
- eval_text_similarity / tokenizer / batch_decode
- rouge_score / rouge_scorer / nltk stemmer

Usage:
    python scripts/analyze_pyspy_raw.py path/to/pyspy.raw

For validation only; no change to production code.
"""

import re
import sys
from pathlib import Path


def parse_line(line: str) -> tuple[str, int]:
    """Extract stack text and trailing sample count. Returns (stack, count)."""
    line = line.rstrip("\n")
    # Trailing format: space + integer
    match = re.search(r" (\d+)$", line)
    if not match:
        return line, 0
    count = int(match.group(1))
    stack = line[: match.start()].rstrip()
    return stack, count


# Substrings that identify each hotspot (line must contain one of these).
HOTSPOT_AVG_LOSSES = [
    "evaluate_probability",
    "utils.py:128",
    "utils.py:129",
    "utils.py:130",
    "utils.py:131",
    "utils.py:132",
    "utils.py:133",
    "trajectory_metrics.py:95",
    "trajectory_metrics.py:96",
    "trajectory_metrics.py:97",
]
HOTSPOT_TOKENIZER = [
    "eval_text_similarity",
    "batch_decode",
    "tokenization_utils_base.py",
]
HOTSPOT_ROUGE = [
    "rouge_score",
    "rouge_scorer",
    "nltk/stem",
    "porter.py",
    "eval_rouge_recall_batch",
]


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_pyspy_raw.py <pyspy.raw>", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    total_samples = 0
    hotspot_avg_losses = 0
    hotspot_tokenizer = 0
    hotspot_rouge = 0

    with path.open() as f:
        for line in f:
            stack, count = parse_line(line)
            if count <= 0:
                continue
            total_samples += count
            if any(s in stack for s in HOTSPOT_AVG_LOSSES):
                hotspot_avg_losses += count
            if any(s in stack for s in HOTSPOT_TOKENIZER):
                hotspot_tokenizer += count
            if any(s in stack for s in HOTSPOT_ROUGE):
                hotspot_rouge += count

    print(f"Total samples: {total_samples}")
    print()
    print("Per-hotspot sample sum and percentage:")
    for name, val in [
        ("avg_losses / evaluate_probability", hotspot_avg_losses),
        ("eval_text_similarity / tokenizer", hotspot_tokenizer),
        ("rouge_score", hotspot_rouge),
    ]:
        pct = (100.0 * val / total_samples) if total_samples else 0
        print(f"  {name}: {val} ({pct:.1f}%)")
    combined = hotspot_avg_losses + hotspot_tokenizer + hotspot_rouge
    # Lines can match multiple hotspots; combined may overcount
    print()
    print("(Note: a stack line can match multiple hotspots; percentages are independent.)")


if __name__ == "__main__":
    main()
