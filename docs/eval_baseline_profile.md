# Evaluation baseline profile (py-spy)

Baseline sample counts from py-spy raw profile for the three evaluation hotspots. Use these numbers to compare after implementing optimizations (re-run the same eval, then `python scripts/analyze_pyspy_raw.py <new.raw>`).

**Profile source:** `reports/main/pyspy-20s-eval-20260211-103408/pyspy-20s.raw` (20s eval, 20 samples).

## Totals

- **Total samples:** 13,387

## Per-hotspot (current / baseline)

- **avg_losses / evaluate_probability** (utils.py L128–133, trajectory_metrics.py L95–97): 3,536 samples (26.4%)
- **eval_text_similarity / tokenizer** (utils.py batch_decode, tokenizer): 8,600 samples (64.2%)
- **rouge_score** (utils.py eval_rouge_recall_batch, rouge_scorer, nltk stemmer): 2,957 samples (22.1%)

(Percentages are of total; a stack line can match multiple hotspots.)

## How to reproduce

```bash
python scripts/analyze_pyspy_raw.py path/to/pyspy-20s.raw
```

Optimizations target: single GPU→CPU transfer for avg_losses, batch_decode-only tokenizer usage, one RougeScorer per batch and optional `use_stemmer=False`.
