"""
Min-k % Prob Attack: https://arxiv.org/pdf/2310.16789.pdf
"""

import numpy as np
import torch
from evals.metrics.mia.all_attacks import Attack
from evals.metrics.utils import tokenwise_logprobs, tokenwise_logprobs_from_logits


class MinKProbAttack(Attack):
    def setup(self, k=0.2, **kwargs):
        self.k = k

    def compute_batch_values(self, batch):
        """Get token-wise log probabilities for the batch."""
        return tokenwise_logprobs(self.model, batch, grad=False)

    def compute_batch_values_from_logits(self, batch, logits):
        """Get token-wise log probabilities from precomputed logits (no model call)."""
        return tokenwise_logprobs_from_logits(batch, logits, return_labels=False)

    def compute_batch_values_from_per_position_scores(self, batch, per_position_scores):
        """Get batch values from precomputed per-position probability scores (e.g. from step-wise score provider).

        per_position_scores: list of list of float, one list per sample (probabilities at each position).
        Returns same format as compute_batch_values: list of 1D tensors of log probs (-log(score)).
        """
        log_probs_batch = []
        for scores in per_position_scores:
            if not scores:
                log_probs_batch.append(torch.tensor([], dtype=torch.float64))
                continue
            arr = np.array(scores, dtype=np.float64)
            lp = -np.log(np.clip(arr, 1e-12, 1.0))
            log_probs_batch.append(torch.from_numpy(lp))
        return log_probs_batch

    def compute_score(self, sample_stats):
        """Score single sample using min-k negative log probs scores attack."""
        lp = sample_stats.float().cpu().numpy()
        if lp.size == 0:
            return 0

        num_k = max(1, int(len(lp) * self.k))
        sorted_vals = np.sort(lp)
        return -np.mean(sorted_vals[:num_k])
