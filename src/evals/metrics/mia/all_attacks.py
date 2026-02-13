"""
Enum class for attacks. Also contains the base attack class.
"""

from enum import Enum
from typing import Any, Dict, List

from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


# Attack definitions
class AllAttacks(str, Enum):
    LOSS = "loss"
    REFERENCE_BASED = "ref"
    ZLIB = "zlib"
    MIN_K = "min_k"
    MIN_K_PLUS_PLUS = "min_k++"
    GRADNORM = "gradnorm"
    RECALL = "recall"


# Base attack class
class Attack:
    def __init__(self, model, data, collator, batch_size, **kwargs):
        """Initialize attack with model and create dataloader."""
        self.model = model
        self.dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
        self.setup(**kwargs)

    def setup(self, **kwargs):
        """Setup attack-specific parameters."""
        pass

    def compute_batch_values(self, batch):
        """Process a batch through model to get needed statistics."""
        raise NotImplementedError

    def compute_score(self, sample_stats):
        """Compute MIA score for a single sample."""
        raise NotImplementedError

    def compute_batch_values_from_logits(self, batch: Dict[str, Any], logits: Any) -> List[Any]:
        """Get batch values (e.g. log-probs) from precomputed logits instead of calling model.
        Used for streaming MIA: process one batch at a time without storing all logits."""
        raise NotImplementedError(
            "Streaming not implemented for this attack; override compute_batch_values_from_logits."
        )

    def process_batch(
        self, batch: Dict[str, Any], batch_values: List[Any]
    ) -> Dict[str, Dict[str, float]]:
        """Turn one batch and its precomputed batch values into per-sample scores.
        Returns {str(idx): {"score": float}} for indices in batch["index"]."""
        indices = batch["index"]
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        elif hasattr(indices, "cpu"):
            indices = indices.cpu().numpy().tolist()
        indices = [indices] if isinstance(indices, (int, np.integer)) else indices
        scores = [self.compute_score(values) for values in batch_values]
        return {str(idx): {"score": float(score)} for idx, score in zip(indices, scores)}

    def attack(self):
        """Run full MIA attack."""
        all_scores = []
        all_indices = []

        for batch in tqdm(self.dataloader, total=len(self.dataloader)):
            indices = batch["index"].cpu().numpy().tolist()
            batch_values = self.compute_batch_values(batch)
            batch_scores = self.process_batch(batch, batch_values)
            for idx in indices:
                all_indices.append(idx)
                all_scores.append(batch_scores[str(idx)]["score"])

        scores_by_index = {
            str(idx): {"score": float(score)}
            for idx, score in zip(all_indices, all_scores)
        }

        return {
            "agg_value": float(np.mean(all_scores)),
            "value_by_index": scores_by_index,
        }
