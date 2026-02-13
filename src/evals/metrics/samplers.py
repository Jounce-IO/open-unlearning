"""
Samplers for evaluation (e.g. length-sorted batching).
"""

from typing import Any, Iterator


class LengthSortedSampler:
    """
    Yields dataset indices in descending order by sequence length
    (by len(instance[length_key])), so batches have minimal padding.
    """

    def __init__(
        self,
        dataset: Any,
        length_key: str = "input_ids",
        descending: bool = True,
    ):
        self.dataset = dataset
        self.length_key = length_key
        self.descending = descending
        n = len(dataset)
        lengths = []
        for i in range(n):
            item = dataset[i]
            seq = item[self.length_key]
            lengths.append(len(seq) if hasattr(seq, "__len__") else 0)
        self._sorted_indices = sorted(
            range(n), key=lambda i: lengths[i], reverse=self.descending
        )

    def __iter__(self) -> Iterator[int]:
        yield from self._sorted_indices

    def __len__(self) -> int:
        return len(self.dataset)
