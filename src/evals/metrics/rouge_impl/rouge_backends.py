"""Registry of ROUGE backends for tests and benchmark."""

from evals.metrics.rouge_impl.backends import (
    ROUGE_BACKENDS,
    get_backend,
    get_all_backends,
)

__all__ = ["ROUGE_BACKENDS", "get_backend", "get_all_backends"]
