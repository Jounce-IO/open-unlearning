"""
Pytest configuration and shared fixtures for trajectory metrics tests.
"""

import pytest
import torch


@pytest.fixture
def device():
    """Return device for tests (CPU for consistency)."""
    return torch.device("cpu")


@pytest.fixture
def vocab_size():
    """Return vocabulary size for tests."""
    return 1000


@pytest.fixture
def sequence_length():
    """Return sequence length for tests."""
    return 20


@pytest.fixture
def num_steps():
    """Return number of diffusion steps for tests."""
    return 8


@pytest.fixture
def sample_logits_history(vocab_size, sequence_length, num_steps):
    """Create sample logits_history for testing."""
    return [torch.randn(1, sequence_length, vocab_size) for _ in range(num_steps)]


@pytest.fixture
def sample_fixation_steps(sequence_length, num_steps):
    """Create sample fixation_steps for testing."""
    return torch.randint(0, num_steps, (1, sequence_length))


@pytest.fixture
def sample_trajectory_tensor(vocab_size, sequence_length, num_steps):
    """Create sample trajectory tensor [V, L, S]."""
    return torch.randn(vocab_size, sequence_length, num_steps)
