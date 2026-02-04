"""
Minimal mock run of trajectory_metrics to trigger debug instrumentation without loading model/dataset.
Use this to verify the on-demand code path and log shapes locally in a few seconds (no GPU needed).

Run from dllm repo root:
  PYTHONPATH=open-unlearning/src uv run python open-unlearning/scripts/debug_trajectory_metrics_mock.py

Logs go to /workspaces/dllm/.cursor/debug.log and to stderr ([DEBUG] lines). For GPU memory
analysis you still need a real eval run (K8s job or full local eval with GPU).
"""
import os
import sys
import torch
from unittest.mock import Mock
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

# Tiny shapes: B=2, L=4, S=3, V=100 (no GPU needed; CPU is fine)
B, L, full_len, S, V = 2, 4, 8, 3, 100
max_prompt_len = full_len - L  # 4


def main():
    # Mock sampler: returns logits_history list of [B, full_len, V], fixation_steps [B, full_len]
    torch.manual_seed(42)
    logits_history = [torch.randn(B, full_len, V) for _ in range(S)]
    fixation_steps = torch.randint(0, S, (B, full_len))

    class SamplerOutput:
        __slots__ = ("logits_history", "fixation_steps")
        def __init__(self, logits_history, fixation_steps):
            self.logits_history = logits_history
            self.fixation_steps = fixation_steps

    sampler = Mock()
    sampler.sample.return_value = SamplerOutput(
        logits_history=logits_history,
        fixation_steps=fixation_steps,
    )

    model = Mock()
    model.sampler = sampler

    # Minimal tokenizer mock (probability metric path may not use it; trajectory_metrics requires it)
    tokenizer = Mock()
    tokenizer.decode = Mock(return_value="")
    tokenizer.encode = Mock(return_value=torch.zeros(1, L, dtype=torch.long))
    tokenizer.eos_token_id = 0

    IGNORE_INDEX = -100

    class TinyDataset:
        def __len__(self):
            return B

        def __getitem__(self, idx):
            input_ids = torch.randint(0, V, (full_len,))
            labels = torch.full((full_len,), IGNORE_INDEX, dtype=torch.long)
            labels[max_prompt_len:] = torch.randint(0, V, (L,))
            return {"input_ids": input_ids, "labels": labels}

    def collate(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "index": torch.tensor(list(range(len(batch)))),
        }

    from torch.utils.data import DataLoader
    from evals.metrics.trajectory_metrics import trajectory_metrics

    dataloader = DataLoader(
        TinyDataset(),
        batch_size=B,
        collate_fn=collate,
    )
    data = TinyDataset()

    # Call underlying metric fn directly (bypass UnlearningMetric.evaluate(model, metric_name, cache))
    metric_fn = trajectory_metrics._metric_fn
    result = metric_fn(
        model,
        metrics=["probability"],
        trajectory_config={"sampler_kwargs": {}},
        data=data,
        collators=collate,
        batch_size=B,
        tokenizer=tokenizer,
    )
    print("trajectory_metrics result keys:", result.keys() if result else None)
    if result and "agg_value" in result:
        print("agg_value trajectory keys:", list(result["agg_value"].keys()))
    return result


if __name__ == "__main__":
    main()
