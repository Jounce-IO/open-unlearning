"""Phase C investigation logger must work with default ``trajectory_logits_storage: dense_r``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

OU_ROOT = Path(__file__).resolve().parents[1]
OU_SRC = OU_ROOT / "src"
if str(OU_SRC) not in sys.path:
    sys.path.insert(0, str(OU_SRC))

from evals.metrics.trajectory_probability_hypothesis import (  # noqa: E402
    TrajectoryProbabilityHypothesisLogger,
    _sample_traj_state,
)
from evals.metrics.trajectory_metrics import _get_logits_at_step  # noqa: E402
from evals.metrics.trajectory_utils import stack_logits_history  # noqa: E402


def test_dense_r_and_list_history_logits_agree_at_steps() -> None:
    B, V, L, S = 2, 12, 5, 4
    lh = [torch.randn(B, L, V) for _ in range(S)]
    R = stack_logits_history(lh)
    F = torch.tensor([[0, 1, 2, 3, 4], [1, 1, 2, 2, 3]], dtype=torch.long)
    for b in range(B):
        st_lh = _sample_traj_state(lh_batch=lh, R_batch=None, b=b, F_b=F[b], S=S, L=L)
        st_r = _sample_traj_state(lh_batch=None, R_batch=R, b=b, F_b=F[b], S=S, L=L)
        for step in range(S):
            for traj in ("steps", "fixation_start"):
                a = _get_logits_at_step(st_lh, traj, step)
                r = _get_logits_at_step(st_r, traj, step)
                assert torch.allclose(a, r, atol=1e-5, rtol=1e-4)


def test_log_batch_dense_r_writes_step_samples(tmp_path: Path) -> None:
    B, V, L, S = 1, 16, 4, 3
    pl = 1
    T = pl + L
    R = torch.randn(B, V, L, S)
    F = torch.zeros(B, L, dtype=torch.long)
    snaps = [torch.randint(1, 10, (B, T), dtype=torch.long) for _ in range(S)]
    props = [torch.randint(1, 10, (B, T), dtype=torch.long) for _ in range(S)]
    labels = torch.full((B, T), -100, dtype=torch.long)
    labels[:, pl : pl + L] = torch.tensor([5, 6, 7, 8], dtype=torch.long)
    input_ids = labels.clone()
    input_ids[input_ids == -100] = 0

    tok = SimpleNamespace(mask_token_id=0, eos_token_id=1)
    tok.decode = lambda ids, **_: " ".join(str(int(x)) for x in ids)

    logger = TrajectoryProbabilityHypothesisLogger(str(tmp_path), rank=0)
    logger.log_batch(
        batch_idx=0,
        model=None,
        tokenizer=tok,
        trajectory_config={
            "logit_alignment": "causal",
            "probability_hypothesis_investigation_trajs": ["steps"],
            "probability_hypothesis_investigation_views": ["full"],
        },
        kwargs={"trajectory_pass_id": "retain__guided_prob"},
        steps_to_use=[0, S - 1],
        trajectory_names=["steps"],
        include_views=["full"],
        lh_batch=None,
        R_batch=R,
        F=F,
        S=S,
        L=L,
        B=B,
        seq_snapshots_batch=snaps,
        prop_snapshots_batch=props,
        labels=labels,
        input_ids=input_ids,
        batch={},
        indices=[0],
        prompt_starts=[pl],
        prompt_lens=[pl],
        effective_lengths=[L],
        prompt_only_input_ids=True,
        gen_labels_per_sample=[labels[0]],
        evaluation_mode="guided_native",
    )
    logger.flush_summary()

    records = [json.loads(line) for line in logger.path.read_text().splitlines() if line.strip()]
    types = [r["record_type"] for r in records]
    assert "batch_skip" not in types
    step_samples = [r for r in records if r["record_type"] == "step_sample"]
    assert len(step_samples) == 2
    assert step_samples[0]["prob_pre_commit_packed"]["prob"] is not None
    summary = next(r for r in records if r["record_type"] == "batch_summary")
    assert summary["logits_storage"] == "dense_r"
