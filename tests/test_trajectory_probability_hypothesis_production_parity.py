"""Production parity: retain__guided_prob defaults (dense_r, B=2, full investigation grid)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

OU_ROOT = Path(__file__).resolve().parents[1]
OU_SRC = OU_ROOT / "src"
if str(OU_SRC) not in sys.path:
    sys.path.insert(0, str(OU_SRC))

from evals.metrics.trajectory_probability_hypothesis import (  # noqa: E402
    TrajectoryProbabilityHypothesisLogger,
)

BATCH_SIZE = 2
TRAJS = ("steps", "fixation_start")
VIEWS = ("full", "eos")


def _run_production_batch(tmp_path: Path) -> list[dict]:
    from types import SimpleNamespace

    B, S, V, L = BATCH_SIZE, 5, 20, 5
    pl = 2
    T = pl + L + 3
    R = torch.randn(B, V, L, S)
    F = torch.zeros(B, L, dtype=torch.long)
    snaps = [torch.randint(1, 12, (B, T), dtype=torch.long) for _ in range(S)]
    props = [torch.randint(1, 12, (B, T), dtype=torch.long) for _ in range(S)]
    labels = torch.full((B, T), -100, dtype=torch.long)
    labels[:, pl : pl + L] = torch.tensor([5, 6, 7, 8, 9])
    input_ids = labels.clone()
    input_ids[input_ids == -100] = 0
    tok = SimpleNamespace(mask_token_id=0, eos_token_id=1)
    tok.decode = lambda ids, **_: "x"

    logger = TrajectoryProbabilityHypothesisLogger(str(tmp_path), rank=0)
    steps_to_use = [0, 2, S - 1]
    logger.log_batch(
        batch_idx=0,
        model=None,
        tokenizer=tok,
        trajectory_config={
            "probability_hypothesis_investigation_trajs": list(TRAJS),
            "probability_hypothesis_investigation_views": list(VIEWS),
        },
        kwargs={"trajectory_pass_id": "retain__guided_prob"},
        steps_to_use=steps_to_use,
        trajectory_names=list(TRAJS),
        include_views=list(VIEWS),
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
        indices=[10, 11],
        prompt_starts=[pl, pl],
        prompt_lens=[pl, pl],
        effective_lengths=[L, L],
        prompt_only_input_ids=True,
        gen_labels_per_sample=[labels[0], labels[1]],
        evaluation_mode="guided_native",
    )
    logger.flush_summary()
    return [
        json.loads(line)
        for line in logger.path.read_text().splitlines()
        if line.strip()
    ]


def test_production_parity_no_batch_skip_and_full_grid(tmp_path: Path) -> None:
    records = _run_production_batch(tmp_path)
    types = [r["record_type"] for r in records]
    assert "batch_skip" not in types
    steps = [r for r in records if r["record_type"] == "step_sample"]
    n_steps = 3
    expected = BATCH_SIZE * n_steps * len(TRAJS) * len(VIEWS)
    assert len(steps) == expected, (
        f"expected {expected} step_sample rows, got {len(steps)}; types={types}"
    )
    summary = next(r for r in records if r["record_type"] == "batch_summary")
    assert summary["logits_storage"] == "dense_r"
    assert summary["B"] == BATCH_SIZE
    for rec in steps:
        assert rec["prob_pre_commit_packed"]["prob"] is not None
        assert "rougeL_recall" in rec["rouge"]
        assert rec["traj_name"] in TRAJS
        assert rec["view"] in VIEWS
