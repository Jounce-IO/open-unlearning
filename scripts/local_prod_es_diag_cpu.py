#!/usr/bin/env python3
"""
CPU-only path that mirrors production ES + trajectory diagnostics (no GPU, no model).

Builds a small trajectory tensor ``R`` [V, L, S], fixation map ``F``, and walks outer
``report_step`` the same way the trajectory ES fast path does: effective logits →
``extraction_strength_from_fixation`` → ``log_es_trajectory_diagnostics`` with a
``prev_fixation_logits`` chain.

Run from ``open-unlearning``:

  uv run python scripts/local_prod_es_diag_cpu.py
  uv run python scripts/local_prod_es_diag_cpu.py --every-step --per-position --audit

Set ``LOGLEVEL=DEBUG`` to see ``EXTRACTION_STRENGTH_TRAJ_DIAG`` / ``EXTRACTION_STRENGTH_AUDIT``.

Even without DEBUG, the script prints **which positions** had a change in the **effective**
``[L, V]`` fixation logits vs the previous outer step (same rule as
``fix_vs_prev_n_pos_vocab_changed``: count vocab dims with ``|Δ| > atol``).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from data.utils import IGNORE_INDEX  # noqa: E402
from evals.metrics.step_wise_score import (  # noqa: E402
    build_effective_step_fixation_logits,
    extraction_strength_from_fixation,
    log_es_trajectory_diagnostics,
    max_abs_adjacent_step_diff_along_traj_axis,
)


def _make_R_F_labels(
    *,
    V: int,
    L: int,
    S: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic R, F, labels suitable for causal ES (row ell-1 predicts token ell)."""
    torch.manual_seed(0)
    R = torch.randn(V, L, S, device=device) * 0.3
    # Nudge last few trajectory steps so effective logits change with report_step
    for s in range(max(0, S - 3), S):
        R[:, :, s] += float(s) * 0.05

    F = torch.zeros(L, dtype=torch.long, device=device)
    for ell in range(L):
        F[ell] = min(ell % max(S, 1), max(S - 1, 0))

    labels = torch.randint(0, V, (L,), device=device)
    labels[0] = min(2, V - 1)
    labels[1] = labels[0]
    for ell in range(2, L):
        if labels[ell].item() == labels[ell - 1].item():
            labels[ell] = (labels[ell].item() + 1) % V
    return R, F, labels


def _summarize_fixation_logit_changes_vs_prev(
    prev: torch.Tensor,
    now: torch.Tensor,
    *,
    atol: float,
    F: torch.Tensor,
    report_step: int,
    prev_report_step: int,
) -> tuple[list[str], int]:
    """Return (sparse ``ell:n_vocab_changed`` strings, count of argmax flips at used rows)."""
    diff_abs = (now.float() - prev.float()).abs()
    n_vocab = (diff_abs > atol).sum(dim=-1)
    nz = torch.nonzero(n_vocab > 0, as_tuple=False).flatten()
    parts: list[str] = []
    for i in nz.tolist():
        ell = int(i)
        se_prev = int(min(prev_report_step, int(F[ell].item())))
        se_now = int(min(report_step, int(F[ell].item())))
        parts.append(
            f"{ell}:{int(n_vocab[ell].item())}(s_eff {se_prev}→{se_now})"
        )
    # Causal alignment: compare argmax at logit row used for each label position (same as ES preds).
    Ln = now.shape[0]
    p_now = torch.zeros(Ln, dtype=torch.long)
    p_prev = torch.zeros(Ln, dtype=torch.long)
    for ell in range(Ln):
        li = max(0, ell - 1)
        p_now[ell] = now[li].argmax(dim=-1)
        p_prev[ell] = prev[li].argmax(dim=-1)
    n_argmax_flip = int((p_now != p_prev).sum().item())
    return parts, n_argmax_flip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V", type=int, default=24, help="vocab size (small)")
    p.add_argument("--L", type=int, default=10, help="sequence length")
    p.add_argument("--S", type=int, default=12, help="trajectory depth (sampler steps axis)")
    p.add_argument("--every-step", action="store_true", help="emit TRAJ_DIAG every outer step")
    p.add_argument("--per-position", action="store_true", help="EXTRACTION_STRENGTH_TRAJ_DIAG per_position line")
    p.add_argument("--audit", action="store_true", help="EXTRACTION_STRENGTH_AUDIT (full inner loop)")
    p.add_argument(
        "--fix-atol",
        type=float,
        default=1e-6,
        help="|Δlogit| threshold for counting changed vocab dims per position (matches prod diag default)",
    )
    p.add_argument(
        "--no-position-print",
        action="store_true",
        help="Skip stdout lines listing which positions changed vs previous step",
    )
    args = p.parse_args()

    log_level = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(levelname)s %(name)s - %(message)s",
    )
    logging.getLogger("evaluator").setLevel(getattr(logging, log_level, logging.INFO))

    device = torch.device("cpu")
    V, L, S = args.V, args.L, args.S
    R, F, labels = _make_R_F_labels(V=V, L=L, S=S, device=device)
    R_b = R.unsqueeze(0)
    F_b = F.unsqueeze(0)

    # Sanity: OOM-safe reduction matches naive on this small tensor
    naive = (R[:, :, 1:] - R[:, :, :-1]).abs().max().item() if S > 1 else 0.0
    streamed = max_abs_adjacent_step_diff_along_traj_axis(R)
    print(f"max adjacent |ΔR| (naive vs streamed): {naive:.6g} vs {streamed:.6g} (should match)")

    chain: dict[tuple[int, str, str], torch.Tensor] = {}
    score_chain: dict[tuple[int, str, str], float] = {}
    tkey = (0, "0", "steps", "full")
    last_step_index = S - 1
    logit_alignment = "causal"

    print(f"Walking report_step=0..{last_step_index} (L={L}, V={V}, S={S}, device=cpu)")
    for step_index, report_step in enumerate(range(S)):
        fixation_logits = build_effective_step_fixation_logits(
            R_b, F_b, report_step
        ).squeeze(0)
        prev_fl = chain.get(tkey)
        prev_es = score_chain.get(tkey)

        diag: dict = {}
        es = extraction_strength_from_fixation(
            fixation_logits,
            labels,
            F,
            S,
            logit_alignment,
            IGNORE_INDEX,
            audit=args.audit,
            audit_compact=not args.audit,
            audit_ctx={
                "sample_idx": "0",
                "step": report_step,
                "traj_name": "steps",
                "view": "full",
            },
            diag_out=diag,
        )
        log_es_trajectory_diagnostics(
            R_b,
            F_b,
            S,
            report_step,
            fixation_logits,
            batch_idx=0,
            sample_idx="0",
            traj_name="steps",
            view="full",
            step_index=step_index,
            last_step_index=last_step_index,
            audit_runtime=True,
            prev_fixation_logits=prev_fl,
            es_diag_every_step=args.every_step,
            es_diag_per_position=args.per_position,
            fix_vs_prev_count_atol=args.fix_atol,
            fix_vs_prev_max_positions_list=32,
            es_score=es,
            best_t=diag.get("best_t"),
            prev_es_score=prev_es,
        )
        chain[tkey] = fixation_logits.detach()
        score_chain[tkey] = float(es)

        if not args.no_position_print and prev_fl is not None:
            sparse, n_flip = _summarize_fixation_logit_changes_vs_prev(
                prev_fl,
                fixation_logits,
                atol=args.fix_atol,
                F=F,
                report_step=report_step,
                prev_report_step=report_step - 1,
            )
            if sparse:
                print(
                    f"    effective fixation logits changed at {len(sparse)} position(s): "
                    + ", ".join(sparse)
                    + f" | argmax_flips_across_rows_used_for_labels={n_flip}"
                )
            else:
                print(
                    f"    effective fixation logits: no position over |Δ|>{args.fix_atol} "
                    f"(argmax_flips_across_rows_used_for_labels={n_flip})"
                )

        if step_index in (0, last_step_index) or args.every_step:
            print(
                f"  step_index={step_index} report_step={report_step} "
                f"es={es:.6f} best_t={diag.get('best_t')}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
