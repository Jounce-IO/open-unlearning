"""
Phase C: log everything needed to validate pre-commit ``lh[s]`` vs post-commit / ROUGE canvas
for trajectory packed probability (single run, JSONL artifact).

Enable via ``trajectory_config.probability_hypothesis_investigation: true`` or env
``TRAJECTORY_PROB_HYPOTHESIS_INVESTIGATION=1``. Output:
``{eval_output_dir}/probability_hypothesis_investigation.jsonl`` (rank 0 only).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from rouge_score import rouge_scorer

from data.utils import IGNORE_INDEX
from evals.metrics.step_wise_score import (
    build_effective_step_fixation_logits_from_history,
    compute_prob_from_fixation_logits,
    compute_prob_packed_shifted_segment_details,
    diffusion_fixation_logits_for_probability,
    sequence_probability_from_scores,
    trajectory_step_logits_to_prob_batch,
)

# Re-export for tests
__all__ = [
    "trajectory_prob_hypothesis_investigation_enabled",
    "run_trajectory_probability_hypothesis_investigation",
    "finalize_trajectory_probability_hypothesis_investigation",
    "TrajectoryProbabilityHypothesisLogger",
]
from evals.metrics.trajectory_utils import diffusion_source_steps_batch
from evals.metrics.utils import eval_rouge_recall_batch

logger = logging.getLogger("evaluator")

_ENV_FLAG = "TRAJECTORY_PROB_HYPOTHESIS_INVESTIGATION"
_LOGGERS: Dict[str, "TrajectoryProbabilityHypothesisLogger"] = {}


def trajectory_prob_hypothesis_investigation_enabled(
    trajectory_config: Optional[Any],
) -> bool:
    if os.environ.get(_ENV_FLAG, "").strip().lower() in ("1", "true", "yes"):
        return True
    if trajectory_config is None:
        return False
    if hasattr(trajectory_config, "get"):
        return bool(trajectory_config.get("probability_hypothesis_investigation", False))
    if isinstance(trajectory_config, dict):
        return bool(trajectory_config.get("probability_hypothesis_investigation", False))
    return False


def _sample_traj_state(
    *,
    lh_batch: Optional[List[torch.Tensor]],
    R_batch: Optional[torch.Tensor],
    b: int,
    F_b: torch.Tensor,
    S: int,
    L: int,
) -> Dict[str, Any]:
    """Per-sample trajectory logits state for :func:`_get_logits_at_step` (dense ``R`` or list ``lh``)."""
    if lh_batch is not None:
        return {"lh": lh_batch, "b": b, "F": F_b, "S": S, "L": L}
    if R_batch is not None:
        return {"R": R_batch[b], "F": F_b, "S": S, "L": L}
    raise ValueError("probability_hypothesis_investigation: need lh_batch or R_batch")


def _coerce_batch_int_list(values: Any, batch_size: int, *, name: str) -> List[int]:
    """Normalize per-sample int metadata to length ``batch_size``."""
    if values is None:
        raise ValueError(f"{name} is required for probability_hypothesis_investigation")
    if isinstance(values, torch.Tensor):
        flat = values.detach().cpu().flatten().tolist()
    elif isinstance(values, (list, tuple)):
        flat = list(values)
    else:
        flat = [values]
    out: List[int] = []
    for x in flat:
        if torch.is_tensor(x):
            out.append(int(x.item()))
        else:
            out.append(int(x))
    if len(out) == 1 and batch_size > 1:
        out = out * batch_size
    if len(out) != batch_size:
        raise ValueError(
            f"{name} length {len(out)} != batch size {batch_size} "
            f"(got {out!r})"
        )
    return out


def _logits_device(
    lh_batch: Optional[List[torch.Tensor]],
    R_batch: Optional[torch.Tensor],
) -> torch.device:
    if lh_batch is not None and len(lh_batch) > 0:
        return lh_batch[0].device
    if R_batch is not None:
        return R_batch.device
    return torch.device("cpu")


def _plain_tc(trajectory_config: Optional[Any]) -> Dict[str, Any]:
    from omegaconf import OmegaConf

    if trajectory_config is None:
        return {}
    if OmegaConf.is_config(trajectory_config):
        return OmegaConf.to_container(trajectory_config, resolve=True) or {}
    if isinstance(trajectory_config, dict):
        return trajectory_config
    return {}


class TrajectoryProbabilityHypothesisLogger:
    """Append JSONL records for one trajectory eval run (rank 0)."""

    def __init__(self, output_dir: str, *, rank: int = 0) -> None:
        self.rank = int(rank)
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / "probability_hypothesis_investigation.jsonl"
        self._manifest_written = False
        self._record_count = 0

    @classmethod
    def get(
        cls,
        output_dir: str,
        *,
        rank: int = 0,
    ) -> "TrajectoryProbabilityHypothesisLogger":
        key = str(Path(output_dir).resolve())
        if key not in _LOGGERS:
            _LOGGERS[key] = cls(output_dir, rank=rank)
        return _LOGGERS[key]

    def write_manifest(self, manifest: Dict[str, Any]) -> None:
        if self.rank != 0:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self._manifest_written:
            rec = {"record_type": "run_manifest", "ts": time.time(), **manifest}
            self._append(rec)
            self._manifest_written = True
            logger.info(
                "probability_hypothesis_investigation: writing to %s",
                self.path,
            )

    def log_batch(
        self,
        *,
        batch_idx: int,
        model: Any,
        tokenizer: Any,
        trajectory_config: Any,
        kwargs: Dict[str, Any],
        steps_to_use: Sequence[int],
        trajectory_names: Sequence[str],
        include_views: Sequence[str],
        lh_batch: Optional[List[torch.Tensor]],
        R_batch: Optional[torch.Tensor],
        F: torch.Tensor,
        S: int,
        L: int,
        B: int,
        seq_snapshots_batch: Optional[List[torch.Tensor]],
        prop_snapshots_batch: Optional[List[torch.Tensor]],
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        batch: Dict[str, Any],
        indices: Any,
        prompt_starts: List[Any],
        prompt_lens: List[Any],
        effective_lengths: List[int],
        prompt_only_input_ids: bool,
        gen_labels_per_sample: List[torch.Tensor],
        evaluation_mode: str,
    ) -> None:
        if self.rank != 0:
            return
        if (
            (lh_batch is None and R_batch is None)
            or seq_snapshots_batch is None
            or prop_snapshots_batch is None
        ):
            self._append(
                {
                    "record_type": "batch_skip",
                    "batch_idx": batch_idx,
                    "reason": "missing_logits_or_snapshots",
                    "has_lh": lh_batch is not None,
                    "has_R": R_batch is not None,
                    "has_seq_snap": seq_snapshots_batch is not None,
                    "has_prop_snap": prop_snapshots_batch is not None,
                }
            )
            return

        tc = _plain_tc(trajectory_config)
        mask_id = getattr(tokenizer, "mask_token_id", None)
        if mask_id is None:
            mask_id = tc.get("mask_token_id")
        if mask_id is None:
            logger.warning(
                "probability_hypothesis_investigation: no mask_token_id; skip batch %s",
                batch_idx,
            )
            return

        from evals.metrics.trajectory_metrics import (
            _stack_sequence_snapshots,
            _trajectory_build_sample_batch_template,
        )

        snap_stack = _stack_sequence_snapshots(seq_snapshots_batch)
        prop_stack = _stack_sequence_snapshots(prop_snapshots_batch)
        pl_list = _coerce_batch_int_list(prompt_lens, B, name="prompt_lens")
        ps_list = _coerce_batch_int_list(prompt_starts, B, name="prompt_starts")
        eff_list = _coerce_batch_int_list(effective_lengths, B, name="effective_lengths")
        logit_alignment = str(tc.get("logit_alignment", "causal"))
        rouge_sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        device = _logits_device(lh_batch, R_batch)
        traj_pass_id = kwargs.get("trajectory_pass_id") or tc.get("trajectory_pass_id")
        logits_storage = (
            "list_history" if lh_batch is not None else "dense_r" if R_batch is not None else None
        )

        inv_trajs = tc.get("probability_hypothesis_investigation_trajs")
        if inv_trajs is None:
            inv_trajs = ["steps", "fixation_start"]
        inv_views = tc.get("probability_hypothesis_investigation_views")
        if inv_views is None:
            inv_views = list(include_views)

        for b in range(B):
            idx_raw = indices[b]
            idx_str = (
                str(idx_raw.item()) if torch.is_tensor(idx_raw) else str(idx_raw)
            )
            _, gt_str, _ = _trajectory_build_sample_batch_template(
                b,
                batch,
                labels,
                input_ids,
                indices,
                prompt_starts,
                prompt_lens,
                L,
                prompt_only_input_ids,
                tokenizer,
            )
            gl = gen_labels_per_sample[b]
            L_eff = eff_list[b]
            pl = pl_list[b]
            ps = ps_list[b]
            n_gen = min(L_eff, int(L))

            row_end = min(int(input_ids.shape[1]), ps + L)
            initial_row = input_ids[b, ps:row_end].detach().long().cpu().tolist()
            if len(initial_row) < L:
                initial_row = initial_row + [0] * (L - len(initial_row))
            initial_row = initial_row[:L]
            initial_gen = initial_row[:n_gen] if n_gen > 0 else []

            for step in steps_to_use:
                step_i = int(step)
                pre_snap_idx = step_i - 1
                post_snap_idx = min(step_i, snap_stack.shape[0] - 1)
                snap_end = min(int(snap_stack.shape[2]), ps + L)
                pre_row = (
                    snap_stack[pre_snap_idx, b, ps:snap_end].detach().long().cpu().tolist()
                    if pre_snap_idx >= 0
                    else initial_row
                )
                if len(pre_row) < L:
                    pre_row = pre_row + [0] * (L - len(pre_row))
                pre_row = pre_row[:L]
                post_row = snap_stack[post_snap_idx, b, ps:snap_end].detach().long().cpu().tolist()
                if len(post_row) < L:
                    post_row = post_row + [0] * (L - len(post_row))
                post_row = post_row[:L]
                pre_gen = pre_row[:n_gen] if n_gen > 0 else []
                post_gen = post_row[:n_gen] if n_gen > 0 else []
                gold_gen = gl[ps : ps + n_gen].detach().long().cpu().tolist()

                F_b = F[b]
                fixation_gen = F_b[:n_gen].detach().long().cpu().tolist()

                for traj_name in trajectory_names:
                    if traj_name not in inv_trajs:
                        continue
                    for view in include_views:
                        if view not in inv_views:
                            continue
                        record = self._record_one(
                            batch_idx=batch_idx,
                            sample_idx=b,
                            index_str=idx_str,
                            trajectory_pass_id=traj_pass_id,
                            evaluation_mode=evaluation_mode,
                            report_step=step_i,
                            traj_name=traj_name,
                            view=view,
                            S=S,
                            L=L,
                            L_eff=L_eff,
                            prompt_len=pl,
                            n_gen=n_gen,
                            logit_alignment=logit_alignment,
                            mask_id=int(mask_id),
                            ground_truth=gt_str,
                            gold_gen_ids=gold_gen,
                            initial_gen_ids=initial_gen,
                            pre_commit_gen_ids=pre_gen,
                            post_commit_gen_ids=post_gen,
                            fixation_steps_gen=fixation_gen,
                            lh_batch=lh_batch,
                            R_batch=R_batch,
                            F_b=F_b,
                            gl=gl,
                            snap_stack=snap_stack,
                            prop_stack=prop_stack,
                            pl_list=pl_list,
                            F=F,
                            effective_lengths_list=eff_list,
                            B=B,
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            rouge_sc=rouge_sc,
                        )
                        self._append(record)

        self._append(
            {
                "record_type": "batch_summary",
                "batch_idx": batch_idx,
                "B": B,
                "steps_logged": list(steps_to_use),
                "samples_logged": B,
                "logits_storage": logits_storage,
            }
        )

    def _record_one(
        self,
        **kw: Any,
    ) -> Dict[str, Any]:
        """Build one JSON-serializable investigation record."""
        b = kw["sample_idx"]
        step_i = kw["report_step"]
        traj_name = kw["traj_name"]
        view = kw["view"]
        lh_batch = kw["lh_batch"]
        R_batch = kw.get("R_batch")
        F_b = kw["F_b"]
        gl = kw["gl"]
        device = kw["device"]
        pl_list = list(kw["pl_list"])
        pl = pl_list[b]
        n_gen = kw["n_gen"]
        L = kw["L"]
        L_eff = kw["L_eff"]
        mask_id = kw["mask_id"]

        from evals.metrics.step_wise_score import build_effective_step_fixation_logits
        from evals.metrics.trajectory_metrics import (
            _get_logits_at_step,
            _trajectory_canvas_gen_texts_batched,
            _trajectory_merge_snapshots_reindex_batched,
        )

        st_b = _sample_traj_state(
            lh_batch=lh_batch,
            R_batch=R_batch,
            b=b,
            F_b=F_b,
            S=kw["S"],
            L=L,
        )
        logits_vl = _get_logits_at_step(st_b, traj_name, step_i)
        logits_step = trajectory_step_logits_to_prob_batch(logits_vl)

        if view == "full":
            logits_view = logits_step
            lab_view = gl.to(device=device, dtype=torch.long).unsqueeze(0)
        else:
            Ls = min(L_eff, logits_step.shape[1])
            logits_view = logits_step[:, :Ls, :].contiguous()
            lab_view = gl[:Ls].to(device=device, dtype=torch.long).unsqueeze(0)

        packed_detail = compute_prob_packed_shifted_segment_details(
            logits_view, lab_view, device, IGNORE_INDEX
        )

        if R_batch is not None:
            eff_logits = build_effective_step_fixation_logits(
                R_batch[b : b + 1], F_b.unsqueeze(0), int(step_i)
            )
        else:
            eff_logits = build_effective_step_fixation_logits_from_history(
                lh_batch, F_b.unsqueeze(0), int(step_i)
            )
        eff_prob = compute_prob_from_fixation_logits(
            eff_logits,
            lab_view,
            device,
            IGNORE_INDEX,
        )[0]
        eff_scores_provider = None
        try:
            from evals.metrics.step_wise_score import FixationStepWiseScoreProvider

            provider = FixationStepWiseScoreProvider(logit_alignment=kw["logit_alignment"])
            scores, _ = provider.get_per_position_scores(
                {"fixation_logits": eff_logits},
                {"labels": lab_view},
                ignore_index=IGNORE_INDEX,
            )
            eff_geom = sequence_probability_from_scores(scores[0][0]) if scores else None
        except Exception as exc:
            eff_geom = None
            eff_scores_provider = f"error:{exc}"

        F_full: torch.Tensor = kw["F"]
        eff_list: List[int] = list(kw["effective_lengths_list"])
        batch_b = int(kw["B"])
        src_batch = diffusion_source_steps_batch(
            traj_name, int(step_i), F_full, kw["S"]
        )
        n_dec_list = [
            min(int(eff_list[bb]), L) if view == "eos" else L for bb in range(batch_b)
        ]
        n_dec = n_dec_list[b]
        merged_canvas = _trajectory_merge_snapshots_reindex_batched(
            kw["snap_stack"],
            int(step_i),
            pl_list,
            L,
            n_dec_list,
            src_batch,
        )
        merged_prop = _trajectory_merge_snapshots_reindex_batched(
            kw["prop_stack"],
            int(step_i),
            pl_list,
            L,
            n_dec_list,
            src_batch,
        )
        gen_texts = _trajectory_canvas_gen_texts_batched(
            merged_canvas,
            merged_prop,
            pl_list,
            L,
            n_dec_list,
            mask_id,
            view,
            eff_list,
            kw["tokenizer"],
            "canvas_plus_step_x0",
        )
        rouge_scores = eval_rouge_recall_batch(
            [gen_texts[b]],
            [kw["ground_truth"]],
            use_stemmer=True,
            scorer=kw["rouge_sc"],
        )
        rouge_l = float(rouge_scores[0].get("rougeL_recall", 0.0))

        rouge_line = merged_canvas[b, pl : pl + n_dec].clone()
        mid = mask_id
        prop_slice = merged_prop[b, pl : pl + n_dec]
        fill = (rouge_line == mid)
        rouge_line[fill] = prop_slice[fill]
        rouge_line_ids = rouge_line.detach().long().cpu().tolist()

        canvas_forward: Dict[str, Any] = {"attempted": False}
        model = kw.get("model")
        if model is not None and n_gen > 0:
            canvas_forward["attempted"] = True
            try:
                post_row_t = kw["snap_stack"][min(step_i, kw["snap_stack"].shape[0] - 1), b, :L]
                inp_line = post_row_t.unsqueeze(0).clone()
                attn = torch.ones_like(inp_line, dtype=torch.long, device=device)
                lab_line = gl.unsqueeze(0)
                fix_logits = diffusion_fixation_logits_for_probability(
                    model,
                    inp_line,
                    attn,
                    lab_line,
                    IGNORE_INDEX,
                )
                cf = compute_prob_from_fixation_logits(
                    fix_logits[:, : lab_view.shape[1], :],
                    lab_view,
                    device,
                    IGNORE_INDEX,
                )[0]
                canvas_forward.update(
                    {
                        "source": "diffusion_fixation_logits_on_post_commit_canvas_row",
                        "prob": cf.get("prob"),
                        "avg_loss": cf.get("avg_loss"),
                    }
                )
                rouge_aligned_inp = rouge_line.unsqueeze(0).clone()
                if rouge_aligned_inp.shape[1] < inp_line.shape[1]:
                    pad = inp_line[:, rouge_aligned_inp.shape[1] :]
                    rouge_aligned_inp = torch.cat([rouge_aligned_inp, pad], dim=1)
                elif rouge_aligned_inp.shape[1] > inp_line.shape[1]:
                    rouge_aligned_inp = rouge_aligned_inp[:, : inp_line.shape[1]]
                fix_logits_rouge = diffusion_fixation_logits_for_probability(
                    model,
                    rouge_aligned_inp,
                    torch.ones_like(rouge_aligned_inp, dtype=torch.long, device=device),
                    lab_line,
                    IGNORE_INDEX,
                )
                cf_r = compute_prob_from_fixation_logits(
                    fix_logits_rouge[:, : lab_view.shape[1], :],
                    lab_view,
                    device,
                    IGNORE_INDEX,
                )[0]
                canvas_forward["rouge_line_prob"] = cf_r.get("prob")
                canvas_forward["rouge_line_avg_loss"] = cf_r.get("avg_loss")
            except Exception as exc:
                canvas_forward["error"] = repr(exc)

        def _count_stats(gen_ids: List[int], gold_ids: List[int]) -> Dict[str, Any]:
            n = min(len(gen_ids), len(gold_ids))
            if n == 0:
                return {"n": 0}
            n_mask = sum(1 for t in gen_ids[:n] if t == mask_id)
            n_mismatch = sum(
                1
                for g, y in zip(gen_ids[:n], gold_ids[:n])
                if g != mask_id and y != IGNORE_INDEX and g != y
            )
            n_match = sum(
                1
                for g, y in zip(gen_ids[:n], gold_ids[:n])
                if g != mask_id and y != IGNORE_INDEX and g == y
            )
            return {
                "n": n,
                "n_mask": n_mask,
                "n_committed": n - n_mask,
                "n_matches_gold": n_match,
                "n_mismatch_gold": n_mismatch,
            }

        pre_ids = kw["pre_commit_gen_ids"]
        post_ids = kw["post_commit_gen_ids"]
        gold_ids = kw["gold_gen_ids"]
        n_rouge_fill = sum(
            1
            for c, r in zip(
                merged_canvas[b, pl : pl + n_dec].tolist(),
                rouge_line_ids,
            )
            if c == mask_id and r != mask_id
        )

        per_token_enriched: List[Dict[str, Any]] = []
        for pt in packed_detail.get("per_token") or []:
            li = int(pt["label_index"])
            if li < pl or li >= pl + n_gen:
                continue
            gi = li - pl
            entry = dict(pt)
            entry["pre_commit_id"] = pre_ids[gi] if gi < len(pre_ids) else None
            entry["post_commit_id"] = post_ids[gi] if gi < len(post_ids) else None
            entry["rouge_line_id"] = rouge_line_ids[gi] if gi < len(rouge_line_ids) else None
            entry["gold_id"] = gold_ids[gi] if gi < len(gold_ids) else None
            entry["fixation_step"] = (
                int(kw["fixation_steps_gen"][gi])
                if gi < len(kw["fixation_steps_gen"])
                else None
            )
            per_token_enriched.append(entry)

        rec: Dict[str, Any] = {
            "record_type": "step_sample",
            "batch_idx": kw["batch_idx"],
            "sample_idx": b,
            "index": kw["index_str"],
            "trajectory_pass_id": kw["trajectory_pass_id"],
            "evaluation_mode": kw["evaluation_mode"],
            "report_step": step_i,
            "traj_name": traj_name,
            "view": view,
            "S": kw["S"],
            "L": L,
            "L_eff": L_eff,
            "prompt_len": pl,
            "n_gen_positions": n_gen,
            "logit_alignment": kw["logit_alignment"],
            "hypothesis": "pre_commit_lh_vs_post_commit_rouge_canvas",
            "prob_pre_commit_packed": {
                "prob": packed_detail.get("prob"),
                "avg_loss": packed_detail.get("avg_loss"),
                "n_valid_tokens": packed_detail.get("n_valid_tokens"),
                "argmax_match_frac": packed_detail.get("argmax_match_frac"),
                "mean_ce_valid": packed_detail.get("mean_ce_valid"),
            },
            "prob_fixation_effective_step": eff_prob,
            "prob_fixation_geom_mean": eff_geom,
            "prob_fixation_geom_error": eff_scores_provider,
            "rouge": {
                "rougeL_recall": rouge_l,
                "gen_text": gen_texts[0] if gen_texts else "",
                "ground_truth": kw["ground_truth"],
            },
            "canvas_forward_prob": canvas_forward,
            "canvas_token_stats": {
                "pre_commit": _count_stats(pre_ids, gold_ids),
                "post_commit": _count_stats(post_ids, gold_ids),
                "rouge_decode_line": _count_stats(rouge_line_ids, gold_ids),
                "n_positions_filled_from_proposal": n_rouge_fill,
            },
            "per_token": per_token_enriched,
            "ratio_last_over_first_pre_commit": None,
        }
        return rec

    def _append(self, record: Dict[str, Any]) -> None:
        if self.rank != 0:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")
        self._record_count += 1

    def flush_summary(self) -> None:
        if self.rank != 0:
            return
        self._append(
            {
                "record_type": "run_complete",
                "total_records": self._record_count,
                "path": str(self.path),
            }
        )
        logger.info(
            "probability_hypothesis_investigation: wrote %s records to %s",
            self._record_count,
            self.path,
        )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, dict)):
        return obj
    return str(obj)


def run_trajectory_probability_hypothesis_investigation(
    *,
    enabled: bool,
    output_dir: Optional[str],
    rank: int,
    batch_idx: int,
    model: Any,
    tokenizer: Any,
    trajectory_config: Any,
    kwargs: Dict[str, Any],
    steps_to_use: Sequence[int],
    trajectory_names: Sequence[str],
    include_views: Sequence[str],
    lh_batch: Optional[List[torch.Tensor]],
    R_batch: Optional[torch.Tensor],
    F: torch.Tensor,
    S: int,
    L: int,
    B: int,
    seq_snapshots_batch: Optional[List[torch.Tensor]],
    prop_snapshots_batch: Optional[List[torch.Tensor]],
    labels: Optional[torch.Tensor],
    input_ids: torch.Tensor,
    batch: Dict[str, Any],
    indices: Any,
    prompt_starts: List[Any],
    prompt_lens: List[Any],
    effective_lengths: List[int],
    prompt_only_input_ids: bool,
    evaluation_mode: str,
) -> None:
    if not enabled or output_dir is None or rank != 0:
        return
    if labels is None:
        return
    gen_labels: List[torch.Tensor] = []
    for b in range(B):
        gl = labels[b]
        if gl.dim() == 1:
            gen_labels.append(gl)
        else:
            gen_labels.append(gl[0])
    inv = TrajectoryProbabilityHypothesisLogger.get(output_dir, rank=rank)
    inv.log_batch(
        batch_idx=batch_idx,
        model=model,
        tokenizer=tokenizer,
        trajectory_config=trajectory_config,
        kwargs=kwargs,
        steps_to_use=steps_to_use,
        trajectory_names=trajectory_names,
        include_views=include_views,
        lh_batch=lh_batch,
        R_batch=R_batch,
        F=F,
        S=S,
        L=L,
        B=B,
        seq_snapshots_batch=seq_snapshots_batch,
        prop_snapshots_batch=prop_snapshots_batch,
        labels=labels,
        input_ids=input_ids,
        batch=batch,
        indices=indices,
        prompt_starts=prompt_starts,
        prompt_lens=prompt_lens,
        effective_lengths=effective_lengths,
        prompt_only_input_ids=prompt_only_input_ids,
        gen_labels_per_sample=gen_labels,
        evaluation_mode=evaluation_mode,
    )

def finalize_trajectory_probability_hypothesis_investigation(
    output_dir: Optional[str], *, rank: int = 0
) -> None:
    if output_dir is None or rank != 0:
        return
    key = str(Path(output_dir).resolve())
    if key in _LOGGERS:
        _LOGGERS[key].flush_summary()
