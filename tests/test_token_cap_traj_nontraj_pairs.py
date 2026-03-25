"""Traj vs non-traj YAML max_new_tokens sets align (TOFU/MUSE)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

OU_ROOT = repo_root

FIXTURE = repo_root / "tests" / "fixtures" / "upstream_open_unlearning_token_caps.json"


def _collect_max_new_tokens(obj, out: list[int]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "max_new_tokens" and isinstance(v, int):
                out.append(v)
            else:
                _collect_max_new_tokens(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_max_new_tokens(v, out)


def _cap_set(rel: str) -> set[int]:
    path = OU_ROOT / rel
    assert path.is_file(), rel
    data = yaml.safe_load(path.read_text())
    found: list[int] = []
    _collect_max_new_tokens(data, found)
    return set(found)


def _effective_nontraj_caps(rel: str) -> set[int]:
    """Non-traj files often omit max_new_tokens and inherit generation defaults."""
    s = _cap_set(rel)
    if s:
        return s
    r = rel.replace("\\", "/").lower()
    if "tofu_metrics" in r:
        return {200}
    if "knowmem" in r:
        return {32}
    return {128}


# (trajectory yaml, non-trajectory yaml)
_TOFU_PAIRS: list[tuple[str, str]] = [
    ("configs/eval/tofu_metrics/trajectory_forget_Q_A_Prob.yaml", "configs/eval/tofu_metrics/forget_Q_A_Prob.yaml"),
    ("configs/eval/tofu_metrics/trajectory_forget_Q_A_ROUGE.yaml", "configs/eval/tofu_metrics/forget_Q_A_ROUGE.yaml"),
    ("configs/eval/tofu_metrics/trajectory_forget_Truth_Ratio.yaml", "configs/eval/tofu_metrics/forget_Truth_Ratio.yaml"),
    ("configs/eval/tofu_metrics/trajectory_extraction_strength.yaml", "configs/eval/tofu_metrics/extraction_strength.yaml"),
    ("configs/eval/tofu_metrics/trajectory_model_utility.yaml", "configs/eval/tofu_metrics/model_utility.yaml"),
    ("configs/eval/tofu_metrics/trajectory_privleak.yaml", "configs/eval/tofu_metrics/privleak.yaml"),
    ("configs/eval/tofu_metrics/trajectory_forget_quality.yaml", "configs/eval/tofu_metrics/forget_quality.yaml"),
]

_MUSE_PAIRS: list[tuple[str, str]] = [
    ("configs/eval/muse_metrics/trajectory_forget_knowmem_ROUGE.yaml", "configs/eval/muse_metrics/forget_knowmem_ROUGE.yaml"),
    ("configs/eval/muse_metrics/trajectory_forget_verbmem_ROUGE.yaml", "configs/eval/muse_metrics/forget_verbmem_ROUGE.yaml"),
    ("configs/eval/muse_metrics/trajectory_extraction_strength.yaml", "configs/eval/muse_metrics/extraction_strength.yaml"),
    ("configs/eval/muse_metrics/trajectory_privleak.yaml", "configs/eval/muse_metrics/privleak.yaml"),
    ("configs/eval/muse_metrics/trajectory_retain_knowmem_ROUGE.yaml", "configs/eval/muse_metrics/retain_knowmem_ROUGE.yaml"),
    ("configs/eval/muse_metrics/trajectory_muse_knowmem.yaml", "configs/eval/muse_metrics/forget_knowmem_ROUGE.yaml"),
    ("configs/eval/muse_metrics/trajectory_muse_verbmem.yaml", "configs/eval/muse_metrics/forget_verbmem_ROUGE.yaml"),
]


@pytest.mark.parametrize("traj_rel,non_rel", _TOFU_PAIRS + _MUSE_PAIRS)
def test_traj_nontraj_max_new_tokens_sets_match(traj_rel: str, non_rel: str) -> None:
    traj_s = _cap_set(traj_rel)
    non_eff = _effective_nontraj_caps(non_rel)
    assert traj_s == non_eff, f"{traj_rel} vs {non_rel}: traj={traj_s!r} non_eff={non_eff!r}"


def test_fork_generation_default_matches_upstream_fixture() -> None:
    data = json.loads(FIXTURE.read_text())
    expected = set(data["paths"]["configs/generation/default.yaml"])
    assert _cap_set("configs/generation/default.yaml") == expected
