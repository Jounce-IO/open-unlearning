"""US3: machine-checked max_new_tokens from specs/006 token-budget-audit §machine-checked."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

OU_ROOT = repo_root

# (relative path from open-unlearning root, sorted list of every max_new_tokens int in file)
_EXPECTED_CAPS: list[tuple[str, list[int]]] = [
    ("configs/generation/default.yaml", [200]),
    ("configs/eval/tofu_metrics/trajectory_all.yaml", [200, 200]),
    ("configs/eval/tofu_metrics/trajectory_privleak.yaml", [200]),
    ("configs/eval/tofu_metrics/trajectory_model_utility.yaml", [200]),
    ("configs/eval/tofu_metrics/trajectory_metrics.yaml", [200]),
    ("configs/eval/tofu_metrics/trajectory_forget_quality.yaml", [200]),
    ("configs/eval/tofu_metrics/trajectory_forget_Truth_Ratio.yaml", [200]),
    ("configs/eval/tofu_metrics/trajectory_forget_Q_A_ROUGE.yaml", [200, 200]),
    ("configs/eval/tofu_metrics/trajectory_forget_Q_A_Prob.yaml", [200]),
    ("configs/eval/tofu_metrics/trajectory_extraction_strength.yaml", [200]),
    ("configs/eval/muse_metrics/trajectory_retain_knowmem_ROUGE.yaml", [32, 32]),
    ("configs/eval/muse_metrics/trajectory_privleak.yaml", [128]),
    ("configs/eval/muse_metrics/trajectory_metrics_minimal.yaml", [32, 32]),
    ("configs/eval/muse_metrics/trajectory_metrics.yaml", [32, 32]),
    ("configs/eval/muse_metrics/trajectory_forget_verbmem_ROUGE.yaml", [128, 128]),
    ("configs/eval/muse_metrics/trajectory_forget_knowmem_ROUGE.yaml", [32, 32]),
    ("configs/eval/muse_metrics/trajectory_extraction_strength.yaml", [128]),
    ("configs/eval/muse_metrics/trajectory_muse_knowmem.yaml", [32, 32]),
    ("configs/eval/muse_metrics/trajectory_muse_verbmem.yaml", [128, 128]),
    ("configs/eval/muse_metrics/retain_knowmem_ROUGE.yaml", [32]),
    ("configs/eval/muse_metrics/forget_verbmem_ROUGE.yaml", [128]),
    ("configs/eval/muse_metrics/forget_knowmem_ROUGE.yaml", [32]),
]


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


@pytest.mark.parametrize("rel_path,expected", _EXPECTED_CAPS)
def test_yaml_max_new_tokens_match_audit(rel_path: str, expected: list[int]) -> None:
    path = OU_ROOT / rel_path
    assert path.is_file(), f"missing {path}"
    data = yaml.safe_load(path.read_text())
    found: list[int] = []
    _collect_max_new_tokens(data, found)
    found.sort()
    exp = sorted(expected)
    assert found == exp, f"{rel_path}: got {found}, expected {exp}"
