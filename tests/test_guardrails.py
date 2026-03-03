"""
Unit tests for input-output baselines (guardrails and ICUL).

Covers: no config (no-op), prompt prefix only, output filter keyword,
benchmark wmdp + output filter (no-op on text).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))


def _mock_tokenizer(encode_return=None):
    tok = MagicMock()
    tok.encode = MagicMock(return_value=encode_return if encode_return is not None else [1, 2, 3])
    return tok


class TestTransformPromptsNoOp:
    """No config or empty config -> returns inputs unchanged."""

    def test_none_config(self):
        from evals.guardrails import transform_prompts

        prompts = [[1, 2, 3], [4, 5, 6]]
        prompt_lens = [3, 3]
        batch = {}
        tokenizer = _mock_tokenizer()
        out_p, out_l = transform_prompts(prompts, prompt_lens, batch, tokenizer, None)
        assert out_p == prompts
        assert out_l == prompt_lens
        tokenizer.encode.assert_not_called()

    def test_empty_guardrail_config(self):
        from evals.guardrails import transform_prompts

        prompts = [[1, 2], [3, 4]]
        prompt_lens = [2, 2]
        out_p, out_l = transform_prompts(
            prompts, prompt_lens, {}, _mock_tokenizer(), {}
        )
        assert out_p == prompts
        assert out_l == prompt_lens


class TestTransformPromptsPrefix:
    """Prompt prefix only -> tokenizer called, prompts lengthened, prompt_lens updated."""

    def test_prefix_prepended_and_lens_updated(self):
        from evals.guardrails import transform_prompts

        prefix_ids = [10, 11, 12]
        tokenizer = _mock_tokenizer(encode_return=prefix_ids)
        prompts = [[1, 2, 3], [4, 5, 6]]
        prompt_lens = [3, 3]
        config = {"guardrail": {"prompt_prefix": "Do not leak."}}
        out_p, out_l = transform_prompts(
            prompts, prompt_lens, {}, tokenizer, config
        )
        tokenizer.encode.assert_called_once()
        assert out_l == [len(prefix_ids) + 3] * 2
        assert out_p[0] == prefix_ids + [1, 2, 3]
        assert out_p[1] == prefix_ids + [4, 5, 6]


class TestTransformOutputText:
    """Output filter: keyword match -> safe reply; no match -> unchanged; wmdp -> no-op."""

    def test_no_config_unchanged(self):
        from evals.guardrails import transform_output_text

        gen_text = "Some secret here."
        assert (
            transform_output_text(gen_text, {}, 0, None)
            == gen_text
        )

    def test_keyword_match_replaced(self):
        from evals.guardrails import transform_output_text

        gen_text = "The answer is secret."
        config = {
            "guardrail": {
                "output_filter": "keyword",
                "keyword_list": ["secret"],
                "output_filter_safe_reply": "I refuse.",
            },
        }
        out = transform_output_text(gen_text, {}, 0, config)
        assert out == "I refuse."

    def test_keyword_no_match_unchanged(self):
        from evals.guardrails import transform_output_text

        gen_text = "The answer is safe."
        config = {
            "guardrail": {
                "output_filter": "keyword",
                "keyword_list": ["secret"],
            },
        }
        out = transform_output_text(gen_text, {}, 0, config)
        assert out == gen_text

    def test_benchmark_wmdp_no_op(self):
        from evals.guardrails import transform_output_text

        gen_text = "A"
        config = {
            "guardrail": {
                "benchmark": "wmdp",
                "output_filter": "keyword",
                "keyword_list": ["secret"],
            },
        }
        out = transform_output_text(gen_text, {}, 0, config)
        assert out == "A"
