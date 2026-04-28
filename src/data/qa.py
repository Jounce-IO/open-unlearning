import logging
import torch
from torch.utils.data import Dataset

from data.utils import load_hf_dataset, preprocess_chat_instance, add_dataset_index

try:
    from datasets import DatasetDict
except ImportError:
    DatasetDict = None

logger = logging.getLogger(__name__)


class QADataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        few_shot_dataset_hf_args=None,
        max_length=512,
        predict_with_generate=False,
        match_dllm_sft_eval_rows: bool = False,
    ):
        super(QADataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        _match_rows = (
            match_dllm_sft_eval_rows
            if isinstance(match_dllm_sft_eval_rows, bool)
            else str(match_dllm_sft_eval_rows).lower().strip() in ("1", "true", "yes", "on")
        )
        raw = load_hf_dataset(**hf_args)
        # dllm ``load_dataset_tofu`` keeps 95% of rows for the exposed ``train`` split (seed=42);
        # ``dllm.tools.sft`` multi-eval evaluates on that train slice. Without this, OU used the
        # full HF split → different rows / ROUGE vs vivid SFT checkpoint-31000.
        if _match_rows:
            path_l = ""
            try:
                from omegaconf import OmegaConf

                if OmegaConf.is_config(hf_args):
                    path_l = str(OmegaConf.select(hf_args, "path") or "").lower()
                elif isinstance(hf_args, dict):
                    path_l = str(hf_args.get("path", "") or "").lower()
            except Exception:
                try:
                    path_l = str(getattr(hf_args, "path", "") or "").lower()
                except Exception:
                    path_l = ""
            if "locuslab/tofu" in path_l and hasattr(raw, "train_test_split"):
                raw = raw.train_test_split(test_size=0.05, seed=42)["train"]
            elif _match_rows and "locuslab/tofu" in path_l:
                logger.warning(
                    "match_dllm_sft_eval_rows=True but dataset has no train_test_split; "
                    "using full split (path=%r)",
                    path_l or hf_args,
                )
        self.data = add_dataset_index(raw)
        self.fs_data = None
        if few_shot_dataset_hf_args is not None:
            raw_data = load_hf_dataset(**few_shot_dataset_hf_args)
            self.fs_data = {}
            # Convert HuggingFace Column objects to lists for concatenation
            self.fs_data[question_key] = list(raw_data[question_key])
            self.fs_data[answer_key] = list(raw_data[answer_key])
        self.template_args = template_args
        self.question_key = question_key
        self.answer_key = answer_key
        self.predict_with_generate = predict_with_generate

    def __len__(self):
        return len(self.data)

    def _process_sample(self, question, answer, index=-1):
        if self.fs_data is None:
            prompt_msgs, response_msgs = [question], [answer]
        else:
            prompt_msgs = self.fs_data[self.question_key] + [question]
            response_msgs = self.fs_data[self.answer_key] + [answer]
        tokenized_data = preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            prompt_msgs,
            response_msgs,
            self.max_length,
            self.predict_with_generate,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
            "index": index,
        }
        return item_dct

    def __getitem__(self, idx):
        question = self.data[idx][self.question_key]
        answer = self.data[idx][self.answer_key]
        index = self.data[idx]["index"]
        if isinstance(answer, str):
            item = self._process_sample(question=question, answer=answer, index=index)
        elif isinstance(answer, list):
            item = {}
            for i, ans in enumerate(answer):
                sample_item = self._process_sample(
                    question=question, answer=ans, index=index
                )
                item[i] = sample_item
        else:
            raise NotImplementedError("answer format not found")
        return item


class MMLUUtilityDataset(QADataset):
    """MMLU one-subject dataset for utility validation (same format as QADataset: input_ids, labels).
    Loads cais/mmlu (or path/name from hf_args), maps choices[answer] to answer text, first 100 by config.
    Cap at 100 is applied in get_data via _cap_dataset_at_100.

    verbalized_reference (default False for backward compatibility):
        When True, assistant targets match dllm SFT utility eval (``mmlu_example_to_messages`` with
        ``verbalized_reference=True``): a full sentence citing the correct letter and choice text.
        When False, targets are only the raw correct choice string (shorter), which inflates
        ROUGE-L recall vs verbalized references for the same generations.
    """
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        max_length=512,
        verbalized_reference: bool = False,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_args = template_args
        self.question_key = question_key
        self.answer_key = answer_key
        self.predict_with_generate = kwargs.get("predict_with_generate", False)
        # Avoid bool("false") == True when Hydra passes a string.
        if isinstance(verbalized_reference, bool):
            self.verbalized_reference = verbalized_reference
        else:
            self.verbalized_reference = str(verbalized_reference).lower().strip() in (
                "1",
                "true",
                "yes",
                "on",
            )
        raw = load_hf_dataset(**hf_args)
        # When split= is in hf_args, load_dataset returns a single Dataset; otherwise DatasetDict.
        if DatasetDict is not None and isinstance(raw, DatasetDict):
            if hasattr(raw, "train") and "train" in raw:
                raw = raw["train"]
            elif hasattr(raw, "test") and "test" in raw:
                raw = raw["test"]
            elif hasattr(raw, "validation") and "validation" in raw:
                raw = raw["validation"]
            else:
                raw = raw[list(raw.keys())[0]]
        # else: raw is already a single Dataset (e.g. load_dataset(..., split="test"))
        self.data = []
        choices_key = "choices"
        answer_key_raw = "answer"
        for i, row in enumerate(raw):
            q = row[question_key]
            choices = row.get(choices_key, [])
            ans_idx = row.get(answer_key_raw, 0)
            if isinstance(ans_idx, str):
                ans_idx = ord(ans_idx.strip().upper()) - ord("A")
            ans_idx_int = min(max(0, int(ans_idx)), len(choices) - 1) if choices else 0
            if choices:
                choice_text = (choices[ans_idx_int] or "").strip()
                letter = chr(ord("A") + ans_idx_int)
                if self.verbalized_reference and choice_text:
                    answer = (
                        f"The correct answer is ({letter}) {choice_text}. "
                        f"Therefore, option {letter} is the best choice."
                    )
                else:
                    answer = choice_text
            else:
                answer = str(ans_idx_int)
            self.data.append({"question": q, "answer": answer, "index": i})
        self.fs_data = None


class QAwithIdkDataset(QADataset):
    def __init__(self, idk_path, return_original=True, *args, **kwargs):
        self.idk_path = idk_path
        self.return_original = return_original
        self.idk_responses = open(self.idk_path, "r").readlines()
        super().__init__(*args, **kwargs)

    def item_with_idk(self, question):
        rand_pos = torch.randint(0, len(self.idk_responses), (1,)).item()
        idk_response = self.idk_responses[rand_pos].strip()
        idk_item = self._process_sample(question=question, answer=idk_response)
        return idk_item

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        if isinstance(item, dict):
            return_item = {"original": item}
            idk_item = self.item_with_idk(question)
            return_item["alternate"] = idk_item
            # return_item = [item, idk_item]
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                idk_item = self.item_with_idk(question)
                return_item["alternate"] = idk_item
                # return_item.append([sample_item, idk_item])
        return return_item if self.return_original else return_item["alternate"]


class QAwithAlternateDataset(QADataset):
    def __init__(self, alternate_key, return_original=True, *args, **kwargs):
        self.alternate_key = alternate_key
        self.return_original = return_original
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        if isinstance(item, dict):
            return_item = {"original": item}
            alt_item = self._process_sample(
                question=question, answer=self.data[idx][self.alternate_key]
            )
            return_item["alternate"] = alt_item
            # return_item = [item, idk_item]
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                alt_item = self._process_sample(
                    question=question, answer=self.data[idx][self.alternate_key]
                )
                return_item["alternate"] = alt_item
                # return_item.append([sample_item, idk_item])
        return return_item if self.return_original else return_item["alternate"]


def _ensure_single_answer(answer, key_label: str, index, dataset_idx: int):
    """Normalize to a single string for tokenization. If list, use first element (no log)."""
    if isinstance(answer, list):
        if len(answer) == 0:
            raise ValueError(
                f"Empty list for {key_label} at dataset index {dataset_idx}; cannot compute dual-answer labels."
            )
        return answer[0]
    return answer


def _wrong_labels_for_sample(question, wrong_answer, process_sample_fn, index):
    """Build labels for wrong answer(s). Returns list of N label tensors or single tensor [L]."""
    if isinstance(wrong_answer, list):
        if len(wrong_answer) == 0:
            raise ValueError(
                "Empty list for wrong_answer; cannot compute dual-answer labels."
            )
        n_opt = len(wrong_answer)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "dual-answer: building %s wrong label tensors (index=%s)",
                n_opt,
                index,
            )
        return [
            process_sample_fn(
                question=question, answer=wrong_answer[k], index=index
            )["labels"]
            for k in range(n_opt)
        ]
    item = process_sample_fn(question=question, answer=wrong_answer, index=index)
    return item["labels"]


class QAwithDualAnswersDataset(QADataset):
    """Dataset yielding both labels_correct and labels_wrong for truth_ratio metrics.

    Uses two answer keys from the same HF split (e.g., paraphrased_answer and
    perturbed_answer in forget10_perturbed). When wrong_answer_key returns a list
    of N options (e.g. TOFU perturbed_answer), yields N wrong label tensors per
    sample so truth_ratio can average over all. When it returns a string, yields
    a single labels_wrong per sample. See evaluation-notes.md "Dual-answer
    datasets and list answers".
    """

    def __init__(
        self,
        correct_answer_key: str,
        wrong_answer_key: str,
        *args,
        **kwargs,
    ):
        self.correct_answer_key = correct_answer_key
        self.wrong_answer_key = wrong_answer_key
        # Parent uses answer_key for __len__ and data loading; use correct as default
        kwargs.setdefault("answer_key", correct_answer_key)
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        question = self.data[idx][self.question_key]
        correct_answer = self.data[idx][self.correct_answer_key]
        wrong_answer = self.data[idx][self.wrong_answer_key]
        index = self.data[idx]["index"]

        correct_answer = _ensure_single_answer(
            correct_answer, self.correct_answer_key, index, idx
        )
        correct_item = self._process_sample(
            question=question, answer=correct_answer, index=index
        )
        wrong_labels = _wrong_labels_for_sample(
            question, wrong_answer, self._process_sample, index
        )

        return {
            "input_ids": correct_item["input_ids"],
            "labels": correct_item["labels"],
            "labels_correct": correct_item["labels"],
            "labels_wrong": wrong_labels,
            "attention_mask": correct_item["attention_mask"],
            "index": index,
        }
