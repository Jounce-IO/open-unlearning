import logging
import torch
import transformers
from typing import Dict, Sequence
from data.utils import IGNORE_INDEX

logger = logging.getLogger(__name__)


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning.

    Pads ``input_ids`` and ``labels`` to their own max lengths per batch (independently).
    So ``input_ids.shape[1]`` and ``labels.shape[1]`` can differ. Do not assume position
    alignment between the two; use content-start indices (e.g. prompt_starts, generation_start)
    when slicing. See docs/trajectory_metrics.md for trajectory conventions.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        padding_side: str = "right",
        index: str = None,
    ):
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.index = index

    def get_instances_from_key(self, instances: Sequence[Dict], key: str):
        ret_instances = [instance[key] for instance in instances]
        return ret_instances

    def _pad_tokens(self, input_ids, padding_value):
        if self.padding_side == "right":
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=padding_value
            )
        else:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.flip(i, dims=[0]) for i in input_ids],
                batch_first=True,
                padding_value=padding_value,
            ).flip(dims=[1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert isinstance(instances[0], dict)
        return_dct = {}
        if "input_ids" not in instances[0]:
            for key in instances[0].keys():
                key_instances = self.get_instances_from_key(
                    instances=instances, key=key
                )
                return_dct[key] = self(key_instances)
        else:
            input_ids = [instance["input_ids"] for instance in instances]
            input_ids = self._pad_tokens(input_ids, self.tokenizer.pad_token_id)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            return_dct.update({"input_ids": input_ids})
            return_dct.update({"attention_mask": attention_mask})
            if "labels" in instances[0]:
                labels = [instance["labels"] for instance in instances]
                labels = self._pad_tokens(labels, IGNORE_INDEX)
                return_dct.update({"labels": labels})
            for extra_label_key in ("labels_correct", "labels_wrong"):
                if extra_label_key in instances[0]:
                    extra_labels = [
                        instance[extra_label_key] for instance in instances
                    ]
                    first = extra_labels[0]
                    if isinstance(first, list):
                        max_N = max(len(x) for x in extra_labels)
                        all_tensors = []
                        for inst in extra_labels:
                            for t in inst:
                                all_tensors.append(t.view(-1) if t.dim() > 1 else t)
                        max_L = max(t.numel() for t in all_tensors)
                        padded = []
                        for inst in extra_labels:
                            tensors = inst
                            rows = []
                            for t in tensors:
                                t_flat = t.view(-1) if t.dim() > 1 else t
                                if t_flat.numel() < max_L:
                                    row = torch.full(
                                        (max_L,),
                                        IGNORE_INDEX,
                                        dtype=t_flat.dtype,
                                        device=t_flat.device,
                                    )
                                    row[: t_flat.numel()] = t_flat
                                else:
                                    row = t_flat[:max_L]
                                rows.append(row)
                            mat = torch.stack(rows)
                            if len(rows) < max_N:
                                pad = torch.full(
                                    (max_N - len(rows), max_L),
                                    IGNORE_INDEX,
                                    dtype=mat.dtype,
                                    device=mat.device,
                                )
                                mat = torch.cat([mat, pad], dim=0)
                            padded.append(mat)
                        extra_labels = torch.stack(padded)
                        if extra_label_key == "labels_wrong" and logger.isEnabledFor(
                            logging.DEBUG
                        ):
                            logger.debug(
                                "collate labels_wrong multi-option: batch=%s N_max=%s L=%s tensor_shape=%s",
                                extra_labels.shape[0],
                                extra_labels.shape[1],
                                extra_labels.shape[2],
                                tuple(extra_labels.shape),
                            )
                    else:
                        extra_labels = self._pad_tokens(extra_labels, IGNORE_INDEX)
                    return_dct.update({extra_label_key: extra_labels})
            if self.index:
                if self.index in instances[0]:
                    return_dct.update(
                        {
                            self.index: torch.tensor(
                                [example[self.index] for example in instances]
                            )
                        }
                    )
                else:
                    raise Warning(f"{self.index} not found in dataset")
        return return_dct
