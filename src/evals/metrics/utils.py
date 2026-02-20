import gc
import os
import sys
from typing import Dict, List, Optional
from tqdm import tqdm
from rouge_score import rouge_scorer
from collections import defaultdict
from omegaconf import OmegaConf
import numpy as np
import scipy as sc
from torch import nn
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, PreTrainedTokenizer
from data.utils import IGNORE_INDEX
import warnings


def _tensor_to_list_of_floats(tensor: torch.Tensor) -> list:
    """Convert 1D tensor to list of Python floats with a single GPU→CPU transfer.

    Avoids repeated syncs; do not keep tensors on GPU after conversion (no GPU cache).
    """
    t = tensor.float().cpu()
    arr = t.numpy()
    return arr.tolist()


def dict_transpose(evals):
    """Transpose a nested dictionary structure to group statistics by item indices."""
    # evals looks like {iidx0: {idx453: {prob: 0.1, loss: 1}},
    #                   iidx1: {idx453: {prob: 0.2, loss: 2}}}
    # multiple answers indexed by intra_item_idx, then item_idx
    # invert the dict, put outermost iidx deepest inside
    # after dict transpose looks like {idx453: {prob: [0.1, 0.2], loss: [1, 2]}
    all_iidxs = list(evals.keys())
    all_idxs = list(evals[all_iidxs[0]].keys())
    all_stat_names = list(evals[all_iidxs[0]][all_idxs[0]].keys())
    evals = {
        idx: {
            stat: [evals[iidx][idx][stat] for iidx in all_iidxs]
            for stat in all_stat_names
        }
        for idx in all_idxs
    }
    return evals


def aggregate_to_1D(x):
    return np.mean(x, axis=tuple(range(1, x.ndim)))


def get_forget_quality(model_tr, reference_tr):
    test_res = sc.stats.ks_2samp(1 / (model_tr + 1e-10), 1 / (reference_tr + 1e-10))
    return {"agg_value": test_res.pvalue}


def _is_tty() -> bool:
    """True if stderr is a TTY (interactive terminal). False for kubectl logs, Docker, etc."""
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


def _should_use_tqdm() -> bool:
    """Use tqdm only when interactive. In K8s/CI, tqdm's timer-based refresh prints a new line
    per refresh (kubectl logs don't handle \\r), producing thousands of duplicate lines."""
    if not _is_tty():
        return False
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return False
    if os.environ.get("CI", "").lower() in ("1", "true", "yes"):
        return False
    return True


def run_batchwise_evals(model, dataloader, batch_eval_fn, batch_eval_fn_args, eval_msg):
    """Run batch-wise evaluations on a dataset using a specified evaluation function. Handles
    multi-answer datasets by organizing evaluations by answer indices and aggregating results."""
    evals = defaultdict(dict)
    total = len(dataloader)
    # In non-TTY / K8s / CI, avoid tqdm: each timer refresh becomes a new line in logs.
    # Use explicit prints at batch boundaries instead.
    if _should_use_tqdm():
        iterator = tqdm(dataloader, desc=eval_msg, total=total)
    else:
        iterator = dataloader
        # Log every 10% or at least every 10 batches, whichever is more frequent
        log_interval = max(1, min(total // 10, 10))

    for batch_idx, batch in enumerate(iterator):
        if not _should_use_tqdm():
            n = batch_idx + 1
            if n % log_interval == 0 or n == total:
                pct = 100 * n / total
                print(f"{eval_msg}: {n}/{total} ({pct:.0f}%)", flush=True)
        # if data arrives in normal format we convert the batch to multiple answer-style
        # like in tofu_perturbed by adding a fake intra_item_index
        if "input_ids" in batch:
            batch = {"0": batch}
        # Assume batch like {"0": {"input_ids": [[]]..., "index": [453, 454..]},
        #                    "1": {"input_ids": [[]]..., "index": [453, 454..]}..}
        assert isinstance(next(iter(batch.values())), dict) and "input_ids" in next(
            iter(batch.values())
        )
        for intra_item_idx, mini_batch in batch.items():
            data_indices = (
                mini_batch.pop("index").cpu().numpy().tolist()
            )  # data item indices
            batch_evals = batch_eval_fn(
                model=model, batch=mini_batch, **batch_eval_fn_args
            )
            indexwise_batch_evals = dict(zip(data_indices, batch_evals))
            assert not (
                evals[intra_item_idx].keys() & indexwise_batch_evals.keys()
            ), "Data indices repeated while iterating dataloader"
            evals[intra_item_idx] |= indexwise_batch_evals
        # Free GPU cache between batches to avoid OOM with large models (e.g. 8B diffusion)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        # Phase 3: per-batch memory when OOM investigation enabled
        if (
            os.environ.get("OOM_INVESTIGATION", "").lower() in ("1", "true", "yes")
            and torch.cuda.is_available()
            and (batch_idx % 10 == 0 or batch_idx == total - 1)
        ):
            alloc = torch.cuda.memory_allocated() / (1024**2)
            res = torch.cuda.memory_reserved() / (1024**2)
            import logging
            logging.getLogger("metrics").info(
                f"[OOM_INVESTIGATION] batch {batch_idx + 1}/{total}: "
                f"memory_allocated_MiB={alloc:.0f} memory_reserved_MiB={res:.0f}"
            )
    # evals looks like {iidx0: {idx453: {prob: 0.1, loss: 1}},
    #                   iidx1: {idx453: {prob: 0.2, loss: 2}}}
    if len(evals) == 1:  # normal single answer dataset, no need for list
        evals = next(iter(evals.values()))
    else:
        # for each index return a dict with all intra_item_idx values in list
        # after dict transpose looks like {idx453: {prob: [0.1, 0.2], loss: [1, 2]}}
        evals = dict_transpose(evals)
    print("Evaluated", len(evals), "examples")
    return evals


def evaluate_probability(model, batch):
    """Evaluate model probabilities and average token-level loss for a given batch."""
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits
    labels = batch["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    # agg loss across tokens
    losses = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    num_token_gt = (batch["labels"] != IGNORE_INDEX).sum(-1)
    avg_losses = losses / num_token_gt
    normalized_probs = torch.exp(-avg_losses)

    # Single GPU→CPU transfer per tensor; no GPU-side cache
    avg_losses_list = _tensor_to_list_of_floats(avg_losses)
    normalized_probs_list = _tensor_to_list_of_floats(normalized_probs)
    # Drop references to free GPU memory before next batch
    del logits, shifted_labels, losses, avg_losses, normalized_probs
    return [
        {"prob": prob, "avg_loss": avg_loss}
        for prob, avg_loss in zip(normalized_probs_list, avg_losses_list)
    ]


def evaluate_probability_confidence_ordered(model, batch):
    """Evaluate forget probability in confidence order (for diffusion LLMs).

    Instead of aggregating in causal order, we use the model's probability for the
    true label at each position (confidence), sort positions by confidence
    (highest first), then take the geometric mean in that order. This mirrors
    diffusion behavior where "top confidence labels" are set first.

    Returns per-sample prob and avg_loss (same semantics as evaluate_probability)
    plus optional confidence_ordered_probs: list of probs in descending confidence order.
    """
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits
    labels = batch["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    bsz, seq_len, V = logits.shape
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)

    results = []
    for i in range(bsz):
        # Per-position probability of the true token (confidence for the label at that position)
        pos_probs = []
        for pos in range(seq_len):
            if shifted_labels[i, pos].item() == IGNORE_INDEX:
                continue
            y = shifted_labels[i, pos].item()
            lp = log_probs[i, pos, y].item()
            pos_probs.append(np.exp(lp))
        if not pos_probs:
            results.append({"prob": None, "avg_loss": None})
            continue
        pos_probs = np.array(pos_probs)
        # Sort by confidence (descending): set top confidence labels first
        pos_probs_sorted = np.sort(pos_probs)[::-1]
        geom_mean = float(np.exp(np.mean(np.log(pos_probs_sorted + 1e-12))))
        avg_loss = float(-np.log(geom_mean + 1e-12))
        results.append({
            "prob": geom_mean,
            "avg_loss": avg_loss,
            "confidence_ordered_probs": pos_probs_sorted.tolist(),
        })
    return results


def tokenwise_logprobs(model, batch, grad=False, return_labels=False):
    """
    Compute token-wise next token prediction logprobs for all labeled tokens for each sample in a batch.
    `grad` decides whether gradients are turned on
    Returns
    log_probs_batch (List[Tensor]): Tensors of size seq_len where seq_len is length of labeled tokens
    labels_batch (List[Tensor]): List of tensors of length N. Returned only if return_labels is True
    """
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.set_grad_enabled(grad):
        output = model(**batch)

    logits = output.logits
    bsz, seq_len, V = logits.shape
    # Trim batch to logits length so next_tokens and log_probs align (avoids RuntimeError when
    # caller passes longer sequences, e.g. trajectory step with full-length batch).
    if batch["input_ids"].shape[1] > seq_len:
        batch["input_ids"] = batch["input_ids"][:, :seq_len]
    if batch.get("labels") is not None and isinstance(batch["labels"], torch.Tensor):
        if batch["labels"].shape[1] > seq_len:
            batch["labels"] = batch["labels"][:, :seq_len]
    if batch.get("attention_mask") is not None and isinstance(batch["attention_mask"], torch.Tensor):
        if batch["attention_mask"].shape[1] > seq_len:
            batch["attention_mask"] = batch["attention_mask"][:, :seq_len]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[:, :-1, :]
    # ^ we don't predict next token for last token, bsz x seq_len-1 x V
    next_tokens = batch["input_ids"][:, 1:].unsqueeze(-1)  # bsz x seq_len-1 x 1
    target_log_probs = torch.gather(log_probs, dim=2, index=next_tokens).squeeze(-1)
    log_probs_batch = []
    labels_batch = []
    for i in range(bsz):
        labels = batch["labels"][i]
        # only focus on tokens which have loss on them (i.e. used in labels)
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][
            :-1
        ]  # -1 to ignore eos prediction
        num_actual_tokens = actual_indices.numel()
        if num_actual_tokens == 0:
            labels_batch.append(torch.tensor([], device=labels.device))
            log_probs_batch.append(torch.tensor([], device=labels.device))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        if start_idx == 0:
            warnings.warn(
                "Index 0 in a datapoint's input_ids must not have loss (unignored labels) on it",
                UserWarning,
            )
        log_probs_batch.append(target_log_probs[i, start_idx - 1 : end_idx])
        labels_batch.append(labels[actual_indices])

    return (log_probs_batch, labels_batch) if return_labels else log_probs_batch


def tokenwise_logprobs_from_logits(batch, logits, return_labels=False):
    """
    Compute token-wise next token prediction logprobs from precomputed logits (no model call).

    Same output as tokenwise_logprobs(model, batch) when model(**batch).logits == logits.
    Caller should pass batch and logits on the same device.

    Returns:
        log_probs_batch (List[Tensor]): Per-sample log-prob tensors.
        If return_labels=True: (log_probs_batch, labels_batch).
    """
    batch = {k: v.to(logits.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    bsz, seq_len, V = logits.shape
    if batch["input_ids"].shape[1] > seq_len:
        batch = {**batch, "input_ids": batch["input_ids"][:, :seq_len]}
    if batch.get("labels") is not None and isinstance(batch["labels"], torch.Tensor):
        if batch["labels"].shape[1] > seq_len:
            batch = {**batch, "labels": batch["labels"][:, :seq_len]}
    if batch.get("attention_mask") is not None and isinstance(batch["attention_mask"], torch.Tensor):
        if batch["attention_mask"].shape[1] > seq_len:
            batch = {**batch, "attention_mask": batch["attention_mask"][:, :seq_len]}
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[:, :-1, :]
    next_tokens = batch["input_ids"][:, 1:].unsqueeze(-1)
    target_log_probs = torch.gather(log_probs, dim=2, index=next_tokens).squeeze(-1)
    log_probs_batch = []
    labels_batch = []
    for i in range(bsz):
        labels = batch["labels"][i]
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][:-1]
        num_actual_tokens = actual_indices.numel()
        if num_actual_tokens == 0:
            labels_batch.append(torch.tensor([], device=labels.device))
            log_probs_batch.append(torch.tensor([], device=labels.device))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        if start_idx == 0:
            warnings.warn(
                "Index 0 in a datapoint's input_ids must not have loss (unignored labels) on it",
                UserWarning,
            )
        log_probs_batch.append(target_log_probs[i, start_idx - 1 : end_idx])
        labels_batch.append(labels[actual_indices])
    return (log_probs_batch, labels_batch) if return_labels else log_probs_batch


def tokenwise_vocab_logprobs(model, batch, grad=False, return_labels=False):
    """Get vocabulary-wise log probabilities for each token in the sequence.

    Returns:
        log_probs_batch (List[Tensor]): List of tensors of shape (N, V) containing log probabilities
        for each sequence, where N is the length of labeled tokens and V is vocab size.
        labels_batch (List[Tensor]): List of tensors of length N. Returned only if return_labels is True
    """
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.set_grad_enabled(grad):
        output = model(**batch)

    logits = output.logits
    bsz, seq_len, V = logits.shape
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[
        :, :-1, :
    ]  # Don't predict for last token

    # Process each sequence in batch separately
    log_probs_batch = []
    labels_batch = []
    for i in range(bsz):
        labels = batch["labels"][i]
        # Only include positions that have labels
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][
            :-1
        ]  # -1 to ignore eos prediction
        if len(actual_indices) == 0:
            labels_batch.append(torch.tensor([], device=labels.device))
            log_probs_batch.append(torch.zeros(0, V, device=labels.device))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        if start_idx == 0:
            warnings.warn(
                "Index 0 in a datapoint's input_ids must not have loss (unignored labels) on it",
                UserWarning,
            )
        # Return full distribution for each position: shape (N, V)
        log_probs_batch.append(log_probs[i, start_idx - 1 : end_idx])
        labels_batch.append(labels[actual_indices])

    return (log_probs_batch, labels_batch) if return_labels else log_probs_batch


class MultiTokenEOSCriteria(StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence. Stopping Criteria forked
    and modified from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/27924d77953491f66a038a09892807065e469358/lm_eval/models/utils.py#L208)"""

    def __init__(
        self,
        sequence: str,
        tokenizer: PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> StoppingCriteriaList:
    return StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )


def eval_rouge_recall_batch(
    gen_outputs: List[str],
    ground_truths: List[str],
    use_stemmer: bool = True,
    scorer: Optional[rouge_scorer.RougeScorer] = None,
) -> List[Dict[str, float]]:
    """Compute ROUGE recall/F1 for (gen, gt) pairs. One scorer per batch (reused for all pairs).

    If scorer is None, creates one RougeScorer with use_stemmer. Pass scorer from caller
    to reuse one scorer per eval_text_similarity call (avoids creating scorer per call).
    No GPU: all data is text; cache only on CPU or disk if needed.
    """
    if scorer is None:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=use_stemmer)
    evals = []
    for gen, gt in zip(gen_outputs, ground_truths):
        rouge_scores = scorer.score(gt, gen)
        evals.append(
            {
                "rouge1_recall": rouge_scores["rouge1"].recall,
                "rougeL_f1": rouge_scores["rougeL"].fmeasure,
                "rougeL_recall": rouge_scores["rougeL"].recall,
            }
        )
    return evals


def eval_rouge_recall_batch_worker(
    gen_outputs: List[str],
    ground_truths: List[str],
    use_stemmer: bool = True,
) -> List[Dict[str, float]]:
    """Same contract as eval_rouge_recall_batch; for use in a process pool.

    Each worker creates its own RougeScorer (no shared state). Picklable; only
    plain data (lists of strings) is passed.
    """
    return eval_rouge_recall_batch(
        gen_outputs, ground_truths, use_stemmer=use_stemmer, scorer=None
    )


def eval_text_similarity(model, tokenizer, batch, generation_args):
    """Evaluate text similarity between model-generated outputs and ground truth using ROUGE scores.

    Decoding is intentionally batched (batch_decode only); no GPU cache for decoded text.
    """
    batch = {k: v.to(model.device) for k, v in batch.items()}
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    input_texts = tokenizer.batch_decode(
        input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    tokens = [label[label != IGNORE_INDEX] for label in labels]
    full_texts = tokenizer.batch_decode(
        tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    ground_truths = [
        full_text.replace(input_text, "").strip()
        for input_text, full_text in zip(input_texts, full_texts)
    ]

    attention_mask = batch["attention_mask"]

    # convert to a simple dict from DictConfig
    generation_args = OmegaConf.to_container(generation_args, resolve=True)
    stopwords = generation_args.pop("stopwords", None)
    if stopwords is not None:
        assert isinstance(stopwords, list)
        sc = stop_sequences_criteria(
            tokenizer, stopwords, input_ids.shape[1], input_ids.shape[0]
        )
        generation_args["stopping_criteria"] = sc
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        **generation_args,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_texts = tokenizer.batch_decode(
        output[:, input_ids.shape[-1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # cut off at stopwords
    if stopwords is None:
        stopwords = []
    stopwords = [tokenizer.decode([tokenizer.eos_token_id])] + stopwords
    for i in range(len(gen_texts)):
        raw_text = gen_texts[i]
        for word in stopwords:
            if word and word in raw_text:
                raw_text = raw_text.split(word)[0]
        raw_text = raw_text.strip()
        gen_texts[i] = raw_text

    # One scorer per eval_text_similarity call; reuse for all pairs (no GPU cache)
    _scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = eval_rouge_recall_batch(gen_texts, ground_truths, scorer=_scorer)
    scores = [
        {
            **rouge_evals,
            "input": input_text,
            "ground_truth": ground_truth,
            "generation": gen_text,
        }
        for rouge_evals, input_text, ground_truth, gen_text in zip(
            scores, input_texts, ground_truths, gen_texts
        )
    ]
    return scores


def extract_target_texts_from_processed_data(tokenizer, batch):
    """Extract and detokenize text from activated positions in the batch."""
    labels = batch["labels"]
    labels = [elem[elem != -100] for elem in labels]
    texts = [
        tokenizer.decode(elem.tolist(), skip_special_tokens=True) for elem in labels
    ]
    return texts
