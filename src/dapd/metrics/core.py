from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data import CausalLMDataCollator, PROMPT_TEMPLATE


def compute_perplexity(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    batch_size: int,
    device: torch.device,
) -> float:
    collator = CausalLMDataCollator(tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            labels = batch["labels"]
            if labels.shape[1] <= 1:
                continue

            shift_logits = outputs.logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            token_count = int((shift_labels != -100).sum().item())
            if token_count == 0:
                continue

            nll_sum = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_nll += float(nll_sum.item())
            total_tokens += token_count

    if total_tokens == 0:
        return float("inf")

    mean_loss = total_nll / total_tokens
    return float(math.exp(min(mean_loss, 20)))


def compute_qa_metrics(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    max_samples: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> dict[str, Any]:
    """Backward-compatible QA metrics API with calibration outputs."""
    return compute_qa_metrics_with_calibration(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
    )


def compute_qa_metrics_with_calibration(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    max_samples: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> dict[str, Any]:
    """Compute QA metrics with calibration diagnostics.

    Returns:
        {
          "accuracy": float,
          "f1": float,
          "ece": float,
          "brier_score": float,
          "per_sample_confidence": list[float],
          "per_sample_correct": list[bool],
        }
    """
    n = min(max_samples, len(dataset))
    if n == 0:
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "ece": 0.0,
            "brier_score": 0.0,
            "per_sample_confidence": [],
            "per_sample_correct": [],
        }

    exact = 0
    f1_sum = 0.0
    per_sample_confidence: list[float] = []
    per_sample_correct: list[bool] = []

    for idx in tqdm(range(n), desc="Evaluating QA", leave=False):
        sample = dataset[idx]
        prompt = PROMPT_TEMPLATE.format(prompt=sample["prompt"])
        gold = str(sample["target"]).strip()

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            lm_out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            first_token_probs = torch.softmax(lm_out.logits[:, -1, :].float(), dim=-1)
            conf = float(first_token_probs.max(dim=-1).values.mean().item())

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        if hasattr(tokenizer, "decode"):
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        else:
            generated = str(generated_ids.tolist()).strip()

        pred_norm = _normalize_text(generated)
        gold_norm = _normalize_text(gold)

        if _answer_matches(pred_norm, gold_norm):
            exact += 1
            per_sample_correct.append(True)
        else:
            per_sample_correct.append(False)

        f1_sum += _token_f1(pred_norm, gold_norm)
        per_sample_confidence.append(conf)

    acc = exact / n
    f1 = f1_sum / n
    binary_correct = [1 if x else 0 for x in per_sample_correct]
    ece = compute_ece(per_sample_confidence, [float(x) for x in binary_correct], n_bins=15)
    brier = compute_brier_score(per_sample_confidence, binary_correct)

    return {
        "accuracy": acc,
        "f1": f1,
        "ece": ece,
        "brier_score": brier,
        "per_sample_confidence": per_sample_confidence,
        "per_sample_correct": per_sample_correct,
    }


def compute_ece(
    confidences: list[float],
    accuracies: list[float],
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error from confidence/accuracy pairs."""
    if not confidences or not accuracies:
        return 0.0
    if len(confidences) != len(accuracies):
        raise ValueError(
            f"confidences and accuracies must have same length, got {len(confidences)} vs {len(accuracies)}"
        )

    n = len(confidences)
    bins = max(1, int(n_bins))
    bin_counts = [0 for _ in range(bins)]
    bin_conf_sum = [0.0 for _ in range(bins)]
    bin_acc_sum = [0.0 for _ in range(bins)]

    for conf, acc in zip(confidences, accuracies):
        c = float(min(1.0, max(0.0, conf)))
        a = float(min(1.0, max(0.0, acc)))
        idx = min(bins - 1, int(c * bins))
        bin_counts[idx] += 1
        bin_conf_sum[idx] += c
        bin_acc_sum[idx] += a

    ece = 0.0
    for idx in range(bins):
        count = bin_counts[idx]
        if count == 0:
            continue
        mean_conf = bin_conf_sum[idx] / count
        mean_acc = bin_acc_sum[idx] / count
        ece += (count / n) * abs(mean_acc - mean_conf)
    return float(ece)


def compute_brier_score(
    probabilities: list[float],
    labels: list[int],
) -> float:
    """Compute Brier score for binary events."""
    if not probabilities or not labels:
        return 0.0
    if len(probabilities) != len(labels):
        raise ValueError(
            f"probabilities and labels must have same length, got {len(probabilities)} vs {len(labels)}"
        )

    total = 0.0
    for p, y in zip(probabilities, labels):
        prob = float(min(1.0, max(0.0, p)))
        target = float(1 if int(y) > 0 else 0)
        total += (prob - target) ** 2
    return float(total / len(probabilities))


def measure_generation_performance(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    samples: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, float]:
    n = min(samples, len(dataset))
    if n == 0:
        return {
            "latency_ms": 0.0,
            "throughput_tokens_per_sec": 0.0,
            "tokens_processed": 0.0,
            "inference_time_sec": 0.0,
            "samples": 0.0,
        }

    prompts = [PROMPT_TEMPLATE.format(prompt=dataset[i]["prompt"]) for i in range(n)]

    warm = tokenizer(prompts[0], return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        _ = model.generate(
            **warm,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    times = []
    tokens_processed = 0

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            start = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            if device.type == "mps" and torch.backends.mps.is_available() and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            end = time.perf_counter()
            times.append(end - start)
            generated = int(max(0, outputs[0].shape[-1] - inputs["input_ids"].shape[1]))
            tokens_processed += generated

    total_time = max(sum(times), 1e-9)
    latency_ms = float((total_time / len(times)) * 1000)
    throughput = float(tokens_processed / total_time)

    return {
        "latency_ms": latency_ms,
        "throughput_tokens_per_sec": throughput,
        "tokens_processed": float(tokens_processed),
        "inference_time_sec": float(total_time),
        "samples": float(n),
    }


def compute_compression_ratio(
    reference_value: float,
    target_value: float,
) -> float:
    # compression_ratio = teacher/reference size over student/target size
    if reference_value <= 0:
        return 0.0
    if target_value <= 0:
        return float("inf")
    return float(reference_value / target_value)


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def _answer_matches(pred_norm: str, gold_norm: str) -> bool:
    """Check whether the predicted text matches the gold answer.

    Models often generate more than just the answer label (e.g., "yes, because ..."
    instead of just "yes", or "a the patient should ..." instead of "a").
    We therefore apply a lenient match strategy:

    1. Exact match after normalization (handles the simple case).
    2. Prefix match: pred starts with gold followed by a word boundary
       (space, end of string). This handles "yes because..." matching "yes".
    3. First-token match: compare only the first token of pred against the
       entire gold when gold is a single token (A/B/C/D, yes/no/maybe).
       This is the standard approach for multiple-choice and yes/no QA.
    """
    if not gold_norm:
        return not pred_norm

    # 1. Exact match
    if pred_norm == gold_norm:
        return True

    gold_tokens = gold_norm.split()
    pred_tokens = pred_norm.split()

    if not pred_tokens:
        return False

    # 2. First-token match for single-token gold answers (A/B/C/D, yes/no/maybe)
    if len(gold_tokens) == 1:
        if pred_tokens[0] == gold_tokens[0]:
            return True

    # 3. Prefix match: pred starts with the full gold phrase
    if len(pred_tokens) >= len(gold_tokens):
        if pred_tokens[: len(gold_tokens)] == gold_tokens:
            return True

    return False


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = pred.split()
    gold_tokens = gold.split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    common = sum((pred_counter & gold_counter).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
