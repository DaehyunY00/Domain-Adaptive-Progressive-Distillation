from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Any

import torch
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

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            mask = (batch["labels"] != -100).float()
            token_count = int(mask.sum().item())
            total_loss += float(outputs.loss.item()) * max(1, token_count)
            total_tokens += token_count

    if total_tokens == 0:
        return float("inf")

    mean_loss = total_loss / total_tokens
    return float(math.exp(min(mean_loss, 20)))


def compute_qa_metrics(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    max_samples: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> dict[str, float]:
    n = min(max_samples, len(dataset))
    if n == 0:
        return {"accuracy": 0.0, "f1": 0.0}

    exact = 0
    f1_sum = 0.0

    for idx in tqdm(range(n), desc="Evaluating QA", leave=False):
        sample = dataset[idx]
        prompt = PROMPT_TEMPLATE.format(prompt=sample["prompt"])
        gold = str(sample["target"]).strip()

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        pred_norm = _normalize_text(generated)
        gold_norm = _normalize_text(gold)

        if pred_norm == gold_norm:
            exact += 1

        f1_sum += _token_f1(pred_norm, gold_norm)

    return {
        "accuracy": exact / n,
        "f1": f1_sum / n,
    }


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
