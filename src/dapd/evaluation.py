from __future__ import annotations

import math
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import PROMPT_TEMPLATE
from .metrics import compute_compression_ratio, compute_perplexity, compute_qa_metrics
from .utils import collect_memory_stats, get_logger, get_model_disk_size_bytes, infer_device


def evaluate_model(
    model_path: str,
    text_dataset: Any,
    lm_dataset: Any,
    config: Any,
    runtime: Any,
    reference_model_path: str | None = None,
) -> dict[str, Any]:
    logger = get_logger("dapd.evaluation", getattr(runtime, "log_level", "INFO"))

    student_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    student_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    device = infer_device(runtime.device)
    student_model.to(device)
    student_model.eval()

    ppl = compute_perplexity(
        model=student_model,
        tokenizer=student_tokenizer,
        dataset=lm_dataset,
        batch_size=config.batch_size,
        device=device,
    )
    qa_metrics = compute_qa_metrics(
        model=student_model,
        tokenizer=student_tokenizer,
        dataset=text_dataset,
        max_samples=config.max_eval_samples,
        max_new_tokens=config.max_new_tokens,
        temperature=config.generation_temperature,
        device=device,
    )

    # Build one prompt batch and reuse it for student/teacher latency fairness.
    eval_prompts = _build_prompt_batch(
        dataset=text_dataset,
        samples=config.num_latency_samples,
    )

    perf = _measure_generation_performance_on_prompts(
        model=student_model,
        tokenizer=student_tokenizer,
        prompts=eval_prompts,
        max_new_tokens=config.max_new_tokens,
        device=device,
    )

    total_params, nonzero_params = _count_model_params(student_model)
    disk_size = get_model_disk_size_bytes(model_path)
    mem = collect_memory_stats(device=device, reset_peak=False)

    latency = {
        "mean_ms": perf["latency_ms"],
        "inference_time_sec": perf["inference_time_sec"],
        "samples": int(perf["samples"]),
    }
    memory_usage = {
        "rss_mb": mem.rss_mb,
        "device_allocated_mb": mem.device_allocated_mb,
        "peak_allocated_mb": mem.peak_allocated_mb,
        "device": mem.device,
    }

    out = {
        "accuracy": qa_metrics["accuracy"],
        "f1": qa_metrics["f1"],
        "perplexity": ppl,
        "model_total_params": total_params,
        "model_nonzero_params": nonzero_params,
        "parameter_sparsity": 1.0 - (nonzero_params / max(1, total_params)),
        "model_disk_size_mb": disk_size / (1024 * 1024),
        "latency_ms": perf["latency_ms"],
        "inference_latency_ms": perf["latency_ms"],
        "latency": latency,
        "throughput_tokens_per_sec": perf["throughput_tokens_per_sec"],
        "tokens_processed": perf["tokens_processed"],
        "inference_time_sec": perf["inference_time_sec"],
        "samples_measured": int(perf["samples"]),
        "memory_usage": memory_usage,
        "memory_usage_mb": mem.rss_mb,
        "device_allocated_mb": mem.device_allocated_mb,
        "peak_allocated_mb": mem.peak_allocated_mb,
        "compression_ratio": 1.0,
        "speedup_vs_teacher": 1.0,
    }

    if reference_model_path:
        ref = _evaluate_reference_performance(
            model_path=reference_model_path,
            prompts=eval_prompts,
            max_new_tokens=config.max_new_tokens,
            device=device,
        )

        out["teacher_latency_ms"] = ref["latency_ms"]
        out["teacher_throughput_tokens_per_sec"] = ref["throughput_tokens_per_sec"]
        out["teacher_tokens_processed"] = ref["tokens_processed"]
        out["teacher_inference_time_sec"] = ref["inference_time_sec"]
        out["speedup_vs_teacher"] = ref["latency_ms"] / max(1e-9, out["inference_latency_ms"])

        ref_params = ref["params"]
        ref_disk_mb = ref["disk_size_mb"]
        out["compression_ratio"] = compute_compression_ratio(ref_params, out["model_total_params"])
        out["disk_compression_ratio"] = compute_compression_ratio(
            ref_disk_mb,
            out["model_disk_size_mb"],
        )
    else:
        out["disk_compression_ratio"] = 1.0

    out["efficiency"] = {
        "compression_ratio": out["compression_ratio"],
        "throughput_tokens_per_sec": out["throughput_tokens_per_sec"],
        "speedup_vs_teacher": out["speedup_vs_teacher"],
        "latency_ms": out["latency_ms"],
        "memory_usage_mb": out["memory_usage_mb"],
        "device_allocated_mb": out["device_allocated_mb"],
        "peak_allocated_mb": out["peak_allocated_mb"],
    }

    logger.info(
        "evaluation done | acc=%.4f f1=%.4f ppl=%.2f latency=%.2fms throughput=%.2f tok/s",
        out["accuracy"],
        out["f1"],
        out["perplexity"],
        out["inference_latency_ms"],
        out["throughput_tokens_per_sec"],
    )

    return out


def _evaluate_reference_performance(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, float]:
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    perf = _measure_generation_performance_on_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    params, _ = _count_model_params(model)
    disk_size_mb = get_model_disk_size_bytes(model_path) / (1024 * 1024)

    return {
        "latency_ms": perf["latency_ms"],
        "throughput_tokens_per_sec": perf["throughput_tokens_per_sec"],
        "tokens_processed": perf["tokens_processed"],
        "inference_time_sec": perf["inference_time_sec"],
        "params": float(params),
        "disk_size_mb": float(disk_size_mb),
    }


def _build_prompt_batch(dataset: Any, samples: int) -> list[str]:
    n = min(max(0, int(samples)), len(dataset))
    prompts: list[str] = []
    for i in range(n):
        prompts.append(PROMPT_TEMPLATE.format(prompt=dataset[i]["prompt"]))
    return prompts


def _count_model_params(model: torch.nn.Module) -> tuple[int, int]:
    total = 0
    nonzero = 0
    for param in model.parameters():
        tensor = param.detach()
        total += tensor.numel()
        nonzero += int(torch.count_nonzero(tensor).item())
    return total, nonzero


def _measure_generation_performance_on_prompts(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, float]:
    if not prompts:
        return {
            "latency_ms": 0.0,
            "throughput_tokens_per_sec": 0.0,
            "tokens_processed": 0.0,
            "inference_time_sec": 0.0,
            "samples": 0.0,
        }

    model.eval()

    total_time = 0.0
    tokens_processed = 0

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

            _synchronize_device(device)
            start = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            _synchronize_device(device)
            elapsed = max(0.0, time.perf_counter() - start)

            total_time += elapsed
            generated_tokens = int(max(0, outputs[0].shape[-1] - inputs["input_ids"].shape[1]))
            tokens_processed += generated_tokens

    safe_time = max(total_time, 1e-9)
    latency_ms = float((safe_time / max(1, len(prompts))) * 1000.0)
    throughput = float(tokens_processed / safe_time)

    if not math.isfinite(throughput) or throughput < 0:
        throughput = 0.0

    return {
        "latency_ms": latency_ms,
        "throughput_tokens_per_sec": throughput,
        "tokens_processed": float(tokens_processed),
        "inference_time_sec": float(total_time),
        "samples": float(len(prompts)),
    }


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        return

    if (
        device.type == "mps"
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "synchronize")
    ):
        torch.mps.synchronize()
