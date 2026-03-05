from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import (
    compute_compression_ratio,
    compute_perplexity,
    compute_qa_metrics,
    measure_generation_performance,
)
from .utils import (
    collect_memory_stats,
    count_parameters,
    get_logger,
    get_model_disk_size_bytes,
    infer_device,
)


def evaluate_model(
    model_path: str,
    text_dataset: Any,
    lm_dataset: Any,
    config: Any,
    runtime: Any,
    reference_model_path: str | None = None,
) -> dict[str, Any]:
    logger = get_logger("dapd.evaluation", getattr(runtime, "log_level", "INFO"))

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = infer_device(runtime.device)
    model.to(device)
    model.eval()

    ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset=lm_dataset,
        batch_size=config.batch_size,
        device=device,
    )
    qa_metrics = compute_qa_metrics(
        model=model,
        tokenizer=tokenizer,
        dataset=text_dataset,
        max_samples=config.max_eval_samples,
        max_new_tokens=config.max_new_tokens,
        temperature=config.generation_temperature,
        device=device,
    )
    perf = measure_generation_performance(
        model=model,
        tokenizer=tokenizer,
        dataset=text_dataset,
        samples=config.num_latency_samples,
        max_new_tokens=config.max_new_tokens,
        device=device,
    )

    total_params, nonzero_params = count_parameters(model)
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
            dataset=text_dataset,
            max_new_tokens=config.max_new_tokens,
            samples=config.num_latency_samples,
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
        out["disk_compression_ratio"] = compute_compression_ratio(ref_disk_mb, out["model_disk_size_mb"])
    else:
        out["disk_compression_ratio"] = 1.0

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
    dataset: Any,
    max_new_tokens: int,
    samples: int,
    device: torch.device,
) -> dict[str, float]:
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    perf = measure_generation_performance(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        samples=samples,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    params, _ = count_parameters(model)
    disk_size_mb = get_model_disk_size_bytes(model_path) / (1024 * 1024)

    return {
        "latency_ms": perf["latency_ms"],
        "throughput_tokens_per_sec": perf["throughput_tokens_per_sec"],
        "tokens_processed": perf["tokens_processed"],
        "inference_time_sec": perf["inference_time_sec"],
        "params": float(params),
        "disk_size_mb": float(disk_size_mb),
    }
