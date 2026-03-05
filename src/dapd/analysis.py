from __future__ import annotations

"""Forward-only analysis utilities for DAPD.

This module provides memory-efficient analysis functions designed for Apple
Silicon MPS environments:
1) Teacher distribution analysis (entropy, confidence, ECE, KL comparison)
2) Distillation dynamics logging callback for Hugging Face Trainer
3) Pruning pattern analysis before/after pruning artifacts
4) OOD comparison between two student models

All analysis paths run inference-only (no optimization/training steps).
"""

import gc
import json
import math
from pathlib import Path
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from .metrics.core import compute_qa_metrics
from .utils import get_logger


def analyze_teacher_distributions(
    general_teacher_path: str,
    domain_teacher_path: str,
    dataset: Any,  # PreparedDatasets.validation_text
    lm_dataset: Any,  # PreparedDatasets.validation_lm
    device: torch.device,
    max_samples: int = 200,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Analyze teacher distributions with sequential model loading.

    Args:
        general_teacher_path: Path/name for the general teacher model.
        domain_teacher_path: Path/name for the domain-adapted teacher model.
        dataset: Text dataset aligned to lm_dataset (used for sanity checks only).
        lm_dataset: Tokenized LM dataset containing input_ids/attention_mask/labels.
        device: Torch device (supports cpu/cuda/mps).
        max_samples: Maximum number of samples to analyze.
        batch_size: Inference batch size for forward-only passes.

    Returns:
        A dict with teacher-wise summary stats, comparison metrics, and per-sample
        entropy/confidence arrays.
    """
    logger = get_logger("dapd.analysis")
    del dataset  # lm_dataset already contains aligned token-level supervision.

    general = _collect_teacher_forward_stats(
        model_path=general_teacher_path,
        lm_dataset=lm_dataset,
        device=device,
        max_samples=max_samples,
        batch_size=batch_size,
        logger=logger,
    )
    _release_model_memory(device=device)

    domain = _collect_teacher_forward_stats(
        model_path=domain_teacher_path,
        lm_dataset=lm_dataset,
        device=device,
        max_samples=max_samples,
        batch_size=batch_size,
        logger=logger,
    )
    _release_model_memory(device=device)

    n = min(len(general["mean_probs"]), len(domain["mean_probs"]))
    kl_per_sample: list[float] = []
    mi_proxy: list[float] = []
    for idx in range(n):
        pg = general["mean_probs"][idx].float()
        pd = domain["mean_probs"][idx].float()
        pg = pg / pg.sum().clamp_min(1e-9)
        pd = pd / pd.sum().clamp_min(1e-9)
        kl = torch.sum(pg * torch.log((pg + 1e-9) / (pd + 1e-9)))
        kl_per_sample.append(float(kl.item()))
        mi_proxy.append(float(domain["confidence"][idx] - general["confidence"][idx]))

    result = {
        "general": {
            "entropy_mean": _safe_mean(general["entropy"]),
            "entropy_std": _safe_std(general["entropy"]),
            "confidence_mean": _safe_mean(general["confidence"]),
            "confidence_std": _safe_std(general["confidence"]),
            "ece": float(general["ece"]),
            "top1_acc": _safe_mean(general["top1_acc"]),
        },
        "domain": {
            "entropy_mean": _safe_mean(domain["entropy"]),
            "entropy_std": _safe_std(domain["entropy"]),
            "confidence_mean": _safe_mean(domain["confidence"]),
            "confidence_std": _safe_std(domain["confidence"]),
            "ece": float(domain["ece"]),
            "top1_acc": _safe_mean(domain["top1_acc"]),
        },
        "comparison": {
            "entropy_reduction": float(
                _safe_mean(general["entropy"]) - _safe_mean(domain["entropy"])
            ),
            "confidence_gain": float(
                _safe_mean(domain["confidence"]) - _safe_mean(general["confidence"])
            ),
            "kl_between_teachers_mean": _safe_mean(kl_per_sample),
            "kl_between_teachers_std": _safe_std(kl_per_sample),
        },
        "per_sample": {
            "general_entropy": [float(x) for x in general["entropy"]],
            "domain_entropy": [float(x) for x in domain["entropy"]],
            "general_conf": [float(x) for x in general["confidence"]],
            "domain_conf": [float(x) for x in domain["confidence"]],
        },
    }
    if kl_per_sample:
        result["per_sample"]["kl_general_to_domain"] = [float(x) for x in kl_per_sample]
    if mi_proxy:
        result["per_sample"]["mi_proxy_conf_gain"] = [float(x) for x in mi_proxy]

    logger.info(
        "teacher distribution analysis complete | n=%s | entropy_reduction=%.4f | confidence_gain=%.4f",
        n,
        result["comparison"]["entropy_reduction"],
        result["comparison"]["confidence_gain"],
    )
    return result


def create_dynamics_callback(output_path: str) -> Any:
    """Create a TrainerCallback that logs distillation dynamics to JSON.

    Args:
        output_path: Destination JSON path for logged training/eval dynamics.

    Returns:
        A TrainerCallback subclass instance compatible with HF Trainer and
        ProgressiveDistillationTrainer.
    """
    logger = get_logger("dapd.analysis")

    class DistillationDynamicsCallback(TrainerCallback):
        def __init__(self) -> None:
            self.output_path = Path(output_path)
            self.records: dict[str, list[dict[str, Any]]] = {
                "log_steps": [],
                "eval_steps": [],
            }

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            del args, control
            logs = logs or {}
            trainer = kwargs.get("trainer")
            row = {
                "global_step": int(getattr(state, "global_step", 0) or 0),
                "temperature": _resolve_temperature(trainer=trainer, state=state),
                "ce_loss": _to_float(logs.get("ce_loss")),
                "kd_loss": _to_float(logs.get("kd_loss")),
                "learning_rate": _to_float(logs.get("learning_rate", logs.get("lr"))),
            }
            if row["ce_loss"] is None and "loss" in logs:
                row["ce_loss"] = _to_float(logs.get("loss"))
            self.records["log_steps"].append(row)
            return control

        def on_evaluate(
            self,
            args: Any,
            state: Any,
            control: Any,
            metrics: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            del args, kwargs
            metrics = metrics or {}
            self.records["eval_steps"].append(
                {
                    "global_step": int(getattr(state, "global_step", 0) or 0),
                    "eval_loss": _to_float(metrics.get("eval_loss")),
                    "eval_accuracy": _to_float(
                        metrics.get("eval_accuracy", metrics.get("accuracy"))
                    ),
                }
            )
            return control

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, state, kwargs
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=2, ensure_ascii=False)
            logger.info("saved distillation dynamics callback records: %s", self.output_path)
            return control

    return DistillationDynamicsCallback()


def analyze_pruning_patterns(
    model_path_before: str,
    model_path_after: str,
    device: torch.device,
) -> dict[str, Any]:
    """Analyze layer-wise pruning patterns from pre/post pruning models.

    Args:
        model_path_before: Model path before pruning.
        model_path_after: Model path after pruning.
        device: Torch device (supports cpu/cuda/mps).

    Returns:
        Layer-level and grouped sparsity summary dictionary.
    """
    logger = get_logger("dapd.analysis")

    before = _collect_model_sparsity(model_path=model_path_before, device=device, logger=logger)
    _release_model_memory(device=device)
    after = _collect_model_sparsity(model_path=model_path_after, device=device, logger=logger)
    _release_model_memory(device=device)

    all_layers = sorted(set(before["layer_attn"]) | set(before["layer_mlp"]) | set(after["layer_attn"]) | set(after["layer_mlp"]))

    layer_sparsity: list[dict[str, float | int]] = []
    ranking: list[tuple[int, float]] = []
    attn_series: list[dict[str, float | int]] = []
    mlp_series: list[dict[str, float | int]] = []

    for layer_idx in all_layers:
        attn_after = _safe_mean(after["layer_attn"].get(layer_idx, []))
        mlp_after = _safe_mean(after["layer_mlp"].get(layer_idx, []))
        layer_sparsity.append(
            {
                "layer_idx": int(layer_idx),
                "attn_sparsity": float(attn_after),
                "mlp_sparsity": float(mlp_after),
            }
        )
        attn_series.append({"layer_idx": int(layer_idx), "sparsity": float(attn_after)})
        mlp_series.append({"layer_idx": int(layer_idx), "sparsity": float(mlp_after)})

        attn_before = _safe_mean(before["layer_attn"].get(layer_idx, []))
        mlp_before = _safe_mean(before["layer_mlp"].get(layer_idx, []))
        delta = ((attn_after + mlp_after) / 2.0) - ((attn_before + mlp_before) / 2.0)
        ranking.append((int(layer_idx), float(delta)))

    ranking_sorted = sorted(ranking, key=lambda x: x[1], reverse=True)
    most_pruned_layers = [idx for idx, _ in ranking_sorted[:3]]
    least_pruned_layers = [idx for idx, _ in ranking_sorted[-3:]]

    result = {
        "layer_sparsity": layer_sparsity,
        "most_pruned_layers": most_pruned_layers,
        "least_pruned_layers": least_pruned_layers,
        "total_sparsity": float(after["total_sparsity"]),
        "attn_head_sparsity_per_layer": attn_series,
        "mlp_sparsity_per_layer": mlp_series,
    }
    logger.info(
        "pruning pattern analysis complete | total_sparsity=%.4f | layers=%s",
        result["total_sparsity"],
        len(layer_sparsity),
    )
    return result


def compute_ood_comparison(
    general_student_path: str,
    domain_student_path: str,
    ood_dataset: Any,
    device: torch.device,
    max_samples: int = 200,
) -> dict[str, Any]:
    """Evaluate OOD performance of general-student vs domain-student models.

    Args:
        general_student_path: Path/name for student distilled from general teacher.
        domain_student_path: Path/name for student distilled from domain teacher.
        ood_dataset: OOD text dataset compatible with compute_qa_metrics.
        device: Torch device (supports cpu/cuda/mps).
        max_samples: Max OOD samples to evaluate.

    Returns:
        A dict with per-model metrics and relative gains.
    """
    logger = get_logger("dapd.analysis")

    general_metrics = _evaluate_qa_model(
        model_path=general_student_path,
        dataset=ood_dataset,
        device=device,
        max_samples=max_samples,
    )
    _release_model_memory(device=device)
    domain_metrics = _evaluate_qa_model(
        model_path=domain_student_path,
        dataset=ood_dataset,
        device=device,
        max_samples=max_samples,
    )
    _release_model_memory(device=device)

    result = {
        "general_student": general_metrics,
        "domain_student": domain_metrics,
        "comparison": {
            "accuracy_gain": float(domain_metrics["accuracy"] - general_metrics["accuracy"]),
            "f1_gain": float(domain_metrics["f1"] - general_metrics["f1"]),
        },
    }
    logger.info(
        "OOD comparison complete | acc_gain=%.4f | f1_gain=%.4f",
        result["comparison"]["accuracy_gain"],
        result["comparison"]["f1_gain"],
    )
    return result


def _collect_teacher_forward_stats(
    model_path: str,
    lm_dataset: Any,
    device: torch.device,
    max_samples: int,
    batch_size: int,
    logger: Any,
) -> dict[str, Any]:
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    logger.info("loaded teacher for analysis: %s", model_path)

    n = min(max(0, int(max_samples)), len(lm_dataset))
    bs = max(1, int(batch_size))

    entropy_per_sample: list[float] = []
    confidence_per_sample: list[float] = []
    top1_per_sample: list[float] = []
    mean_probs: list[torch.Tensor] = []
    all_confidences: list[torch.Tensor] = []
    all_correctness: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, n, bs):
            samples = [lm_dataset[idx] for idx in range(start, min(start + bs, n))]
            if not samples:
                continue
            batch = _collate_lm_samples(samples=samples, device=device)

            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits[:, :-1, :].float()
            labels = batch["labels"][:, 1:]
            valid = labels != -100
            if valid.sum().item() == 0:
                continue

            probs = torch.softmax(logits, dim=-1)
            token_entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
            pred = probs.argmax(dim=-1)
            pred_conf = probs.max(dim=-1).values

            safe_labels = labels.clamp_min(0)
            label_prob = probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

            batch_size_actual = probs.shape[0]
            for row_idx in range(batch_size_actual):
                mask = valid[row_idx]
                valid_count = int(mask.sum().item())
                if valid_count == 0:
                    continue
                entropy_per_sample.append(float(token_entropy[row_idx][mask].mean().item()))
                confidence_per_sample.append(float(label_prob[row_idx][mask].mean().item()))
                corr = (pred[row_idx][mask] == labels[row_idx][mask]).float()
                top1_per_sample.append(float(corr.mean().item()))
                all_confidences.append(pred_conf[row_idx][mask].detach().cpu())
                all_correctness.append(corr.detach().cpu())

                sample_mean_probs = probs[row_idx][mask].mean(dim=0).detach().cpu()
                sample_mean_probs = sample_mean_probs / sample_mean_probs.sum().clamp_min(1e-9)
                mean_probs.append(sample_mean_probs)

    del model
    return {
        "entropy": entropy_per_sample,
        "confidence": confidence_per_sample,
        "top1_acc": top1_per_sample,
        "ece": _compute_ece(confidences=all_confidences, correctness=all_correctness, n_bins=15),
        "mean_probs": mean_probs,
    }


def _collate_lm_samples(samples: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    max_len = max(len(sample["input_ids"]) for sample in samples)
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    label_rows: list[list[int]] = []

    for sample in samples:
        input_ids = list(sample["input_ids"])
        attention_mask = list(sample.get("attention_mask", [1] * len(input_ids)))
        labels = list(sample.get("labels", input_ids))
        if len(labels) < len(input_ids):
            labels = labels + [-100] * (len(input_ids) - len(labels))

        pad_len = max_len - len(input_ids)
        input_rows.append(input_ids + [0] * pad_len)
        mask_rows.append(attention_mask + [0] * pad_len)
        label_rows.append(labels + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long, device=device),
        "labels": torch.tensor(label_rows, dtype=torch.long, device=device),
    }


def _compute_ece(
    confidences: list[torch.Tensor],
    correctness: list[torch.Tensor],
    n_bins: int = 15,
) -> float:
    if not confidences or not correctness:
        return 0.0
    conf = torch.cat(confidences).float().clamp(min=0.0, max=1.0)
    corr = torch.cat(correctness).float().clamp(min=0.0, max=1.0)
    if conf.numel() == 0:
        return 0.0

    bins = torch.linspace(0.0, 1.0, steps=max(2, int(n_bins) + 1))
    total = float(conf.numel())
    ece = 0.0
    for idx in range(len(bins) - 1):
        lo = bins[idx]
        hi = bins[idx + 1]
        in_bin = (conf >= lo) & (conf <= hi if idx == len(bins) - 2 else conf < hi)
        count = int(in_bin.sum().item())
        if count == 0:
            continue
        acc_bin = float(corr[in_bin].mean().item())
        conf_bin = float(conf[in_bin].mean().item())
        ece += abs(acc_bin - conf_bin) * (count / total)
    return float(ece)


def _resolve_temperature(trainer: Any, state: Any) -> float | None:
    if trainer is None:
        return None

    schedule_obj = getattr(trainer, "temperature_schedule", None)
    if isinstance(schedule_obj, (int, float)):
        return float(schedule_obj)

    if callable(schedule_obj):
        try:
            return float(schedule_obj(int(getattr(state, "global_step", 0) or 0)))
        except Exception:
            pass

    base = getattr(trainer, "base_temperature", None)
    min_temp = getattr(trainer, "min_temperature", None)
    schedule_name = getattr(trainer, "temperature_schedule", None)
    if base is None or min_temp is None or schedule_name is None:
        return None

    step = int(getattr(state, "global_step", 0) or 0)
    max_steps = int(getattr(state, "max_steps", 0) or 0)
    if max_steps <= 0:
        trainer_args = getattr(trainer, "args", None)
        max_steps = int(getattr(trainer_args, "max_steps", 0) or 0)
    max_steps = max(1, max_steps)
    return float(_scheduled_temperature(step, max_steps, float(base), float(min_temp), str(schedule_name)))


def _scheduled_temperature(
    step: int,
    max_steps: int,
    base_temperature: float,
    min_temperature: float,
    schedule: str,
) -> float:
    mode = schedule.lower().strip()
    if mode == "constant":
        return float(base_temperature)
    if max_steps <= 1:
        return float(base_temperature)

    progress = min(max(step / (max_steps - 1), 0.0), 1.0)
    if mode == "linear":
        return float(base_temperature + (min_temperature - base_temperature) * progress)
    if mode == "cosine":
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_temperature + (base_temperature - min_temperature) * cosine)
    return float(base_temperature)


def _collect_model_sparsity(model_path: str, device: torch.device, logger: Any) -> dict[str, Any]:
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    logger.info("loaded model for pruning analysis: %s", model_path)

    layer_attn: dict[int, list[float]] = {}
    layer_mlp: dict[int, list[float]] = {}
    total_zero = 0.0
    total_count = 0.0

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = module.weight.detach()
        count = float(weight.numel())
        zero = float((weight == 0).sum().item())
        sparsity = zero / max(1.0, count)

        total_zero += zero
        total_count += count

        layer_idx = _extract_layer_index(name)
        if layer_idx is None:
            continue
        if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")):
            layer_attn.setdefault(layer_idx, []).append(float(sparsity))
        elif name.endswith(("gate_proj", "up_proj", "down_proj")):
            layer_mlp.setdefault(layer_idx, []).append(float(sparsity))

    del model
    return {
        "layer_attn": layer_attn,
        "layer_mlp": layer_mlp,
        "total_sparsity": float(total_zero / max(1.0, total_count)),
    }


def _extract_layer_index(module_name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", module_name)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _evaluate_qa_model(
    model_path: str,
    dataset: Any,
    device: torch.device,
    max_samples: int,
) -> dict[str, float]:
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()
    metrics = compute_qa_metrics(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_samples=int(max_samples),
        max_new_tokens=32,
        temperature=0.0,
        device=device,
    )
    del model
    return {"accuracy": float(metrics["accuracy"]), "f1": float(metrics["f1"])}


def _release_model_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(torch.tensor(values, dtype=torch.float32).mean().item())


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(torch.tensor(values, dtype=torch.float32).std(unbiased=False).item())


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None

