from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import logging
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import yaml

_MPS_PEAK_ALLOCATED_BYTES = 0


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some ops may not have deterministic kernels on every backend.
            pass
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def infer_device(device: str = "auto") -> torch.device:
    normalized = str(device).lower().strip()
    if normalized == "auto":
        if _is_apple_silicon() and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    requested = torch.device(normalized)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device but CUDA is not available.")
    if requested.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS device but MPS is not available on this host.")
    return requested


def validate_runtime_precision(device: torch.device, fp16: bool, bf16: bool) -> None:
    if fp16 and device.type != "cuda":
        raise ValueError(
            "runtime.fp16=true is only supported with CUDA in Hugging Face Trainer. "
            "Set runtime.fp16=false for MPS/CPU."
        )

    if bf16 and device.type != "cuda":
        raise ValueError(
            "runtime.bf16=true is only supported with CUDA in this pipeline. "
            "Set runtime.bf16=false for MPS/CPU."
        )


def validate_quantization_config(device: torch.device, use_qlora: bool) -> None:
    if not use_qlora:
        return

    if device.type != "cuda":
        raise RuntimeError(
            "QLoRA(bitsandbytes 4-bit) is CUDA-only. On Apple Silicon(MPS), use LoRA "
            "(use_qlora=false). Optional torchao quantization is not enabled in this pipeline."
        )

    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "use_qlora=true requires bitsandbytes. Install optional deps with: "
            "python -m pip install -e .[qlora]"
        )


def configure_model_for_training(model: torch.nn.Module, gradient_checkpointing: bool) -> bool | None:
    previous_use_cache = None
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        previous_use_cache = bool(model.config.use_cache)

    if gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    return previous_use_cache


def restore_model_use_cache(model: torch.nn.Module, previous_use_cache: bool | None) -> None:
    if previous_use_cache is None:
        return
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = previous_use_cache


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def dump_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = 0
    nonzero = 0
    for param in model.parameters():
        p = param.detach()
        total += p.numel()
        nonzero += int(torch.count_nonzero(p).item())
    return total, nonzero


def get_model_disk_size_bytes(model_dir: str | Path) -> int:
    model_dir = Path(model_dir)
    total = 0
    for p in model_dir.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


@dataclass
class MemoryStats:
    rss_mb: float
    device: str
    device_allocated_mb: float
    peak_allocated_mb: float


def collect_memory_stats(device: torch.device, reset_peak: bool = False) -> MemoryStats:
    global _MPS_PEAK_ALLOCATED_BYTES

    process = psutil.Process()
    rss_bytes = process.memory_info().rss
    device_allocated = 0
    peak_allocated = 0

    if device.type == "cuda" and torch.cuda.is_available():
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(device)
        device_allocated = int(torch.cuda.memory_allocated(device))
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
    elif device.type == "mps" and torch.backends.mps.is_available():
        current_allocated = 0
        if hasattr(torch.mps, "current_allocated_memory"):
            current_allocated = int(torch.mps.current_allocated_memory())

        if reset_peak:
            _MPS_PEAK_ALLOCATED_BYTES = current_allocated
        else:
            _MPS_PEAK_ALLOCATED_BYTES = max(_MPS_PEAK_ALLOCATED_BYTES, current_allocated)

        device_allocated = current_allocated
        peak_allocated = _MPS_PEAK_ALLOCATED_BYTES

    return MemoryStats(
        rss_mb=rss_bytes / (1024 * 1024),
        device=device.type,
        device_allocated_mb=device_allocated / (1024 * 1024),
        peak_allocated_mb=peak_allocated / (1024 * 1024),
    )


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}
