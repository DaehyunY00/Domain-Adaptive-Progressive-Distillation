from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .baseline_comparison import generate_baseline_comparison
from .distillation_intervention import run_distillation_interventions
from .flops import estimate_model_flops_gmac, summarize_flops_reduction
from .pruning_patterns import analyze_pruning_patterns as _legacy_analyze_pruning_patterns
from .teacher_calibration import analyze_teacher_calibration, compute_brier_score, compute_ece
from .teacher_distribution import (
    analyze_teacher_distributions as _legacy_analyze_teacher_distributions,
    compute_confidence_distribution,
    compute_entropy_distribution,
    compute_kl_divergence_between_teachers,
)
from .teacher_information import (
    analyze_teacher_information,
    compute_confidence_distribution as compute_information_confidence_distribution,
    compute_entropy_distribution as compute_information_entropy_distribution,
    compute_mutual_information,
)
from .temperature_analysis import run_temperature_schedule_analysis

_FORWARD_ANALYSIS_MODULE = None


def _load_forward_analysis_module() -> Any:
    global _FORWARD_ANALYSIS_MODULE
    if _FORWARD_ANALYSIS_MODULE is not None:
        return _FORWARD_ANALYSIS_MODULE

    file_path = Path(__file__).resolve().parents[1] / "analysis.py"
    spec = importlib.util.spec_from_file_location("dapd._forward_analysis", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load forward analysis module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _FORWARD_ANALYSIS_MODULE = module
    return module


def create_dynamics_callback(output_path: str) -> Any:
    module = _load_forward_analysis_module()
    return module.create_dynamics_callback(output_path)


def analyze_teacher_distributions(*args: Any, **kwargs: Any) -> Any:
    # Forward-only API from src/dapd/analysis.py
    forward_keys = {"lm_dataset", "device", "batch_size"}
    legacy_keys = {"runtime", "output_dir"}
    has_forward = any(key in kwargs for key in forward_keys)
    has_legacy = any(key in kwargs for key in legacy_keys)

    if has_forward and has_legacy:
        raise TypeError(
            "analyze_teacher_distributions received mixed forward/legacy kwargs. "
            "Use either (lm_dataset, device, ...) or (runtime, output_dir, ...)."
        )

    if has_forward:
        module = _load_forward_analysis_module()
        return module.analyze_teacher_distributions(*args, **kwargs)

    # Legacy API from src/dapd/analysis/teacher_distribution.py
    return _legacy_analyze_teacher_distributions(*args, **kwargs)


def analyze_pruning_patterns(*args: Any, **kwargs: Any) -> Any:
    # Forward-only API from src/dapd/analysis.py
    if "model_path_before" in kwargs or "model_path_after" in kwargs:
        module = _load_forward_analysis_module()
        return module.analyze_pruning_patterns(*args, **kwargs)
    # Legacy report-driven API from src/dapd/analysis/pruning_patterns.py
    return _legacy_analyze_pruning_patterns(*args, **kwargs)


def compute_ood_comparison(*args: Any, **kwargs: Any) -> Any:
    module = _load_forward_analysis_module()
    return module.compute_ood_comparison(*args, **kwargs)


__all__ = [
    "analyze_teacher_calibration",
    "compute_ece",
    "compute_brier_score",
    "analyze_teacher_information",
    "compute_mutual_information",
    "compute_information_confidence_distribution",
    "compute_information_entropy_distribution",
    "run_distillation_interventions",
    "analyze_pruning_patterns",
    "estimate_model_flops_gmac",
    "summarize_flops_reduction",
    "generate_baseline_comparison",
    "analyze_teacher_distributions",
    "compute_confidence_distribution",
    "compute_entropy_distribution",
    "compute_kl_divergence_between_teachers",
    "run_temperature_schedule_analysis",
    "create_dynamics_callback",
    "compute_ood_comparison",
]
