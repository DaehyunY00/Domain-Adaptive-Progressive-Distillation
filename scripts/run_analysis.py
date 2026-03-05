#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


def _ensure_local_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.is_dir():
        src_path = str(src_dir)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)


_ensure_local_src_on_path()

from transformers import AutoTokenizer

from dapd.analysis import analyze_pruning_patterns, analyze_teacher_distributions, compute_ood_comparison
from dapd.config import PipelineConfig
from dapd.data import (
    bioasq_proxy_fallback_used,
    build_external_eval_dataset,
    build_unified_dataset,
    prepare_datasets_from_unified,
    reset_bioasq_proxy_fallback_flag,
)
from dapd.utils import dump_json, ensure_dir, get_logger, infer_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-training analysis suite for DAPD")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--general_teacher",
        type=str,
        default=None,
        help="General teacher path/name (default: config adaptation.teacher_model_name_or_path)",
    )
    parser.add_argument(
        "--domain_teacher",
        type=str,
        default=None,
        help="Domain teacher path (default: inferred from adaptation output dir)",
    )
    parser.add_argument(
        "--distilled_student",
        type=str,
        default=None,
        help="Distilled student path (default: <distillation.output_dir>/final)",
    )
    parser.add_argument(
        "--pruned_student",
        type=str,
        default=None,
        help="Pruned student path (default: <pruning.output_dir>/final)",
    )
    parser.add_argument(
        "--ood_dataset",
        type=str,
        default="bioasq",
        help="OOD dataset name (default: bioasq)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Analysis output directory (default: config.analysis.output_dir)",
    )
    args = parser.parse_args()

    reset_bioasq_proxy_fallback_flag()
    cfg = PipelineConfig.from_yaml(args.config)
    logger = get_logger("dapd.run_analysis", getattr(cfg.runtime, "log_level", "INFO"))
    device = infer_device(cfg.runtime.device)

    output_dir = ensure_dir(args.output_dir or cfg.analysis.output_dir)
    general_teacher = args.general_teacher or cfg.adaptation.teacher_model_name_or_path
    domain_teacher = args.domain_teacher or _resolve_domain_teacher_path(cfg.adaptation.output_dir)
    distilled_student = args.distilled_student or str((Path(cfg.distillation.output_dir) / "final").resolve())
    pruned_student = args.pruned_student or str((Path(cfg.pruning.output_dir) / "final").resolve())

    logger.info("analysis device=%s", device)
    logger.info("general_teacher=%s", general_teacher)
    logger.info("domain_teacher=%s", domain_teacher)
    logger.info("distilled_student=%s", distilled_student)
    logger.info("pruned_student=%s", pruned_student)

    results: dict[str, Any] = {
        "paths": {
            "general_teacher": general_teacher,
            "domain_teacher": domain_teacher,
            "distilled_student": distilled_student,
            "pruned_student": pruned_student,
            "output_dir": str(output_dir),
        },
        "device": str(device),
        "ood_dataset": args.ood_dataset,
    }

    # 1) Teacher distribution analysis
    try:
        unified = build_unified_dataset(cfg.data)
        teacher_tokenizer = AutoTokenizer.from_pretrained(general_teacher, use_fast=True)
        teacher_data = prepare_datasets_from_unified(
            unified=unified,
            config=cfg.data,
            tokenizer=teacher_tokenizer,
        )
        teacher_analysis = analyze_teacher_distributions(
            general_teacher_path=general_teacher,
            domain_teacher_path=domain_teacher,
            dataset=teacher_data.validation_text,
            lm_dataset=teacher_data.validation_lm,
            device=device,
            max_samples=max(
                1,
                min(
                    int(getattr(cfg.evaluation, "teacher_analysis_samples", 200)),
                    len(teacher_data.validation_text),
                ),
            ),
            batch_size=max(1, int(getattr(cfg.evaluation, "batch_size", 1))),
        )
        teacher_path = Path(output_dir) / "teacher_analysis.json"
        dump_json(teacher_analysis, teacher_path)
        results["teacher_analysis"] = teacher_analysis
        results["teacher_analysis_path"] = str(teacher_path.resolve())
    except Exception as exc:
        logger.warning("teacher analysis failed: %s", exc)
        results["teacher_analysis"] = {"error": str(exc)}
        results["teacher_analysis_path"] = None

    # 2) Distillation dynamics summary
    try:
        dynamics_path = _resolve_dynamics_path(cfg, distilled_student)
        dynamics_summary = _summarize_dynamics(dynamics_path)
        dynamics_summary_path = Path(output_dir) / "dynamics_summary.json"
        dump_json(dynamics_summary, dynamics_summary_path)
        results["dynamics_summary"] = dynamics_summary
        results["dynamics_summary_path"] = str(dynamics_summary_path.resolve())
    except Exception as exc:
        logger.warning("dynamics summary failed: %s", exc)
        results["dynamics_summary"] = {"error": str(exc)}
        results["dynamics_summary_path"] = None

    # 3) Pruning pattern analysis
    try:
        pruning_analysis = analyze_pruning_patterns(
            model_path_before=distilled_student,
            model_path_after=pruned_student,
            device=device,
        )
        pruning_path = Path(output_dir) / "pruning_analysis.json"
        dump_json(pruning_analysis, pruning_path)
        results["pruning_analysis"] = pruning_analysis
        results["pruning_analysis_path"] = str(pruning_path.resolve())
    except Exception as exc:
        logger.warning("pruning analysis failed: %s", exc)
        results["pruning_analysis"] = {"error": str(exc)}
        results["pruning_analysis_path"] = None

    # 4) OOD comparison
    try:
        ood_dataset = build_external_eval_dataset(
            dataset_name=args.ood_dataset,
            cache_dir=cfg.data.cache_dir,
            seed=cfg.data.seed,
            max_eval_samples=getattr(cfg.evaluation, "ood_max_eval_samples", cfg.evaluation.max_eval_samples),
        )
        ood_comparison = compute_ood_comparison(
            general_student_path=distilled_student,
            domain_student_path=pruned_student,
            ood_dataset=ood_dataset,
            device=device,
            max_samples=int(getattr(cfg.evaluation, "ood_max_eval_samples", cfg.evaluation.max_eval_samples)),
        )
        ood_path = Path(output_dir) / "ood_comparison.json"
        dump_json(ood_comparison, ood_path)
        results["ood_comparison"] = ood_comparison
        results["ood_comparison_path"] = str(ood_path.resolve())
    except Exception as exc:
        logger.warning("OOD comparison failed: %s", exc)
        results["ood_comparison"] = {"error": str(exc)}
        results["ood_comparison_path"] = None

    summary_path = Path(output_dir) / "analysis_summary.json"
    dump_json(results, summary_path)
    if bioasq_proxy_fallback_used():
        raise RuntimeError(
            "BioASQ fallback to pubmed_qa proxy was used during analysis. "
            "OOD comparison is not valid for reporting."
        )
    print("\nAnalysis Summary")
    _print_summary_table(results)
    print(f"\nSaved analysis summary JSON: {summary_path}")


def _resolve_domain_teacher_path(adaptation_output_dir: str) -> str:
    root = Path(adaptation_output_dir).expanduser().resolve()
    candidates = [root / "merged", root / "final", root / "adapter", root]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(root / "merged")


def _resolve_dynamics_path(cfg: PipelineConfig, distilled_student_path: str) -> Path:
    candidates = [
        Path(cfg.distillation.output_dir) / "distillation_dynamics.json",
        Path(distilled_student_path).resolve().parent / "distillation_dynamics.json",
        Path(cfg.distillation.output_dir).resolve().parent / "distillation_dynamics.json",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    # Return the first candidate to keep path metadata even when file is missing.
    return candidates[0].resolve()


def _summarize_dynamics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path), "reason": "missing_file"}

    raw = json.loads(path.read_text(encoding="utf-8"))
    log_steps = raw.get("log_steps", []) or []
    eval_steps = raw.get("eval_steps", []) or []

    temperature = _extract_numeric(log_steps, "temperature")
    ce_loss = _extract_numeric(log_steps, "ce_loss")
    kd_loss = _extract_numeric(log_steps, "kd_loss")
    lr = _extract_numeric(log_steps, "learning_rate")
    eval_loss = _extract_numeric(eval_steps, "eval_loss")
    eval_acc = _extract_numeric(eval_steps, "eval_accuracy")

    return {
        "available": True,
        "path": str(path),
        "num_log_steps": int(len(log_steps)),
        "num_eval_steps": int(len(eval_steps)),
        "temperature": _series_stats(temperature),
        "ce_loss": _series_stats(ce_loss),
        "kd_loss": _series_stats(kd_loss),
        "learning_rate": _series_stats(lr),
        "eval_loss": _series_stats(eval_loss),
        "eval_accuracy": _series_stats(eval_acc),
        "global_step_start": _global_step(log_steps[0]) if log_steps else 0,
        "global_step_end": _global_step(log_steps[-1]) if log_steps else 0,
    }


def _extract_numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        raw = row.get(key)
        if raw is None:
            continue
        try:
            val = float(raw)
        except Exception:
            continue
        if math.isfinite(val):
            vals.append(val)
    return vals


def _series_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = float(statistics.fmean(values))
    std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return {
        "count": float(len(values)),
        "mean": mean,
        "std": std,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _global_step(row: dict[str, Any]) -> int:
    try:
        return int(row.get("global_step", 0))
    except Exception:
        return 0


def _print_summary_table(results: dict[str, Any]) -> None:
    rows: list[tuple[str, str]] = []

    teacher = results.get("teacher_analysis", {}) or {}
    comparison = teacher.get("comparison", {}) if isinstance(teacher, dict) else {}
    rows.append(("teacher_entropy_reduction", _fmt(comparison.get("entropy_reduction"))))
    rows.append(("teacher_confidence_gain", _fmt(comparison.get("confidence_gain"))))

    dynamics = results.get("dynamics_summary", {}) or {}
    ce_mean = _nested_get(dynamics, ["ce_loss", "mean"])
    kd_mean = _nested_get(dynamics, ["kd_loss", "mean"])
    rows.append(("dynamics_ce_loss_mean", _fmt(ce_mean)))
    rows.append(("dynamics_kd_loss_mean", _fmt(kd_mean)))

    pruning = results.get("pruning_analysis", {}) or {}
    rows.append(("pruning_total_sparsity", _fmt(pruning.get("total_sparsity"))))

    ood = results.get("ood_comparison", {}) or {}
    ood_cmp = ood.get("comparison", {}) if isinstance(ood, dict) else {}
    rows.append(("ood_accuracy_gain", _fmt(ood_cmp.get("accuracy_gain"))))
    rows.append(("ood_f1_gain", _fmt(ood_cmp.get("f1_gain"))))

    width = max((len(name) for name, _ in rows), default=10)
    print("metric".ljust(width) + " | value")
    print("-" * width + "-+-" + "-" * 12)
    for name, value in rows:
        print(name.ljust(width) + " | " + value)


def _nested_get(data: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
        if cur is None:
            return None
    return cur


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
