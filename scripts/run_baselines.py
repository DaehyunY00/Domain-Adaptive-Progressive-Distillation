#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import statistics
import sys
from dataclasses import asdict
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

from dapd.adaptation import run_domain_adaptation
from dapd.config import PipelineConfig
from dapd.data import (
    bioasq_proxy_fallback_used,
    build_unified_dataset,
    prepare_datasets_from_unified,
    reset_bioasq_proxy_fallback_flag,
)
from dapd.evaluation import evaluate_model
from dapd.pipeline import DAPDPipeline
from dapd.pruning import run_structured_pruning
from dapd.utils import dump_json, ensure_dir, get_logger, set_seed


BASELINES = ("direct_kd", "lora_only", "no_distill_prune")
MULTI_SEEDS = (42, 123, 2024)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DAPD baseline comparisons")
    parser.add_argument("--config", type=str, required=True, help="Base YAML config")
    parser.add_argument(
        "--baselines",
        type=str,
        default=",".join(BASELINES),
        help="Comma-separated baselines. Supported: direct_kd,lora_only,no_distill_prune",
    )
    parser.add_argument(
        "--multi_seed",
        action="store_true",
        help="Run each baseline with fixed seeds: 42,123,2024",
    )
    args = parser.parse_args()
    reset_bioasq_proxy_fallback_flag()

    requested = [b.strip() for b in args.baselines.split(",") if b.strip()]
    unknown = [b for b in requested if b not in BASELINES]
    if unknown:
        raise ValueError(f"Unsupported baselines: {unknown}. Supported: {list(BASELINES)}")

    base_cfg = PipelineConfig.from_yaml(args.config)
    baseline_root = Path(base_cfg.distillation.output_dir).resolve().parent / "baselines"
    ensure_dir(baseline_root)

    rows: list[dict[str, Any]] = []
    for baseline in requested:
        seeds = list(MULTI_SEEDS) if args.multi_seed else [int(base_cfg.runtime.seed)]
        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            run_root = ensure_dir(baseline_root / baseline / f"seed_{seed}")
            _configure_run(cfg=cfg, seed=seed, run_root=run_root)
            _apply_baseline(cfg=cfg, baseline=baseline)

            print(f"\n=== Running baseline: {baseline} (seed={seed}) ===")
            if baseline == "no_distill_prune":
                summary = _run_no_distill_prune_baseline(cfg=cfg)
            else:
                summary = DAPDPipeline(cfg).run()

            metrics = summary.get("evaluation") or {}
            row = {
                "baseline": baseline,
                "seed": int(seed),
                "accuracy": _metric(metrics, "accuracy"),
                "f1": _metric(metrics, "f1"),
                "ece": _metric(metrics, "ece"),
                "brier_score": _metric(metrics, "brier_score"),
                "perplexity": _metric(metrics, "perplexity"),
                "compression_ratio": _metric(metrics, "compression_ratio"),
                "throughput_tokens_per_sec": _metric(metrics, "throughput_tokens_per_sec"),
                "speedup_vs_teacher": _metric(metrics, "speedup_vs_teacher"),
                "latency_ms": _metric(metrics, "latency_ms"),
                "memory_usage_mb": _metric(metrics, "memory_usage_mb"),
                "summary_path": str((run_root / "pipeline_summary.json").resolve()),
            }
            rows.append(row)

    aggregate = _aggregate_by_baseline(rows)
    summary_path = baseline_root / "baseline_summary.json"
    dump_json(
        {
            "multi_seed": bool(args.multi_seed),
            "seeds": list(MULTI_SEEDS) if args.multi_seed else [int(base_cfg.runtime.seed)],
            "runs": rows,
            "aggregate": aggregate,
        },
        summary_path,
    )

    print("\nBaseline Summary")
    _print_table(rows)
    if aggregate:
        print("\nAggregate (mean ± std by baseline)")
        _print_aggregate(aggregate)
    if bioasq_proxy_fallback_used():
        raise RuntimeError(
            "BioASQ fallback to pubmed_qa proxy was used during baseline runs. "
            "OOD claims are not valid."
        )
    print(f"\nSaved baseline summary JSON: {summary_path}")


def _configure_run(cfg: PipelineConfig, seed: int, run_root: Path) -> None:
    cfg.runtime.seed = int(seed)
    cfg.data.seed = int(seed)
    cfg.adaptation.output_dir = str(run_root / "domain_teacher")
    cfg.distillation.output_dir = str(run_root / "distilled_student")
    cfg.pruning.output_dir = str(run_root / "pruned_student")
    cfg.evaluation.output_file = str(run_root / "eval_metrics.json")
    cfg.evaluation.ood_output_file = str(run_root / "eval_metrics_ood.json")
    cfg.data.tokenized_cache_dir = str(run_root / "cache" / "tokenized")
    if hasattr(cfg, "analysis"):
        cfg.analysis.output_dir = str(run_root / "analysis")


def _apply_baseline(cfg: PipelineConfig, baseline: str) -> None:
    if baseline == "direct_kd":
        cfg.adaptation.enabled = False
        cfg.distillation.use_kl = True
        cfg.pruning.enabled = False
        return

    if baseline == "lora_only":
        cfg.adaptation.enabled = True
        cfg.distillation.use_kl = False
        cfg.distillation.alpha = 1.0
        cfg.pruning.enabled = False
        return

    if baseline == "no_distill_prune":
        cfg.adaptation.enabled = True
        cfg.pruning.enabled = True
        return

    raise ValueError(f"Unsupported baseline: {baseline}")


def _run_no_distill_prune_baseline(cfg: PipelineConfig) -> dict[str, Any]:
    logger = get_logger("dapd.baselines", getattr(cfg.runtime, "log_level", "INFO"))
    set_seed(cfg.runtime.seed, deterministic=cfg.runtime.deterministic)

    ensure_dir(cfg.adaptation.output_dir)
    ensure_dir(cfg.pruning.output_dir)
    artifact_root = Path(cfg.distillation.output_dir).resolve().parent
    ensure_dir(artifact_root)

    unified = build_unified_dataset(cfg.data)
    teacher_tokenizer = AutoTokenizer.from_pretrained(cfg.adaptation.teacher_model_name_or_path, use_fast=True)
    teacher_data = prepare_datasets_from_unified(
        unified=unified,
        config=cfg.data,
        tokenizer=teacher_tokenizer,
    )

    logger.info("baseline(no_distill_prune) step1: domain adaptation")
    adapt_artifacts = run_domain_adaptation(
        config=cfg.adaptation,
        runtime=cfg.runtime,
        datasets=teacher_data,
    )

    logger.info("baseline(no_distill_prune) step2: prune adapted teacher directly")
    prune_artifacts = run_structured_pruning(
        config=cfg.pruning,
        runtime=cfg.runtime,
        model_path=adapt_artifacts.teacher_path,
        calibration_dataset=teacher_data.validation_lm,
    )
    final_model_path = prune_artifacts.model_path

    logger.info("baseline(no_distill_prune) step3: evaluate pruned teacher")
    eval_tokenizer = AutoTokenizer.from_pretrained(final_model_path, use_fast=True)
    eval_data = prepare_datasets_from_unified(
        unified=unified,
        config=cfg.data,
        tokenizer=eval_tokenizer,
    )
    eval_metrics = evaluate_model(
        model_path=final_model_path,
        text_dataset=eval_data.test_text,
        lm_dataset=eval_data.test_lm,
        config=cfg.evaluation,
        runtime=cfg.runtime,
        reference_model_path=adapt_artifacts.teacher_path,
    )
    dump_json(eval_metrics, cfg.evaluation.output_file)

    summary = {
        "baseline": "no_distill_prune",
        "config": cfg.to_dict(),
        "adaptation": asdict(adapt_artifacts),
        "teacher_logits_source": None,
        "distillation": None,
        "pruning": asdict(prune_artifacts),
        "final_model_path": final_model_path,
        "evaluation": eval_metrics,
    }
    summary_path = Path(cfg.distillation.output_dir).resolve().parent / "pipeline_summary.json"
    dump_json(summary, summary_path)
    return summary


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _aggregate_by_baseline(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    metric_keys = [
        "accuracy",
        "f1",
        "ece",
        "brier_score",
        "perplexity",
        "compression_ratio",
        "throughput_tokens_per_sec",
        "speedup_vs_teacher",
        "latency_ms",
        "memory_usage_mb",
    ]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for baseline in sorted({str(row["baseline"]) for row in rows}):
        baseline_rows = [row for row in rows if row.get("baseline") == baseline]
        baseline_stats: dict[str, dict[str, float]] = {}
        for key in metric_keys:
            vals = [float(row[key]) for row in baseline_rows if row.get(key) is not None]
            if not vals:
                continue
            mean = float(statistics.fmean(vals))
            std = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
            baseline_stats[key] = {"mean": mean, "std": std}
        out[baseline] = baseline_stats
    return out


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "baseline",
        "seed",
        "accuracy",
        "f1",
        "ece",
        "brier_score",
        "perplexity",
        "compression_ratio",
        "throughput_tokens_per_sec",
        "speedup_vs_teacher",
        "latency_ms",
        "memory_usage_mb",
    ]
    widths: dict[str, int] = {}
    for h in headers:
        widths[h] = max(len(h), max((len(_fmt(row.get(h))) for row in rows), default=0))

    def line(values: list[str]) -> str:
        return " | ".join(v.ljust(widths[h]) for h, v in zip(headers, values))

    print(line(headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(line([_fmt(row.get(h)) for h in headers]))


def _print_aggregate(aggregate: dict[str, dict[str, dict[str, float]]]) -> None:
    metric_order = ["accuracy", "f1", "ece", "brier_score", "perplexity", "compression_ratio", "latency_ms"]
    for baseline, stats in aggregate.items():
        print(f"{baseline}:")
        for key in metric_order:
            vals = stats.get(key)
            if not vals:
                continue
            print(f"  {key}: {vals['mean']:.4f} ± {vals['std']:.4f}")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
