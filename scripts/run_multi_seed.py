#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
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

from dapd.config import PipelineConfig
from dapd.pipeline import DAPDPipeline
from dapd.utils import dump_json, ensure_dir


DEFAULT_SEEDS = (42, 123, 777)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DAPD pipeline with multiple seeds")
    parser.add_argument("--config", type=str, required=True, help="Base YAML config")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SEEDS),
        help="Comma-separated seed list (default: 42,123,777)",
    )
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    if not seeds:
        raise ValueError("No valid seeds provided.")

    base_cfg = PipelineConfig.from_yaml(args.config)
    summary_root = Path(base_cfg.distillation.output_dir).resolve().parent
    multi_seed_root = ensure_dir(summary_root / "multi_seed")

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        run_root = ensure_dir(multi_seed_root / f"seed_{seed}")
        _configure_seed_run(cfg=cfg, seed=seed, run_root=run_root)

        print(f"\n=== Running seed: {seed} ===")
        summary = DAPDPipeline(cfg).run()
        metrics = summary.get("evaluation") or {}

        row = {
            "seed": seed,
            "accuracy": _metric(metrics, "accuracy"),
            "f1": _metric(metrics, "f1"),
            "perplexity": _metric(metrics, "perplexity"),
            "compression_ratio": _metric(metrics, "compression_ratio"),
            "throughput_tokens_per_sec": _metric(metrics, "throughput_tokens_per_sec"),
            "speedup_vs_teacher": _metric(metrics, "speedup_vs_teacher"),
            "latency_ms": _metric(metrics, "latency_ms"),
            "memory_usage_mb": _metric(metrics, "memory_usage_mb"),
            "expected_calibration_error": _metric(metrics, "expected_calibration_error"),
            "brier_score": _metric(metrics, "brier_score"),
            "summary_path": str((run_root / "pipeline_summary.json").resolve()),
        }
        rows.append(row)

    aggregate = _aggregate_metrics(rows)
    out = {
        "seeds": rows,
        "aggregate": aggregate,
    }
    summary_path = Path(summary_root) / "statistical_summary.json"
    dump_json(out, summary_path)

    print("\nMulti-seed Summary")
    _print_table(rows, aggregate)
    print(f"\nSaved summary: {summary_path}")


def _parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    return seeds


def _configure_seed_run(cfg: PipelineConfig, seed: int, run_root: Path) -> None:
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


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_keys = [
        "accuracy",
        "f1",
        "perplexity",
        "compression_ratio",
        "throughput_tokens_per_sec",
        "speedup_vs_teacher",
        "latency_ms",
        "memory_usage_mb",
        "expected_calibration_error",
        "brier_score",
    ]
    out: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        vals = [float(row[key]) for row in rows if row.get(key) is not None]
        if not vals:
            continue
        mean = float(statistics.fmean(vals))
        std = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
        out[key] = {"mean": mean, "std": std}
    return out


def _print_table(rows: list[dict[str, Any]], aggregate: dict[str, dict[str, float]]) -> None:
    headers = [
        "seed",
        "accuracy",
        "f1",
        "perplexity",
        "compression_ratio",
        "latency_ms",
    ]
    widths: dict[str, int] = {}
    for h in headers:
        widths[h] = max(len(h), max((len(_fmt(row.get(h))) for row in rows), default=0))

    def _line(vals: list[str]) -> str:
        return " | ".join(v.ljust(widths[h]) for h, v in zip(headers, vals))

    print(_line(headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(_line([_fmt(row.get(h)) for h in headers]))

    print("\nAggregate (mean ± std)")
    for key in ["accuracy", "f1", "perplexity", "compression_ratio", "latency_ms"]:
        stats = aggregate.get(key)
        if not stats:
            continue
        print(f"{key}: {stats['mean']:.4f} ± {stats['std']:.4f}")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
