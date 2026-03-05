#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
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


VARIANTS = ("full", "no_adapt", "no_kd", "no_prune")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DAPD ablation variants")
    parser.add_argument("--config", type=str, required=True, help="Base YAML config")
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(VARIANTS),
        help="Comma-separated variants. Supported: full,no_adapt,no_kd,no_prune",
    )
    args = parser.parse_args()

    requested_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in requested_variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"Unsupported variants: {unknown}. Supported: {list(VARIANTS)}")

    base_cfg = PipelineConfig.from_yaml(args.config)
    ablation_root = Path(base_cfg.distillation.output_dir).resolve().parent / "ablation"
    ensure_dir(ablation_root)

    rows: list[dict[str, Any]] = []
    for variant in requested_variants:
        cfg = copy.deepcopy(base_cfg)
        _apply_variant(cfg, variant=variant, ablation_root=ablation_root)

        print(f"\n=== Running variant: {variant} ===")
        summary = DAPDPipeline(cfg).run()
        metrics = summary.get("evaluation") or {}

        row = {
            "variant": variant,
            "accuracy": _metric(metrics, "accuracy"),
            "f1": _metric(metrics, "f1"),
            "perplexity": _metric(metrics, "perplexity"),
            "compression_ratio": _metric(metrics, "compression_ratio"),
            "throughput_tokens_per_sec": _metric(metrics, "throughput_tokens_per_sec"),
            "speedup_vs_teacher": _metric(metrics, "speedup_vs_teacher"),
            "latency_ms": _metric(metrics, "latency_ms"),
            "memory_usage_mb": _metric(metrics, "memory_usage_mb"),
            "summary_path": str((ablation_root / variant / "pipeline_summary.json").resolve()),
        }
        rows.append(row)

    table_path = ablation_root / "ablation_summary.json"
    dump_json({"variants": rows}, table_path)

    print("\nAblation Summary")
    _print_table(rows)
    print(f"\nSaved ablation summary JSON: {table_path}")


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _apply_variant(cfg: PipelineConfig, variant: str, ablation_root: Path) -> None:
    run_root = ensure_dir(ablation_root / variant)
    cfg.adaptation.output_dir = str(run_root / "domain_teacher")
    cfg.distillation.output_dir = str(run_root / "distilled_student")
    cfg.pruning.output_dir = str(run_root / "pruned_student")
    cfg.evaluation.output_file = str(run_root / "eval_metrics.json")
    cfg.data.tokenized_cache_dir = str(run_root / "cache" / "tokenized")

    # Defaults for full pipeline.
    cfg.adaptation.enabled = True
    cfg.distillation.use_kl = True
    cfg.pruning.enabled = True

    if variant == "full":
        return

    if variant == "no_adapt":
        cfg.adaptation.enabled = False
        return

    if variant == "no_kd":
        cfg.distillation.use_kl = False
        return

    if variant == "no_prune":
        cfg.pruning.enabled = False
        return

    raise ValueError(f"Unsupported variant: {variant}")


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "variant",
        "accuracy",
        "f1",
        "perplexity",
        "compression_ratio",
        "throughput_tokens_per_sec",
        "speedup_vs_teacher",
        "latency_ms",
        "memory_usage_mb",
    ]

    widths: dict[str, int] = {}
    for h in headers:
        cell_max = max([len(_fmt(row.get(h))) for row in rows], default=0)
        widths[h] = max(len(h), cell_max)

    def line(values: list[str]) -> str:
        return " | ".join(v.ljust(widths[h]) for h, v in zip(headers, values))

    print(line(headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(line([_fmt(row.get(h)) for h in headers]))


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
