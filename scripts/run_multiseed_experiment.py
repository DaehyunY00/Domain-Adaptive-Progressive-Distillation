#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
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

DEFAULT_SEEDS = [42, 123, 777]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DAPD multi-seed experiment")
    parser.add_argument("--config", type=str, required=True, help="Base config YAML path")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SEEDS),
        help="Comma separated seeds (default: 42,123,777)",
    )
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    base_cfg = PipelineConfig.from_yaml(args.config)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        run_root = ensure_dir(Path("runs/dapd/multiseed") / f"seed_{seed}")
        _override_paths(cfg, run_root=run_root, seed=seed)

        print(f"\n=== Running seed {seed} ===")
        summary = DAPDPipeline(cfg).run()
        metric = summary.get("evaluation") or {}
        rows.append(
            {
                "seed": int(seed),
                "accuracy": _to_float(metric.get("accuracy")),
                "f1": _to_float(metric.get("f1")),
                "perplexity": _to_float(metric.get("perplexity")),
                "compression_ratio": _to_float(metric.get("compression_ratio")),
                "throughput_tokens_per_sec": _to_float(metric.get("throughput_tokens_per_sec")),
                "expected_calibration_error": _to_float(metric.get("expected_calibration_error")),
                "brier_score": _to_float(metric.get("brier_score")),
            }
        )

    agg = _aggregate(rows)
    out = {"runs": rows, "aggregate": agg}
    out_path = Path("runs/dapd/statistical_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(out, out_path)
    print(f"\nSaved statistical summary: {out_path}")


def _override_paths(cfg: PipelineConfig, run_root: Path, seed: int) -> None:
    cfg.runtime.seed = int(seed)
    cfg.data.seed = int(seed)
    cfg.adaptation.output_dir = str(run_root / "domain_teacher")
    cfg.distillation.output_dir = str(run_root / "distilled_student")
    cfg.pruning.output_dir = str(run_root / "pruned_student")
    cfg.evaluation.output_file = str(run_root / "eval_metrics.json")
    cfg.evaluation.ood_output_file = str(run_root / "eval_metrics_ood.json")
    cfg.data.tokenized_cache_dir = str(run_root / "cache" / "tokenized")
    cfg.analysis.output_dir = str(run_root / "analysis")


def _parse_seeds(value: str) -> list[int]:
    out: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = [
        "accuracy",
        "f1",
        "perplexity",
        "compression_ratio",
        "throughput_tokens_per_sec",
        "expected_calibration_error",
        "brier_score",
    ]
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if not values:
            continue
        out[key] = {
            "mean": float(statistics.fmean(values)),
            "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        }
    return out


if __name__ == "__main__":
    main()
