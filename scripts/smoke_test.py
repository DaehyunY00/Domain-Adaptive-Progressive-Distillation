#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
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
from dapd.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end DAPD smoke test on MPS-safe config")
    parser.add_argument("--device", type=str, default="auto", help="Runtime device override (default: auto)")
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip actual training loops by setting epochs to zero (data/model/pipeline wiring test)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "configs" / "dapd_mps_fast.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")

    cfg = PipelineConfig.from_yaml(cfg_path)
    run_root = Path(repo_root / "runs" / "smoke_test").resolve()
    ensure_dir(run_root)

    _configure_for_smoke_test(cfg=cfg, run_root=run_root, device=args.device, skip_training=args.skip_training)

    summary = DAPDPipeline(cfg).run()
    _materialize_root_analysis_files(summary=summary, run_root=run_root)
    _verify_outputs(run_root=run_root)
    _print_eval_metrics(run_root=run_root)

    print("SMOKE TEST PASSED")


def _configure_for_smoke_test(
    cfg: PipelineConfig,
    run_root: Path,
    device: str,
    skip_training: bool,
) -> None:
    cfg.runtime.device = str(device)
    cfg.runtime.seed = 42
    cfg.data.seed = 42

    cfg.data.max_train_samples = 50
    cfg.data.max_eval_samples = 20
    cfg.data.ood_max_samples = min(int(getattr(cfg.data, "ood_max_samples", 20)), 20)

    cfg.adaptation.output_dir = str(run_root / "domain_teacher")
    cfg.distillation.output_dir = str(run_root / "distilled_student")
    cfg.pruning.output_dir = str(run_root / "pruned_student")
    cfg.evaluation.output_file = str(run_root / "eval_metrics.json")
    cfg.evaluation.ood_output_file = str(run_root / "eval_metrics_ood.json")
    cfg.data.tokenized_cache_dir = str(run_root / "cache" / "tokenized")

    cfg.evaluation.num_latency_samples = min(int(getattr(cfg.evaluation, "num_latency_samples", 10)), 10)
    cfg.evaluation.max_eval_samples = 20
    cfg.evaluation.latency_benchmark_runs = min(int(getattr(cfg.evaluation, "latency_benchmark_runs", 50)), 50)
    cfg.evaluation.latency_warmup_runs = min(int(getattr(cfg.evaluation, "latency_warmup_runs", 3)), 3)
    cfg.evaluation.num_warmup_runs = min(int(getattr(cfg.evaluation, "num_warmup_runs", 3)), 3)

    if skip_training:
        cfg.adaptation.num_train_epochs = 0
        cfg.distillation.num_train_epochs = 0
        cfg.pruning.calibration_batches = 1
        cfg.pruning.calibration_batch_size = 1


def _materialize_root_analysis_files(summary: dict[str, Any], run_root: Path) -> None:
    analysis = summary.get("analysis", {}) or {}

    dynamics_src = analysis.get("dynamics_log_path")
    if dynamics_src and Path(dynamics_src).exists():
        shutil.copy2(dynamics_src, run_root / "distillation_dynamics.json")

    pruning_src = analysis.get("pruning_analysis_path")
    if pruning_src and Path(pruning_src).exists():
        shutil.copy2(pruning_src, run_root / "pruning_analysis.json")


def _verify_outputs(run_root: Path) -> None:
    required = [
        run_root / "domain_teacher",
        run_root / "distilled_student" / "final",
        run_root / "pruned_student" / "final",
        run_root / "eval_metrics.json",
        run_root / "distillation_dynamics.json",
        run_root / "pruning_analysis.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Smoke test output validation failed. Missing paths:\n- " + "\n- ".join(missing))


def _print_eval_metrics(run_root: Path) -> None:
    eval_path = run_root / "eval_metrics.json"
    metrics = json.loads(eval_path.read_text(encoding="utf-8"))
    keys = [
        "accuracy",
        "f1",
        "perplexity",
        "latency_ms",
        "throughput_tokens_per_sec",
        "compression_ratio",
        "disk_size_compression_ratio",
    ]
    print("Smoke eval metrics:")
    for key in keys:
        print(f"  {key}: {metrics.get(key)}")


if __name__ == "__main__":
    main()
