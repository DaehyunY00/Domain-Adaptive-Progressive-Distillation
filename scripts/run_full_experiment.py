#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
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
from dapd.utils import dump_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full DAPD experiment suite (pipeline, ablation, baselines, analysis)"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to base YAML config")
    parser.add_argument("--skip_ablation", action="store_true", help="Skip ablation runs")
    parser.add_argument("--skip_baselines", action="store_true", help="Skip baseline runs")
    parser.add_argument("--skip_analysis", action="store_true", help="Skip post-training analysis")
    parser.add_argument(
        "--run_multi_seed",
        action="store_true",
        help="Also run the multi-seed pipeline after the main suite",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="full,no_adapt,no_kd,no_prune,constant_temp",
        help="Comma-separated ablation variants",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="direct_kd,lora_only,no_distill_prune",
        help="Comma-separated baseline names",
    )
    parser.add_argument(
        "--baseline_multi_seed",
        action="store_true",
        help="Run baseline comparisons with the built-in fixed seeds",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,777",
        help="Comma-separated seed list for --run_multi_seed",
    )
    parser.add_argument(
        "--analysis_ood_dataset",
        type=str,
        default="bioasq",
        help="OOD dataset passed to run_analysis.py",
    )
    parser.add_argument(
        "--analysis_output_dir",
        type=str,
        default=None,
        help="Override analysis output directory",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser().resolve()
    cfg = PipelineConfig.from_yaml(config_path)
    artifact_root = Path(cfg.distillation.output_dir).resolve().parent
    suite_summary_path = artifact_root / "full_experiment_summary.json"

    suite: dict[str, Any] = {
        "config_path": str(config_path),
        "artifact_root": str(artifact_root),
        "requested": {
            "run_pipeline": True,
            "run_ablation": not args.skip_ablation,
            "run_baselines": not args.skip_baselines,
            "run_analysis": not args.skip_analysis,
            "run_multi_seed": bool(args.run_multi_seed),
            "variants": args.variants,
            "baselines": args.baselines,
            "baseline_multi_seed": bool(args.baseline_multi_seed),
            "seeds": args.seeds,
        },
        "paths": {},
        "stages": {},
    }
    _write_suite_summary(suite_summary_path, suite)

    pipeline_summary_path = artifact_root / "pipeline_summary.json"
    pipeline_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_pipeline.py"),
        "--config",
        str(config_path),
    ]
    suite["stages"]["pipeline"] = _run_stage(
        name="pipeline",
        command=pipeline_cmd,
        cwd=repo_root,
        expected_summary_path=pipeline_summary_path,
    )
    _write_suite_summary(suite_summary_path, suite)
    _assert_stage_success(suite["stages"]["pipeline"])

    pipeline_summary = _load_json(pipeline_summary_path)
    suite["paths"] = _extract_model_paths(pipeline_summary)
    _write_suite_summary(suite_summary_path, suite)

    if not args.skip_ablation:
        ablation_summary_path = artifact_root / "ablation" / "ablation_summary.json"
        ablation_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "run_ablation.py"),
            "--config",
            str(config_path),
            "--variants",
            args.variants,
        ]
        suite["stages"]["ablation"] = _run_stage(
            name="ablation",
            command=ablation_cmd,
            cwd=repo_root,
            expected_summary_path=ablation_summary_path,
        )
        _write_suite_summary(suite_summary_path, suite)
        _assert_stage_success(suite["stages"]["ablation"])

    if not args.skip_baselines:
        baseline_summary_path = artifact_root / "baselines" / "baseline_summary.json"
        baseline_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "run_baselines.py"),
            "--config",
            str(config_path),
            "--baselines",
            args.baselines,
        ]
        if args.baseline_multi_seed:
            baseline_cmd.append("--multi_seed")
        suite["stages"]["baselines"] = _run_stage(
            name="baselines",
            command=baseline_cmd,
            cwd=repo_root,
            expected_summary_path=baseline_summary_path,
        )
        _write_suite_summary(suite_summary_path, suite)
        _assert_stage_success(suite["stages"]["baselines"])

    if not args.skip_analysis:
        analysis_output_dir = _resolve_output_dir(
            repo_root=repo_root,
            configured_path=args.analysis_output_dir or cfg.analysis.output_dir,
        )
        analysis_summary_path = analysis_output_dir / "analysis_summary.json"
        analysis_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "run_analysis.py"),
            "--config",
            str(config_path),
            "--domain_teacher",
            str(suite["paths"]["domain_teacher"]),
            "--distilled_student",
            str(suite["paths"]["distilled_student"]),
            "--pruned_student",
            str(suite["paths"]["pruned_student"]),
            "--ood_dataset",
            args.analysis_ood_dataset,
        ]
        if args.analysis_output_dir:
            analysis_cmd.extend(["--output_dir", args.analysis_output_dir])
        suite["stages"]["analysis"] = _run_stage(
            name="analysis",
            command=analysis_cmd,
            cwd=repo_root,
            expected_summary_path=analysis_summary_path,
        )
        _write_suite_summary(suite_summary_path, suite)
        _assert_stage_success(suite["stages"]["analysis"])

    if args.run_multi_seed:
        statistical_summary_path = artifact_root / "statistical_summary.json"
        multi_seed_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "run_multi_seed.py"),
            "--config",
            str(config_path),
            "--seeds",
            args.seeds,
        ]
        suite["stages"]["multi_seed"] = _run_stage(
            name="multi_seed",
            command=multi_seed_cmd,
            cwd=repo_root,
            expected_summary_path=statistical_summary_path,
        )
        _write_suite_summary(suite_summary_path, suite)
        _assert_stage_success(suite["stages"]["multi_seed"])

    suite["status"] = "completed"
    _write_suite_summary(suite_summary_path, suite)
    print(json.dumps(suite, indent=2, ensure_ascii=False))
    print(f"\nFull experiment suite completed. Summary saved to: {suite_summary_path}")


def _run_stage(
    name: str,
    command: list[str],
    cwd: Path,
    expected_summary_path: Path | None = None,
) -> dict[str, Any]:
    print(f"\n=== Running stage: {name} ===")
    print("$", shlex.join(command))
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env=os.environ.copy(),
        check=False,
    )

    result: dict[str, Any] = {
        "status": "completed" if proc.returncode == 0 else "failed",
        "returncode": int(proc.returncode),
        "command": command,
    }
    if expected_summary_path is not None:
        result["summary_path"] = str(expected_summary_path.resolve())
        result["summary_exists"] = bool(expected_summary_path.exists())
    return result


def _assert_stage_success(stage: dict[str, Any]) -> None:
    if int(stage.get("returncode", 1)) != 0:
        raise SystemExit(int(stage["returncode"]))
    if stage.get("summary_path") and not bool(stage.get("summary_exists")):
        raise FileNotFoundError(f"Expected summary file was not created: {stage['summary_path']}")


def _resolve_output_dir(repo_root: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _extract_model_paths(pipeline_summary: dict[str, Any]) -> dict[str, str]:
    adaptation = pipeline_summary.get("adaptation") or {}
    distillation = pipeline_summary.get("distillation") or {}
    pruning = pipeline_summary.get("pruning") or {}

    domain_teacher = adaptation.get("teacher_path")
    distilled_student = distillation.get("student_path")
    pruned_student = pruning.get("model_path") or pipeline_summary.get("final_model_path")

    if not domain_teacher or not distilled_student or not pruned_student:
        raise ValueError("Pipeline summary does not contain the paths required for downstream stages.")

    return {
        "domain_teacher": str(domain_teacher),
        "distilled_student": str(distilled_student),
        "pruned_student": str(pruned_student),
        "final_model_path": str(pipeline_summary.get("final_model_path", pruned_student)),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_suite_summary(path: Path, payload: dict[str, Any]) -> None:
    dump_json(payload, path)


if __name__ == "__main__":
    main()
