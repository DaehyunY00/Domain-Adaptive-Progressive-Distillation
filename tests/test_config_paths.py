from __future__ import annotations

from pathlib import Path

from dapd.config import PipelineConfig


def test_config_paths_under_configs_tree_resolve_from_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config_dir = project_root / "configs" / "experiments"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "test.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                '  cache_dir: "./cache/hf"',
                '  tokenized_cache_dir: "./runs/example/cache/tokenized"',
                "adaptation:",
                '  output_dir: "./runs/example/domain_teacher"',
                "distillation:",
                '  output_dir: "./runs/example/distilled_student"',
                "pruning:",
                '  output_dir: "./runs/example/pruned_student"',
                "evaluation:",
                '  output_file: "./runs/example/eval_metrics.json"',
                '  ood_output_file: "./runs/example/eval_metrics_ood.json"',
                "analysis:",
                '  output_dir: "./runs/example/analysis"',
            ]
        ),
        encoding="utf-8",
    )

    cfg = PipelineConfig.from_yaml(config_path)

    assert cfg.data.cache_dir == str((project_root / "cache" / "hf").resolve())
    assert cfg.data.tokenized_cache_dir == str((project_root / "runs" / "example" / "cache" / "tokenized").resolve())
    assert cfg.adaptation.output_dir == str((project_root / "runs" / "example" / "domain_teacher").resolve())
    assert cfg.evaluation.output_file == str((project_root / "runs" / "example" / "eval_metrics.json").resolve())
