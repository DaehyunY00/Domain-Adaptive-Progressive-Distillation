from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from dapd.config import PipelineConfig
from dapd.distillation import DistillationArtifacts, TeacherLogitsSource
from dapd.pipeline import DAPDPipeline


def test_pipeline_skips_adaptation_when_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = PipelineConfig()
    cfg.adaptation.enabled = False
    cfg.adaptation.teacher_model_name_or_path = "dummy/base-teacher"
    cfg.adaptation.output_dir = str(tmp_path / "domain_teacher")
    cfg.distillation.output_dir = str(tmp_path / "distilled_student")
    cfg.pruning.output_dir = str(tmp_path / "pruned_student")
    cfg.evaluation.output_file = str(tmp_path / "eval_metrics.json")
    cfg.pruning.enabled = False
    cfg.evaluation.enabled = False

    calls = {"adapt": 0}

    def _fake_adapt(*args, **kwargs):
        del args, kwargs
        calls["adapt"] += 1
        raise AssertionError(
            "run_domain_adaptation should not be called "
            "when adaptation.enabled=false"
        )

    monkeypatch.setattr("dapd.pipeline.run_domain_adaptation", _fake_adapt)
    monkeypatch.setattr(
        "dapd.pipeline.build_unified_dataset",
        lambda *args, **kwargs: {"train": [1], "validation": [1], "test": [1]},
    )
    monkeypatch.setattr(
        "dapd.pipeline.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "dapd.pipeline.prepare_datasets_from_unified",
        lambda *args, **kwargs: SimpleNamespace(
            train_lm=[],
            validation_lm=[],
            test_lm=[],
            validation_text=[],
            test_text=[],
        ),
    )
    monkeypatch.setattr(
        "dapd.pipeline.prepare_teacher_logits_source",
        lambda *args, **kwargs: TeacherLogitsSource(
            teacher_model=None,
            use_kl=False,
            teacher_path="dummy/base-teacher",
        ),
    )
    monkeypatch.setattr(
        "dapd.pipeline.run_progressive_distillation",
        lambda *args, **kwargs: DistillationArtifacts(
            student_path=str(tmp_path / "distilled_student" / "final"),
            used_kl=False,
            distillation_temperature_start=2.0,
            distillation_temperature_end=2.0,
        ),
    )

    summary = DAPDPipeline(cfg).run()

    assert calls["adapt"] == 0
    assert summary["adaptation"]["teacher_path"] == "dummy/base-teacher"
