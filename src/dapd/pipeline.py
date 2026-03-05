from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .adaptation import AdaptationArtifacts, run_domain_adaptation
from .config import PipelineConfig
from .data import build_unified_dataset, prepare_datasets_from_unified
from .distillation import (
    DistillationArtifacts,
    TeacherLogitsSource,
    prepare_teacher_logits_source,
    run_progressive_distillation,
)
from .evaluation import evaluate_model
from .pruning import PruningArtifacts, run_structured_pruning
from .utils import dump_json, dump_yaml, ensure_dir, get_logger, set_seed


class DAPDPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "DAPDPipeline":
        return cls(PipelineConfig.from_yaml(config_path))

    def run(self) -> dict[str, Any]:
        cfg = self.config
        logger = get_logger("dapd.pipeline", getattr(cfg.runtime, "log_level", "INFO"))
        set_seed(cfg.runtime.seed, deterministic=cfg.runtime.deterministic)

        ensure_dir(cfg.adaptation.output_dir)
        ensure_dir(cfg.distillation.output_dir)
        ensure_dir(cfg.pruning.output_dir)
        artifact_root = Path(cfg.distillation.output_dir).resolve().parent
        ensure_dir(artifact_root)

        config_used_path = artifact_root / "config_used.yaml"
        dump_yaml(cfg.to_dict(), config_used_path)
        logger.info("saved config snapshot to %s", config_used_path)

        unified = build_unified_dataset(cfg.data)
        logger.info(
            "dataset ready | train=%s validation=%s test=%s",
            len(unified["train"]),
            len(unified["validation"]),
            len(unified["test"]),
        )

        # Step 1: Domain adaptation (general teacher -> domain teacher)
        teacher_tokenizer = AutoTokenizer.from_pretrained(cfg.adaptation.teacher_model_name_or_path, use_fast=True)
        teacher_data = prepare_datasets_from_unified(unified=unified, config=cfg.data, tokenizer=teacher_tokenizer)

        adapt_artifacts: AdaptationArtifacts = run_domain_adaptation(
            config=cfg.adaptation,
            runtime=cfg.runtime,
            datasets=teacher_data,
        )
        logger.info("step1 complete | domain_teacher=%s", adapt_artifacts.teacher_path)

        # Step 2: Prepare teacher logits source for progressive distillation.
        teacher_logits_source: TeacherLogitsSource = prepare_teacher_logits_source(
            distillation_config=cfg.distillation,
            runtime=cfg.runtime,
            adaptation_artifacts=adapt_artifacts,
            teacher_base_model_name_or_path=cfg.adaptation.teacher_model_name_or_path,
        )
        logger.info("step2 complete | teacher_logits_enabled=%s", teacher_logits_source.use_kl)

        # Step 3: Distill student from domain-adapted teacher.
        student_tokenizer = AutoTokenizer.from_pretrained(cfg.distillation.student_model_name_or_path, use_fast=True)
        student_data = prepare_datasets_from_unified(unified=unified, config=cfg.data, tokenizer=student_tokenizer)

        distill_artifacts: DistillationArtifacts = run_progressive_distillation(
            config=cfg.distillation,
            runtime=cfg.runtime,
            datasets=student_data,
            teacher_logits_source=teacher_logits_source,
        )
        logger.info("step3 complete | distilled_student=%s", distill_artifacts.student_path)

        final_model_path = distill_artifacts.student_path
        pruning_artifacts: PruningArtifacts | None = None

        # Step 4: Structured pruning.
        if cfg.pruning.enabled:
            pruning_artifacts = run_structured_pruning(
                config=cfg.pruning,
                runtime=cfg.runtime,
                model_path=distill_artifacts.student_path,
                calibration_dataset=student_data.validation_lm,
            )
            final_model_path = pruning_artifacts.model_path
            logger.info("step4 complete | pruned_student=%s", final_model_path)

        eval_metrics = None
        # Step 5: Evaluation + benchmark metrics.
        if cfg.evaluation.enabled:
            eval_tokenizer = AutoTokenizer.from_pretrained(final_model_path, use_fast=True)
            eval_data = prepare_datasets_from_unified(unified=unified, config=cfg.data, tokenizer=eval_tokenizer)
            eval_metrics = evaluate_model(
                model_path=final_model_path,
                text_dataset=eval_data.test_text,
                lm_dataset=eval_data.test_lm,
                config=cfg.evaluation,
                runtime=cfg.runtime,
                reference_model_path=adapt_artifacts.teacher_path,
            )
            dump_json(eval_metrics, cfg.evaluation.output_file)
            logger.info("step5 complete | eval_metrics=%s", cfg.evaluation.output_file)

        summary = {
            "config": cfg.to_dict(),
            "adaptation": asdict(adapt_artifacts),
            "teacher_logits_source": {
                "teacher_path": teacher_logits_source.teacher_path,
                "use_kl": teacher_logits_source.use_kl,
            },
            "distillation": asdict(distill_artifacts),
            "pruning": asdict(pruning_artifacts) if pruning_artifacts else None,
            "final_model_path": final_model_path,
            "evaluation": eval_metrics,
            "config_used_path": str(config_used_path),
        }

        summary_path = _summary_path(cfg)
        dump_json(summary, summary_path)
        logger.info("pipeline summary saved to %s", summary_path)
        return summary


def _summary_path(config: PipelineConfig) -> str:
    parent = Path(config.distillation.output_dir).resolve().parent
    return str(parent / "pipeline_summary.json")
