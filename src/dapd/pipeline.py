from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .analysis import (
    analyze_pruning_patterns,
    analyze_teacher_distributions,
    run_temperature_schedule_analysis,
)
from .adaptation import AdaptationArtifacts, run_domain_adaptation
from .config import PipelineConfig
from .data import (
    build_external_eval_dataset,
    build_unified_dataset,
    prepare_datasets_from_unified,
    tokenize_for_causal_lm,
)
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
        if cfg.analysis.enabled:
            ensure_dir(cfg.analysis.output_dir)

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

        need_teacher_tokenized = bool(
            cfg.adaptation.enabled or (cfg.analysis.enabled and cfg.analysis.run_teacher_distribution)
        )
        teacher_data = None
        if need_teacher_tokenized:
            teacher_tokenizer = AutoTokenizer.from_pretrained(
                cfg.adaptation.teacher_model_name_or_path,
                use_fast=True,
            )
            teacher_data = prepare_datasets_from_unified(
                unified=unified,
                config=cfg.data,
                tokenizer=teacher_tokenizer,
            )

        # Step 1: Domain adaptation (general teacher -> domain teacher).
        if cfg.adaptation.enabled:
            if teacher_data is None:
                raise RuntimeError("teacher_data must be prepared before domain adaptation")
            adapt_artifacts = run_domain_adaptation(
                config=cfg.adaptation,
                runtime=cfg.runtime,
                datasets=teacher_data,
            )
            logger.info("step1 complete | domain_teacher=%s", adapt_artifacts.teacher_path)
        else:
            adapt_artifacts = AdaptationArtifacts(
                teacher_path=cfg.adaptation.teacher_model_name_or_path,
                adapter_path=cfg.adaptation.teacher_model_name_or_path,
                merged_teacher_path=None,
                model_size_mb=0.0,
            )
            logger.info(
                "step1 skipped | using base teacher directly: %s",
                adapt_artifacts.teacher_path,
            )

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
        ood_metrics = None
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

            if cfg.evaluation.run_ood_test:
                try:
                    ood_text = build_external_eval_dataset(
                        dataset_name=cfg.evaluation.ood_test_dataset,
                        cache_dir=cfg.data.cache_dir,
                        seed=cfg.data.seed,
                        max_eval_samples=cfg.evaluation.ood_max_eval_samples,
                    )
                    ood_lm = tokenize_for_causal_lm(
                        dataset=ood_text,
                        tokenizer=eval_tokenizer,
                        max_length=cfg.data.max_length,
                        num_proc=cfg.data.num_proc,
                        split_name=f"ood_{cfg.evaluation.ood_test_dataset.lower()}",
                        tokenized_cache_dir=cfg.data.tokenized_cache_dir,
                        enable_map_cache=cfg.data.enable_map_cache,
                        dataset_names=[cfg.evaluation.ood_test_dataset],
                        seed=cfg.data.seed,
                        preprocessing_version=cfg.data.preprocessing_version,
                    )
                    ood_metrics = evaluate_model(
                        model_path=final_model_path,
                        text_dataset=ood_text,
                        lm_dataset=ood_lm,
                        config=cfg.evaluation,
                        runtime=cfg.runtime,
                        reference_model_path=adapt_artifacts.teacher_path,
                    )
                    ood_metrics["ood_dataset"] = cfg.evaluation.ood_test_dataset
                    ood_metrics["train_datasets"] = list(cfg.data.datasets)
                    if eval_metrics is not None:
                        ood_metrics["performance_drop"] = {
                            "accuracy_drop": float(eval_metrics["accuracy"] - ood_metrics["accuracy"]),
                            "f1_drop": float(eval_metrics["f1"] - ood_metrics["f1"]),
                            "perplexity_increase": float(
                                ood_metrics["perplexity"] - eval_metrics["perplexity"]
                            ),
                        }
                    dump_json(ood_metrics, cfg.evaluation.ood_output_file)
                    logger.info("step5 OOD complete | metrics=%s", cfg.evaluation.ood_output_file)
                except Exception as exc:
                    logger.warning("OOD evaluation skipped: %s", exc)
                    ood_metrics = {
                        "ood_dataset": cfg.evaluation.ood_test_dataset,
                        "error": str(exc),
                    }

        analysis_results: dict[str, Any] = {}
        if cfg.analysis.enabled:
            if cfg.analysis.run_teacher_distribution:
                try:
                    if teacher_data is None:
                        teacher_tokenizer = AutoTokenizer.from_pretrained(
                            cfg.adaptation.teacher_model_name_or_path,
                            use_fast=True,
                        )
                        teacher_data = prepare_datasets_from_unified(
                            unified=unified,
                            config=cfg.data,
                            tokenizer=teacher_tokenizer,
                        )
                    analysis_results["teacher_distribution"] = analyze_teacher_distributions(
                        general_teacher_path=cfg.adaptation.teacher_model_name_or_path,
                        domain_teacher_path=adapt_artifacts.teacher_path,
                        dataset=teacher_data.validation_lm,
                        runtime=cfg.runtime,
                        output_dir=cfg.analysis.output_dir,
                        max_samples=cfg.analysis.max_samples,
                    )
                except Exception as exc:
                    logger.warning("teacher distribution analysis skipped: %s", exc)
                    analysis_results["teacher_distribution"] = {"error": str(exc)}

            if cfg.analysis.run_temperature_analysis:
                try:
                    analysis_results["temperature_analysis"] = run_temperature_schedule_analysis(
                        teacher_model_path=adapt_artifacts.teacher_path,
                        student_model_path=distill_artifacts.student_path,
                        train_dataset=student_data.train_lm,
                        validation_dataset=student_data.validation_lm,
                        runtime=cfg.runtime,
                        output_dir=cfg.analysis.output_dir,
                        alpha=cfg.distillation.alpha,
                        temperature=cfg.distillation.temperature,
                        min_temperature=cfg.distillation.min_temperature,
                        steps=cfg.analysis.temperature_steps,
                    )
                except Exception as exc:
                    logger.warning("temperature analysis skipped: %s", exc)
                    analysis_results["temperature_analysis"] = {"error": str(exc)}

            if cfg.analysis.run_pruning_patterns:
                try:
                    if pruning_artifacts is None:
                        raise ValueError("pruning is disabled; no pruning report available")
                    analysis_results["pruning_patterns"] = analyze_pruning_patterns(
                        pruning_report_path=pruning_artifacts.pruning_report_path,
                        output_dir=cfg.analysis.output_dir,
                    )
                except Exception as exc:
                    logger.warning("pruning pattern analysis skipped: %s", exc)
                    analysis_results["pruning_patterns"] = {"error": str(exc)}

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
            "evaluation_ood": ood_metrics,
            "analysis": analysis_results if cfg.analysis.enabled else None,
            "config_used_path": str(config_used_path),
        }

        summary_path = _summary_path(cfg)
        dump_json(summary, summary_path)
        logger.info("pipeline summary saved to %s", summary_path)
        return summary


def _summary_path(config: PipelineConfig) -> str:
    parent = Path(config.distillation.output_dir).resolve().parent
    return str(parent / "pipeline_summary.json")
