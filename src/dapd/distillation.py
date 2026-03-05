from __future__ import annotations

"""Progressive distillation utilities for DAPD.

This module implements temperature-based knowledge distillation for causal language
models with padding-safe masking and tokenizer compatibility checks.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any
import warnings

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .adaptation import AdaptationArtifacts, load_adapted_teacher_for_inference
from .data import CausalLMDataCollator, PreparedDatasets
from .utils import (
    collect_memory_stats,
    configure_model_for_training,
    ensure_dir,
    get_logger,
    infer_device,
    restore_model_use_cache,
    validate_runtime_precision,
)


@dataclass
class TeacherLogitsSource:
    """Source information for teacher logits during distillation.

    Attributes:
        teacher_model: Loaded teacher model used for forward passes.
        use_kl: Whether KL distillation is enabled.
        teacher_path: Filesystem path of the domain-adapted teacher artifact.
    """

    teacher_model: torch.nn.Module | None
    use_kl: bool
    teacher_path: str


@dataclass
class DistillationArtifacts:
    """Artifacts produced after progressive distillation."""

    student_path: str
    used_kl: bool
    distillation_temperature_start: float
    distillation_temperature_end: float


class ProgressiveDistillationTrainer(Trainer):
    """Trainer implementing CE + temperature-scaled KL distillation.

    Total loss:
        loss = alpha * CE(student_logits, labels)
             + (1 - alpha) * KL(log_softmax(student/T), softmax(teacher/T)) * T^2

    For causal LM, the KL term is computed on shifted prediction positions
    (`logits[:, :-1]` against labels/mask shifted by one position).
    """

    def __init__(
        self,
        *args: Any,
        teacher_logits_source: TeacherLogitsSource,
        alpha: float,
        base_temperature: float,
        min_temperature: float,
        temperature_schedule: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_logits_source = teacher_logits_source
        self.alpha = float(alpha)
        self.base_temperature = float(base_temperature)
        self.min_temperature = float(min_temperature)
        self.temperature_schedule = str(temperature_schedule)
        self.logger = get_logger("dapd.distillation")

        teacher_model = self.teacher_logits_source.teacher_model
        if teacher_model is not None:
            teacher_model.eval()
            for parameter in teacher_model.parameters():
                parameter.requires_grad_(False)
            if self.args.device.type != "cpu":
                teacher_model.to(self.args.device)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any | None = None,
    ) -> Any:
        """Compute CE-only or CE+KD loss for a training batch."""
        del num_items_in_batch

        labels = inputs.get("labels")
        outputs = model(**inputs)
        ce_loss = outputs.loss
        if ce_loss is None:
            raise ValueError("Model output does not include loss; ensure labels are provided.")

        if not self.teacher_logits_source.use_kl:
            return (ce_loss, outputs) if return_outputs else ce_loss

        teacher_model = self.teacher_logits_source.teacher_model
        if teacher_model is None:
            raise RuntimeError("teacher_model is required when KL distillation is enabled")

        max_steps = max(1, int(self.state.max_steps or self.args.max_steps or 1))
        current_step = int(self.state.global_step or 0)
        temperature = _scheduled_temperature(
            step=current_step,
            max_steps=max_steps,
            base_temperature=self.base_temperature,
            min_temperature=self.min_temperature,
            schedule=self.temperature_schedule,
        )

        teacher_inputs: dict[str, torch.Tensor] = {"input_ids": inputs["input_ids"]}
        if inputs.get("attention_mask") is not None:
            teacher_inputs["attention_mask"] = inputs["attention_mask"]

        with torch.no_grad():
            teacher_outputs = teacher_model(**teacher_inputs)

        student_logits = outputs.logits
        teacher_logits = teacher_outputs.logits.detach().to(student_logits.device)

        kd_loss = _compute_masked_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            attention_mask=inputs.get("attention_mask"),
            temperature=temperature,
        )

        loss = self.alpha * ce_loss + (1.0 - self.alpha) * kd_loss

        if self.state.global_step % max(1, int(self.args.logging_steps)) == 0:
            self.logger.info(
                "step=%s ce=%.4f kd=%.4f temp=%.4f alpha=%.3f",
                self.state.global_step,
                float(ce_loss.detach().item()),
                float(kd_loss.detach().item()),
                float(temperature),
                float(self.alpha),
            )

        return (loss, outputs) if return_outputs else loss


def prepare_teacher_logits_source(
    distillation_config: Any,
    runtime: Any,
    adaptation_artifacts: AdaptationArtifacts,
    teacher_base_model_name_or_path: str,
) -> TeacherLogitsSource:
    """Prepare a teacher source and validate whether KL is safe to use.

    KL distillation is allowed only when teacher/student tokenizers are compatible.
    If incompatible, behavior follows `allow_kl_fallback_to_ce`.
    """
    device = infer_device(runtime.device)
    validate_runtime_precision(device, runtime.fp16, runtime.bf16)
    _validate_distillation_hparams(
        alpha=distillation_config.alpha,
        temperature=distillation_config.temperature,
        min_temperature=distillation_config.min_temperature,
        temperature_schedule=distillation_config.temperature_schedule,
    )

    if not bool(getattr(distillation_config, "use_kl", True)):
        return TeacherLogitsSource(
            teacher_model=None,
            use_kl=False,
            teacher_path=adaptation_artifacts.teacher_path,
        )

    teacher_tokenizer = AutoTokenizer.from_pretrained(adaptation_artifacts.teacher_path, use_fast=True)
    student_tokenizer = AutoTokenizer.from_pretrained(distillation_config.student_model_name_or_path, use_fast=True)

    use_kl = _resolve_kl_usage(
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
        allow_kl_fallback_to_ce=distillation_config.allow_kl_fallback_to_ce,
    )

    teacher_model = None
    if use_kl:
        teacher_model, _ = load_adapted_teacher_for_inference(
            adapt_artifacts=adaptation_artifacts,
            base_model_name_or_path=teacher_base_model_name_or_path,
        )

    return TeacherLogitsSource(
        teacher_model=teacher_model,
        use_kl=use_kl,
        teacher_path=adaptation_artifacts.teacher_path,
    )


def run_progressive_distillation(
    config: Any,
    runtime: Any,
    datasets: PreparedDatasets,
    teacher_logits_source: TeacherLogitsSource,
) -> DistillationArtifacts:
    """Run student training with progressive KD and return output artifacts."""
    output_dir = ensure_dir(config.output_dir)
    device = infer_device(runtime.device)
    validate_runtime_precision(device, runtime.fp16, runtime.bf16)
    _validate_distillation_hparams(
        alpha=config.alpha,
        temperature=config.temperature,
        min_temperature=config.min_temperature,
        temperature_schedule=config.temperature_schedule,
    )

    student_model = AutoModelForCausalLM.from_pretrained(config.student_model_name_or_path, trust_remote_code=True)
    student_tokenizer = AutoTokenizer.from_pretrained(config.student_model_name_or_path, use_fast=True)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    previous_use_cache = configure_model_for_training(
        model=student_model,
        gradient_checkpointing=config.gradient_checkpointing,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        max_grad_norm=config.max_grad_norm,
        fp16=runtime.fp16,
        bf16=runtime.bf16,
        evaluation_strategy=config.evaluation_strategy,
        save_strategy=config.save_strategy,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=runtime.dataloader_num_workers,
        dataloader_pin_memory=device.type == "cuda",
    )

    trainer = ProgressiveDistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=datasets.train_lm,
        eval_dataset=datasets.validation_lm,
        data_collator=CausalLMDataCollator(student_tokenizer),
        tokenizer=student_tokenizer,
        teacher_logits_source=teacher_logits_source,
        alpha=config.alpha,
        base_temperature=config.temperature,
        min_temperature=config.min_temperature,
        temperature_schedule=config.temperature_schedule,
    )

    mem_before = collect_memory_stats(device=device, reset_peak=True)
    trainer.train()
    mem_after = collect_memory_stats(device=device, reset_peak=False)
    restore_model_use_cache(student_model, previous_use_cache)

    logger = get_logger("dapd.distillation")
    logger.info(
        "[memory] rss_mb=%.1f device=%s allocated_mb=%.1f peak_mb=%.1f (start_rss_mb=%.1f)",
        mem_after.rss_mb,
        mem_after.device,
        mem_after.device_allocated_mb,
        mem_after.peak_allocated_mb,
        mem_before.rss_mb,
    )

    final_dir = Path(output_dir) / "final"
    ensure_dir(final_dir)
    trainer.model.save_pretrained(str(final_dir))
    student_tokenizer.save_pretrained(str(final_dir))

    temp_end = _scheduled_temperature(
        step=max(1, int(trainer.state.max_steps or 1)) - 1,
        max_steps=max(1, int(trainer.state.max_steps or 1)),
        base_temperature=config.temperature,
        min_temperature=config.min_temperature,
        schedule=config.temperature_schedule,
    )

    return DistillationArtifacts(
        student_path=str(final_dir),
        used_kl=teacher_logits_source.use_kl,
        distillation_temperature_start=float(config.temperature),
        distillation_temperature_end=float(temp_end),
    )


def _is_tokenizer_compatible(tok_a: Any, tok_b: Any) -> bool:
    """Return True if two tokenizers are compatible for token-level KL.

    Compatibility requires equal vocab size, equal vocab mapping when available,
    equal special token maps, and matching id->token conversion on probe ids.
    """
    if tok_a.vocab_size != tok_b.vocab_size:
        return False

    if hasattr(tok_a, "get_vocab") and hasattr(tok_b, "get_vocab"):
        if tok_a.get_vocab() != tok_b.get_vocab():
            return False

    if getattr(tok_a, "special_tokens_map", {}) != getattr(tok_b, "special_tokens_map", {}):
        return False

    if hasattr(tok_a, "convert_ids_to_tokens") and hasattr(tok_b, "convert_ids_to_tokens"):
        probe_ids = [0, 1, 2, 10, 100, max(0, tok_a.vocab_size - 1)]
        for token_id in probe_ids:
            if token_id >= tok_a.vocab_size:
                continue
            if tok_a.convert_ids_to_tokens(token_id) != tok_b.convert_ids_to_tokens(token_id):
                return False

    return True


def _resolve_kl_usage(
    teacher_tokenizer: Any,
    student_tokenizer: Any,
    allow_kl_fallback_to_ce: bool,
) -> bool:
    """Resolve whether KL distillation can be enabled.

    Raises:
        ValueError: If tokenizers are incompatible and fallback is disabled.
    """
    compatible = _is_tokenizer_compatible(teacher_tokenizer, student_tokenizer)
    if compatible:
        return True

    message = (
        "Teacher and student tokenizers are not compatible for KL distillation. "
        "Set distillation.allow_kl_fallback_to_ce=true to continue with CE-only training."
    )

    if allow_kl_fallback_to_ce:
        warnings.warn(message, stacklevel=2)
        return False

    raise ValueError(message)


def _compute_masked_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor:
    """Compute padding-safe KL distillation loss for causal LM.

    The KL term is computed with causal shift and temperature scaling:
      soft_teacher = softmax(teacher_logits / T)
      soft_student = log_softmax(student_logits / T)
      kd_loss = KL(soft_student, soft_teacher) * T^2

    Args:
        student_logits: Student logits with shape [batch, seq_len, vocab].
        teacher_logits: Teacher logits with shape [batch, seq_len, vocab].
        labels: Optional labels where padding/ignored tokens are marked as -100.
        attention_mask: Optional attention mask; used when labels are unavailable.
        temperature: Distillation temperature T (> 0).

    Returns:
        Scalar KL distillation loss tensor.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for KL distillation")
    if student_logits.ndim != 3 or teacher_logits.ndim != 3:
        raise ValueError("student_logits and teacher_logits must be rank-3 tensors [B, S, V]")

    if student_logits.size(1) != teacher_logits.size(1):
        raise ValueError(
            f"KL requires same sequence length, got student={student_logits.size(1)} "
            f"teacher={teacher_logits.size(1)}"
        )
    if student_logits.size(-1) != teacher_logits.size(-1):
        raise ValueError(
            f"KL requires same vocab size, got student={student_logits.size(-1)} "
            f"teacher={teacher_logits.size(-1)}"
        )

    seq_len = int(student_logits.size(1))
    if seq_len <= 1:
        return student_logits.new_zeros(())

    # Causal LM alignment: position t predicts token at t+1.
    student_shifted = student_logits[:, :-1, :].float() / float(temperature)
    teacher_shifted = teacher_logits[:, :-1, :].float() / float(temperature)

    student_log_probs = F.log_softmax(student_shifted, dim=-1)
    teacher_probs = F.softmax(teacher_shifted, dim=-1)
    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)

    mask = _build_causal_mask(
        labels=labels,
        attention_mask=attention_mask,
        pred_len=token_kl.size(1),
        batch_size=token_kl.size(0),
        device=token_kl.device,
        dtype=token_kl.dtype,
    )

    valid_count = mask.sum()
    if valid_count.item() <= 0.0:
        return token_kl.new_zeros(())

    return ((token_kl * mask).sum() / valid_count) * (float(temperature) ** 2)


def _build_causal_mask(
    labels: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    pred_len: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build per-token mask for shifted causal-LM predictions."""
    if pred_len <= 0:
        return torch.zeros((batch_size, 0), device=device, dtype=dtype)

    if labels is not None:
        if labels.ndim != 2:
            raise ValueError("labels must be rank-2 tensor [B, S]")
        shifted_labels = labels[:, 1 : 1 + pred_len]
        return (shifted_labels != -100).to(device=device, dtype=dtype)

    if attention_mask is not None:
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must be rank-2 tensor [B, S]")
        shifted_mask = attention_mask[:, 1 : 1 + pred_len]
        return shifted_mask.to(device=device, dtype=dtype)

    return torch.ones((batch_size, pred_len), device=device, dtype=dtype)


def _scheduled_temperature(
    step: int,
    max_steps: int,
    base_temperature: float,
    min_temperature: float,
    schedule: str,
) -> float:
    """Return scheduled distillation temperature for current step."""
    mode = schedule.lower().strip()
    if mode == "constant":
        return float(base_temperature)

    if max_steps <= 1:
        return float(base_temperature)

    progress = min(max(step / (max_steps - 1), 0.0), 1.0)

    if mode == "linear":
        return float(base_temperature + (min_temperature - base_temperature) * progress)

    if mode == "cosine":
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_temperature + (base_temperature - min_temperature) * cosine)

    raise ValueError(
        f"Unsupported temperature_schedule='{schedule}'. Use one of: constant, linear, cosine"
    )


def _validate_distillation_hparams(
    alpha: float,
    temperature: float,
    min_temperature: float,
    temperature_schedule: str,
) -> None:
    """Validate key distillation hyperparameters."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if min_temperature <= 0.0:
        raise ValueError(f"min_temperature must be > 0, got {min_temperature}")
    if min_temperature > temperature:
        raise ValueError(
            f"min_temperature ({min_temperature}) must be <= temperature ({temperature})"
        )

    mode = temperature_schedule.lower().strip()
    if mode not in {"constant", "linear", "cosine"}:
        raise ValueError(
            f"temperature_schedule must be one of ['constant', 'linear', 'cosine'], got '{temperature_schedule}'"
        )
