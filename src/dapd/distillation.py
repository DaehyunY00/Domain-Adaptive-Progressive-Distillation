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
    get_recommended_teacher_dtype,
    infer_device,
    resolve_training_strategy_kwargs,
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
    dynamics_log_path: str | None = None


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
            # teacher는 항상 CPU에 유지한다.
            # MPS(Apple Silicon)에서 student + teacher를 동시에 MPS에 올리면
            # 7GB+ MPS 사용으로 OOM이 발생한다.
            # teacher를 CPU에 두면 MPS는 student(2GB) + gradient(2GB)만 사용.
            # teacher logits는 compute_loss에서 청크 단위로 MPS에 전송한다.
            teacher_model.to("cpu")

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

        if not torch.isfinite(ce_loss):
            if _has_no_supervised_tokens(labels):
                safe_zero = outputs.logits.new_zeros(())
                self.logger.warning("Skipping batch with no supervised target tokens after truncation.")
                return (safe_zero, outputs) if return_outputs else safe_zero
            self.logger.warning("Non-finite CE loss detected; zeroing this batch loss for stability.")
            ce_loss = torch.nan_to_num(ce_loss, nan=0.0, posinf=0.0, neginf=0.0)

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

        # teacher는 CPU에 있으므로 inputs도 CPU로 이동해서 forward 수행.
        # student forward와 별개의 그래프이므로 no_grad + CPU 연산으로 MPS 부하 없음.
        teacher_inputs: dict[str, torch.Tensor] = {
            "input_ids": inputs["input_ids"].to("cpu"),
        }
        if inputs.get("attention_mask") is not None:
            teacher_inputs["attention_mask"] = inputs["attention_mask"].to("cpu")

        with torch.no_grad():
            teacher_outputs = teacher_model(**teacher_inputs)

        student_logits = outputs.logits
        # teacher_logits는 CPU에 유지한다.
        # _compute_masked_kl_loss 내부 청크 루프에서 청크 단위로 student device(MPS)로 이동.
        # 전체를 MPS로 한 번에 복사하면 [B, S, V] 전체가 MPS에 올라가지만,
        # 청크별 이동은 최대 [B, 4, 152064] = ~1.2MB만 MPS에 상주한다.
        teacher_logits = teacher_outputs.logits.detach()  # CPU 유지
        del teacher_outputs

        kd_loss = _compute_masked_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            attention_mask=inputs.get("attention_mask"),
            temperature=temperature,
        )
        if not torch.isfinite(kd_loss):
            self.logger.warning("Non-finite KD loss detected; zeroing this batch KD loss for stability.")
            kd_loss = torch.nan_to_num(kd_loss, nan=0.0, posinf=0.0, neginf=0.0)

        loss = self.alpha * ce_loss + (1.0 - self.alpha) * kd_loss
        if not torch.isfinite(loss):
            self.logger.warning("Combined distillation loss became non-finite; skipping batch update.")
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        if self.state.global_step % max(1, int(self.args.logging_steps)) == 0:
            self.logger.info(
                "step=%s ce=%.4f kd=%.4f temp=%.4f alpha=%.3f",
                self.state.global_step,
                float(ce_loss.detach().item()),
                float(kd_loss.detach().item()),
                float(temperature),
                float(self.alpha),
            )
            if self.model.training:
                self.log(
                    {
                        "temperature": float(temperature),
                        "ce_loss": float(ce_loss.detach().item()),
                        "kd_loss": float(kd_loss.detach().item()),
                    }
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
        # MPS(Apple Silicon)에서는 teacher를 bfloat16으로 로드해 메모리 3GB 절감.
        # teacher는 eval-only(requires_grad=False)이므로 bfloat16 사용 안전.
        teacher_dtype = get_recommended_teacher_dtype(device)
        teacher_model, _ = load_adapted_teacher_for_inference(
            adapt_artifacts=adaptation_artifacts,
            base_model_name_or_path=teacher_base_model_name_or_path,
            torch_dtype=teacher_dtype,
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
    dynamics_log_path: str | None = None,
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

    # low_cpu_mem_usage: 로딩 시 CPU peak를 model_size 수준으로 억제한다.
    # MPS(Apple Silicon)에서 student는 float32로 로드해 MPS가 직접 학습을 처리.
    # teacher는 이미 CPU bfloat16으로 별도 로드되어 있음(prepare_teacher_logits_source).
    student_model = AutoModelForCausalLM.from_pretrained(
        config.student_model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
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
        # Adafactor: 2nd moment를 인수분해해 저장 → optimizer state 5.31GB → ~0.1GB 절감.
        # lm_head[152064,896] AdamW 2nd moment 545MB → 0.6MB (1782x 절감).
        # 논문 품질: Shazeer & Stern (2018) 수준의 표준 대안 옵티마이저.
        optim=getattr(config, "optim", "adamw_torch"),
        **resolve_training_strategy_kwargs(
            evaluation_strategy=config.evaluation_strategy,
            save_strategy=config.save_strategy,
        ),
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=runtime.dataloader_num_workers,
        dataloader_pin_memory=device.type == "cuda",
    )

    callbacks = None
    if dynamics_log_path is not None:
        from .analysis import create_dynamics_callback

        callbacks = [create_dynamics_callback(dynamics_log_path)]

    trainer = ProgressiveDistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=datasets.train_lm,
        eval_dataset=datasets.validation_lm,
        data_collator=CausalLMDataCollator(student_tokenizer),
        tokenizer=student_tokenizer,
        callbacks=callbacks,
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
        dynamics_log_path=dynamics_log_path,
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

    mask = _build_causal_mask(
        labels=labels,
        attention_mask=attention_mask,
        pred_len=seq_len - 1,
        batch_size=student_logits.size(0),
        device=student_logits.device,
        dtype=student_logits.dtype,
    )

    valid_count = mask.sum()
    if valid_count.item() <= 0.0:
        return student_logits.new_zeros(())

    # Compute KL in small token chunks to avoid large [B, S, V] temporary allocations
    # on MPS with large vocabularies.
    pred_len = int(seq_len - 1)
    vocab_size = int(student_logits.size(-1))
    chunk_tokens = _resolve_kl_chunk_tokens(
        device=student_logits.device,
        pred_len=pred_len,
        vocab_size=vocab_size,
    )
    temp = float(temperature)
    kl_sum = student_logits.new_zeros(())

    for start in range(0, pred_len, chunk_tokens):
        end = min(pred_len, start + chunk_tokens)
        mask_chunk = mask[:, start:end]
        if mask_chunk.sum().item() <= 0.0:
            continue

        # Causal LM alignment: position t predicts token at t+1.
        # teacher_logits가 CPU에 있을 경우 .to(device, dtype)으로 MPS에 올린다.
        # 청크 크기(4 tokens × 152064 vocab × 4 bytes) = ~2.3MB → MPS 부하 미미.
        student_chunk = student_logits[:, start:end, :] / temp
        teacher_chunk = teacher_logits[:, start:end, :].to(
            device=student_logits.device,
            dtype=student_logits.dtype,
        ) / temp

        student_log_probs = F.log_softmax(student_chunk, dim=-1)
        teacher_probs = F.softmax(teacher_chunk, dim=-1)
        token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        kl_sum = kl_sum + (token_kl * mask_chunk).sum()

        del student_chunk, teacher_chunk, student_log_probs, teacher_probs, token_kl

    return (kl_sum / valid_count) * (temp**2)


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


def _resolve_kl_chunk_tokens(
    device: torch.device,
    pred_len: int,
    vocab_size: int,
) -> int:
    """KL 계산의 토큰 청크 크기를 결정해 peak 메모리를 줄인다.

    MPS에서 [B, chunk, V] 임시 텐서가 큰 vocab(Qwen: 152,064)과 결합하면
    메모리 급증이 발생한다. 청크 크기를 줄여 peak를 제어한다.

    Qwen2.5 vocab=152,064 기준 메모리:
      float32: chunk=8  → 8 × 152,064 × 4 bytes ≈ 4.6MB per chunk
      float32: chunk=4  → 4 × 152,064 × 4 bytes ≈ 2.3MB per chunk  (M4 16GB 권장)
    """
    if pred_len <= 0:
        return 1
    if device.type == "mps":
        # M4 16GB Unified Memory: vocab >= 100K이면 chunk=4로 peak 메모리 억제.
        # vocab < 100K이면 chunk=8 유지.
        if vocab_size >= 100_000:
            return min(pred_len, 4)
        return min(pred_len, 8)
    return pred_len


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


def _has_no_supervised_tokens(labels: torch.Tensor | None) -> bool:
    if labels is None or labels.ndim != 2 or labels.size(1) <= 1:
        return False
    return bool((labels[:, 1:] != -100).sum().item() == 0)
