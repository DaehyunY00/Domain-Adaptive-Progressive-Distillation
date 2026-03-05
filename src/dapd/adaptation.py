from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from .data import CausalLMDataCollator, PreparedDatasets
from .utils import (
    collect_memory_stats,
    configure_model_for_training,
    ensure_dir,
    get_logger,
    get_model_disk_size_bytes,
    infer_device,
    restore_model_use_cache,
    validate_quantization_config,
    validate_runtime_precision,
)


@dataclass
class AdaptationArtifacts:
    teacher_path: str
    adapter_path: str
    merged_teacher_path: str | None
    model_size_mb: float


def run_domain_adaptation(config: Any, runtime: Any, datasets: PreparedDatasets) -> AdaptationArtifacts:
    logger = get_logger("dapd.adaptation", getattr(runtime, "log_level", "INFO"))
    output_dir = ensure_dir(config.output_dir)
    device = infer_device(runtime.device)
    validate_runtime_precision(device, runtime.fp16, runtime.bf16)
    validate_quantization_config(device, config.use_qlora)

    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
    }

    if runtime.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif runtime.fp16:
        model_kwargs["torch_dtype"] = torch.float16

    if config.use_qlora:
        model_kwargs["quantization_config"] = _qlora_config(runtime=runtime)
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.teacher_model_name_or_path, **model_kwargs)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    previous_use_cache = configure_model_for_training(
        model=model, gradient_checkpointing=config.gradient_checkpointing
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_lm,
        eval_dataset=datasets.validation_lm,
        data_collator=CausalLMDataCollator(tokenizer),
        tokenizer=tokenizer,
    )

    mem_before = collect_memory_stats(device=device, reset_peak=True)
    trainer.train()
    mem_after = collect_memory_stats(device=device, reset_peak=False)
    restore_model_use_cache(model, previous_use_cache)
    logger.info(
        "[memory] rss_mb=%.1f device=%s allocated_mb=%.1f peak_mb=%.1f (start_rss_mb=%.1f)",
        mem_after.rss_mb,
        mem_after.device,
        mem_after.device_allocated_mb,
        mem_after.peak_allocated_mb,
        mem_before.rss_mb,
    )

    adapter_dir = output_dir / "adapter"
    ensure_dir(adapter_dir)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    merged_path = _try_merge_adapter(model, adapter_dir.parent / "merged", tokenizer)
    final_teacher_path = str(merged_path or adapter_dir)
    size_mb = get_model_disk_size_bytes(final_teacher_path) / (1024 * 1024)
    logger.info("domain adaptation completed | teacher_path=%s | model_size_mb=%.1f", final_teacher_path, size_mb)
    return AdaptationArtifacts(
        teacher_path=final_teacher_path,
        adapter_path=str(adapter_dir),
        merged_teacher_path=str(merged_path) if merged_path else None,
        model_size_mb=float(size_mb),
    )


def load_adapted_teacher_for_inference(adapt_artifacts: AdaptationArtifacts, base_model_name_or_path: str) -> tuple[Any, Any]:
    model_path = Path(adapt_artifacts.teacher_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)

    if _is_adapter_dir(model_path):
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, trust_remote_code=True)
        teacher_model = PeftModel.from_pretrained(base_model, str(model_path))
    else:
        teacher_model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher_model.eval()
    return teacher_model, tokenizer


def _is_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists()


def _try_merge_adapter(model: Any, merged_dir: Path, tokenizer: Any) -> Path | None:
    if not hasattr(model, "merge_and_unload"):
        return None

    try:
        merged_model = model.merge_and_unload()
        ensure_dir(merged_dir)
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        return merged_dir
    except Exception:
        return None


def _qlora_config(runtime: Any) -> BitsAndBytesConfig:
    compute_dtype = torch.float16
    if getattr(runtime, "bf16", False):
        compute_dtype = torch.bfloat16
    elif getattr(runtime, "fp16", False):
        compute_dtype = torch.float16

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
