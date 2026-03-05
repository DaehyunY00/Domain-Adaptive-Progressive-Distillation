from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    datasets: list[str] = field(default_factory=lambda: ["pubmed_qa", "sciq", "medmcqa"])
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    seed: int = 42
    max_train_samples: int | None = None
    max_eval_samples: int | None = 512
    max_length: int = 512
    num_proc: int = 2
    preprocessing_batch_size: int = 1000
    cache_dir: str | None = None
    tokenized_cache_dir: str = "runs/dapd/cache/tokenized"
    enable_map_cache: bool = True


@dataclass
class AdaptationConfig:
    teacher_model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct"
    output_dir: str = "runs/dapd/domain_teacher"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    use_qlora: bool = False
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0


@dataclass
class DistillationConfig:
    student_model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "runs/dapd/distilled_student"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    gradient_checkpointing: bool = True
    alpha: float = 0.7
    temperature: float = 2.0
    temperature_schedule: str = "constant"  # constant | linear | cosine
    min_temperature: float = 1.0
    allow_kl_fallback_to_ce: bool = False
    max_grad_norm: float = 1.0


@dataclass
class PruningConfig:
    enabled: bool = True
    method: str = "structured"
    output_dir: str = "runs/dapd/pruned_student"
    prune_ratio: float = 0.2
    beta: float = 0.5
    calibration_batches: int = 50
    calibration_batch_size: int = 2
    enable_attention_head_pruning: bool = True
    enable_mlp_pruning: bool = True
    enable_layer_pruning: bool = False
    layer_prune_ratio: float = 0.0
    min_heads_per_layer: int = 1
    min_mlp_neurons: int = 32


@dataclass
class EvaluationConfig:
    enabled: bool = True
    output_file: str = "runs/dapd/eval_metrics.json"
    max_eval_samples: int = 200
    batch_size: int = 2
    max_new_tokens: int = 32
    generation_temperature: float = 0.0
    num_latency_samples: int = 20


@dataclass
class RuntimeConfig:
    seed: int = 42
    deterministic: bool = True
    device: str = "auto"
    fp16: bool = False
    bf16: bool = False
    dataloader_num_workers: int = 0
    log_level: str = "INFO"


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "PipelineConfig":
        config_path = Path(config_path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        base_dir = config_path.parent

        data = DataConfig(**raw.get("data", {}))
        adaptation = AdaptationConfig(**raw.get("adaptation", {}))
        distillation = DistillationConfig(**raw.get("distillation", {}))
        pruning = PruningConfig(**raw.get("pruning", {}))
        evaluation = EvaluationConfig(**raw.get("evaluation", {}))
        runtime = RuntimeConfig(**raw.get("runtime", {}))

        if data.cache_dir:
            data.cache_dir = _resolve_path(base_dir, data.cache_dir)
        data.tokenized_cache_dir = _resolve_path(base_dir, data.tokenized_cache_dir)

        adaptation.output_dir = _resolve_path(base_dir, adaptation.output_dir)
        distillation.output_dir = _resolve_path(base_dir, distillation.output_dir)
        pruning.output_dir = _resolve_path(base_dir, pruning.output_dir)
        evaluation.output_file = _resolve_path(base_dir, evaluation.output_file)

        return cls(
            data=data,
            adaptation=adaptation,
            distillation=distillation,
            pruning=pruning,
            evaluation=evaluation,
            runtime=runtime,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": vars(self.data),
            "adaptation": vars(self.adaptation),
            "distillation": vars(self.distillation),
            "pruning": vars(self.pruning),
            "evaluation": vars(self.evaluation),
            "runtime": vars(self.runtime),
        }


def _resolve_path(base_dir: Path, value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())
