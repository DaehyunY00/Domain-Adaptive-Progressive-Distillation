from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from dapd import evaluation as evaluation_module
from dapd.metrics import compute_compression_ratio


class DummyBatch(dict):
    def to(self, device: torch.device) -> "DummyBatch":
        for key, value in list(self.items()):
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device)
        return self


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        truncation: bool = True,
    ) -> DummyBatch:
        del truncation
        assert return_tensors == "pt"
        base = max(2, min(8, len(text.split()) + 1))
        ids = torch.arange(1, base + 1, dtype=torch.long).unsqueeze(0)
        return DummyBatch({"input_ids": ids, "attention_mask": torch.ones_like(ids)})


class DummyEvalModel(nn.Module):
    def __init__(self, param_count: int) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(param_count, dtype=torch.float32))

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        max_new_tokens = int(kwargs.get("max_new_tokens", 4))
        new_tokens = torch.full(
            (input_ids.shape[0], max_new_tokens),
            2,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, new_tokens], dim=1)


def test_compute_compression_ratio_teacher_over_student() -> None:
    ratio = compute_compression_ratio(teacher_params := 200.0, student_params := 100.0)
    assert ratio == pytest.approx(teacher_params / student_params)
    assert compute_compression_ratio(0.0, 100.0) == pytest.approx(0.0)
    assert math.isinf(compute_compression_ratio(100.0, 0.0))


def test_evaluate_model_compression_ratio_gt_one_with_bigger_teacher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    student_model = DummyEvalModel(param_count=10)
    teacher_model = DummyEvalModel(param_count=30)

    def _load_model(path: str, **kwargs: Any) -> nn.Module:
        del kwargs
        return student_model if path == "student" else teacher_model

    monkeypatch.setattr(evaluation_module.AutoModelForCausalLM, "from_pretrained", _load_model)
    monkeypatch.setattr(
        evaluation_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )

    monkeypatch.setattr(evaluation_module, "compute_perplexity", lambda *args, **kwargs: 12.3)
    monkeypatch.setattr(
        evaluation_module,
        "compute_qa_metrics",
        lambda *args, **kwargs: {"accuracy": 0.4, "f1": 0.5},
    )
    monkeypatch.setattr(
        evaluation_module,
        "collect_memory_stats",
        lambda *args, **kwargs: SimpleNamespace(
            rss_mb=123.0,
            device_allocated_mb=12.0,
            peak_allocated_mb=34.0,
            device="cpu",
        ),
    )
    monkeypatch.setattr(
        evaluation_module,
        "get_model_disk_size_bytes",
        lambda *args, **kwargs: 10 * 1024 * 1024,
    )
    monkeypatch.setattr(
        evaluation_module,
        "infer_device",
        lambda *args, **kwargs: torch.device("cpu"),
    )

    cfg = SimpleNamespace(
        batch_size=1,
        max_eval_samples=2,
        max_new_tokens=4,
        generation_temperature=0.0,
        num_latency_samples=2,
    )
    runtime = SimpleNamespace(device="cpu", log_level="INFO")
    text_dataset = [{"prompt": "p1", "target": "a"}, {"prompt": "p2", "target": "b"}]

    out = evaluation_module.evaluate_model(
        model_path="student",
        text_dataset=text_dataset,
        lm_dataset=[],
        config=cfg,
        runtime=runtime,
        reference_model_path="teacher",
    )

    assert out["compression_ratio"] > 1.0
    assert out["efficiency"]["compression_ratio"] > 1.0
    assert out["speedup_vs_teacher"] >= 0.0


def test_throughput_is_finite_and_non_negative() -> None:
    model = DummyEvalModel(param_count=8)
    tokenizer = DummyTokenizer()

    perf = evaluation_module._measure_generation_performance_on_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=["hello world", "test prompt"],
        max_new_tokens=6,
        device=torch.device("cpu"),
    )

    assert math.isfinite(perf["throughput_tokens_per_sec"])
    assert perf["throughput_tokens_per_sec"] >= 0.0
    assert perf["tokens_processed"] >= 0.0
