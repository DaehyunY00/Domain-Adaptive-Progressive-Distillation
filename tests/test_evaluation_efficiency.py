from __future__ import annotations

from types import SimpleNamespace
import math

import pytest
import torch

from dapd import evaluation as evaluation_module
from dapd.metrics import compute_compression_ratio


class DummyModel:
    def to(self, _device: torch.device) -> "DummyModel":
        return self

    def eval(self) -> "DummyModel":
        return self


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token = None
        self.eos_token = "<eos>"


def test_compute_compression_ratio_teacher_over_student() -> None:
    assert compute_compression_ratio(teacher_params := 200.0, student_params := 100.0) == pytest.approx(
        teacher_params / student_params
    )
    assert compute_compression_ratio(0.0, 100.0) == pytest.approx(0.0)
    assert math.isinf(compute_compression_ratio(100.0, 0.0))


def test_evaluate_model_reports_efficiency_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluation_module.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: DummyModel(),
    )
    monkeypatch.setattr(
        evaluation_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )

    monkeypatch.setattr(evaluation_module, "compute_perplexity", lambda *args, **kwargs: 12.3)
    monkeypatch.setattr(evaluation_module, "compute_qa_metrics", lambda *args, **kwargs: {"accuracy": 0.4, "f1": 0.5})
    monkeypatch.setattr(
        evaluation_module,
        "measure_generation_performance",
        lambda *args, **kwargs: {
            "latency_ms": 10.0,
            "throughput_tokens_per_sec": 250.0,
            "tokens_processed": 500.0,
            "inference_time_sec": 2.0,
            "samples": 5.0,
        },
    )
    monkeypatch.setattr(evaluation_module, "count_parameters", lambda *args, **kwargs: (100, 90))
    monkeypatch.setattr(evaluation_module, "get_model_disk_size_bytes", lambda *args, **kwargs: 50 * 1024 * 1024)
    monkeypatch.setattr(
        evaluation_module,
        "collect_memory_stats",
        lambda *args, **kwargs: SimpleNamespace(
            rss_mb=321.0,
            device_allocated_mb=12.0,
            peak_allocated_mb=34.0,
            device="cpu",
        ),
    )
    monkeypatch.setattr(evaluation_module, "infer_device", lambda *args, **kwargs: torch.device("cpu"))
    monkeypatch.setattr(
        evaluation_module,
        "_evaluate_reference_performance",
        lambda *args, **kwargs: {
            "latency_ms": 20.0,
            "throughput_tokens_per_sec": 125.0,
            "tokens_processed": 500.0,
            "inference_time_sec": 4.0,
            "params": 200.0,
            "disk_size_mb": 100.0,
        },
    )

    cfg = SimpleNamespace(
        batch_size=1,
        max_eval_samples=10,
        max_new_tokens=8,
        generation_temperature=0.0,
        num_latency_samples=5,
    )
    runtime = SimpleNamespace(device="cpu", log_level="INFO")

    out = evaluation_module.evaluate_model(
        model_path="student",
        text_dataset=[],
        lm_dataset=[],
        config=cfg,
        runtime=runtime,
        reference_model_path="teacher",
    )

    assert out["compression_ratio"] == pytest.approx(2.0)
    assert out["throughput_tokens_per_sec"] == pytest.approx(250.0)
    assert out["latency"]["mean_ms"] == pytest.approx(10.0)
    assert out["memory_usage"]["rss_mb"] == pytest.approx(321.0)
    assert out["memory_usage"]["peak_allocated_mb"] == pytest.approx(34.0)
