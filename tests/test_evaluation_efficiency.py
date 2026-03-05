from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
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

    def pad(
        self,
        features: list[dict[str, list[int]]],
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        max_len = max(len(f["input_ids"]) for f in features)

        def _pad(key: str, value: int) -> torch.Tensor:
            rows = []
            for feat in features:
                row = list(feat[key])
                if len(row) < max_len:
                    row = row + [value] * (max_len - len(row))
                rows.append(row)
            return torch.tensor(rows, dtype=torch.long)

        out = {
            "input_ids": _pad("input_ids", 0),
            "attention_mask": _pad("attention_mask", 0),
        }
        if "labels" in features[0]:
            out["labels"] = _pad("labels", -100)
        return out


class DummyEvalModel(nn.Module):
    def __init__(self, param_count: int) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(param_count, dtype=torch.float32))
        self.vocab_size = 8

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> Any:
        del attention_mask
        batch, seq_len = input_ids.shape
        logits = self.param[0].to(input_ids.device) + torch.arange(
            self.vocab_size,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits = logits.view(1, 1, self.vocab_size).expand(batch, seq_len, self.vocab_size).contiguous()

        loss = None
        if labels is not None and seq_len > 1:
            shift_logits = logits[:, :-1, :].reshape(-1, self.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        return SimpleNamespace(logits=logits, loss=loss)

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
        "compute_qa_metrics_with_calibration",
        lambda *args, **kwargs: {
            "accuracy": 0.4,
            "f1": 0.5,
            "ece": 0.1,
            "brier_score": 0.2,
            "per_sample_confidence": [0.7, 0.8],
            "per_sample_correct": [True, False],
        },
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
    assert "disk_size_compression_ratio" in out
    assert "sparse_disk_size_mb" in out
    assert out["efficiency"]["compression_ratio"] > 1.0
    assert "disk_size_compression_ratio" in out["efficiency"]
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


def test_latency_benchmark_respects_timed_runs() -> None:
    model = DummyEvalModel(param_count=8)
    tokenizer = DummyTokenizer()
    max_new_tokens = 3
    timed_runs = 5

    perf = evaluation_module._measure_generation_performance_on_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=["a b c"],
        max_new_tokens=max_new_tokens,
        device=torch.device("cpu"),
        warmup_runs=2,
        timed_runs=timed_runs,
    )

    assert perf["samples"] == pytest.approx(float(timed_runs))
    assert perf["tokens_processed"] == pytest.approx(float(timed_runs * max_new_tokens))


def test_calibration_metrics_are_finite() -> None:
    model = DummyEvalModel(param_count=8)
    tokenizer = DummyTokenizer()
    dataset = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
        {"input_ids": [2, 3], "attention_mask": [1, 1], "labels": [-100, 3]},
    ]

    out = evaluation_module._compute_token_calibration_metrics(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=1,
        bins=10,
        device=torch.device("cpu"),
    )

    assert math.isfinite(out["expected_calibration_error"])
    assert math.isfinite(out["brier_score"])
    assert out["expected_calibration_error"] >= 0.0
    assert out["brier_score"] >= 0.0


def test_infer_sparse_disk_size_mb_from_final_sparse(tmp_path: Path) -> None:
    final_dir = tmp_path / "pruned_student" / "final"
    sparse_dir = tmp_path / "pruned_student" / "final_sparse"
    final_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    sparse_file = sparse_dir / "pytorch_model_sparse.pt"
    sparse_file.write_bytes(b"x" * 1024)

    size_mb = evaluation_module._infer_sparse_disk_size_mb(str(final_dir))
    assert size_mb is not None
    assert size_mb > 0.0
