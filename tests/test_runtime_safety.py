from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dapd.utils import (
    collect_memory_stats,
    configure_model_for_training,
    infer_device,
    restore_model_use_cache,
    validate_quantization_config,
    validate_runtime_precision,
)


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.gc_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.gc_enabled = True


def test_configure_model_for_training_disables_use_cache_with_gc() -> None:
    model = DummyModel()

    previous = configure_model_for_training(model, gradient_checkpointing=True)

    assert previous is True
    assert model.gc_enabled is True
    assert model.config.use_cache is False

    restore_model_use_cache(model, previous)
    assert model.config.use_cache is True


def test_validate_runtime_precision_blocks_mps_fp16_bf16() -> None:
    with pytest.raises(ValueError):
        validate_runtime_precision(torch.device("mps"), fp16=True, bf16=False)

    with pytest.raises(ValueError):
        validate_runtime_precision(torch.device("mps"), fp16=False, bf16=True)


def test_validate_runtime_precision_allows_cuda_mixed_precision() -> None:
    validate_runtime_precision(torch.device("cuda"), fp16=True, bf16=False)
    validate_runtime_precision(torch.device("cuda"), fp16=False, bf16=True)


def test_infer_device_explicit_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError):
        infer_device("cuda")


def test_validate_quantization_config_blocks_non_cuda() -> None:
    with pytest.raises(RuntimeError):
        validate_quantization_config(torch.device("mps"), use_qlora=True)


def test_collect_memory_stats_cpu() -> None:
    stats = collect_memory_stats(torch.device("cpu"), reset_peak=True)
    assert stats.rss_mb > 0
    assert stats.device == "cpu"
    assert stats.device_allocated_mb == 0
    assert stats.peak_allocated_mb == 0
