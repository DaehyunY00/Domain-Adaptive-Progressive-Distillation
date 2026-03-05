from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from dapd.pruning import run_structured_pruning


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token = None
        self.eos_token = "<eos>"

    def pad(
        self,
        features: list[dict[str, list[int]]],
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        max_len = max(len(f["input_ids"]) for f in features)

        def _pad(key: str, pad_value: int) -> torch.Tensor:
            rows = []
            for f in features:
                row = list(f[key])
                if len(row) < max_len:
                    row = row + [pad_value] * (max_len - len(row))
                rows.append(row)
            return torch.tensor(rows, dtype=torch.long)

        return {
            "input_ids": _pad("input_ids", 0),
            "attention_mask": _pad("attention_mask", 0),
            "labels": _pad("labels", -100),
        }

    def save_pretrained(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "tokenizer.json").write_text("{}", encoding="utf-8")


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.k_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.v_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.o_proj = nn.Linear(4, 4, bias=False)

        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(4, 6, bias=False)
        self.mlp.up_proj = nn.Linear(4, 6, bias=False)
        self.mlp.down_proj = nn.Linear(6, 4, bias=False)


class DummyCausalLM(nn.Module):
    def __init__(self, with_prune_heads_api: bool = False) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock()])
        self.lm_head = nn.Linear(4, 32, bias=False)
        self.config = SimpleNamespace(hidden_size=4, num_attention_heads=2, use_cache=True)

        self._with_prune_heads_api = with_prune_heads_api
        self.prune_head_calls: list[dict[int, list[int]]] = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> Any:
        del attention_mask
        x = self.embed(input_ids)

        for layer in self.model.layers:
            q = layer.self_attn.q_proj(x)
            k = layer.self_attn.k_proj(x)
            v = layer.self_attn.v_proj(x)
            attn = layer.self_attn.o_proj((q + k + v) / 3.0)

            gate = torch.sigmoid(layer.mlp.gate_proj(x))
            up = layer.mlp.up_proj(x)
            x = x + attn + layer.mlp.down_proj(gate * up)

        logits = self.lm_head(x)
        return SimpleNamespace(logits=logits)

    def prune_heads(self, heads_to_prune: dict[int, list[int]]) -> None:
        if not self._with_prune_heads_api:
            raise AttributeError("prune_heads API unavailable")
        self.prune_head_calls.append(heads_to_prune)

    def save_pretrained(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out / "pytorch_model.bin")


def _dummy_calibration_dataset() -> list[dict[str, list[int]]]:
    return [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
        {"input_ids": [3, 4], "attention_mask": [1, 1], "labels": [-100, 4]},
    ]


def test_run_structured_pruning_masking_mode_smoke(monkeypatch: Any, tmp_path: Path) -> None:
    model = DummyCausalLM(with_prune_heads_api=False)

    monkeypatch.setattr(
        "dapd.pruning.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        "dapd.pruning.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )

    model_path = tmp_path / "model"
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "dummy.bin").write_bytes(b"x")

    cfg = SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        method="structured",
        pruning_mode="masking",
        prune_ratio=0.5,
        beta=0.5,
        calibration_batches=1,
        calibration_batch_size=1,
        enable_attention_head_pruning=True,
        enable_mlp_pruning=True,
        enable_layer_pruning=False,
        layer_prune_ratio=0.0,
        min_heads_per_layer=1,
        min_mlp_neurons=2,
    )
    runtime = SimpleNamespace(device="cpu", log_level="INFO")

    artifacts = run_structured_pruning(
        config=cfg,
        runtime=runtime,
        model_path=str(model_path),
        calibration_dataset=_dummy_calibration_dataset(),
    )

    assert Path(artifacts.model_path).exists()
    assert artifacts.pruning_mode == "masking"

    report = json.loads(Path(artifacts.pruning_report_path).read_text(encoding="utf-8"))
    assert report["pruning_mode_used"] == "masking"
    assert "estimated_speedup_potential" in report
    assert isinstance(report.get("attention_patterns"), list)
    assert isinstance(report.get("mlp_patterns"), list)


def test_run_structured_pruning_physical_head_api_if_available(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    model = DummyCausalLM(with_prune_heads_api=True)

    monkeypatch.setattr(
        "dapd.pruning.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        "dapd.pruning.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )

    model_path = tmp_path / "model"
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "dummy.bin").write_bytes(b"x")

    cfg = SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        method="structured",
        pruning_mode="physical",
        prune_ratio=0.5,
        beta=0.5,
        calibration_batches=1,
        calibration_batch_size=1,
        enable_attention_head_pruning=True,
        enable_mlp_pruning=False,
        enable_layer_pruning=False,
        layer_prune_ratio=0.0,
        min_heads_per_layer=1,
        min_mlp_neurons=2,
    )
    runtime = SimpleNamespace(device="cpu", log_level="INFO")

    artifacts = run_structured_pruning(
        config=cfg,
        runtime=runtime,
        model_path=str(model_path),
        calibration_dataset=_dummy_calibration_dataset(),
    )

    assert artifacts.physical_attention_pruning_succeeded is True
    assert len(model.prune_head_calls) == 1
