from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from dapd.pruning import ActivationStats, _prune_attention_heads, _prune_layers, _prune_mlp_neurons


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


class TinyModel(nn.Module):
    def __init__(self, num_layers: int = 1) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock() for _ in range(num_layers)])
        self.config = SimpleNamespace(hidden_size=4, num_attention_heads=2)


def _linear_modules(model: nn.Module) -> dict[str, nn.Linear]:
    return {name: mod for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}


def _all_ones(model: nn.Module) -> None:
    with torch.no_grad():
        for _, module in _linear_modules(model).items():
            module.weight.fill_(1.0)


def test_attention_head_pruning_uses_activation_importance() -> None:
    model = TinyModel(num_layers=1)
    _all_ones(model)
    modules = _linear_modules(model)

    q_name = "model.layers.0.self_attn.q_proj"
    k_name = "model.layers.0.self_attn.k_proj"
    v_name = "model.layers.0.self_attn.v_proj"
    o_name = "model.layers.0.self_attn.o_proj"

    low_high = torch.tensor([0.01, 0.01, 1.0, 1.0], dtype=torch.float32)
    activations = ActivationStats(
        output={q_name: low_high, k_name: low_high, v_name: low_high},
        input={o_name: low_high},
    )

    config = SimpleNamespace(prune_ratio=0.5, beta=0.0, min_heads_per_layer=1)
    pruned, total, physical = _prune_attention_heads(
        model=model,
        modules=modules,
        activations=activations,
        config=config,
    )

    assert (pruned, total) == (1, 2)
    assert physical is False

    q_proj = modules[q_name]
    o_proj = modules[o_name]
    assert torch.allclose(q_proj.weight[:2, :], torch.zeros_like(q_proj.weight[:2, :]))
    assert not torch.allclose(q_proj.weight[2:, :], torch.zeros_like(q_proj.weight[2:, :]))
    assert torch.allclose(o_proj.weight[:, :2], torch.zeros_like(o_proj.weight[:, :2]))
    assert not torch.allclose(o_proj.weight[:, 2:], torch.zeros_like(o_proj.weight[:, 2:]))


def test_mlp_pruning_zeros_low_importance_neurons() -> None:
    model = TinyModel(num_layers=1)
    _all_ones(model)
    modules = _linear_modules(model)

    gate_name = "model.layers.0.mlp.gate_proj"
    up_name = "model.layers.0.mlp.up_proj"
    down_name = "model.layers.0.mlp.down_proj"

    score = torch.tensor([0.00, 0.10, 0.20, 0.90, 1.00, 1.10], dtype=torch.float32)
    activations = ActivationStats(
        output={gate_name: score, up_name: score},
        input={down_name: score},
    )

    config = SimpleNamespace(prune_ratio=0.5, beta=0.0, min_mlp_neurons=3)
    pruned, total, physical = _prune_mlp_neurons(
        model=model,
        modules=modules,
        activations=activations,
        config=config,
    )

    assert (pruned, total) == (3, 6)
    assert physical is False

    gate = modules[gate_name]
    up = modules[up_name]
    down = modules[down_name]

    assert torch.allclose(gate.weight[:3, :], torch.zeros_like(gate.weight[:3, :]))
    assert torch.allclose(up.weight[:3, :], torch.zeros_like(up.weight[:3, :]))
    assert torch.allclose(down.weight[:, :3], torch.zeros_like(down.weight[:, :3]))
    assert not torch.allclose(gate.weight[3:, :], torch.zeros_like(gate.weight[3:, :]))


def test_layer_pruning_zeros_low_importance_block() -> None:
    model = TinyModel(num_layers=3)
    modules = _linear_modules(model)

    with torch.no_grad():
        for name, module in modules.items():
            if name.startswith("model.layers.0"):
                module.weight.fill_(0.1)
            elif name.startswith("model.layers.1"):
                module.weight.fill_(1.0)
            elif name.startswith("model.layers.2"):
                module.weight.fill_(2.0)

    activations = ActivationStats(output={}, input={})
    config = SimpleNamespace(beta=1.0, layer_prune_ratio=0.34)
    pruned, total = _prune_layers(
        model=model,
        modules=modules,
        activations=activations,
        config=config,
    )

    assert (pruned, total) == (1, 3)

    for name, module in modules.items():
        if name.startswith("model.layers.0"):
            assert torch.allclose(module.weight, torch.zeros_like(module.weight))
        if name.startswith("model.layers.1"):
            assert not torch.allclose(module.weight, torch.zeros_like(module.weight))
        if name.startswith("model.layers.2"):
            assert not torch.allclose(module.weight, torch.zeros_like(module.weight))
