from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import CausalLMDataCollator
from .utils import (
    count_parameters,
    dump_json,
    ensure_dir,
    get_logger,
    get_model_disk_size_bytes,
    infer_device,
)

_ATTN_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_SUFFIXES = ("gate_proj", "up_proj", "down_proj")


@dataclass
class PruningArtifacts:
    model_path: str
    method: str
    pruned_attention_heads: int
    total_attention_heads: int
    pruned_mlp_neurons: int
    total_mlp_neurons: int
    pruned_layers: int
    total_layers: int
    parameter_sparsity: float
    model_size_before_mb: float
    model_size_after_mb: float
    pruning_report_path: str


@dataclass
class ActivationStats:
    """Activation statistics aggregated during calibration forward passes.

    Attributes:
        output: Mean absolute activation per output channel for each linear module.
        input: Mean absolute activation per input channel for each linear module.
    """

    output: dict[str, torch.Tensor]
    input: dict[str, torch.Tensor]


def run_structured_pruning(
    config: Any,
    runtime: Any,
    model_path: str,
    calibration_dataset: Any,
) -> PruningArtifacts:
    """Run structured pruning (attention heads, MLP neurons, optional layers).

    Importance score is computed as:
        importance = beta * |weight| + (1 - beta) * activation_score
    where activation score comes from calibration forward passes.
    """
    logger = get_logger("dapd.pruning", getattr(runtime, "log_level", "INFO"))
    _validate_pruning_config(config)

    output_dir = ensure_dir(config.output_dir)

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    device = infer_device(runtime.device)
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    previous_use_cache = None
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        previous_use_cache = bool(model.config.use_cache)
        model.config.use_cache = False

    model_size_before_mb = get_model_disk_size_bytes(model_path) / (1024 * 1024)

    activations = _collect_activation_importance(
        model=model,
        tokenizer=tokenizer,
        dataset=calibration_dataset,
        batch_size=config.calibration_batch_size,
        calibration_batches=config.calibration_batches,
        device=device,
    )

    modules = {name: mod for name, mod in model.named_modules() if isinstance(mod, torch.nn.Linear)}

    pruned_heads = 0
    total_heads = 0
    pruned_neurons = 0
    total_neurons = 0
    pruned_layers = 0
    total_layers = 0

    if config.enable_attention_head_pruning:
        p, t = _prune_attention_heads(model=model, modules=modules, activations=activations, config=config)
        pruned_heads += p
        total_heads += t

    if config.enable_mlp_pruning:
        p, t = _prune_mlp_neurons(model=model, modules=modules, activations=activations, config=config)
        pruned_neurons += p
        total_neurons += t

    if config.enable_layer_pruning and config.layer_prune_ratio > 0:
        p, t = _prune_layers(model=model, modules=modules, activations=activations, config=config)
        pruned_layers += p
        total_layers += t

    if previous_use_cache is not None and hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = previous_use_cache

    final_dir = Path(output_dir) / "final"
    ensure_dir(final_dir)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    total_params, nonzero_params = count_parameters(model)
    sparsity = 1.0 - (nonzero_params / max(1, total_params))
    model_size_after_mb = get_model_disk_size_bytes(final_dir) / (1024 * 1024)

    report = {
        "method": config.method,
        "prune_ratio": config.prune_ratio,
        "beta": config.beta,
        "pruned_attention_heads": pruned_heads,
        "total_attention_heads": total_heads,
        "pruned_mlp_neurons": pruned_neurons,
        "total_mlp_neurons": total_neurons,
        "pruned_layers": pruned_layers,
        "total_layers": total_layers,
        "parameter_sparsity": float(sparsity),
        "model_size_before_mb": float(model_size_before_mb),
        "model_size_after_mb": float(model_size_after_mb),
    }
    report_path = Path(output_dir) / "pruning_report.json"
    dump_json(report, report_path)

    logger.info(
        "structured pruning done | heads %s/%s | mlp %s/%s | layers %s/%s | size %.1fMB -> %.1fMB",
        pruned_heads,
        total_heads,
        pruned_neurons,
        total_neurons,
        pruned_layers,
        total_layers,
        model_size_before_mb,
        model_size_after_mb,
    )

    return PruningArtifacts(
        model_path=str(final_dir),
        method=config.method,
        pruned_attention_heads=pruned_heads,
        total_attention_heads=total_heads,
        pruned_mlp_neurons=pruned_neurons,
        total_mlp_neurons=total_neurons,
        pruned_layers=pruned_layers,
        total_layers=total_layers,
        parameter_sparsity=float(sparsity),
        model_size_before_mb=float(model_size_before_mb),
        model_size_after_mb=float(model_size_after_mb),
        pruning_report_path=str(report_path),
    )


def _collect_activation_importance(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: Any,
    batch_size: int,
    calibration_batches: int,
    device: torch.device,
) -> ActivationStats:
    """Collect mean absolute input/output activations for linear modules."""
    collator = CausalLMDataCollator(tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    output_sums: dict[str, torch.Tensor] = {}
    output_counts: dict[str, int] = {}
    input_sums: dict[str, torch.Tensor] = {}
    input_counts: dict[str, int] = {}
    handles: list[Any] = []

    def _hook(name: str):
        def fn(_module: Any, module_input: tuple[Any, ...], module_output: Any) -> None:
            out_tensor = _extract_first_tensor(module_output)
            if out_tensor is not None:
                _accumulate_feature_stats(
                    store=output_sums,
                    counts=output_counts,
                    key=name,
                    vec=_reduce_feature_abs_mean(out_tensor),
                )

            in_tensor = _extract_first_tensor(module_input)
            if in_tensor is not None:
                _accumulate_feature_stats(
                    store=input_sums,
                    counts=input_counts,
                    key=name,
                    vec=_reduce_feature_abs_mean(in_tensor),
                )

        return fn

    for name, linear in model.named_modules():
        if isinstance(linear, torch.nn.Linear):
            handles.append(linear.register_forward_hook(_hook(name)))

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= calibration_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))

    for handle in handles:
        handle.remove()

    output_stats = {name: output_sums[name] / max(1, output_counts[name]) for name in output_sums}
    input_stats = {name: input_sums[name] / max(1, input_counts[name]) for name in input_sums}
    return ActivationStats(output=output_stats, input=input_stats)


def _prune_attention_heads(
    model: torch.nn.Module,
    modules: dict[str, torch.nn.Linear],
    activations: ActivationStats,
    config: Any,
) -> tuple[int, int]:
    """Prune low-importance attention heads by zeroing aligned projection channels."""
    pruned = 0
    total = 0

    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    num_heads = getattr(getattr(model, "config", None), "num_attention_heads", None)
    if hidden_size is None or num_heads is None or num_heads <= 0:
        return pruned, total

    head_dim = hidden_size // num_heads
    if head_dim <= 0 or head_dim * num_heads != hidden_size:
        return pruned, total

    for q_name, q_proj in modules.items():
        if not q_name.endswith("q_proj"):
            continue

        prefix = q_name[: -len("q_proj")]
        k_name, v_name, o_name = prefix + "k_proj", prefix + "v_proj", prefix + "o_proj"
        if k_name not in modules or v_name not in modules or o_name not in modules:
            continue

        k_proj = modules[k_name]
        v_proj = modules[v_name]
        o_proj = modules[o_name]

        if q_proj.out_features != hidden_size:
            continue
        if k_proj.out_features != hidden_size or v_proj.out_features != hidden_size:
            continue
        if o_proj.in_features != hidden_size:
            continue

        q_weight = q_proj.weight.detach().abs().mean(dim=1)
        k_weight = k_proj.weight.detach().abs().mean(dim=1)
        v_weight = v_proj.weight.detach().abs().mean(dim=1)
        o_weight = o_proj.weight.detach().abs().mean(dim=0)
        weight_score = (q_weight + k_weight + v_weight + o_weight) / 4.0

        q_act = _activation_vector(activations.output, q_name, q_weight)
        k_act = _activation_vector(activations.output, k_name, q_weight)
        v_act = _activation_vector(activations.output, v_name, q_weight)
        o_act = _activation_vector(activations.input, o_name, q_weight)
        act_score = (q_act + k_act + v_act + o_act) / 4.0

        channel_score = _combine_importance(weight_score, act_score, beta=float(config.beta))
        head_scores = channel_score.reshape(num_heads, head_dim).mean(dim=1)

        n_prune = int(num_heads * float(config.prune_ratio))
        n_prune = min(max(0, n_prune), max(0, num_heads - int(config.min_heads_per_layer)))

        total += num_heads
        if n_prune <= 0:
            continue

        prune_heads = torch.topk(head_scores, k=n_prune, largest=False).indices.tolist()
        channel_mask = torch.zeros(hidden_size, dtype=torch.bool, device=q_proj.weight.device)
        for head_idx in prune_heads:
            start = head_idx * head_dim
            channel_mask[start : start + head_dim] = True

        with torch.no_grad():
            q_proj.weight[channel_mask, :] = 0.0
            k_proj.weight[channel_mask, :] = 0.0
            v_proj.weight[channel_mask, :] = 0.0
            o_proj.weight[:, channel_mask] = 0.0
            if q_proj.bias is not None:
                q_proj.bias[channel_mask] = 0.0
            if k_proj.bias is not None:
                k_proj.bias[channel_mask] = 0.0
            if v_proj.bias is not None:
                v_proj.bias[channel_mask] = 0.0

        pruned += n_prune

    return pruned, total


def _prune_mlp_neurons(
    model: torch.nn.Module,
    modules: dict[str, torch.nn.Linear],
    activations: ActivationStats,
    config: Any,
) -> tuple[int, int]:
    """Prune low-importance MLP neurons by zeroing gate/up rows and down columns."""
    del model
    pruned = 0
    total = 0

    for gate_name, gate_proj in modules.items():
        if not gate_name.endswith("gate_proj"):
            continue

        prefix = gate_name[: -len("gate_proj")]
        up_name, down_name = prefix + "up_proj", prefix + "down_proj"
        if up_name not in modules or down_name not in modules:
            continue

        up_proj = modules[up_name]
        down_proj = modules[down_name]

        intermediate = gate_proj.out_features
        if intermediate <= int(config.min_mlp_neurons):
            continue

        if up_proj.out_features != intermediate or down_proj.in_features != intermediate:
            continue

        gate_weight = gate_proj.weight.detach().abs().mean(dim=1)
        up_weight = up_proj.weight.detach().abs().mean(dim=1)
        down_weight = down_proj.weight.detach().abs().mean(dim=0)
        weight_score = (gate_weight + up_weight + down_weight) / 3.0

        gate_act = _activation_vector(activations.output, gate_name, weight_score)
        up_act = _activation_vector(activations.output, up_name, weight_score)
        down_act = _activation_vector(activations.input, down_name, weight_score)
        act_score = (gate_act + up_act + down_act) / 3.0

        neuron_score = _combine_importance(weight_score, act_score, beta=float(config.beta))

        n_prune = int(intermediate * float(config.prune_ratio))
        n_prune = min(max(0, n_prune), max(0, intermediate - int(config.min_mlp_neurons)))

        total += intermediate
        if n_prune <= 0:
            continue

        prune_idx = torch.topk(neuron_score, k=n_prune, largest=False).indices

        with torch.no_grad():
            gate_proj.weight[prune_idx, :] = 0.0
            up_proj.weight[prune_idx, :] = 0.0
            down_proj.weight[:, prune_idx] = 0.0
            if gate_proj.bias is not None:
                gate_proj.bias[prune_idx] = 0.0
            if up_proj.bias is not None:
                up_proj.bias[prune_idx] = 0.0

        pruned += n_prune

    return pruned, total


def _prune_layers(
    model: torch.nn.Module,
    modules: dict[str, torch.nn.Linear],
    activations: ActivationStats,
    config: Any,
) -> tuple[int, int]:
    """Optionally prune low-importance transformer blocks by zeroing whole layers."""
    layer_prefix_scores: dict[str, list[float]] = {}

    for name, module in modules.items():
        if not _is_layer_scoring_module(name):
            continue

        prefix = _layer_block_prefix(name)
        weight_score = float(module.weight.detach().abs().mean().item())

        output_act = activations.output.get(name)
        input_act = activations.input.get(name)
        activation_terms: list[float] = []
        if output_act is not None:
            activation_terms.append(float(output_act.abs().mean().item()))
        if input_act is not None:
            activation_terms.append(float(input_act.abs().mean().item()))
        activation_score = float(sum(activation_terms) / len(activation_terms)) if activation_terms else 0.0

        score = float(float(config.beta) * weight_score + (1.0 - float(config.beta)) * activation_score)
        layer_prefix_scores.setdefault(prefix, []).append(score)

    if not layer_prefix_scores:
        return 0, 0

    layer_scores: list[tuple[str, float]] = []
    for prefix, scores in layer_prefix_scores.items():
        layer_scores.append((prefix, float(sum(scores) / len(scores))))

    total_layers = len(layer_scores)
    n_prune = int(total_layers * float(config.layer_prune_ratio))
    n_prune = min(max(0, n_prune), max(0, total_layers - 1))
    if n_prune <= 0:
        return 0, total_layers

    layer_scores.sort(key=lambda x: x[1])
    prune_prefixes = [name for name, _ in layer_scores[:n_prune]]

    pruned = 0
    for prefix in prune_prefixes:
        try:
            submodule = model.get_submodule(prefix)
        except Exception:
            continue

        with torch.no_grad():
            for parameter in submodule.parameters(recurse=True):
                parameter.zero_()
        pruned += 1

    return pruned, total_layers


def _reduce_feature_abs_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce any tensor to feature vector by averaging all dims except the last."""
    detached = tensor.detach().abs().float()
    if detached.ndim == 0:
        return detached.reshape(1).cpu()
    if detached.ndim == 1:
        return detached.cpu()
    reduce_dims = tuple(range(detached.ndim - 1))
    return detached.mean(dim=reduce_dims).cpu()


def _extract_first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _extract_first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _accumulate_feature_stats(
    store: dict[str, torch.Tensor],
    counts: dict[str, int],
    key: str,
    vec: torch.Tensor,
) -> None:
    if key not in store:
        store[key] = vec
        counts[key] = 1
        return

    if store[key].shape != vec.shape:
        return

    store[key] += vec
    counts[key] += 1


def _activation_vector(
    source: dict[str, torch.Tensor],
    name: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    vec = source.get(name)
    if vec is None or vec.numel() != reference.numel():
        return torch.zeros_like(reference)
    return vec.to(device=reference.device, dtype=reference.dtype)


def _combine_importance(
    weight_score: torch.Tensor,
    activation_score: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    return beta * _normalize(weight_score) + (1.0 - beta) * _normalize(activation_score)


def _normalize(vec: torch.Tensor) -> torch.Tensor:
    denom = vec.abs().max().clamp_min(1e-8)
    return vec / denom


def _is_layer_scoring_module(name: str) -> bool:
    return name.endswith(_ATTN_SUFFIXES) or name.endswith(_MLP_SUFFIXES)


def _layer_block_prefix(module_name: str) -> str:
    suffixes = (
        ".self_attn.q_proj",
        ".self_attn.k_proj",
        ".self_attn.v_proj",
        ".self_attn.o_proj",
        ".attention.q_proj",
        ".attention.k_proj",
        ".attention.v_proj",
        ".attention.o_proj",
        ".mlp.gate_proj",
        ".mlp.up_proj",
        ".mlp.down_proj",
    )
    for suffix in suffixes:
        if module_name.endswith(suffix):
            return module_name[: -len(suffix)]

    if "." in module_name:
        return module_name.rsplit(".", 1)[0]
    return module_name


def _validate_pruning_config(config: Any) -> None:
    method = str(config.method).lower().strip()
    if method != "structured":
        raise ValueError(f"Unsupported pruning method '{config.method}'. Only 'structured' is implemented.")
    if not (0.0 <= float(config.prune_ratio) < 1.0):
        raise ValueError(f"prune_ratio must be in [0, 1), got {config.prune_ratio}")
    if not (0.0 <= float(config.beta) <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {config.beta}")
    if int(config.calibration_batches) < 1:
        raise ValueError("calibration_batches must be >= 1")
    if int(config.calibration_batch_size) < 1:
        raise ValueError("calibration_batch_size must be >= 1")
    if bool(config.enable_layer_pruning) and not (0.0 <= float(config.layer_prune_ratio) < 1.0):
        raise ValueError(f"layer_prune_ratio must be in [0, 1), got {config.layer_prune_ratio}")
    if int(config.min_heads_per_layer) < 1:
        raise ValueError("min_heads_per_layer must be >= 1")
    if int(config.min_mlp_neurons) < 1:
        raise ValueError("min_mlp_neurons must be >= 1")
