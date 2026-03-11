from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .analysis.flops import estimate_model_flops_gmac, summarize_flops_reduction
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
    pruning_mode: str
    physical_attention_pruning_succeeded: bool
    physical_mlp_pruning_succeeded: bool
    pruned_attention_heads: int
    total_attention_heads: int
    pruned_mlp_neurons: int
    total_mlp_neurons: int
    pruned_layers: int
    total_layers: int
    parameter_sparsity: float
    model_size_before_mb: float
    model_size_after_mb: float
    flops_before_gmac: float | None
    flops_after_gmac: float | None
    flops_reduction_ratio: float | None
    estimated_speedup_potential: float
    pruning_report_path: str
    sparse_model_path: str | None = None
    sparse_compression_ratio: float = 1.0
    dense_size_mb: float = 0.0


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
    """Run structured pruning with masking or optional physical pruning.

    Importance score:
      importance = beta * |weight| + (1 - beta) * activation_score

    pruning_mode:
      - masking: always zero-out selected structured units
      - physical: try physical pruning for supported components, fallback to masking
    """
    logger = get_logger("dapd.pruning", getattr(runtime, "log_level", "INFO"))
    _validate_pruning_config(config)

    output_dir = ensure_dir(config.output_dir)

    # low_cpu_mem_usage: 로딩 시 CPU peak를 model_size 수준으로 억제한다.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
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
    total_params_before, _ = count_parameters(model)
    flops_before_gmac = estimate_model_flops_gmac(model=model, seq_len=128, batch_size=1, device=device)

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
    physical_head_success = False
    physical_mlp_success = False
    attention_patterns: list[dict[str, Any]] = []
    mlp_patterns: list[dict[str, Any]] = []

    if config.enable_attention_head_pruning:
        p, t, physical_head_success = _prune_attention_heads(
            model=model,
            modules=modules,
            activations=activations,
            config=config,
            logger=logger,
            details=attention_patterns,
        )
        pruned_heads += p
        total_heads += t
        if physical_head_success:
            modules = {
                name: mod
                for name, mod in model.named_modules()
                if isinstance(mod, torch.nn.Linear)
            }

    if config.enable_mlp_pruning:
        p, t, physical_mlp_success = _prune_mlp_neurons(
            model=model,
            modules=modules,
            activations=activations,
            config=config,
            logger=logger,
            details=mlp_patterns,
        )
        pruned_neurons += p
        total_neurons += t
        if physical_mlp_success:
            modules = {
                name: mod
                for name, mod in model.named_modules()
                if isinstance(mod, torch.nn.Linear)
            }

    if config.enable_layer_pruning and config.layer_prune_ratio > 0:
        p, t = _prune_layers(model=model, modules=modules, activations=activations, config=config)
        pruned_layers += p
        total_layers += t

    if (
        previous_use_cache is not None
        and hasattr(model, "config")
        and hasattr(model.config, "use_cache")
    ):
        model.config.use_cache = previous_use_cache

    final_dir = Path(output_dir) / "final"
    ensure_dir(final_dir)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    sparse_dir = Path(output_dir) / "final_sparse"
    sparse_stats = save_sparse_model(
        model=model,
        output_dir=sparse_dir,
        tokenizer=tokenizer,
        sparsity_threshold=0.3,
    )

    total_params, nonzero_params = count_parameters(model)
    sparsity = 1.0 - (nonzero_params / max(1, total_params))
    model_size_after_mb = get_model_disk_size_bytes(final_dir) / (1024 * 1024)
    parameter_compression_ratio = float(total_params_before / max(1, total_params))
    disk_compression_ratio = float(model_size_before_mb / max(1e-9, model_size_after_mb))
    flops_after_gmac = estimate_model_flops_gmac(model=model, seq_len=128, batch_size=1, device=device)
    flops_stats = summarize_flops_reduction(
        flops_before_gmac=flops_before_gmac,
        flops_after_gmac=flops_after_gmac,
    )

    mode_used = _resolve_pruning_mode_used(
        requested_mode=str(getattr(config, "pruning_mode", "masking")),
        head_physical_success=physical_head_success,
        mlp_physical_success=physical_mlp_success,
    )

    estimated_speedup = _estimate_speedup_potential(
        pruned_heads=pruned_heads,
        total_heads=total_heads,
        pruned_neurons=pruned_neurons,
        total_neurons=total_neurons,
        pruned_layers=pruned_layers,
        total_layers=total_layers,
        physical_head_success=physical_head_success,
        physical_mlp_success=physical_mlp_success,
    )

    report = {
        "method": config.method,
        "pruning_mode_requested": str(getattr(config, "pruning_mode", "masking")),
        "pruning_mode_used": mode_used,
        "prune_ratio": config.prune_ratio,
        "beta": config.beta,
        "physical_pruning": {
            "attention_heads_succeeded": physical_head_success,
            "mlp_neurons_succeeded": physical_mlp_success,
        },
        "attention_patterns": attention_patterns,
        "mlp_patterns": mlp_patterns,
        "pruned_attention_heads": pruned_heads,
        "total_attention_heads": total_heads,
        "pruned_mlp_neurons": pruned_neurons,
        "total_mlp_neurons": total_neurons,
        "pruned_layers": pruned_layers,
        "total_layers": total_layers,
        "parameter_sparsity": float(sparsity),
        "model_size_before_mb": float(model_size_before_mb),
        "model_size_after_mb": float(model_size_after_mb),
        "compression": {
            "parameter_compression_ratio": parameter_compression_ratio,
            "disk_size_before_mb": float(model_size_before_mb),
            "disk_size_after_mb": float(model_size_after_mb),
            "disk_compression_ratio": disk_compression_ratio,
            "sparse_size_mb": float(sparse_stats["sparse_size_mb"]),
            "sparse_compression_ratio": float(sparse_stats["sparse_compression_ratio"]),
        },
        "flops": flops_stats,
        "estimated_speedup_potential": float(estimated_speedup),
    }
    report_path = Path(output_dir) / "pruning_report.json"
    dump_json(report, report_path)

    logger.info(
        (
            "structured pruning done | mode=%s | heads %s/%s (physical=%s) | "
            "mlp %s/%s (physical=%s) | layers %s/%s | est_speedup=%.2fx | "
            "size %.1fMB -> %.1fMB | sparse_ratio=%.2fx"
        ),
        mode_used,
        pruned_heads,
        total_heads,
        physical_head_success,
        pruned_neurons,
        total_neurons,
        physical_mlp_success,
        pruned_layers,
        total_layers,
        estimated_speedup,
        model_size_before_mb,
        model_size_after_mb,
        float(sparse_stats["sparse_compression_ratio"]),
    )

    return PruningArtifacts(
        model_path=str(final_dir),
        method=config.method,
        pruning_mode=mode_used,
        physical_attention_pruning_succeeded=physical_head_success,
        physical_mlp_pruning_succeeded=physical_mlp_success,
        pruned_attention_heads=pruned_heads,
        total_attention_heads=total_heads,
        pruned_mlp_neurons=pruned_neurons,
        total_mlp_neurons=total_neurons,
        pruned_layers=pruned_layers,
        total_layers=total_layers,
        parameter_sparsity=float(sparsity),
        model_size_before_mb=float(model_size_before_mb),
        model_size_after_mb=float(model_size_after_mb),
        flops_before_gmac=flops_stats["flops_before_gmac"],
        flops_after_gmac=flops_stats["flops_after_gmac"],
        flops_reduction_ratio=flops_stats["flops_reduction_ratio"],
        estimated_speedup_potential=float(estimated_speedup),
        pruning_report_path=str(report_path),
        sparse_model_path=str(sparse_dir),
        sparse_compression_ratio=float(sparse_stats["sparse_compression_ratio"]),
        dense_size_mb=float(sparse_stats["dense_size_mb"]),
    )


def save_sparse_model(
    model: torch.nn.Module,
    output_dir: str | Path,
    tokenizer: Any,
    sparsity_threshold: float = 0.3,
) -> dict[str, float]:
    """Save sparse-aware model weights and return disk-size compression metrics.

    For each tensor in the state dict, tensors named like `*.weight` with zero
    fraction above `sparsity_threshold` are converted to sparse COO tensors.
    All tensors are moved to CPU before serialization.
    """
    out_dir = ensure_dir(output_dir)
    threshold = float(max(0.0, min(1.0, sparsity_threshold)))

    dense_state: dict[str, torch.Tensor] = {}
    sparse_state: dict[str, torch.Tensor] = {}

    for name, tensor in model.state_dict().items():
        cpu_tensor = tensor.detach().cpu()
        dense_state[name] = cpu_tensor

        save_tensor = cpu_tensor
        if "weight" in name and cpu_tensor.numel() > 0:
            zero_fraction = float((cpu_tensor == 0).sum().item()) / float(cpu_tensor.numel())
            if zero_fraction > threshold:
                try:
                    # MPS-safe path: always move to CPU before sparse conversion.
                    save_tensor = cpu_tensor.to_sparse()
                except Exception:
                    save_tensor = cpu_tensor
        sparse_state[name] = save_tensor

    dense_ref_path = Path(out_dir) / "_dense_reference.pt"
    sparse_state_path = Path(out_dir) / "pytorch_model_sparse.pt"

    torch.save(dense_state, dense_ref_path)
    dense_bytes = float(dense_ref_path.stat().st_size) if dense_ref_path.exists() else 0.0

    torch.save(sparse_state, sparse_state_path)
    sparse_bytes = float(sparse_state_path.stat().st_size) if sparse_state_path.exists() else 0.0

    if dense_ref_path.exists():
        dense_ref_path.unlink()

    if hasattr(model, "config") and getattr(model, "config", None) is not None:
        cfg = getattr(model, "config")
        if hasattr(cfg, "save_pretrained"):
            cfg.save_pretrained(str(out_dir))
    if hasattr(model, "generation_config") and getattr(model, "generation_config", None) is not None:
        generation_cfg = getattr(model, "generation_config")
        if hasattr(generation_cfg, "save_pretrained"):
            generation_cfg.save_pretrained(str(out_dir))

    tokenizer.save_pretrained(str(out_dir))

    dense_size_mb = float(dense_bytes / (1024 * 1024))
    sparse_size_mb = float(sparse_bytes / (1024 * 1024))
    sparse_compression_ratio = float(dense_bytes / max(1.0, sparse_bytes))
    return {
        "dense_size_mb": dense_size_mb,
        "sparse_size_mb": sparse_size_mb,
        "sparse_compression_ratio": sparse_compression_ratio,
    }


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
    logger: Any | None = None,
    details: list[dict[str, Any]] | None = None,
) -> tuple[int, int, bool]:
    """Prune low-importance attention heads with physical fallback behavior.

    Supports both standard MHA (Multi-Head Attention) and GQA (Grouped Query
    Attention) architectures such as Qwen2.5, Llama-3, Mistral.
    In MHA, q/k/v all have out_features == hidden_size.
    In GQA, k/v have out_features == num_key_value_heads * head_dim < hidden_size.
    Pruning is applied to Q heads (q_proj rows, o_proj cols); k/v masking is
    only applied when k/v out_features equal hidden_size (MHA case).

    Returns:
      (pruned_heads, total_heads, physical_success)
    """
    physical_mode = str(getattr(config, "pruning_mode", "masking")).lower().strip() == "physical"

    plans: list[dict[str, Any]] = []
    pruned = 0
    total = 0

    model_cfg = getattr(model, "config", None)
    hidden_size = getattr(model_cfg, "hidden_size", None)
    num_heads = getattr(model_cfg, "num_attention_heads", None)
    if hidden_size is None or num_heads is None or num_heads <= 0:
        return 0, 0, False

    head_dim = hidden_size // num_heads
    if head_dim <= 0 or head_dim * num_heads != hidden_size:
        return 0, 0, False

    # num_key_value_heads < num_attention_heads in GQA architectures.
    # Fall back to num_heads (MHA) when the attribute is absent.
    num_kv_heads: int = int(getattr(model_cfg, "num_key_value_heads", num_heads) or num_heads)
    is_gqa = num_kv_heads != num_heads

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

        # For GQA: k/v out_features == num_kv_heads * head_dim (< hidden_size).
        # For MHA: k/v out_features == hidden_size.
        kv_expected = num_kv_heads * head_dim
        if k_proj.out_features not in (hidden_size, kv_expected):
            continue
        if v_proj.out_features not in (hidden_size, kv_expected):
            continue
        if o_proj.in_features != hidden_size:
            continue

        q_weight = q_proj.weight.detach().abs().mean(dim=1)
        o_weight = o_proj.weight.detach().abs().mean(dim=0)

        if is_gqa:
            # In GQA, score is derived only from Q and O projections since k/v
            # channels don't correspond 1-to-1 with Q head channels.
            weight_score = (q_weight + o_weight) / 2.0
            q_act = _activation_vector(activations.output, q_name, q_weight)
            o_act = _activation_vector(activations.input, o_name, q_weight)
            act_score = (q_act + o_act) / 2.0
        else:
            k_weight = k_proj.weight.detach().abs().mean(dim=1)
            v_weight = v_proj.weight.detach().abs().mean(dim=1)
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
        # Q-channel mask spans all Q head channels (size == hidden_size).
        q_channel_mask = torch.zeros(hidden_size, dtype=torch.bool, device=q_proj.weight.device)
        for head_idx in prune_heads:
            start = head_idx * head_dim
            q_channel_mask[start : start + head_dim] = True

        # KV-channel mask: only used in MHA (full hidden_size) case.
        # In GQA k/v rows don't align with Q head indices so we skip k/v masking.
        kv_channel_mask: torch.Tensor | None = None
        if not is_gqa:
            kv_channel_mask = q_channel_mask

        plans.append(
            {
                "q_name": q_name,
                "k_name": k_name,
                "v_name": v_name,
                "o_name": o_name,
                "q_proj": q_proj,
                "k_proj": k_proj,
                "v_proj": v_proj,
                "o_proj": o_proj,
                "prune_heads": prune_heads,
                "channel_mask": q_channel_mask,
                "kv_channel_mask": kv_channel_mask,
                "layer_idx": _extract_layer_index(q_name),
            }
        )
        if details is not None:
            details.append(
                {
                    "module": q_name,
                    "layer_index": _extract_layer_index(q_name),
                    "num_heads": int(num_heads),
                    "num_kv_heads": int(num_kv_heads),
                    "is_gqa": bool(is_gqa),
                    "pruned_heads": [int(x) for x in prune_heads],
                    "head_scores": [float(x) for x in head_scores.detach().cpu().tolist()],
                }
            )
        pruned += n_prune

    if not plans:
        return 0, total, False

    if physical_mode:
        heads_to_prune: dict[int, list[int]] = {}
        for plan in plans:
            layer_idx = plan["layer_idx"]
            if layer_idx is None:
                heads_to_prune = {}
                break
            heads_to_prune.setdefault(layer_idx, []).extend(plan["prune_heads"])

        if heads_to_prune:
            deduped = {idx: sorted(set(v)) for idx, v in heads_to_prune.items() if v}
            if deduped and physical_prune_attention_heads(
                model=model,
                heads_to_prune=deduped,
                logger=logger,
            ):
                return pruned, total, True

        if logger is not None:
            logger.warning(
                "Physical attention head pruning unavailable for this architecture. "
                "Falling back to masking."
            )

    _apply_attention_masking(plans)
    return pruned, total, False


def _prune_mlp_neurons(
    model: torch.nn.Module,
    modules: dict[str, torch.nn.Linear],
    activations: ActivationStats,
    config: Any,
    logger: Any | None = None,
    details: list[dict[str, Any]] | None = None,
) -> tuple[int, int, bool]:
    """Prune low-importance MLP neurons with optional physical reduction.

    Returns:
      (pruned_neurons, total_neurons, physical_success)
    """
    physical_mode = str(getattr(config, "pruning_mode", "masking")).lower().strip() == "physical"

    plans: list[dict[str, Any]] = []
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

        plans.append(
            {
                "gate_name": gate_name,
                "up_name": up_name,
                "down_name": down_name,
                "gate_proj": gate_proj,
                "up_proj": up_proj,
                "down_proj": down_proj,
                "prune_idx": prune_idx,
                "intermediate": intermediate,
            }
        )
        if details is not None:
            pruned_idx_list = [int(x) for x in prune_idx.detach().cpu().tolist()]
            neuron_score_cpu = neuron_score.detach().cpu()
            details.append(
                {
                    "module": gate_name,
                    "layer_index": _extract_layer_index(gate_name),
                    "total_neurons": int(intermediate),
                    "pruned_neurons": pruned_idx_list,
                    "pruned_importance": [float(neuron_score_cpu[idx].item()) for idx in pruned_idx_list],
                    "importance_mean": float(neuron_score_cpu.mean().item()),
                    "importance_std": float(neuron_score_cpu.std(unbiased=False).item()),
                }
            )
        pruned += n_prune

    if not plans:
        return 0, total, False

    if physical_mode:
        if physical_prune_mlp_neurons(model=model, plans=plans, config=config, logger=logger):
            return pruned, total, True

        if logger is not None:
            logger.warning(
                "Physical MLP pruning unavailable for this architecture. Falling back to masking."
            )

    _apply_mlp_masking(plans)
    return pruned, total, False


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
        activation_score = (
            float(sum(activation_terms) / len(activation_terms))
            if activation_terms
            else 0.0
        )

        score = float(
            float(config.beta) * weight_score
            + (1.0 - float(config.beta)) * activation_score
        )
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


def _apply_attention_masking(plans: list[dict[str, Any]]) -> None:
    for plan in plans:
        q_proj = plan["q_proj"]
        k_proj = plan["k_proj"]
        v_proj = plan["v_proj"]
        o_proj = plan["o_proj"]
        # Q-channel mask (always hidden_size channels).
        channel_mask = plan["channel_mask"]
        # KV-channel mask: same as Q mask for MHA; None for GQA
        # (k/v rows don't align with Q head channels in GQA).
        kv_channel_mask = plan.get("kv_channel_mask", channel_mask)

        with torch.no_grad():
            q_proj.weight[channel_mask, :] = 0.0
            o_proj.weight[:, channel_mask] = 0.0
            if q_proj.bias is not None:
                q_proj.bias[channel_mask] = 0.0

            if kv_channel_mask is not None:
                k_proj.weight[kv_channel_mask, :] = 0.0
                v_proj.weight[kv_channel_mask, :] = 0.0
                if k_proj.bias is not None:
                    k_proj.bias[kv_channel_mask] = 0.0
                if v_proj.bias is not None:
                    v_proj.bias[kv_channel_mask] = 0.0


def _apply_mlp_masking(plans: list[dict[str, Any]]) -> None:
    for plan in plans:
        gate_proj = plan["gate_proj"]
        up_proj = plan["up_proj"]
        down_proj = plan["down_proj"]
        prune_idx = plan["prune_idx"]

        with torch.no_grad():
            gate_proj.weight[prune_idx, :] = 0.0
            up_proj.weight[prune_idx, :] = 0.0
            down_proj.weight[:, prune_idx] = 0.0
            if gate_proj.bias is not None:
                gate_proj.bias[prune_idx] = 0.0
            if up_proj.bias is not None:
                up_proj.bias[prune_idx] = 0.0


def physical_prune_attention_heads(
    model: torch.nn.Module,
    heads_to_prune: dict[int, list[int]] | None = None,
    logger: Any | None = None,
) -> bool:
    """Physically remove attention heads using architecture-native prune APIs."""
    heads_to_prune = heads_to_prune or {}
    if not heads_to_prune:
        return False

    for target in (model, getattr(model, "model", None)):
        if target is None:
            continue
        prune_fn = getattr(target, "prune_heads", None)
        if not callable(prune_fn):
            continue

        try:
            prune_fn(heads_to_prune)
            return True
        except Exception as exc:
            if logger is not None:
                logger.warning("Physical head prune API failed: %s", exc)

    return False


def physical_prune_mlp_neurons(
    model: torch.nn.Module,
    plans: list[dict[str, Any]] | None = None,
    config: Any | None = None,
    logger: Any | None = None,
) -> bool:
    """Physically remove MLP neurons for LLaMA-like gate/up/down blocks."""
    plans = plans or []
    if not plans:
        return False
    if config is None:
        return False

    if not _is_llama_like_mlp_model(model):
        if logger is not None:
            logger.warning(
                "Physical MLP pruning is supported only for "
                "LLaMA-like gate/up/down blocks."
            )
        return False

    try:
        for plan in plans:
            gate_name = plan["gate_name"]
            up_name = plan["up_name"]
            down_name = plan["down_name"]
            prune_idx = plan["prune_idx"]
            intermediate = int(plan["intermediate"])

            keep_mask = torch.ones(intermediate, dtype=torch.bool, device=prune_idx.device)
            keep_mask[prune_idx] = False
            keep_idx = torch.where(keep_mask)[0]
            if keep_idx.numel() < int(config.min_mlp_neurons):
                return False

            parent_path = gate_name.rsplit(".", 1)[0]
            parent_module = model.get_submodule(parent_path)

            gate_proj = model.get_submodule(gate_name)
            up_proj = model.get_submodule(up_name)
            down_proj = model.get_submodule(down_name)

            if not (
                isinstance(gate_proj, torch.nn.Linear)
                and isinstance(up_proj, torch.nn.Linear)
                and isinstance(down_proj, torch.nn.Linear)
            ):
                return False

            new_intermediate = int(keep_idx.numel())
            new_gate = torch.nn.Linear(
                gate_proj.in_features,
                new_intermediate,
                bias=gate_proj.bias is not None,
            )
            new_up = torch.nn.Linear(
                up_proj.in_features,
                new_intermediate,
                bias=up_proj.bias is not None,
            )
            new_down = torch.nn.Linear(
                new_intermediate,
                down_proj.out_features,
                bias=down_proj.bias is not None,
            )

            new_gate.to(device=gate_proj.weight.device, dtype=gate_proj.weight.dtype)
            new_up.to(device=up_proj.weight.device, dtype=up_proj.weight.dtype)
            new_down.to(device=down_proj.weight.device, dtype=down_proj.weight.dtype)

            with torch.no_grad():
                new_gate.weight.copy_(gate_proj.weight[keep_idx, :])
                if gate_proj.bias is not None and new_gate.bias is not None:
                    new_gate.bias.copy_(gate_proj.bias[keep_idx])

                new_up.weight.copy_(up_proj.weight[keep_idx, :])
                if up_proj.bias is not None and new_up.bias is not None:
                    new_up.bias.copy_(up_proj.bias[keep_idx])

                new_down.weight.copy_(down_proj.weight[:, keep_idx])
                if down_proj.bias is not None and new_down.bias is not None:
                    # Keep full down_proj bias intentionally: MLP neuron pruning changes
                    # down_proj input dimension only. Output dimension is unchanged.
                    new_down.bias.copy_(down_proj.bias)

            setattr(parent_module, "gate_proj", new_gate)
            setattr(parent_module, "up_proj", new_up)
            setattr(parent_module, "down_proj", new_down)
    except Exception as exc:
        if logger is not None:
            logger.warning("Physical MLP pruning failed: %s", exc)
        return False

    return True


def _extract_layer_index(module_name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", module_name)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _is_llama_like_mlp_model(model: torch.nn.Module) -> bool:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or not isinstance(layers, torch.nn.ModuleList) or len(layers) == 0:
        return False

    first = layers[0]
    mlp = getattr(first, "mlp", None)
    if mlp is None:
        return False

    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    down = getattr(mlp, "down_proj", None)
    return (
        isinstance(gate, torch.nn.Linear)
        and isinstance(up, torch.nn.Linear)
        and isinstance(down, torch.nn.Linear)
    )


def _resolve_pruning_mode_used(
    requested_mode: str,
    head_physical_success: bool,
    mlp_physical_success: bool,
) -> str:
    req = requested_mode.lower().strip()
    if req != "physical":
        return "masking"

    if head_physical_success and mlp_physical_success:
        return "physical"
    if head_physical_success or mlp_physical_success:
        return "mixed"
    return "masking"


def _estimate_speedup_potential(
    pruned_heads: int,
    total_heads: int,
    pruned_neurons: int,
    total_neurons: int,
    pruned_layers: int,
    total_layers: int,
    physical_head_success: bool,
    physical_mlp_success: bool,
) -> float:
    """Heuristic speedup estimate from structured pruning ratios.

    This is an estimate for reporting only, not a guaranteed runtime speedup.
    """
    head_frac = pruned_heads / max(1, total_heads)
    mlp_frac = pruned_neurons / max(1, total_neurons)
    layer_frac = pruned_layers / max(1, total_layers) if total_layers > 0 else 0.0

    head_coeff = 0.30 if physical_head_success else 0.05
    mlp_coeff = 0.45 if physical_mlp_success else 0.08
    layer_coeff = 0.60

    gain = (head_coeff * head_frac) + (mlp_coeff * mlp_frac) + (layer_coeff * layer_frac)
    return float(max(1.0, 1.0 + gain))


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
        raise ValueError(
            f"Unsupported pruning method '{config.method}'. "
            "Only 'structured' is implemented."
        )

    mode = str(getattr(config, "pruning_mode", "masking")).lower().strip()
    if mode not in {"masking", "physical"}:
        raise ValueError(f"pruning_mode must be one of ['masking', 'physical'], got '{mode}'")

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
