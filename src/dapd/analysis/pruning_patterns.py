from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..utils import dump_json, ensure_dir, get_logger


def analyze_pruning_patterns(
    pruning_report_path: str,
    output_dir: str = "runs/dapd/analysis",
) -> dict[str, Any]:
    """Build pruning-pattern summaries and visualization artifacts from pruning report."""
    logger = get_logger("dapd.analysis.pruning_patterns")
    out_dir = ensure_dir(output_dir)
    out_path = Path(out_dir) / "pruning_patterns.json"

    report_file = Path(pruning_report_path)
    if not report_file.exists():
        result = {"error": f"missing_pruning_report:{pruning_report_path}"}
        dump_json(result, out_path)
        return result

    report = json.loads(report_file.read_text(encoding="utf-8"))
    attention_rows = report.get("attention_patterns", []) or []
    mlp_rows = report.get("mlp_patterns", []) or []

    head_heatmap = _build_attention_heatmap(attention_rows)
    neuron_importance = _collect_pruned_mlp_importance(mlp_rows)

    head_plot_path = Path(out_dir) / "attention_head_importance_heatmap.png"
    mlp_plot_path = Path(out_dir) / "mlp_neuron_importance_histogram.png"

    head_plot_saved = _save_attention_heatmap(head_heatmap, head_plot_path)
    mlp_plot_saved = _save_histogram(
        values=neuron_importance,
        path=mlp_plot_path,
        title="Pruned MLP Neuron Importance",
        xlabel="Importance",
    )

    result = {
        "pruning_report_path": str(report_file.resolve()),
        "pruning_mode_used": report.get("pruning_mode_used"),
        "pruned_attention_heads": int(report.get("pruned_attention_heads", 0)),
        "total_attention_heads": int(report.get("total_attention_heads", 0)),
        "pruned_mlp_neurons": int(report.get("pruned_mlp_neurons", 0)),
        "total_mlp_neurons": int(report.get("total_mlp_neurons", 0)),
        "attention_layers_analyzed": int(head_heatmap.shape[0]) if head_heatmap.numel() > 0 else 0,
        "mlp_groups_analyzed": int(len(mlp_rows)),
        "mlp_pruned_importance": _summary(neuron_importance),
        "plots": {
            "attention_head_importance_heatmap": str(head_plot_path) if head_plot_saved else None,
            "mlp_neuron_importance_histogram": str(mlp_plot_path) if mlp_plot_saved else None,
        },
    }
    dump_json(result, out_path)
    logger.info("pruning pattern analysis saved: %s", out_path)
    return result


def _build_attention_heatmap(attention_rows: list[dict[str, Any]]) -> torch.Tensor:
    if not attention_rows:
        return torch.empty((0, 0), dtype=torch.float32)

    ordered = sorted(
        attention_rows,
        key=lambda x: (int(x.get("layer_index", -1)), str(x.get("module", ""))),
    )
    max_heads = max((len(row.get("head_scores", []) or []) for row in ordered), default=0)
    if max_heads <= 0:
        return torch.empty((0, 0), dtype=torch.float32)

    matrix = torch.full((len(ordered), max_heads), float("nan"), dtype=torch.float32)
    for idx, row in enumerate(ordered):
        scores = row.get("head_scores", []) or []
        if not scores:
            continue
        matrix[idx, : len(scores)] = torch.tensor(scores, dtype=torch.float32)
    return matrix


def _collect_pruned_mlp_importance(mlp_rows: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in mlp_rows:
        imp = row.get("pruned_importance", []) or []
        values.extend(float(x) for x in imp)
    return values


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0}
    x = torch.tensor(values, dtype=torch.float32)
    return {
        "count": float(x.numel()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "p50": float(torch.quantile(x, 0.50).item()),
        "p90": float(torch.quantile(x, 0.90).item()),
    }


def _save_attention_heatmap(matrix: torch.Tensor, path: Path) -> bool:
    if matrix.numel() == 0:
        return False
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-error
    except Exception:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(9.0, 4.5))
    ax = figure.add_subplot(111)
    m = ax.imshow(matrix.numpy(), aspect="auto", interpolation="nearest")
    ax.set_title("Attention Head Importance Heatmap")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")
    figure.colorbar(m, ax=ax)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def _save_histogram(values: list[float], path: Path, title: str, xlabel: str) -> bool:
    if not values:
        return False
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-error
    except Exception:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.0, 4.8))
    plt.hist(values, bins=50, alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return True
