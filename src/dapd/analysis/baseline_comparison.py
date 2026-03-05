from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import dump_json, ensure_dir

BASELINE_TEMPLATE: list[dict[str, Any]] = [
    {
        "method": "LoRA fine-tuning",
        "adaptation": "Yes",
        "distillation": "No",
        "pruning": "No",
        "training_cost": "Medium",
    },
    {
        "method": "Direct KD",
        "adaptation": "No",
        "distillation": "Yes",
        "pruning": "No",
        "training_cost": "Medium",
    },
    {
        "method": "SparseGPT",
        "adaptation": "Optional",
        "distillation": "No",
        "pruning": "Yes",
        "training_cost": "Low",
    },
    {
        "method": "LLM-Pruner",
        "adaptation": "Optional",
        "distillation": "No",
        "pruning": "Yes",
        "training_cost": "Medium",
    },
    {
        "method": "DAPD",
        "adaptation": "Yes",
        "distillation": "Yes",
        "pruning": "Yes",
        "training_cost": "Medium-High",
    },
]


def generate_baseline_comparison(
    method_metrics: dict[str, dict[str, Any]] | None = None,
    output_dir: str = "runs/dapd/analysis",
) -> dict[str, Any]:
    """Generate baseline comparison table in JSON + Markdown."""
    method_metrics = method_metrics or {}
    rows: list[dict[str, Any]] = []
    for base in BASELINE_TEMPLATE:
        method = str(base["method"])
        metric = method_metrics.get(method, {})
        row = dict(base)
        row["accuracy"] = _to_float_or_none(metric.get("accuracy"))
        row["f1"] = _to_float_or_none(metric.get("f1"))
        row["notes"] = metric.get("notes")
        rows.append(row)

    out_dir = ensure_dir(output_dir)
    json_path = Path(out_dir) / "baseline_comparison.json"
    md_path = Path(out_dir) / "baseline_comparison.md"

    payload = {
        "columns": ["Method", "Adaptation", "Distillation", "Pruning", "Training Cost", "Accuracy", "F1"],
        "rows": rows,
    }
    dump_json(payload, json_path)
    md_path.write_text(_to_markdown(rows), encoding="utf-8")

    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "rows": rows,
    }


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Adaptation | Distillation | Pruning | Training Cost | Accuracy | F1 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        acc = _fmt(row.get("accuracy"))
        f1 = _fmt(row.get("f1"))
        lines.append(
            f"| {row['method']} | {row['adaptation']} | {row['distillation']} | "
            f"{row['pruning']} | {row['training_cost']} | {acc} | {f1} |"
        )
    return "\n".join(lines) + "\n"


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
