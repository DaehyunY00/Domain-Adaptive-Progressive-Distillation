from __future__ import annotations

from typing import Any

import torch


def estimate_model_flops_gmac(
    model: torch.nn.Module,
    seq_len: int = 128,
    batch_size: int = 1,
    device: torch.device | None = None,
) -> float | None:
    """Estimate model FLOPs (GMAC) with fvcore; returns None if unavailable."""
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore
    except Exception:
        return None

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()
    input_ids = torch.ones((int(batch_size), int(seq_len)), dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        try:
            analysis = FlopCountAnalysis(model, (input_ids, attention_mask))
            total_flops = float(analysis.total())
        except Exception:
            return None

    if total_flops < 0 or not torch.isfinite(torch.tensor(total_flops)):
        return None
    return float(total_flops / 1e9)


def summarize_flops_reduction(
    flops_before_gmac: float | None,
    flops_after_gmac: float | None,
) -> dict[str, float | None]:
    if flops_before_gmac is None or flops_after_gmac is None or flops_before_gmac <= 0:
        return {
            "flops_before_gmac": flops_before_gmac,
            "flops_after_gmac": flops_after_gmac,
            "flops_reduction_ratio": None,
            "flops_speedup_estimate": None,
        }

    ratio = float(max(0.0, 1.0 - (flops_after_gmac / flops_before_gmac)))
    speedup = float(flops_before_gmac / max(1e-12, flops_after_gmac))
    return {
        "flops_before_gmac": float(flops_before_gmac),
        "flops_after_gmac": float(flops_after_gmac),
        "flops_reduction_ratio": ratio,
        "flops_speedup_estimate": speedup,
    }


def supports_flops_estimation() -> bool:
    try:
        import fvcore  # noqa: F401
    except Exception:
        return False
    return True


def format_flops_for_report(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "flops_before_gmac": stats.get("flops_before_gmac"),
        "flops_after_gmac": stats.get("flops_after_gmac"),
        "flops_reduction_ratio": stats.get("flops_reduction_ratio"),
        "flops_speedup_estimate": stats.get("flops_speedup_estimate"),
    }
