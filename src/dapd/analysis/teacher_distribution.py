from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from ..utils import dump_json, ensure_dir, get_logger, infer_device


def compute_entropy_distribution(teacher_logits: torch.Tensor) -> list[float]:
    """Return per-token entropy values from teacher logits."""
    logits = _as_logits_tensor(teacher_logits)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    return entropy.reshape(-1).detach().cpu().tolist()


def compute_confidence_distribution(teacher_logits: torch.Tensor) -> list[float]:
    """Return per-token max-probability confidence values from teacher logits."""
    logits = _as_logits_tensor(teacher_logits)
    probs = torch.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values
    return confidence.reshape(-1).detach().cpu().tolist()


def compute_kl_divergence_between_teachers(
    general_teacher: torch.Tensor,
    domain_teacher: torch.Tensor,
) -> float:
    """Compute mean KL(domain_teacher || general_teacher) over all tokens."""
    general_logits = _as_logits_tensor(general_teacher)
    domain_logits = _as_logits_tensor(domain_teacher)
    if general_logits.shape != domain_logits.shape:
        raise ValueError(
            "KL requires same logits shape for both teachers; "
            f"got {tuple(general_logits.shape)} vs {tuple(domain_logits.shape)}"
        )

    general_log_probs = F.log_softmax(general_logits, dim=-1)
    domain_probs = F.softmax(domain_logits, dim=-1)
    token_kl = F.kl_div(general_log_probs, domain_probs, reduction="none").sum(dim=-1)
    return float(token_kl.mean().item())


def analyze_teacher_distributions(
    general_teacher_path: str,
    domain_teacher_path: str,
    dataset: Any,
    runtime: Any,
    output_dir: str = "runs/dapd/analysis",
    max_samples: int = 128,
) -> dict[str, Any]:
    """Analyze teacher entropy/confidence shifts and save JSON + histograms."""
    logger = get_logger("dapd.analysis.teacher_distribution", getattr(runtime, "log_level", "INFO"))
    out_dir = ensure_dir(output_dir)
    out_path = Path(out_dir) / "teacher_distribution.json"

    if len(dataset) == 0:
        result = {"error": "empty_dataset"}
        dump_json(result, out_path)
        return result

    device = infer_device(runtime.device)
    general_teacher = AutoModelForCausalLM.from_pretrained(general_teacher_path, trust_remote_code=True).to(device)
    domain_teacher = AutoModelForCausalLM.from_pretrained(domain_teacher_path, trust_remote_code=True).to(device)
    general_teacher.eval()
    domain_teacher.eval()

    n = min(max(1, int(max_samples)), len(dataset))
    general_entropy: list[float] = []
    domain_entropy: list[float] = []
    general_confidence: list[float] = []
    domain_confidence: list[float] = []
    kl_values: list[float] = []

    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]
            batch = _sample_to_batch(sample=sample, device=device)
            mask = _sample_mask(sample=sample, seq_len=batch["input_ids"].shape[1], device=device)

            general_logits = general_teacher(**batch).logits.float()
            domain_logits = domain_teacher(**batch).logits.float()
            if general_logits.shape[-1] != domain_logits.shape[-1]:
                raise ValueError(
                    "Teacher vocab mismatch in distribution analysis: "
                    f"{general_logits.shape[-1]} vs {domain_logits.shape[-1]}"
                )

            g_entropy = _masked_scalar_distribution(
                values=torch.as_tensor(compute_entropy_distribution(general_logits), device=device)
                .reshape_as(mask),
                mask=mask,
            )
            d_entropy = _masked_scalar_distribution(
                values=torch.as_tensor(compute_entropy_distribution(domain_logits), device=device)
                .reshape_as(mask),
                mask=mask,
            )
            g_conf = _masked_scalar_distribution(
                values=torch.as_tensor(compute_confidence_distribution(general_logits), device=device)
                .reshape_as(mask),
                mask=mask,
            )
            d_conf = _masked_scalar_distribution(
                values=torch.as_tensor(compute_confidence_distribution(domain_logits), device=device)
                .reshape_as(mask),
                mask=mask,
            )

            general_entropy.extend(g_entropy)
            domain_entropy.extend(d_entropy)
            general_confidence.extend(g_conf)
            domain_confidence.extend(d_conf)

            kl_token = _token_kl_domain_to_general(general_logits=general_logits, domain_logits=domain_logits)
            kl_values.extend(_masked_scalar_distribution(values=kl_token, mask=mask))

    entropy_plot = Path(out_dir) / "teacher_entropy_histogram.png"
    confidence_plot = Path(out_dir) / "teacher_confidence_histogram.png"
    entropy_plot_saved = _save_overlay_histogram(
        general_values=general_entropy,
        domain_values=domain_entropy,
        path=entropy_plot,
        title="Teacher Entropy Distribution",
        xlabel="Entropy",
    )
    confidence_plot_saved = _save_overlay_histogram(
        general_values=general_confidence,
        domain_values=domain_confidence,
        path=confidence_plot,
        title="Teacher Confidence Distribution",
        xlabel="Confidence",
    )

    result = {
        "samples_used": n,
        "general_teacher_path": general_teacher_path,
        "domain_teacher_path": domain_teacher_path,
        "general_entropy": _summarize(general_entropy),
        "domain_entropy": _summarize(domain_entropy),
        "general_confidence": _summarize(general_confidence),
        "domain_confidence": _summarize(domain_confidence),
        "kl_domain_to_general": _summarize(kl_values),
        "plots": {
            "entropy_histogram": str(entropy_plot) if entropy_plot_saved else None,
            "confidence_histogram": str(confidence_plot) if confidence_plot_saved else None,
        },
    }
    dump_json(result, out_path)
    logger.info("teacher distribution analysis saved: %s", out_path)
    return result


def _sample_to_batch(sample: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(sample["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = sample.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    else:
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device).unsqueeze(0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _sample_mask(sample: dict[str, Any], seq_len: int, device: torch.device) -> torch.Tensor:
    labels = sample.get("labels")
    if labels is not None:
        label_tensor = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0)
        return (label_tensor != -100).float()

    attention_mask = sample.get("attention_mask")
    if attention_mask is not None:
        return torch.tensor(attention_mask, dtype=torch.float32, device=device).unsqueeze(0)

    return torch.ones((1, seq_len), dtype=torch.float32, device=device)


def _token_kl_domain_to_general(
    general_logits: torch.Tensor,
    domain_logits: torch.Tensor,
) -> torch.Tensor:
    general_log_probs = F.log_softmax(general_logits, dim=-1)
    domain_probs = F.softmax(domain_logits, dim=-1)
    return F.kl_div(general_log_probs, domain_probs, reduction="none").sum(dim=-1)


def _masked_scalar_distribution(values: torch.Tensor, mask: torch.Tensor) -> list[float]:
    active = mask > 0
    if active.sum().item() == 0:
        return []
    return values[active].detach().cpu().tolist()


def _as_logits_tensor(logits: torch.Tensor | Any) -> torch.Tensor:
    tensor = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)
    if tensor.ndim != 3:
        raise ValueError("Expected logits shape [batch, seq_len, vocab_size]")
    return tensor.float()


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "count": 0.0}
    x = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "p50": float(torch.quantile(x, 0.50).item()),
        "p90": float(torch.quantile(x, 0.90).item()),
        "count": float(x.numel()),
    }


def _save_overlay_histogram(
    general_values: list[float],
    domain_values: list[float],
    path: Path,
    title: str,
    xlabel: str,
) -> bool:
    if not general_values and not domain_values:
        return False

    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-error
    except Exception:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.5))
    if general_values:
        plt.hist(general_values, bins=50, alpha=0.55, label="General teacher")
    if domain_values:
        plt.hist(domain_values, bins=50, alpha=0.55, label="Domain teacher")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Token count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return True
