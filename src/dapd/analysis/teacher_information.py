from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from ..utils import dump_json, ensure_dir, get_logger, infer_device


def compute_mutual_information(teacher_probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Estimate I(P_teacher ; Y_task) with a plug-in approximation."""
    probs = _to_probs(teacher_probs)
    y = labels.long().reshape(-1)
    if probs.shape[0] != y.shape[0]:
        raise ValueError(f"teacher_probs rows ({probs.shape[0]}) and labels ({y.shape[0]}) must match")

    num_classes = probs.shape[-1]
    y = y.clamp(min=0, max=num_classes - 1)
    prior = torch.bincount(y, minlength=num_classes).float()
    prior = prior / prior.sum().clamp_min(1e-12)
    entropy_y = -(prior * prior.clamp_min(1e-12).log()).sum()

    p_true = probs.gather(-1, y.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
    conditional_entropy = -(p_true.log()).mean()
    mi = torch.clamp(entropy_y - conditional_entropy, min=0.0)
    return float(mi.item())


def compute_confidence_distribution(teacher_probs: torch.Tensor) -> list[float]:
    probs = _to_probs(teacher_probs)
    return probs.max(dim=-1).values.detach().cpu().tolist()


def compute_entropy_distribution(teacher_probs: torch.Tensor) -> list[float]:
    probs = _to_probs(teacher_probs)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    return entropy.detach().cpu().tolist()


def analyze_teacher_information(
    general_teacher_path: str,
    domain_teacher_path: str,
    dataset: Any,
    runtime: Any,
    output_dir: str = "runs/dapd/analysis",
    max_samples: int = 128,
) -> dict[str, Any]:
    """Analyze teacher information metrics with entropy/confidence distributions."""
    logger = get_logger("dapd.analysis.teacher_information", getattr(runtime, "log_level", "INFO"))
    out_dir = ensure_dir(output_dir)
    out_path = Path(out_dir) / "teacher_information.json"

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
    general_probs: list[torch.Tensor] = []
    domain_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for idx in range(n):
            sample = dataset[idx]
            batch = _sample_to_batch(sample=sample, device=device)
            labels = _shifted_labels(sample=sample, seq_len=batch["input_ids"].shape[1], device=device)

            logits_g = general_teacher(**batch).logits[:, :-1, :].float()
            logits_d = domain_teacher(**batch).logits[:, :-1, :].float()
            if logits_g.shape[-1] != logits_d.shape[-1]:
                raise ValueError(
                    "Teacher vocab mismatch in information analysis: "
                    f"{logits_g.shape[-1]} vs {logits_d.shape[-1]}"
                )

            valid = labels != -100
            if valid.sum().item() == 0:
                continue

            probs_g = torch.softmax(logits_g, dim=-1)[valid]
            probs_d = torch.softmax(logits_d, dim=-1)[valid]
            y = labels[valid]
            general_probs.append(probs_g.detach().cpu())
            domain_probs.append(probs_d.detach().cpu())
            all_labels.append(y.detach().cpu())

    if not all_labels:
        result = {"error": "no_valid_tokens"}
        dump_json(result, out_path)
        return result

    y_all = torch.cat(all_labels, dim=0)
    probs_general = torch.cat(general_probs, dim=0)
    probs_domain = torch.cat(domain_probs, dim=0)

    entropy_general = compute_entropy_distribution(probs_general)
    entropy_domain = compute_entropy_distribution(probs_domain)
    confidence_general = compute_confidence_distribution(probs_general)
    confidence_domain = compute_confidence_distribution(probs_domain)

    entropy_plot = Path(out_dir) / "teacher_information_entropy_histogram.png"
    confidence_plot = Path(out_dir) / "teacher_information_confidence_histogram.png"
    entropy_saved = _save_overlay_histogram(
        general_values=entropy_general,
        domain_values=entropy_domain,
        path=entropy_plot,
        title="Teacher Entropy Distribution",
        xlabel="Entropy",
    )
    confidence_saved = _save_overlay_histogram(
        general_values=confidence_general,
        domain_values=confidence_domain,
        path=confidence_plot,
        title="Teacher Confidence Distribution",
        xlabel="Confidence",
    )

    result = {
        "samples_used": n,
        "tokens_evaluated": int(y_all.numel()),
        "general_teacher_path": general_teacher_path,
        "domain_teacher_path": domain_teacher_path,
        "general_teacher": {
            "mutual_information": compute_mutual_information(probs_general, y_all),
            "entropy": _summary(entropy_general),
            "confidence": _summary(confidence_general),
        },
        "domain_teacher": {
            "mutual_information": compute_mutual_information(probs_domain, y_all),
            "entropy": _summary(entropy_domain),
            "confidence": _summary(confidence_domain),
        },
        "plots": {
            "entropy_histogram": str(entropy_plot) if entropy_saved else None,
            "confidence_histogram": str(confidence_plot) if confidence_saved else None,
        },
    }
    result["delta"] = {
        "mutual_information_gain": float(
            result["domain_teacher"]["mutual_information"] - result["general_teacher"]["mutual_information"]
        )
    }
    dump_json(result, out_path)
    logger.info("teacher information analysis saved: %s", out_path)
    return result


def _to_probs(teacher_probs: torch.Tensor) -> torch.Tensor:
    probs = teacher_probs.float()
    if probs.ndim != 2:
        raise ValueError(f"teacher_probs must be rank-2 [N, C], got shape={tuple(probs.shape)}")
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs


def _sample_to_batch(sample: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(sample["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = sample.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    else:
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device).unsqueeze(0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _shifted_labels(sample: dict[str, Any], seq_len: int, device: torch.device) -> torch.Tensor:
    labels = sample.get("labels")
    if labels is None:
        return torch.full((1, max(0, seq_len - 1)), -100, dtype=torch.long, device=device)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0)
    return label_tensor[:, 1:]


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0}
    x = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "p50": float(torch.quantile(x, 0.50).item()),
        "p90": float(torch.quantile(x, 0.90).item()),
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
