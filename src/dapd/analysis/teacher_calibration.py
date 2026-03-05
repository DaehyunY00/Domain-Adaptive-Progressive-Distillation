from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from ..utils import dump_json, ensure_dir, get_logger, infer_device


def compute_ece(
    pred_probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Compute expected calibration error (ECE) for multiclass predictions."""
    probs = _to_probs(pred_probs)
    y = _to_labels(labels)
    if probs.shape[0] != y.shape[0]:
        raise ValueError(f"pred_probs rows ({probs.shape[0]}) and labels ({y.shape[0]}) must match")

    confidence, pred = probs.max(dim=-1)
    correctness = (pred == y).float()
    bins = torch.linspace(0.0, 1.0, steps=max(2, int(n_bins) + 1), device=probs.device)

    total = float(confidence.numel())
    if total == 0:
        return 0.0

    ece = 0.0
    for idx in range(len(bins) - 1):
        lo = bins[idx]
        hi = bins[idx + 1]
        in_bin = (confidence >= lo) & (confidence <= hi if idx == len(bins) - 2 else confidence < hi)
        count = int(in_bin.sum().item())
        if count == 0:
            continue
        acc_bin = float(correctness[in_bin].mean().item())
        conf_bin = float(confidence[in_bin].mean().item())
        ece += abs(acc_bin - conf_bin) * (count / total)
    return float(ece)


def compute_brier_score(pred_probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute multiclass Brier score."""
    probs = _to_probs(pred_probs)
    y = _to_labels(labels)
    if probs.shape[0] != y.shape[0]:
        raise ValueError(f"pred_probs rows ({probs.shape[0]}) and labels ({y.shape[0]}) must match")

    num_classes = probs.shape[-1]
    one_hot = torch.nn.functional.one_hot(y.clamp(min=0, max=num_classes - 1), num_classes=num_classes).float()
    brier = ((probs - one_hot) ** 2).sum(dim=-1).mean()
    return float(brier.item())


def plot_reliability_diagram(
    pred_probs: torch.Tensor,
    labels: torch.Tensor,
    output_path: str | Path,
    n_bins: int = 15,
) -> bool:
    """Save reliability diagram; returns False when matplotlib is unavailable."""
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-error
    except Exception:
        return False

    probs = _to_probs(pred_probs).detach().cpu()
    y = _to_labels(labels).detach().cpu()
    if probs.numel() == 0:
        return False

    confidence, pred = probs.max(dim=-1)
    correctness = (pred == y).float()
    bins = torch.linspace(0.0, 1.0, steps=max(2, int(n_bins) + 1))

    bin_conf: list[float] = []
    bin_acc: list[float] = []
    bin_count: list[int] = []
    for idx in range(len(bins) - 1):
        lo = bins[idx]
        hi = bins[idx + 1]
        in_bin = (confidence >= lo) & (confidence <= hi if idx == len(bins) - 2 else confidence < hi)
        count = int(in_bin.sum().item())
        if count == 0:
            bin_conf.append(float((lo + hi).item() / 2.0))
            bin_acc.append(0.0)
            bin_count.append(0)
            continue
        bin_conf.append(float(confidence[in_bin].mean().item()))
        bin_acc.append(float(correctness[in_bin].mean().item()))
        bin_count.append(count)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    xs = torch.tensor(bin_conf).numpy()
    ys = torch.tensor(bin_acc).numpy()
    widths = 1.0 / max(1, int(n_bins))

    plt.figure(figsize=(6.5, 5.0))
    plt.bar(xs, ys, width=widths * 0.9, alpha=0.75, label="Accuracy")
    plt.plot([0, 1], [0, 1], "--", color="black", linewidth=1.0, label="Perfect calibration")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return True


def analyze_teacher_calibration(
    general_teacher_path: str,
    domain_teacher_path: str,
    dataset: Any,
    runtime: Any,
    output_dir: str = "runs/dapd/analysis",
    max_samples: int = 128,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Compare calibration of general/domain teacher on token-level targets."""
    logger = get_logger("dapd.analysis.teacher_calibration", getattr(runtime, "log_level", "INFO"))
    out_dir = ensure_dir(output_dir)
    out_path = Path(out_dir) / "teacher_calibration.json"

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
    probs_general: list[torch.Tensor] = []
    probs_domain: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for idx in range(n):
            sample = dataset[idx]
            batch = _sample_to_batch(sample=sample, device=device)
            labels = _shifted_labels(sample=sample, seq_len=batch["input_ids"].shape[1], device=device)

            out_g = general_teacher(**batch).logits[:, :-1, :].float()
            out_d = domain_teacher(**batch).logits[:, :-1, :].float()
            if out_g.shape[-1] != out_d.shape[-1]:
                raise ValueError(
                    "Teacher vocab mismatch in calibration analysis: "
                    f"{out_g.shape[-1]} vs {out_d.shape[-1]}"
                )

            valid = labels != -100
            if valid.sum().item() == 0:
                continue

            probs_g = torch.softmax(out_g, dim=-1)[valid]
            probs_d = torch.softmax(out_d, dim=-1)[valid]
            y = labels[valid]

            probs_general.append(probs_g.detach().cpu())
            probs_domain.append(probs_d.detach().cpu())
            all_labels.append(y.detach().cpu())

    if not all_labels:
        result = {"error": "no_valid_tokens"}
        dump_json(result, out_path)
        return result

    y_all = torch.cat(all_labels, dim=0)
    pg_all = torch.cat(probs_general, dim=0)
    pd_all = torch.cat(probs_domain, dim=0)

    rel_general_path = Path(out_dir) / "reliability_general_teacher.png"
    rel_domain_path = Path(out_dir) / "reliability_domain_teacher.png"
    rel_general_saved = plot_reliability_diagram(pg_all, y_all, rel_general_path, n_bins=n_bins)
    rel_domain_saved = plot_reliability_diagram(pd_all, y_all, rel_domain_path, n_bins=n_bins)

    result = {
        "samples_used": n,
        "tokens_evaluated": int(y_all.numel()),
        "general_teacher_path": general_teacher_path,
        "domain_teacher_path": domain_teacher_path,
        "general_teacher": {
            "expected_calibration_error": compute_ece(pg_all, y_all, n_bins=n_bins),
            "brier_score": compute_brier_score(pg_all, y_all),
            "confidence": _confidence_summary(pg_all),
        },
        "domain_teacher": {
            "expected_calibration_error": compute_ece(pd_all, y_all, n_bins=n_bins),
            "brier_score": compute_brier_score(pd_all, y_all),
            "confidence": _confidence_summary(pd_all),
        },
        "plots": {
            "reliability_general_teacher": str(rel_general_path) if rel_general_saved else None,
            "reliability_domain_teacher": str(rel_domain_path) if rel_domain_saved else None,
        },
    }
    result["delta"] = {
        "ece_improvement": float(
            result["general_teacher"]["expected_calibration_error"]
            - result["domain_teacher"]["expected_calibration_error"]
        ),
        "brier_improvement": float(
            result["general_teacher"]["brier_score"] - result["domain_teacher"]["brier_score"]
        ),
    }
    dump_json(result, out_path)
    logger.info("teacher calibration analysis saved: %s", out_path)
    return result


def _to_probs(pred_probs: torch.Tensor) -> torch.Tensor:
    probs = pred_probs.float()
    if probs.ndim != 2:
        raise ValueError(f"pred_probs must be rank-2 [N, C], got shape={tuple(probs.shape)}")
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs


def _to_labels(labels: torch.Tensor) -> torch.Tensor:
    y = labels.long().reshape(-1)
    return y


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


def _confidence_summary(pred_probs: torch.Tensor) -> dict[str, float]:
    conf = pred_probs.max(dim=-1).values
    return {
        "mean": float(conf.mean().item()),
        "std": float(conf.std(unbiased=False).item()),
        "p50": float(torch.quantile(conf, 0.50).item()),
        "p90": float(torch.quantile(conf, 0.90).item()),
    }
