from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from ..distillation import _compute_masked_kl_loss
from ..utils import dump_json, ensure_dir, get_logger, infer_device


def run_distillation_interventions(
    student_model_path: str,
    domain_teacher_path: str,
    dataset: Any,
    runtime: Any,
    output_dir: str = "runs/dapd/analysis",
    max_samples: int = 64,
    alpha: float = 0.7,
    temperature: float = 2.0,
    teacher_small_path: str | None = None,
    teacher_large_path: str | None = None,
) -> dict[str, Any]:
    """Run intervention analyses for KD supervision signal quality.

    Experiment A:
      replace teacher distribution with uniform distribution.
    Experiment B:
      replace soft teacher distribution with hard top-1 supervision.
    Experiment C:
      compare KD signal from small vs large teacher models.
    """
    logger = get_logger("dapd.analysis.distillation_intervention", getattr(runtime, "log_level", "INFO"))
    out_dir = ensure_dir(output_dir)
    out_path = Path(out_dir) / "intervention_results.json"

    if len(dataset) == 0:
        result = {"error": "empty_dataset"}
        dump_json(result, out_path)
        return result

    device = infer_device(runtime.device)
    student = AutoModelForCausalLM.from_pretrained(student_model_path, trust_remote_code=True).to(device).eval()
    domain_teacher = AutoModelForCausalLM.from_pretrained(domain_teacher_path, trust_remote_code=True).to(device).eval()

    teacher_small_path = teacher_small_path or student_model_path
    teacher_large_path = teacher_large_path or domain_teacher_path
    teacher_small = AutoModelForCausalLM.from_pretrained(teacher_small_path, trust_remote_code=True).to(device).eval()
    teacher_large = AutoModelForCausalLM.from_pretrained(teacher_large_path, trust_remote_code=True).to(device).eval()

    n = min(max(1, int(max_samples)), len(dataset))
    exp_a = _Accumulator()
    exp_b = _Accumulator()
    exp_c_small = _Accumulator()
    exp_c_large = _Accumulator()

    with torch.no_grad():
        for idx in range(n):
            sample = dataset[idx]
            batch = _sample_to_batch(sample=sample, device=device)
            labels = batch["labels"]
            attention_mask = batch.get("attention_mask")
            if batch["input_ids"].shape[1] <= 1:
                continue

            student_out = student(
                input_ids=batch["input_ids"],
                attention_mask=attention_mask,
                labels=labels,
            )
            student_logits = student_out.logits.float()
            ce_loss = float(student_out.loss.item()) if student_out.loss is not None else 0.0

            t_domain = domain_teacher(input_ids=batch["input_ids"], attention_mask=attention_mask).logits.float()
            t_small = teacher_small(input_ids=batch["input_ids"], attention_mask=attention_mask).logits.float()
            t_large = teacher_large(input_ids=batch["input_ids"], attention_mask=attention_mask).logits.float()

            vocab_size = student_logits.shape[-1]
            if t_domain.shape[-1] != vocab_size:
                raise ValueError(
                    "Intervention A/B requires matching vocab size between student and domain teacher; "
                    f"got student={vocab_size} teacher={t_domain.shape[-1]}"
                )

            # Experiment A: uniform teacher distribution.
            kd_uniform = _uniform_kd_loss(
                student_logits=student_logits,
                labels=labels,
                attention_mask=attention_mask,
                temperature=temperature,
            )
            exp_a.add(_combine_loss(ce=ce_loss, kd=kd_uniform, alpha=alpha), kd_uniform, ce_loss)

            # Experiment B: hard top-1 teacher supervision.
            kd_hard = _hard_teacher_loss(
                student_logits=student_logits,
                teacher_logits=t_domain,
                labels=labels,
                temperature=temperature,
            )
            exp_b.add(_combine_loss(ce=ce_loss, kd=kd_hard, alpha=alpha), kd_hard, ce_loss)

            # Experiment C: teacher capacity effect.
            if t_small.shape[-1] == vocab_size:
                kd_small = float(
                    _compute_masked_kl_loss(
                        student_logits=student_logits,
                        teacher_logits=t_small.detach(),
                        labels=labels,
                        attention_mask=attention_mask,
                        temperature=temperature,
                    ).item()
                )
                exp_c_small.add(_combine_loss(ce=ce_loss, kd=kd_small, alpha=alpha), kd_small, ce_loss)

            if t_large.shape[-1] == vocab_size:
                kd_large = float(
                    _compute_masked_kl_loss(
                        student_logits=student_logits,
                        teacher_logits=t_large.detach(),
                        labels=labels,
                        attention_mask=attention_mask,
                        temperature=temperature,
                    ).item()
                )
                exp_c_large.add(_combine_loss(ce=ce_loss, kd=kd_large, alpha=alpha), kd_large, ce_loss)

    result = {
        "student_model_path": student_model_path,
        "domain_teacher_path": domain_teacher_path,
        "teacher_small_path": teacher_small_path,
        "teacher_large_path": teacher_large_path,
        "samples_used": n,
        "alpha": float(alpha),
        "temperature": float(temperature),
        "experiment_a_uniform_teacher": exp_a.to_dict(),
        "experiment_b_hard_top1_teacher": exp_b.to_dict(),
        "experiment_c_teacher_capacity": {
            "small_teacher": exp_c_small.to_dict(),
            "large_teacher": exp_c_large.to_dict(),
            "delta_kd_loss_large_minus_small": float(exp_c_large.kd_mean - exp_c_small.kd_mean),
            "delta_total_loss_large_minus_small": float(exp_c_large.total_mean - exp_c_small.total_mean),
        },
    }
    dump_json(result, out_path)
    logger.info("distillation intervention results saved: %s", out_path)
    return result


class _Accumulator:
    def __init__(self) -> None:
        self._n = 0
        self._total = 0.0
        self._kd = 0.0
        self._ce = 0.0

    def add(self, total_loss: float, kd_loss: float, ce_loss: float) -> None:
        self._n += 1
        self._total += float(total_loss)
        self._kd += float(kd_loss)
        self._ce += float(ce_loss)

    @property
    def total_mean(self) -> float:
        return self._total / max(1, self._n)

    @property
    def kd_mean(self) -> float:
        return self._kd / max(1, self._n)

    @property
    def ce_mean(self) -> float:
        return self._ce / max(1, self._n)

    def to_dict(self) -> dict[str, float]:
        return {
            "num_batches": float(self._n),
            "total_loss_mean": float(self.total_mean),
            "kd_loss_mean": float(self.kd_mean),
            "ce_loss_mean": float(self.ce_mean),
        }


def _combine_loss(ce: float, kd: float, alpha: float) -> float:
    return float(alpha * ce + (1.0 - alpha) * kd)


def _sample_to_batch(sample: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(sample["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    attention_mask_raw = sample.get("attention_mask")
    labels_raw = sample.get("labels")

    if attention_mask_raw is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    else:
        attention_mask = torch.tensor(attention_mask_raw, dtype=torch.long, device=device).unsqueeze(0)

    if labels_raw is None:
        labels = input_ids.clone()
        labels[:, 0] = -100
    else:
        labels = torch.tensor(labels_raw, dtype=torch.long, device=device).unsqueeze(0)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _uniform_kd_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    temperature: float,
) -> float:
    if student_logits.shape[1] <= 1:
        return 0.0
    student_shifted = student_logits[:, :-1, :].float() / max(float(temperature), 1e-6)
    log_probs = F.log_softmax(student_shifted, dim=-1)
    uniform = torch.full_like(log_probs, 1.0 / student_shifted.shape[-1])
    token_kl = F.kl_div(log_probs, uniform, reduction="none").sum(dim=-1)
    mask = _build_mask(labels=labels, attention_mask=attention_mask, pred_len=token_kl.shape[1], device=token_kl.device)
    valid = mask.sum().item()
    if valid <= 0:
        return 0.0
    kd = ((token_kl * mask).sum() / mask.sum()) * (float(temperature) ** 2)
    return float(kd.item())


def _hard_teacher_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> float:
    if student_logits.shape[1] <= 1:
        return 0.0
    student_shifted = student_logits[:, :-1, :].reshape(-1, student_logits.size(-1)) / max(float(temperature), 1e-6)
    teacher_shifted = teacher_logits[:, :-1, :]
    hard_targets = teacher_shifted.argmax(dim=-1).reshape(-1)
    shifted_labels = labels[:, 1:].reshape(-1)
    valid = shifted_labels != -100
    if valid.sum().item() == 0:
        return 0.0
    loss = F.cross_entropy(student_shifted[valid], hard_targets[valid], reduction="mean")
    return float((loss * (float(temperature) ** 2)).item())


def _build_mask(
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    pred_len: int,
    device: torch.device,
) -> torch.Tensor:
    if labels is not None:
        return (labels[:, 1 : 1 + pred_len] != -100).float().to(device)
    if attention_mask is not None:
        return attention_mask[:, 1 : 1 + pred_len].float().to(device)
    return torch.ones((1, pred_len), dtype=torch.float32, device=device)
