from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from ..distillation import _compute_masked_kl_loss, _scheduled_temperature
from ..utils import dump_json, ensure_dir, get_logger, infer_device

TEMPERATURE_SCHEDULES = ("constant", "linear", "cosine")


def run_temperature_schedule_analysis(
    teacher_model_path: str,
    student_model_path: str,
    train_dataset: Any,
    validation_dataset: Any,
    runtime: Any,
    output_dir: str = "runs/dapd/analysis",
    alpha: float = 0.7,
    temperature: float = 2.0,
    min_temperature: float = 1.0,
    steps: int = 100,
) -> dict[str, Any]:
    """Analyze KD behavior for constant/linear/cosine temperature schedules.

    This is a forward-only analysis pass (no optimizer step) to keep runtime and
    memory predictable on Apple Silicon. It tracks:
      - KD loss trajectory
      - training-stability proxy (loss std + loss delta std)
      - temperature-scaled token accuracy on validation data
    """
    logger = get_logger("dapd.analysis.temperature", getattr(runtime, "log_level", "INFO"))
    out_dir = ensure_dir(output_dir)
    out_path = Path(out_dir) / "temperature_analysis.json"

    if len(train_dataset) == 0:
        result = {"error": "empty_train_dataset"}
        dump_json(result, out_path)
        return result

    device = infer_device(runtime.device)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path, trust_remote_code=True).to(device)
    teacher_model.eval()

    schedule_results: dict[str, Any] = {}
    analysis_steps = max(1, int(steps))
    for schedule in TEMPERATURE_SCHEDULES:
        student_model = AutoModelForCausalLM.from_pretrained(student_model_path, trust_remote_code=True).to(device)
        student_model.eval()
        if hasattr(student_model, "config") and hasattr(student_model.config, "use_cache"):
            student_model.config.use_cache = False

        loss_trace: list[float] = []
        kd_trace: list[float] = []
        ce_trace: list[float] = []
        temp_trace: list[float] = []
        step_trace: list[int] = []

        with torch.no_grad():
            for step in range(analysis_steps):
                sample = train_dataset[step % len(train_dataset)]
                batch = _sample_to_batch(sample=sample, device=device)
                if batch["input_ids"].shape[1] <= 1:
                    continue

                student_out = student_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                teacher_out = teacher_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                student_logits = student_out.logits.float()
                teacher_logits = teacher_out.logits.detach().float()
                if student_logits.shape[-1] != teacher_logits.shape[-1]:
                    raise ValueError(
                        "Tokenizer/vocab mismatch in temperature analysis: "
                        f"student={student_logits.shape[-1]} teacher={teacher_logits.shape[-1]}"
                    )

                temp = _scheduled_temperature(
                    step=step,
                    max_steps=analysis_steps,
                    base_temperature=temperature,
                    min_temperature=min_temperature,
                    schedule=schedule,
                )
                kd_loss = _compute_masked_kl_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                    temperature=temp,
                )
                ce_loss = _ce_from_shifted_logits(student_logits=student_logits, labels=batch["labels"])
                total_loss = alpha * ce_loss + (1.0 - alpha) * kd_loss

                step_trace.append(step)
                temp_trace.append(float(temp))
                kd_trace.append(float(kd_loss.item()))
                ce_trace.append(float(ce_loss.item()))
                loss_trace.append(float(total_loss.item()))

        val_acc = _temperature_scaled_token_accuracy(
            model=student_model,
            dataset=validation_dataset,
            device=device,
            temperature=temp_trace[-1] if temp_trace else temperature,
        )
        schedule_results[schedule] = {
            "steps": step_trace,
            "temperature": temp_trace,
            "kd_loss": kd_trace,
            "ce_loss": ce_trace,
            "total_loss": loss_trace,
            "training_stability": _stability_stats(loss_trace),
            "validation_accuracy": float(val_acc),
        }
        logger.info(
            "temperature analysis | schedule=%s | kd_mean=%.4f | val_acc=%.4f",
            schedule,
            float(torch.tensor(kd_trace).mean().item()) if kd_trace else 0.0,
            float(val_acc),
        )

    loss_plot_path = Path(out_dir) / "temperature_loss_vs_step.png"
    temp_plot_path = Path(out_dir) / "temperature_vs_step.png"
    loss_plot_saved = _save_schedule_plot(
        schedule_results=schedule_results,
        y_key="kd_loss",
        path=loss_plot_path,
        title="KD Loss vs Step",
        ylabel="KD Loss",
    )
    temp_plot_saved = _save_schedule_plot(
        schedule_results=schedule_results,
        y_key="temperature",
        path=temp_plot_path,
        title="Temperature vs Step",
        ylabel="Temperature",
    )

    result = {
        "teacher_model_path": teacher_model_path,
        "student_model_path": student_model_path,
        "alpha": float(alpha),
        "base_temperature": float(temperature),
        "min_temperature": float(min_temperature),
        "steps": analysis_steps,
        "schedules": schedule_results,
        "plots": {
            "loss_vs_step": str(loss_plot_path) if loss_plot_saved else None,
            "temperature_vs_step": str(temp_plot_path) if temp_plot_saved else None,
        },
    }
    dump_json(result, out_path)
    logger.info("temperature analysis saved: %s", out_path)
    return result


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

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _ce_from_shifted_logits(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if student_logits.shape[1] <= 1:
        return student_logits.new_zeros(())
    shifted_logits = student_logits[:, :-1, :].reshape(-1, student_logits.size(-1))
    shifted_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shifted_logits, shifted_labels, ignore_index=-100)


def _stability_stats(loss_trace: list[float]) -> dict[str, float]:
    if not loss_trace:
        return {"loss_std": 0.0, "loss_delta_std": 0.0}
    x = torch.tensor(loss_trace, dtype=torch.float32)
    if x.numel() <= 1:
        return {"loss_std": 0.0, "loss_delta_std": 0.0}
    return {
        "loss_std": float(x.std(unbiased=False).item()),
        "loss_delta_std": float(torch.diff(x).std(unbiased=False).item()),
    }


def _temperature_scaled_token_accuracy(
    model: Any,
    dataset: Any,
    device: torch.device,
    temperature: float,
    confidence_threshold: float = 0.5,
    max_samples: int = 128,
) -> float:
    if len(dataset) == 0:
        return 0.0

    n = min(max(1, int(max_samples)), len(dataset))
    correct = 0.0
    total = 0.0

    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]
            batch = _sample_to_batch(sample=sample, device=device)
            if batch["input_ids"].shape[1] <= 1:
                continue

            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits[:, :-1, :].float() / max(float(temperature), 1e-6)
            labels = batch["labels"][:, 1:]
            valid = labels != -100
            if valid.sum().item() == 0:
                continue

            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            calibrated = valid & (conf >= float(confidence_threshold))
            use_mask = calibrated if calibrated.any().item() else valid
            corr = (pred == labels) & use_mask
            correct += float(corr.sum().item())
            total += float(use_mask.sum().item())

    if total <= 0:
        return 0.0
    return float(correct / total)


def _save_schedule_plot(
    schedule_results: dict[str, Any],
    y_key: str,
    path: Path,
    title: str,
    ylabel: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-error
    except Exception:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.0, 4.8))
    for schedule in TEMPERATURE_SCHEDULES:
        row = schedule_results.get(schedule, {})
        xs = row.get("steps", [])
        ys = row.get(y_key, [])
        if xs and ys:
            plt.plot(xs, ys, label=schedule)

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return True
