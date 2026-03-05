# DAPD Full Pipeline

Domain-Adaptive Progressive Distillation (DAPD) for small LLMs.

## DAPD Architecture

```text
General Teacher
   |
   |  (1) Domain Adaptation (LoRA on MPS / QLoRA on CUDA)
   v
Domain Teacher  -------------------------------+
   |                                           |
   |  (2) Progressive Distillation             |
   |      L = alpha*CE + (1-alpha)*T^2*KL(T||S)
   v                                           |
Distilled Student                              |
   |                                           |
   |  (3) Structured Pruning                   |
   v                                           |
Pruned Student                                 |
   |                                           |
   +------------ (4) Evaluation ---------------+
             acc / F1 / ppl / latency / memory
             + compression_ratio / throughput / speedup_vs_teacher
```

## Key Design Choices

1. Apple Silicon(MPS): `use_qlora: false` + LoRA 권장
2. QLoRA(bitsandbytes 4-bit): CUDA-only
3. KL distillation: teacher/student tokenizer compatibility required
4. Reproducibility: deterministic seed, dataset/tokenization cache, config snapshot logging

## Project Structure

```text
.
├── configs/
│   ├── dapd_example.yaml
│   └── dapd_mps_safe.yaml
├── scripts/
│   ├── run_pipeline.py
│   └── run_ablation.py
└── src/dapd/
    ├── adaptation.py
    ├── config.py
    ├── data.py
    ├── distillation.py
    ├── evaluation.py
    ├── metrics/
    │   ├── __init__.py
    │   └── core.py
    ├── pipeline.py
    ├── pruning.py
    └── utils.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

CUDA + QLoRA optional deps:

```bash
python -m pip install -e .[qlora]
```

## Run (Mac MPS default)

`configs/dapd_example.yaml`은 MacBook MPS(16GB) 안전 기본값입니다.

```bash
PYTHONPATH=src python scripts/run_pipeline.py --config configs/dapd_example.yaml
```

## Run Ablation (Paper-Ready)

Run 4 ablations with programmatic YAML overrides:

- `full`: adaptation + KL distillation + pruning
- `no_adapt`: skip adaptation (use base teacher directly)
- `no_kd`: CE-only student training (`distillation.use_kl=false`)
- `no_prune`: skip pruning

```bash
PYTHONPATH=src python scripts/run_ablation.py --config configs/dapd_example.yaml
```

Artifacts are written to `runs/dapd/ablation/{variant}/...`.

Artifacts are saved under `runs/dapd/...`:

- `runs/dapd/config_used.yaml`
- `runs/dapd/pipeline_summary.json`
- `runs/dapd/eval_metrics.json`
- stage outputs: `runs/dapd/domain_teacher`, `runs/dapd/distilled_student`, `runs/dapd/pruned_student`

## Evaluation JSON Example

`runs/dapd/eval_metrics.json` includes efficiency metrics:

```json
{
  "compression_ratio": 2.4,
  "throughput_tokens_per_sec": 132.8,
  "speedup_vs_teacher": 1.7,
  "latency_ms": 84.2,
  "memory_usage_mb": 6134.5,
  "efficiency": {
    "compression_ratio": 2.4,
    "throughput_tokens_per_sec": 132.8,
    "speedup_vs_teacher": 1.7
  }
}
```

## Progressive Distillation Details

Distillation objective:

```text
L = alpha * CE(student, labels)
  + (1 - alpha) * T^2 * KL( softmax(teacher/T) || softmax(student/T) )
```

Implementation details:

- padding tokens are masked (`labels == -100`)
- teacher logits are detached (`torch.no_grad()`)
- tokenizer mismatch:
  - default: fail-fast (`allow_kl_fallback_to_ce: false`)
  - optional: CE-only fallback (`allow_kl_fallback_to_ce: true`)
- temperature scheduling supported: `constant`, `linear`, `cosine`

## Structured Pruning Details

Current pruning method: `structured`

Supports:

- attention head pruning
- MLP neuron pruning
- optional layer pruning

Importance score:

```text
importance = beta * weight_magnitude + (1 - beta) * activation_score
```

Pruning report is exported to:

- `runs/dapd/.../pruning_report.json`

## Troubleshooting

1. `QLoRA(bitsandbytes 4-bit) is CUDA-only...`
- 원인: MPS/CPU에서 `use_qlora: true`
- 해결: `adaptation.use_qlora: false`

2. `Teacher and student tokenizers are not compatible for KL distillation...`
- 해결:
  - teacher/student를 같은 tokenizer family로 사용
  - 또는 `distillation.allow_kl_fallback_to_ce: true`

3. `runtime.fp16=true ... Set runtime.fp16=false for MPS/CPU`
- 해결: MPS에서는 `runtime.fp16=false`, `runtime.bf16=false`

4. MPS OOM
- `data.max_length` 감소 (예: 256 -> 192)
- `gradient_accumulation_steps` 증가
- `max_train_samples`/`max_eval_samples` 감소
- teacher/student 더 작은 모델 사용

## Reproducibility Instructions

1. Fix random seed
- `runtime.seed: 42`
- `runtime.deterministic: true`

2. Fix dataset/tokenization cache
- `data.cache_dir`
- `data.tokenized_cache_dir`
- `data.enable_map_cache: true`

3. Keep config snapshot
- `runs/dapd/config_used.yaml`

4. Keep environment fixed
- same machine, same Python/PyTorch/Transformers versions

## Benchmark Table Template

| Method | Teacher | Student | Pruning | Acc | F1 | PPL | Latency (ms) | Throughput (tok/s) | Memory (MB) | Compression Ratio | Speedup vs Teacher |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Adapt only | - | Student | No |  |  |  |  |  |  |  |  |
| Distill only | General | Student | No |  |  |  |  |  |  |  |  |
| Distill + Prune | General | Student | Yes |  |  |  |  |  |  |  |  |
| Adapt -> Distill | Domain | Student | No |  |  |  |  |  |  |  |  |
| Full DAPD | Domain | Student | Yes |  |  |  |  |  |  |  |  |

## Notes

- Public datasets require internet access for first download.
- Default settings prioritize stability/reproducibility on MPS; tune hyperparameters for final paper-quality results.
