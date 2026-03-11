# DAPD: Domain-Adaptive Progressive Distillation

End-to-end research pipeline for compressing a large domain-adapted LLM into a smaller student model on Apple Silicon (MPS). The pipeline covers five stages: domain adaptation, knowledge distillation, structured pruning, evaluation, and analysis.

**Models:** Qwen2.5-1.5B-Instruct (teacher) → Qwen2.5-0.5B-Instruct (student)
**Datasets:** PubMedQA + MedMCQA (training), BioASQ (OOD evaluation)
**Target hardware:** MacBook Pro M4 16GB Unified Memory

## Pipeline Overview

The five stages run in sequence via `scripts/run_pipeline.py`:

**Stage 1 — Domain Adaptation (LoRA)**
Fine-tunes the teacher model (Qwen2.5-1.5B-Instruct) on the target domain using LoRA (r=8, α=16) applied to `q_proj`, `k_proj`, `v_proj`, `o_proj`. The adapted weights are merged and saved as the domain teacher.

**Stage 2 — Teacher Logits Preparation**
The domain teacher is moved to CPU. During distillation, teacher forward passes run on CPU and logit tensors are transferred to MPS in small chunks (4 tokens × vocab) to bound peak MPS memory to ~2 MB per step.

**Stage 3 — Progressive Knowledge Distillation (CE + KL)**
Trains the student (Qwen2.5-0.5B-Instruct) with a combined loss:
- Cross-entropy loss weight α = 0.7
- KL-divergence loss weight (1-α) = 0.3
- Temperature schedule: linear decay from T=2.0 → T=1.0 over training

**Stage 4 — Structured Pruning**
Applies magnitude-based structured pruning (masking mode, ratio=0.1) to attention heads and MLP neurons, calibrated on 2 batches of domain data.

**Stage 5 — Evaluation**
Reports accuracy/F1/perplexity for the domain-teacher, distilled-student, and pruned-student checkpoints, plus latency and memory compression metrics. Optional OOD evaluation on BioASQ.

## Repository Structure

```text
src/dapd/
  config.py           # Dataclass configs for all pipeline stages
  pipeline.py         # DAPDPipeline: orchestrates all 5 stages
  adaptation.py       # Stage 1: LoRA domain adaptation
  distillation.py     # Stage 3: Progressive KD with teacher CPU offload
  pruning.py          # Stage 4: Structured pruning
  evaluation.py       # Stage 5: Accuracy, latency, compression metrics
  data.py             # Dataset loading and tokenization (pubmed_qa, medmcqa, bioasq)
  analysis.py         # Forward-pass analysis utilities
  analysis/           # Analysis submodules (teacher distribution, pruning patterns, etc.)
  metrics/            # Evaluation metric helpers

scripts/
  run_pipeline.py          # Run the full 5-stage pipeline
  run_full_experiment.py   # Pipeline + ablation + baselines + analysis
  run_ablation.py          # Ablation study (constant_temp variant)
  run_baselines.py         # Baseline model comparisons
  run_analysis.py          # Post-training analysis
  smoke_test.py            # Quick end-to-end smoke test (~30 min)

configs/
  dapd_m4_16gb.yaml    # Primary config: M4 16GB (teacher CPU offload + Adafactor)
  dapd_mps_safe.yaml   # Conservative MPS config
  dapd_mps_fast.yaml   # Faster MPS config (larger batches)
  dapd_example.yaml    # Annotated reference config
```

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional (CUDA-only QLoRA path):
```bash
python -m pip install -e ".[qlora]"
```

## Quick Start — Apple Silicon M4 16GB

### 1) Check environment
```bash
make check_env
```

### 2) Smoke test (~30 min)
```bash
make smoke_test
```

### 3) Full pipeline (~overnight)
```bash
make run_m4
```

This is equivalent to:
```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTHONPATH=src \
  python scripts/run_pipeline.py --config configs/dapd_m4_16gb.yaml
```

`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` disables PyTorch's internal MPS memory cap so macOS manages MPS allocation directly. This is required with the 16GB budget.

### 4) Full experiment suite (pipeline + ablation + baselines + analysis)
```bash
make run_full_suite
```

### 5) Ablation study
```bash
make run_ablation
```

### 6) Baseline comparisons
```bash
make run_baselines
```

### 7) Post-training analysis
```bash
make run_analysis
```

## Direct Script Commands

Full pipeline (M4 16GB):
```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTHONPATH=src \
  python scripts/run_pipeline.py --config configs/dapd_m4_16gb.yaml
```

Full pipeline (generic MPS safe):
```bash
PYTHONPATH=src python scripts/run_pipeline.py --config configs/dapd_mps_safe.yaml
```

Full experiment suite:
```bash
PYTHONPATH=src python scripts/run_full_experiment.py --config configs/dapd_mps_safe.yaml
```

With multi-seed runs:
```bash
PYTHONPATH=src python scripts/run_full_experiment.py \
  --config configs/dapd_mps_safe.yaml \
  --run_multi_seed
```

Ablation study:
```bash
PYTHONPATH=src python scripts/run_ablation.py --config configs/dapd_mps_safe.yaml
```

Baseline comparisons:
```bash
PYTHONPATH=src python scripts/run_baselines.py --config configs/dapd_mps_safe.yaml
```

Post-training analysis (requires completed pipeline run):
```bash
PYTHONPATH=src python scripts/run_analysis.py --config configs/dapd_mps_safe.yaml \
  --domain_teacher runs/dapd_mps/domain_teacher/merged \
  --distilled_student runs/dapd_mps/distilled_student/final \
  --pruned_student runs/dapd_mps/pruned_student/final
```

Smoke test options:
```bash
PYTHONPATH=src python scripts/smoke_test.py --device auto
PYTHONPATH=src python scripts/smoke_test.py --skip_training
```

## dapd_m4_16gb.yaml Key Settings

| Section | Parameter | Value | Reason |
|---------|-----------|-------|--------|
| data | max_length | 64 | Reduces activation memory by ~67% vs 192 |
| data | max_train_samples | 800 | Safety margin for MPS budget |
| data | ood_max_samples | 100 | Reduces OOD eval peak memory |
| adaptation | lora_target_modules | q/k/v/o only (4 modules) | Reduces LoRA trainable params by ~43% |
| adaptation | optim | adafactor | Optimizer state 43× smaller than AdamW |
| distillation | optim | adafactor | lm_head[152064,896] state: 545 MB → 0.6 MB |
| distillation | temperature | 2.0 → 1.0 linear | Progressive KD schedule |
| distillation | alpha | 0.7 | CE weight; 0.3 for KL |
| pruning | calibration_batches | 2 | Reduces calibration peak memory |
| evaluation | latency_benchmark_runs | 30 | Shorter evaluation runtime |
| runtime | device | auto | Selects MPS automatically on Apple Silicon |
| runtime | fp16/bf16 | false | MPS Trainer does not support fp16/bf16 flags |
| runtime | dataloader_num_workers | 0 | Prevents multiprocessing OOM on MPS |

## MPS Memory Budget (M4 16GB)

With teacher CPU offload and Adafactor optimizer (`dapd_m4_16gb.yaml`):

| Stage | MPS | CPU | Notes |
|-------|-----|-----|-------|
| Domain Adaptation (LoRA) | ~2.5 GB | ~4 GB | Teacher on MPS; Adafactor state on CPU |
| Distillation (KD) | ~4.2 GB | ~7 GB | Teacher on CPU; chunked KL transfer to MPS |
| Pruning | ~2.0 GB | ~3 GB | Student only on MPS |
| Evaluation | ~2.0 GB | ~3 GB | One model at a time |

Distillation MPS breakdown:
- Student Qwen2.5-0.5B in float32: ~2.0 GB
- Student gradients (float32): ~2.0 GB
- Activations (batch=1, seq=64): ~0.2 GB
- KL chunk [1, 4, 152064] per step: ~2 MB (transient)
- **Total MPS: ~4.2 GB** ✅ (previously 7.7 GB → OOM at 519 MB allocation)

## Memory Optimization Techniques

**1. Teacher CPU Offload**
The teacher (Qwen2.5-1.5B) loads in bfloat16 to CPU (~3 GB). Teacher forward passes run on CPU per distillation step, then logit tensors are chunked (4 tokens at a time) and transferred to MPS for KL computation. This keeps teacher activations off MPS entirely.

**2. Adafactor Optimizer** (Shazeer & Stern, 2018)
Adafactor stores a factored approximation of the second moment (row vector + column vector) instead of the full matrix. For the largest weight in the student, `lm_head` of shape [152064, 896]:
- AdamW 2nd moment: 152064 × 896 × 4 bytes = **545 MB**
- Adafactor factored state: (152064 + 896) × 4 bytes = **0.6 MB**
- Reduction: **~891×** for lm_head alone; total MPS 10.62 GB → 5.31 GB

**3. Chunked KL Divergence**
For vocabularies ≥ 100K on MPS, KL is computed in chunks of 4 tokens per step instead of the full sequence at once. This caps peak MPS memory per KL computation to ~2 MB regardless of sequence length.

**4. `low_cpu_mem_usage=True`**
Applied to every `from_pretrained()` call across all pipeline stages (adaptation, distillation, evaluation, pruning, analysis). This loads model weights directly to their target dtype and device without creating a full float32 copy on CPU first.

**5. `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`**
Removes PyTorch's internal MPS allocation limit. Without this, PyTorch raises OOM before macOS's actual physical limit is reached. Set via environment variable before running `scripts/run_pipeline.py`.

## Output Artifacts

For `configs/dapd_m4_16gb.yaml`, artifacts are saved under `runs/dapd_m4/`:

```text
runs/dapd_m4/
  config_used.yaml
  pipeline_summary.json
  eval_metrics.json
  eval_metrics_ood.json          (if OOD enabled)
  domain_teacher/
    merged/                      (LoRA-merged teacher checkpoint)
  distilled_student/
    final/                       (distilled student checkpoint)
    distillation_dynamics.json   (loss/temp per step)
  pruned_student/
    final/                       (pruned student checkpoint)
    pruning_analysis.json        (head/neuron sparsity report)
  analysis/                      (if analysis.enabled: true)
```

For `configs/dapd_mps_safe.yaml` runs, replace `dapd_m4` with `dapd_mps` above.

The full experiment suite additionally produces:
```text
runs/dapd_mps/
  full_experiment_summary.json
  ablation/
  baselines/
  analysis/
```

## BioASQ OOD Important Note

If `bioasq` cannot be loaded from Hugging Face (`kroshan/BioASQ`), the loader falls back to `pubmed_qa` as an OOD proxy and logs an error. To prevent invalid OOD claims, the following scripts fail explicitly when this fallback activates:

- `scripts/run_pipeline.py`
- `scripts/run_ablation.py`
- `scripts/run_baselines.py`
- `scripts/run_analysis.py`

If this error appears, provide the real BioASQ data and rerun.

## Reproducibility

- Set `runtime.seed` (default: 42) and `runtime.deterministic: true`
- Keep `data.enable_map_cache: true` to reuse tokenized splits across runs
- Archive `config_used.yaml` and `pipeline_summary.json` per run for exact reproduction
- The tokenized cache key includes tokenizer ID, prompt template hash, max_length, seed, and dataset fingerprint — changing any of these invalidates the cache

## Test & Validation

Run full test suite:
```bash
PYTHONPATH=src pytest -q
```

Quick import checks:
```bash
PYTHONPATH=src python -c "from dapd.pipeline import DAPDPipeline; print('Pipeline OK')"
PYTHONPATH=src python -c "from dapd.analysis import analyze_teacher_distributions; print('Analysis OK')"
```

## Troubleshooting

**`RuntimeError: MPS backend out of memory`**
Use `make run_m4` instead of `make run_mps`. If OOM persists, reduce `data.max_length` further (e.g., 32) or reduce `data.max_train_samples`. Ensure `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` is set.

**`BioASQ dataset unavailable`**
The fallback to pubmed_qa invalidates OOD claims. Download the real BioASQ dataset from [bioasq.org](http://bioasq.org) and provide it via `data.ood_datasets`.

**`evaluation_strategy` deprecation warning**
Harmless. HuggingFace Trainer renamed this to `eval_strategy` in recent versions; both are accepted.

**Slow tokenization on first run**
The tokenized cache is built on first run and reused on subsequent runs. With `data.enable_map_cache: true`, subsequent runs skip tokenization entirely.
