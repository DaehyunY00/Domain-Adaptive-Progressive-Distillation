# DAPD: Domain-Adaptive Progressive Distillation

End-to-end research pipeline for small LLM compression on Apple Silicon (MPS), including:
- domain adaptation (LoRA)
- progressive distillation (CE + KD with temperature schedule)
- structured pruning (masking/physical when supported)
- evaluation (accuracy/F1/perplexity + latency/memory/compression)
- analysis utilities (teacher distribution, pruning patterns, OOD comparison)

## Current Code Status

Implemented and wired:
- Data pipeline for `pubmed_qa`, `medmcqa`, `bioasq` (OOD path supported)
- MPS-safe runtime defaults (`configs/dapd_mps_safe.yaml`, `configs/dapd_mps_fast.yaml`)
- Distillation dynamics logging (`distillation_dynamics.json`)
- Pruning analysis export (`pruning_analysis.json`)
- Baseline runner (`scripts/run_baselines.py`)
- Ablation runner with `constant_temp` variant (`scripts/run_ablation.py`)
- Post-training analysis runner (`scripts/run_analysis.py`)
- Smoke test script (`scripts/smoke_test.py`)

Validation status:
- Test suite passes locally via `PYTHONPATH=src pytest -q`

## Repository Structure

```text
src/dapd/
  adaptation.py
  distillation.py
  pruning.py
  evaluation.py
  pipeline.py
  data.py
  config.py
  analysis.py
  analysis/
  metrics/

scripts/
  run_pipeline.py
  run_ablation.py
  run_baselines.py
  run_analysis.py
  smoke_test.py

configs/
  dapd_mps_safe.yaml
  dapd_mps_fast.yaml
  dapd_example.yaml
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

## Quick Start (Apple Silicon M4, 16GB)

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
make run_mps
```

### 4) Ablation
```bash
make run_ablation
```

### 5) Baselines
```bash
make run_baselines
```

### 6) Analysis
```bash
make run_analysis
```

## Direct Script Commands

Run full pipeline:
```bash
PYTHONPATH=src python scripts/run_pipeline.py --config configs/dapd_mps_safe.yaml
```

Run ablation:
```bash
PYTHONPATH=src python scripts/run_ablation.py --config configs/dapd_mps_safe.yaml
```

Run baselines:
```bash
PYTHONPATH=src python scripts/run_baselines.py --config configs/dapd_mps_safe.yaml
```

Run analysis:
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

## Output Artifacts

Main run artifacts are saved under the configured run root, for example:

```text
runs/dapd_mps/
  config_used.yaml
  pipeline_summary.json
  eval_metrics.json
  eval_metrics_ood.json (if OOD enabled)
  domain_teacher/
  distilled_student/
  pruned_student/
```

Smoke test artifacts:
```text
runs/smoke_test/
  domain_teacher/
  distilled_student/final/
  pruned_student/final/
  eval_metrics.json
  distillation_dynamics.json
  pruning_analysis.json
```

## BioASQ OOD Important Note

If `bioasq` cannot be loaded from Hugging Face (`kroshan/BioASQ`), the loader falls back to `pubmed_qa` proxy for continuity.

To prevent invalid OOD reporting, experiment scripts now fail explicitly when this fallback is used:
- `scripts/run_pipeline.py`
- `scripts/run_ablation.py`
- `scripts/run_baselines.py`
- `scripts/run_analysis.py`

If this error appears, provide real BioASQ data and rerun.

## MPS Memory Guidance

Approximate peak unified memory (M4 16GB):
- Domain Adaptation (LoRA): ~5-6 GB
- Distillation (teacher + student): ~8-9 GB
- Pruning + Evaluation: ~3-4 GB

If you hit OOM:
- use `configs/dapd_mps_fast.yaml`
- reduce `data.max_length`
- reduce `evaluation.num_latency_samples`
- keep `runtime.fp16=false`, `runtime.bf16=false` on Trainer path for MPS stability

## Reproducibility

- Set `runtime.seed` and `runtime.deterministic=true`
- Keep `data.enable_map_cache=true`
- Track `config_used.yaml` and `pipeline_summary.json` per run

## Test & Validation

Run full tests:
```bash
PYTHONPATH=src pytest -q
```

Quick import checks:
```bash
PYTHONPATH=src python -c "from dapd.pipeline import DAPDPipeline; print('Pipeline OK')"
PYTHONPATH=src python -c "from dapd.analysis import analyze_teacher_distributions; print('Analysis OK')"
```
