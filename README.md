# DAPD Full Pipeline

`Research_plan.md`의 설계를 코드로 옮긴 **Domain-Adaptive Progressive Distillation (DAPD)** 파이프라인입니다.

구성 단계:
1. Dataset preprocessing (PubMedQA / SciQ / MedMCQA)
2. Domain adaptation (LoRA/QLoRA)
3. Progressive distillation (`alpha * KL + (1-alpha) * task loss`)
4. Structured pruning (`beta * |w| + (1-beta) * activation importance`)
5. Evaluation (Accuracy, F1, Perplexity, model size, latency, memory)

## 프로젝트 구조

```text
.
├── Research_plan.md
├── pyproject.toml
├── configs/
│   └── dapd_example.yaml
├── scripts/
│   └── run_pipeline.py
└── src/
    └── dapd/
        ├── __init__.py
        ├── config.py
        ├── data.py
        ├── adaptation.py
        ├── distillation.py
        ├── pruning.py
        ├── evaluation.py
        ├── pipeline.py
        └── utils.py
```

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

QLoRA(4bit, CUDA) 사용 시:

```bash
python -m pip install -e .[qlora]
```

## 실행 (Apple Silicon MPS 기본)

`configs/dapd_example.yaml`은 MacBook MPS(16GB) 기준 안전값으로 설정되어 있습니다.

```bash
PYTHONPATH=src python scripts/run_pipeline.py --config configs/dapd_example.yaml
```

CUDA + QLoRA를 사용하려면(옵션):

```bash
# 1) CUDA 환경에서만
# 2) YAML에서 adaptation.use_qlora: true
PYTHONPATH=src python scripts/run_pipeline.py --config configs/dapd_example.yaml
```

실행 결과:
- 단계별 모델 아티팩트: `runs/dapd/...`
- 평가 지표 JSON: `runs/dapd/eval_metrics.json`
- 파이프라인 요약: `runs/dapd/pipeline_summary.json`

## 제약 조건 / 핵심 동작

- Teacher/Student 모델은 YAML에서 변경 가능
- Apple Silicon(MPS)에서는 `use_qlora: false` (LoRA 권장)
- QLoRA(bitsandbytes 4bit)는 CUDA 전용이며 MPS/CPU에서 에러를 발생시킵니다.
- KL distillation은 teacher/student tokenizer 호환성이 필요합니다.
  - 기본 동작: 비호환이면 fail-fast (`ValueError`)
  - 강제 CE fallback이 필요하면 `distillation.allow_kl_fallback_to_ce: true`

## Troubleshooting (자주 발생하는 오류)

1. `QLoRA(bitsandbytes 4-bit) is CUDA-only...`
- 원인: MPS/CPU에서 `use_qlora: true`
- 해결: `adaptation.use_qlora: false`로 변경 (LoRA 사용)

2. `Teacher and student tokenizers are not compatible for KL distillation...`
- 원인: teacher/student tokenizer 불일치
- 해결:
  - 같은 모델 family 사용 (예: Qwen2.5 계열)
  - 또는 `distillation.allow_kl_fallback_to_ce: true`로 CE-only 진행

3. `runtime.fp16=true ... Set runtime.fp16=false for MPS/CPU`
- 원인: MPS에서 fp16/bf16 Trainer 설정
- 해결: `runtime.fp16: false`, `runtime.bf16: false`

4. MPS 메모리 부족(OOM) 또는 학습 중단
- 해결 우선순위:
  - `data.max_length` 축소 (예: 256 -> 192)
  - `gradient_accumulation_steps` 증가
  - `max_train_samples`, `max_eval_samples` 축소
  - 더 작은 teacher/student 모델 선택

## Reproducibility (재현성)

1. 시드 고정
- `runtime.seed`를 고정 (예: 42)

2. 데이터 캐시 고정
- `data.cache_dir`를 고정 경로로 유지 (예: `./cache/hf`)
- 동일 캐시/동일 config로 재실행

3. 동일 설정 파일 보관
- 실행에 사용한 YAML 파일을 실험 결과와 함께 보관
- 결과 아티팩트(`runs/dapd/pipeline_summary.json`, `runs/dapd/eval_metrics.json`)와 함께 관리

4. 단일 머신/버전 고정
- 같은 Mac, 같은 Python/PyTorch/Transformers 버전에서 비교

## 참고

- 공개 데이터셋 로딩에 네트워크가 필요합니다.
- 기본 설정은 재현 가능성과 구조 검증 중심이며, 실제 논문급 성능을 위해 epoch/steps/hparam 튜닝이 필요합니다.
