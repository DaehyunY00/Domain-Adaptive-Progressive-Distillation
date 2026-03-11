# DAPD Colab 실행 가이드

> **Domain-Adaptive Progressive Distillation** — Google Colab 실험 환경 설정 및 실행 절차

---

## 목차

1. [환경 선택 기준](#1-환경-선택-기준)
2. [Colab 초기 설정](#2-colab-초기-설정)
3. [노트북 실행 순서](#3-노트북-실행-순서)
4. [실험 규모별 설정](#4-실험-규모별-설정)
5. [전체 실험 로드맵](#5-전체-실험-로드맵)
6. [결과 파일 위치](#6-결과-파일-위치)
7. [세션 종료 후 재시작](#7-세션-종료-후-재시작)
8. [트러블슈팅](#8-트러블슈팅)

---

## 1. 환경 선택 기준

### 실험 규모별 권장 환경

| 실험 | 설정 | 권장 환경 | 예상 시간 |
|---|---|---|---|
| **개발/검증** | v2 (5k샘플, 128토큰) | M4 로컬 또는 T4 무료 | 2.4h / 15분 |
| **파일럿** | v2 baseline × 5 | Colab 무료 T4 | 1.3h (총) |
| **논문 메인** | full (전체, 256토큰) | Colab Pro A100 | 49분/회 |
| **논문 전체** | full × 9회 + multi-seed | Colab Pro A100 | ~10h |

### GPU별 특성

| GPU | VRAM | 속도(fp32) | Colab 플랜 | 세션 한도 |
|---|---|---|---|---|
| T4 | 16GB | ~8 TFLOPS | 무료 | 12h |
| V100 | 16GB | ~14 TFLOPS | Pro | 24h |
| **A100** | **40GB** | **~20 TFLOPS (bf16 312T)** | **Pro** | **24h** |

> **권장**: Colab Pro + A100으로 논문 실험 전체를 하루 안에 완료

---

## 2. Colab 초기 설정

### Step 1. 런타임 설정

1. Colab 상단 메뉴: **런타임 → 런타임 유형 변경**
2. 하드웨어 가속기: **GPU** 선택
3. GPU 유형: **A100** (Pro) 또는 **T4** (무료) 선택
4. **저장** 클릭 후 런타임 재시작

### Step 2. Google Drive 프로젝트 구조 확인

Drive에 프로젝트가 다음 구조로 존재해야 합니다:

```
내 드라이브/
└── Domain_Adaption/          ← DRIVE_PROJECT_PATH
    ├── src/
    │   └── dapd/
    ├── scripts/
    ├── configs/
    │   ├── dapd_m4_16gb.yaml
    │   └── dapd_full.yaml
    ├── pyproject.toml
    ├── Makefile
    └── DAPD_Colab.ipynb      ← 이 노트북
```

> Drive 경로가 다르면 노트북 **섹션 1**의 `DRIVE_PROJECT_PATH` 변수를 수정하세요.

### Step 3. 노트북 열기

**방법 A** — Drive에서 직접 열기:
```
내 드라이브 → Domain_Adaption → DAPD_Colab.ipynb 더블클릭
```

**방법 B** — Colab에서 업로드:
```
colab.research.google.com → 파일 업로드 → DAPD_Colab.ipynb 선택
```

---

## 3. 노트북 실행 순서

노트북의 각 섹션을 **순서대로** 실행하세요.

```
섹션 0  GPU 확인        → GPU 이름 및 VRAM 출력
섹션 1  Drive 마운트    → /content/drive 연결
섹션 2  디렉토리 설정   → 소스 복사 + runs/ → Drive 심볼릭 링크
섹션 3  패키지 설치     → dapd 패키지 및 의존성 설치 (~3분)
섹션 4  Config 생성     → Colab 전용 YAML 설정 생성
섹션 5  모델 다운로드   → (선택) HF 모델 사전 캐시
────────────────────────────────────────────────────────
섹션 6  메인 파이프라인  실행 ← 핵심 실험
섹션 7  Baseline 실험   실행
섹션 8  Ablation Study  실행
섹션 9  Multi-seed 실험 실행
────────────────────────────────────────────────────────
섹션 10 결과 정리       → Drive 저장 확인
```

> **섹션 0~5는 매 세션 시작 시 항상 실행해야 합니다.**
> 섹션 6~9는 이전에 완료된 것은 건너뛰어도 됩니다.

---

## 4. 실험 규모별 설정

노트북의 `EXPERIMENT_SCALE` 변수로 실험 규모를 선택합니다.

### v2 설정 (Colab 무료 T4에서 실행 가능)

```python
EXPERIMENT_SCALE = 'v2'
```

| 항목 | 값 |
|---|---|
| 학습 데이터 | 5,000 샘플 |
| max_length | 128 토큰 |
| 도메인 적응 | 2 epoch |
| 지식 증류 | 3 epoch |
| Precision | bf16 (CUDA) |
| 1회 소요시간 | ~15분 (T4) |

### full 설정 (Colab Pro A100 권장)

```python
EXPERIMENT_SCALE = 'full'
```

| 항목 | 값 |
|---|---|
| 학습 데이터 | 전체 (~390k 샘플) |
| max_length | 256 토큰 |
| 도메인 적응 | 3 epoch |
| 지식 증류 | 5 epoch |
| Precision | bf16 (A100) |
| 1회 소요시간 | ~49분 (A100) |

---

## 5. 전체 실험 로드맵

논문 제출에 필요한 모든 실험의 실행 순서와 예상 시간입니다.

### A100 기준 전체 일정

```
Day 1 (A100, 약 10시간)
├── [49분]  메인 파이프라인 (full scale, seed=42)
├── [4.1h]  Baseline 5종 (zero_shot, student_sft, direct_kd, lora_only, no_distill_prune)
├── [3.3h]  Ablation 4종 (no_adapt, no_kd, no_prune, constant_temp)
└── [2.5h]  Multi-seed (seed 42, 123, 777)
```

### 실행 커맨드 요약

```bash
# 1. 메인 파이프라인
python scripts/run_pipeline.py --config configs/dapd_colab_full.yaml

# 2. Baseline 전체
python scripts/run_baselines.py \
  --config configs/dapd_colab_full.yaml \
  --baselines zero_shot,student_sft,direct_kd,lora_only,no_distill_prune

# 3. Ablation
python scripts/run_ablation.py --config configs/dapd_colab_full.yaml

# 4. Multi-seed
python scripts/run_multi_seed.py \
  --config configs/dapd_colab_full.yaml \
  --seeds 42,123,777
```

---

## 6. 결과 파일 위치

실험 완료 후 Drive에 저장되는 주요 파일들:

```
Domain_Adaption/runs/
├── pipeline_log.txt              ← 메인 파이프라인 로그
├── baselines_log.txt             ← Baseline 로그
├── ablation_log.txt              ← Ablation 로그
├── multiseed_log.txt             ← Multi-seed 로그
│
└── dapd_full/                    ← 실험 결과 (scale=full 기준)
    ├── eval_metrics.json         ★ 최종 평가 지표
    ├── eval_metrics_ood.json     ★ OOD (BioASQ) 평가
    ├── pipeline_summary.json     ★ 전체 파이프라인 요약
    │
    ├── domain_teacher/           ← 도메인 적응된 Teacher 모델
    │   └── merged/               ← LoRA merge된 최종 Teacher
    │
    ├── distilled_student/        ← 지식 증류된 Student 모델
    │   ├── final/                ← 최종 Student 모델
    │   └── distillation_dynamics.json  ← 훈련 dynamics 로그
    │
    ├── pruned_student/           ← 가지치기된 Student 모델
    │   ├── final/
    │   └── pruning_analysis.json ← Pruning 분석
    │
    ├── analysis/                 ← 심층 분석 결과
    │   ├── temperature_analysis.json
    │   └── teacher_distribution_analysis.json (GPU 전용)
    │
    ├── baselines/                ← Baseline 실험 결과
    │   └── baseline_summary.json
    │
    ├── ablation/                 ← Ablation 실험 결과
    │   └── ablation_summary.json
    │
    └── multi_seed/               ← Multi-seed 실험 결과
        ├── seed_42/
        ├── seed_123/
        └── seed_777/
```

---

## 7. 세션 종료 후 재시작

Colab 세션이 종료되더라도 Drive에 결과가 저장되어 있으므로 실험을 이어서 진행할 수 있습니다.

### 재시작 절차

```
1. 노트북 열기
2. 섹션 0~3 실행 (GPU 확인 + Drive 마운트 + 소스 복사 + 패키지 설치)
3. 섹션 4 실행 (Config 재생성)
4. 완료되지 않은 실험 섹션부터 재실행
```

> **체크포인트 자동 재개**: HuggingFace Trainer는 중단 지점의 체크포인트에서 자동으로 재개합니다.
> `runs/dapd_full/distilled_student/checkpoint-XXX/` 폴더가 있으면 자동으로 해당 지점부터 이어서 학습합니다.

### 주의사항

- **섹션 2의 디렉토리 설정은 반드시 재실행**: 심볼릭 링크가 세션마다 초기화됩니다
- **패키지 설치(섹션 3)도 매 세션 재실행 필요**: Colab 로컬 환경은 세션마다 초기화됩니다

---

## 8. 트러블슈팅

### CUDA Out of Memory

```python
# 셀 시작 부분에 추가
import torch, gc
gc.collect()
torch.cuda.empty_cache()
```

또는 config에서 배치 크기 줄이기:
```yaml
# configs/dapd_colab_full.yaml 수정
adaptation:
  per_device_train_batch_size: 2   # 4 → 2
  gradient_accumulation_steps: 16  # 8 → 16 (effective batch 유지)
```

### Drive 경로 오류

```python
# 실제 Drive 구조 확인
import os
for f in os.listdir('/content/drive/MyDrive'):
    print(f)
# 출력을 보고 DRIVE_PROJECT_PATH 수정
```

### 패키지 버전 충돌

```bash
# 강제 재설치
!pip install -q --force-reinstall transformers==4.42.0 peft==0.11.0
```

### HuggingFace 모델 다운로드 실패

```python
# 토큰 인증 (필요한 경우)
from huggingface_hub import login
login(token='hf_YOUR_ACCESS_TOKEN')

# 또는 환경변수로 설정
import os
os.environ['HF_TOKEN'] = 'hf_YOUR_ACCESS_TOKEN'
```

### T4에서 full-scale 실행 중 OOM

T4(16GB)에서 full-scale은 메모리 부족이 발생할 수 있습니다.
**v2 scale로 전환**하거나, 아래 설정을 추가하세요:

```python
# 섹션 4 실행 후 config를 수동으로 수정
import yaml

with open(colab_config_path) as f:
    cfg = yaml.safe_load(f)

# 메모리 절약 설정
cfg['adaptation']['per_device_train_batch_size'] = 1
cfg['adaptation']['gradient_accumulation_steps'] = 32
cfg['distillation']['per_device_train_batch_size'] = 1
cfg['data']['max_length'] = 128  # 256 → 128

with open(colab_config_path, 'w') as f:
    yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

print('메모리 절약 설정 적용 완료')
```

### 실험 도중 세션 끊김 방지

```javascript
// Colab에서 F12 → Console 탭에 아래 코드 붙여넣기
// (브라우저 연결 유지용)
function KeepAlive() {
  document.querySelector('#top-toolbar').click();
  setTimeout(KeepAlive, 60000);
}
KeepAlive();
```

---

## 참고

- **로컬 M4 실험**: `make run_m4` (2.4h, 개발/검증용)
- **Colab v2 실험**: `EXPERIMENT_SCALE = 'v2'` (15분/회, baseline 파일럿)
- **Colab full 실험**: `EXPERIMENT_SCALE = 'full'` (49분/회, 논문 본실험)
- **결과 분석**: 실험 완료 후 `DAPD_TopConference_Review.docx` 수치 업데이트
