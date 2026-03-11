# Full Pipeline Experiment Review

## Run Command

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTHONPATH=src \
  python scripts/run_pipeline.py --config configs/dapd_m4_16gb.yaml
```

## Artifact Location

- Main artifact root: `configs/runs/dapd_m4`
- Config snapshot: `configs/runs/dapd_m4/config_used.yaml`
- Pipeline summary: `configs/runs/dapd_m4/pipeline_summary.json`
- In-domain evaluation: `configs/runs/dapd_m4/eval_metrics.json`
- OOD evaluation: `configs/runs/dapd_m4/eval_metrics_ood.json`
- Distillation dynamics: `configs/runs/dapd_m4/distilled_student/distillation_dynamics.json`
- Pruning report: `configs/runs/dapd_m4/pruned_student/pruning_report.json`
- Pruning analysis: `configs/runs/dapd_m4/pruned_student/pruning_analysis.json`

## Executive Summary

이번 full pipeline은 **기계적으로는 완료**되었지만, 현재 산출물 기준으로는 **성능 실험 결과를 논문/보고용으로 해석하기 어렵습니다**.

핵심 이유는 아래 3가지입니다.

1. **최종 평가 결과가 사실상 비어 있음**
   - In-domain / OOD 모두 `accuracy=0.0`, `f1=0.0`, `perplexity=Infinity`
   - `per_sample_confidence=[]`, `per_sample_correct=[]`, `samples_measured=0`
   - 즉, 평가가 끝났더라도 **실제 유효 샘플이 0개였던 것으로 보입니다**

2. **Distillation 단계가 매우 불안정함**
   - distillation dynamics에 `NaN` CE loss가 다수 존재
   - step 10, step 20에서 학습 loss가 각각 `8.50e13`, `1.16e11` 수준으로 폭증
   - gradient norm도 `3467.66`, `918.54`로 매우 큼

3. **Pruning 효과가 구조적 sparsity 외에는 거의 없음**
   - MLP 뉴런만 pruning 되었고 attention head pruning은 적용되지 않음
   - dense checkpoint 크기 감소 없음
   - 예상 속도 향상은 `1.008x` 수준으로 사실상 미미

## Configuration Snapshot

- Teacher: `Qwen/Qwen2.5-1.5B-Instruct`
- Student: `Qwen/Qwen2.5-0.5B-Instruct`
- Train samples: `800`
- Eval samples: `64`
- OOD max samples: `100`
- Sequence length: `64`
- Adaptation accumulation: `32`
- Distillation accumulation: `32`
- Pruning: structured masking, ratio `0.1`
- Latency benchmark runs: `30`

## Stage-by-Stage Review

### 1. Domain Adaptation

결과는 상대적으로 가장 정상적입니다.

- 최종 teacher artifact: `configs/runs/dapd_m4/domain_teacher/merged`
- 저장 크기: 약 `2959.58 MB`
- optimizer steps: `25`
- logged train loss:
  - step 10: `1.1813`
  - step 20: `0.5075`

해석:

- LoRA adaptation 자체는 정상 수렴한 것으로 보입니다.
- 초기 domain adaptation 단계는 현재 full pipeline에서 **가장 신뢰 가능한 단계**입니다.

### 2. Progressive Distillation

결과는 불안정합니다.

- 최종 student artifact: `configs/runs/dapd_m4/distilled_student/final`
- temperature schedule: `2.0 -> 1.0`
- optimizer steps: `25`
- dynamics log entries: `99`
- `ce_loss = NaN` 발생 횟수: `39`
- `kd_loss = 0.0` 발생 횟수: `39`

주요 로그:

- logged loss @ step 10: `85046204353740.8`
- grad norm @ step 10: `3467.66`
- logged loss @ step 20: `116230612582.4`
- grad norm @ step 20: `918.54`

해석:

- adaptation과 달리 distillation은 **수치적으로 불안정**합니다.
- `NaN` CE loss가 약 `39.4%` (`39/99`) 로그에서 나타났고, 고정 구간에서 `kd_loss=0.0`와 같이 나타납니다.
- 이는 단순한 성능 저하가 아니라, **학습 자체가 흔들렸을 가능성**을 강하게 시사합니다.
- 따라서 이후 pruning 및 evaluation 결과도 이 student 품질에 영향을 받았다고 보는 것이 타당합니다.

### 3. Structured Pruning

결과는 “sparsity는 생겼지만 실효성은 낮은 pruning”에 가깝습니다.

- pruning mode: `masking`
- attention head pruning: `0 / 0`
- MLP neuron pruning: `11664 / 116736`
- parameter sparsity: `0.06346` (`6.35%`)
- estimated speedup potential: `1.00799x`
- model size before pruning: `1899.76 MB`
- model size after pruning: `1899.76 MB`
- sparse export size: `2403.99 MB`

해석:

- 실제 pruning은 **MLP 쪽에만 균일하게 약 10%** 들어갔습니다.
- attention head pruning은 전혀 동작하지 않았습니다.
- dense checkpoint 기준으로는 pruning 전후 파일 크기가 동일합니다.
- 오히려 sparse export는 `2403.99 MB`로 dense 모델보다 큽니다.
- 즉, 현재 pruning은
  - 파라미터 nonzero 감소는 만들었지만
  - 저장 크기 감소나 실질 추론 가속으로는 거의 이어지지 않았습니다.

추가 관찰:

- `pruning_analysis.json` 기준 24개 layer 모두 동일한 패턴입니다.
- layer별 MLP sparsity가 모두 `0.09991776198148727`로 사실상 동일합니다.
- 이는 데이터 기반 차등 pruning이라기보다 **전역 ratio 기반 균일 마스킹**에 가깝습니다.

### 4. Final Evaluation

현재 evaluation 산출물은 **유효 성능 측정값으로 보기 어렵습니다**.

#### In-domain (`eval_metrics.json`)

- accuracy: `0.0`
- f1: `0.0`
- perplexity: `Infinity`
- latency_ms: `0.0`
- throughput_tokens_per_sec: `0.0`
- teacher_latency_ms: `0.0`
- samples_measured: `0`
- per-sample arrays: empty

#### OOD (`eval_metrics_ood.json`)

- accuracy: `0.0`
- f1: `0.0`
- perplexity: `Infinity`
- latency_ms: `0.0`
- throughput_tokens_per_sec: `0.0`
- samples_measured: `0`
- per-sample arrays: empty

해석:

- 이 결과는 “성능이 나빴다”가 아니라, **평가 입력이 비어 있었을 가능성**이 큽니다.
- 근거:
  - per-sample 결과가 전부 비어 있음
  - latency benchmark 결과도 전부 0
  - perplexity가 `Infinity`
  - tokenized cache 아래에 `train_*.arrow`, `validation_*.arrow`는 있지만 `test_*.arrow`는 보이지 않음

따라서 현재의 최종 성능 지표는 **실험 결과가 아니라 비어 있는 평가 산출물**로 해석하는 것이 맞습니다.

## What Is Still Meaningful?

현재 run에서 의미 있게 읽을 수 있는 정보는 아래에 한정됩니다.

- Domain adaptation은 정상적으로 진행됨
- Distillation은 실행은 되었지만 매우 불안정함
- Pruning은 MLP masking 위주로 제한적으로 적용됨
- 최종 성능 평가는 유효하지 않음

즉, 이번 run은 **“pipeline dry-run + partial training diagnostics”**로는 가치가 있지만, **최종 성능 보고 실험**으로는 아직 부족합니다.

## Important Nuances

### 1. Reported compression ratio는 pruning 효과가 아님

`eval_metrics.json`의

- `compression_ratio = 3.1247`
- `disk_size_compression_ratio = 1.5579`

는 언뜻 좋아 보이지만, 이것은 대부분

- teacher(`1.5B`) vs student(`0.5B`)

의 **아키텍처 차이**에서 오는 이득입니다.

반면 pruning 자체는

- dense size before = `1899.76 MB`
- dense size after = `1899.76 MB`

이므로, **추가적인 checkpoint 축소 효과는 없었습니다**.

### 2. Sparse export는 현재 배포 친화적이지 않음

- dense final model: `1899.76 MB`
- sparse export: `2403.99 MB`

즉, 현재 serialization 방식에서는 sparse representation이 오히려 더 큽니다.

### 3. Artifact path가 `runs/`가 아니라 `configs/runs/` 밑에 생성됨

이번 실행 결과는 `runs/dapd_m4`가 아니라 `configs/runs/dapd_m4`에 저장되었습니다.

이는 config 경로 기준 상대 경로 해석 때문입니다. 실행 자체에는 문제가 없지만,

- artifact 관리
- README 예시 경로
- 후속 분석 스크립트

와 일관성이 깨질 수 있습니다.

## Overall Verdict

이번 full pipeline run에 대한 종합 판단은 다음과 같습니다.

- **Pipeline execution:** 성공
- **Adaptation quality:** 양호
- **Distillation quality:** 불안정
- **Pruning utility:** 제한적
- **Final evaluation validity:** 낮음 / 사실상 무효
- **Reporting readiness:** 미달

## Recommended Next Actions

우선순위 순으로 정리하면:

1. **평가 split이 비어 있지 않은지 먼저 수정**
   - empty test split 처리
   - 필요하면 test가 비어 있을 때 validation fallback 추가

2. **pruned model에 대해 evaluation만 다시 수행**
   - 현재는 학습을 다시 하기보다 evaluation 유효성부터 회복하는 것이 우선

3. **distillation instability 원인 점검**
   - CE loss NaN 배치 추적
   - learning rate 추가 하향
   - KL scaling / temperature schedule 점검
   - gradient clipping과 label validity 확인

4. **pruning 전략 재검토**
   - masking 대신 physical pruning 가능 범위 확대
   - attention head pruning이 0/0인 이유 점검
   - sparse serialization 이득이 없는 현재 저장 형식 개선

5. **artifact path 정리**
   - `configs/runs/...` 대신 `runs/...`로 고정되게 경로 정책 수정

## Final Assessment

현재 결과만 놓고 보면, 이번 실험은

- “DAPD full pipeline이 M4 16GB에서 전체 단계까지 도는가?”

라는 질문에는 **예**에 가깝지만,

- “DAPD가 실제로 유효한 성능/효율 결과를 냈는가?”

라는 질문에는 **아직 아니오**라고 판단하는 것이 가장 정확합니다.

