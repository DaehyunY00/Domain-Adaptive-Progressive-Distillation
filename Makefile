.PHONY: all check_env smoke_test \
        run_m4 run_m4_ablation run_m4_baselines run_m4_baselines_multiseed \
        run_m4_analysis run_m4_suite \
        run_full run_full_multiseed run_full_ablation run_full_baselines \
        run_full_analysis run_full_suite \
        run_mps run_mps_suite \
        help

# ════════════════════════════════════════════════════════════════════════════
# Help
# ════════════════════════════════════════════════════════════════════════════

help:
	@echo ""
	@echo "DAPD Experiment Targets"
	@echo "============================================================="
	@echo ""
	@echo "  Environment & Quick Test"
	@echo "  -------------------------------------------------------------"
	@echo "  make check_env               -- 환경 확인 + 코드 fix 검증"
	@echo "  make smoke_test              -- quick end-to-end test (~30분)"
	@echo ""
	@echo "  M4 16GB  (configs/dapd_m4_16gb.yaml v2)"
	@echo "  -------------------------------------------------------------"
	@echo "  make run_m4                  -- 메인 파이프라인"
	@echo "  make run_m4_ablation         -- Ablation (5 variants)"
	@echo "  make run_m4_baselines        -- Baseline 5개 (single seed)"
	@echo "  make run_m4_baselines_multiseed -- Baseline × 3 seeds"
	@echo "  make run_m4_analysis         -- Post-training 분석"
	@echo "  make run_m4_suite            -- 위 4개 순차 실행"
	@echo ""
	@echo "  GPU Full Scale  (configs/dapd_full.yaml — A100/RTX4090)"
	@echo "  -------------------------------------------------------------"
	@echo "  make run_full                -- 메인 파이프라인 전체 데이터"
	@echo "  make run_full_ablation       -- Ablation (full scale)"
	@echo "  make run_full_baselines      -- Baseline + multi-seed"
	@echo "  make run_full_analysis       -- 분석"
	@echo "  make run_full_suite          -- 위 모두 순차 실행"
	@echo ""
	@echo "  MPS Safe (configs/dapd_mps_safe.yaml)"
	@echo "  -------------------------------------------------------------"
	@echo "  make run_mps                 -- MPS safe 파이프라인"
	@echo "  make run_mps_suite           -- MPS suite"
	@echo "============================================================="
	@echo ""

# ════════════════════════════════════════════════════════════════════════════
# 환경 확인
# ════════════════════════════════════════════════════════════════════════════

check_env:
	@python -c "import torch; print('PyTorch:', torch.__version__)"
	@python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
	@python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
	@PYTHONPATH=src python -c "from dapd.pipeline import DAPDPipeline; print('pipeline OK')"
	@PYTHONPATH=src python -c "from dapd.analysis import analyze_teacher_distributions; print('analysis OK')"
	@PYTHONPATH=src python -c "from dapd.metrics.core import _answer_matches; print('FIX-01 answer_matches OK')"
	@PYTHONPATH=src python -c "from dapd.pruning import _prune_attention_heads; print('FIX-02 GQA pruning OK')"
	@PYTHONPATH=src python -c "from dapd.data import _map_bioasq; print('FIX-03 bioasq mapper OK')"
	@PYTHONPATH=src python -c "from dapd.analysis import create_dynamics_callback; print('FIX-04 dynamics callback OK')"

smoke_test:
	PYTHONPATH=src python scripts/smoke_test.py

# ════════════════════════════════════════════════════════════════════════════
# M4 16GB — Apple Silicon (dapd_m4_16gb.yaml v2)
# ════════════════════════════════════════════════════════════════════════════
# v2 변경사항: max_length=128, max_train_samples=5000, epochs=2/3,
#              alpha=0.5, prune_ratio=0.2, analysis enabled

# [1단계] 메인 파이프라인: domain adaptation → KD → pruning → evaluation → analysis
run_m4:
	PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTHONPATH=src \
		python scripts/run_pipeline.py --config configs/dapd_m4_16gb.yaml

# [2단계] Ablation study: full | no_adapt | no_kd | no_prune | constant_temp
run_m4_ablation:
	PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTHONPATH=src \
		python scripts/run_ablation.py --config configs/dapd_m4_16gb.yaml

# [3단계] Baseline 비교 (single seed):
#   zero_shot | student_sft | direct_kd | lora_only | no_distill_prune
run_m4_baselines:
	PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTHONPATH=src \
		python scripts/run_baselines.py --config configs/dapd_m4_16gb.yaml \
		--baselines zero_shot,student_sft,direct_kd,lora_only,no_distill_prune

# [3단계-확장] Baseline × multi-seed (42, 123, 2024)
run_m4_baselines_multiseed:
	PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTHONPATH=src \
		python scripts/run_baselines.py --config configs/dapd_m4_16gb.yaml \
		--baselines zero_shot,student_sft,direct_kd,lora_only,no_distill_prune \
		--multi_seed

# [4단계] Post-training analysis (run_m4 완료 후 실행):
run_m4_analysis:
	PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTHONPATH=src \
		python scripts/run_analysis.py --config configs/dapd_m4_16gb.yaml \
		--domain_teacher    runs/dapd_m4/domain_teacher/merged \
		--distilled_student runs/dapd_m4/distilled_student/final \
		--pruned_student    runs/dapd_m4/pruned_student/final

# [전체] M4 suite: pipeline → ablation → baselines → analysis
run_m4_suite: run_m4 run_m4_ablation run_m4_baselines run_m4_analysis
	@echo ""
	@echo "=== M4 suite 완료 ==="
	@echo "파이프라인: runs/dapd_m4/pipeline_summary.json"
	@echo "Ablation:   runs/dapd_m4/ablation/ablation_summary.json"
	@echo "Baselines:  runs/dapd_m4/baselines/baseline_summary.json"
	@echo "Analysis:   runs/dapd_m4/analysis/"

# ════════════════════════════════════════════════════════════════════════════
# Full Scale — CUDA GPU (dapd_full.yaml — A100 / RTX4090 권장)
# ════════════════════════════════════════════════════════════════════════════
# 전체 데이터, max_length=256, epochs=3/5, bf16=true, AdamW optimizer

run_full:
	PYTHONPATH=src python scripts/run_pipeline.py --config configs/dapd_full.yaml

run_full_ablation:
	PYTHONPATH=src python scripts/run_ablation.py --config configs/dapd_full.yaml

# Baseline + multi-seed (3 seeds)
run_full_baselines:
	PYTHONPATH=src python scripts/run_baselines.py --config configs/dapd_full.yaml \
		--baselines zero_shot,student_sft,direct_kd,lora_only,no_distill_prune \
		--multi_seed

run_full_analysis:
	PYTHONPATH=src python scripts/run_analysis.py --config configs/dapd_full.yaml \
		--domain_teacher    runs/dapd_full/domain_teacher/merged \
		--distilled_student runs/dapd_full/distilled_student/final \
		--pruned_student    runs/dapd_full/pruned_student/final

run_full_suite: run_full run_full_ablation run_full_baselines run_full_analysis
	@echo ""
	@echo "=== Full suite 완료 ==="
	@echo "파이프라인: runs/dapd_full/pipeline_summary.json"
	@echo "Ablation:   runs/dapd_full/ablation/ablation_summary.json"
	@echo "Baselines:  runs/dapd_full/baselines/baseline_summary.json"
	@echo "Analysis:   runs/dapd_full/analysis/"

# ════════════════════════════════════════════════════════════════════════════
# MPS Safe (기존 호환성 유지)
# ════════════════════════════════════════════════════════════════════════════

run_mps:
	PYTHONPATH=src python scripts/run_pipeline.py --config configs/dapd_mps_safe.yaml

run_mps_suite:
	PYTHONPATH=src python scripts/run_full_experiment.py --config configs/dapd_mps_safe.yaml
