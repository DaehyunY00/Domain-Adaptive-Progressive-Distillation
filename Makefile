.PHONY: smoke_test run_mps run_ablation run_baselines run_analysis check_env

# Quick smoke test (~30 min on M4)
smoke_test:
	PYTHONPATH=src python scripts/smoke_test.py

# Full pipeline on MPS safe config
run_mps:
	PYTHONPATH=src python scripts/run_pipeline.py --config configs/dapd_mps_safe.yaml

# Ablation study (run after run_mps)
run_ablation:
	PYTHONPATH=src python scripts/run_ablation.py --config configs/dapd_mps_safe.yaml

# Baseline comparison
run_baselines:
	PYTHONPATH=src python scripts/run_baselines.py --config configs/dapd_mps_safe.yaml

# Analysis experiments (requires completed pipeline run)
run_analysis:
	PYTHONPATH=src python scripts/run_analysis.py --config configs/dapd_mps_safe.yaml \
		--domain_teacher runs/dapd_mps/domain_teacher/merged \
		--distilled_student runs/dapd_mps/distilled_student/final \
		--pruned_student runs/dapd_mps/pruned_student/final

# Check environment
check_env:
	python -c "import torch; print('PyTorch:', torch.__version__); print('MPS:', torch.backends.mps.is_available())"
	PYTHONPATH=src python -c "from dapd.pipeline import DAPDPipeline; print('DAPD import OK')"
	PYTHONPATH=src python -c "from dapd.analysis import analyze_teacher_distributions; print('Analysis module OK')"
