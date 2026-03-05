# DAPD Experiments (6) — 실행 명령어 가이드

아래 명령어는 저장소 루트(`/Users/daehyunyoo/Library/CloudStorage/GoogleDrive-dhyoo970111@gmail.com/내 드라이브/Domain_Adaption`)에서 실행합니다.

## 0) 환경 준비

```bash
python -m pip install -e ".[dev]"
python -m pip install matplotlib
export PYTHONPATH=src
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## 0-1) 실험용 config 자동 생성

```bash
python - <<'PY'
from pathlib import Path
import copy
import yaml

base = yaml.safe_load(Path("configs/dapd_example.yaml").read_text(encoding="utf-8"))
out_dir = Path("configs/experiments")
out_dir.mkdir(parents=True, exist_ok=True)

def set_common(cfg, exp_name):
    run_root = f"./runs/dapd/experiments/{exp_name}"
    cfg["adaptation"]["output_dir"] = f"{run_root}/domain_teacher"
    cfg["distillation"]["output_dir"] = f"{run_root}/distilled_student"
    cfg["pruning"]["output_dir"] = f"{run_root}/pruned_student"
    cfg["evaluation"]["output_file"] = f"{run_root}/eval_metrics.json"
    cfg["evaluation"]["ood_output_file"] = f"{run_root}/eval_metrics_ood.json"
    cfg["data"]["tokenized_cache_dir"] = f"{run_root}/cache/tokenized"
    cfg.setdefault("analysis", {})
    cfg["analysis"]["output_dir"] = f"{run_root}/analysis"
    return cfg

def dump(name, cfg):
    p = out_dir / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("wrote", p)

# Exp1: Teacher entropy
cfg = set_common(copy.deepcopy(base), "exp1_teacher_entropy")
cfg["pruning"]["enabled"] = False
cfg["evaluation"]["enabled"] = False
cfg["analysis"]["enabled"] = True
cfg["analysis"]["run_teacher_distribution"] = True
cfg["analysis"]["run_temperature_analysis"] = False
cfg["analysis"]["run_pruning_patterns"] = False
dump("exp1_teacher_entropy", cfg)

# Exp3: Distillation efficiency (domain teacher)
cfg = set_common(copy.deepcopy(base), "exp3_domain")
cfg["pruning"]["enabled"] = False
cfg["evaluation"]["enabled"] = True
cfg["analysis"]["enabled"] = False
dump("exp3_domain", cfg)

# Exp3: Distillation efficiency (general teacher)
cfg = set_common(copy.deepcopy(base), "exp3_general")
cfg["adaptation"]["enabled"] = False
cfg["pruning"]["enabled"] = False
cfg["evaluation"]["enabled"] = True
cfg["analysis"]["enabled"] = False
dump("exp3_general", cfg)

# Exp4: Temperature curriculum
cfg = set_common(copy.deepcopy(base), "exp4_temperature")
cfg["pruning"]["enabled"] = False
cfg["evaluation"]["enabled"] = False
cfg["analysis"]["enabled"] = True
cfg["analysis"]["run_teacher_distribution"] = False
cfg["analysis"]["run_temperature_analysis"] = True
cfg["analysis"]["run_pruning_patterns"] = False
cfg["analysis"]["temperature_steps"] = 120
dump("exp4_temperature", cfg)

# Exp5: Head analysis (domain teacher)
cfg = set_common(copy.deepcopy(base), "exp5_domain_heads")
cfg["evaluation"]["enabled"] = False
cfg["pruning"]["enabled"] = True
cfg["analysis"]["enabled"] = True
cfg["analysis"]["run_teacher_distribution"] = False
cfg["analysis"]["run_temperature_analysis"] = False
cfg["analysis"]["run_pruning_patterns"] = True
dump("exp5_domain_heads", cfg)

# Exp5: Head analysis (general teacher)
cfg = set_common(copy.deepcopy(base), "exp5_general_heads")
cfg["adaptation"]["enabled"] = False
cfg["evaluation"]["enabled"] = False
cfg["pruning"]["enabled"] = True
cfg["analysis"]["enabled"] = True
cfg["analysis"]["run_teacher_distribution"] = False
cfg["analysis"]["run_temperature_analysis"] = False
cfg["analysis"]["run_pruning_patterns"] = True
dump("exp5_general_heads", cfg)

# Exp6: OOD (domain teacher)
cfg = set_common(copy.deepcopy(base), "exp6_domain_ood")
cfg["data"]["datasets"] = ["pubmed_qa"]
cfg["pruning"]["enabled"] = False
cfg["evaluation"]["enabled"] = True
cfg["evaluation"]["run_ood_test"] = True
cfg["evaluation"]["ood_test_dataset"] = "bioasq"
cfg["analysis"]["enabled"] = False
dump("exp6_domain_ood", cfg)

# Exp6: OOD (general teacher)
cfg = set_common(copy.deepcopy(base), "exp6_general_ood")
cfg["data"]["datasets"] = ["pubmed_qa"]
cfg["adaptation"]["enabled"] = False
cfg["pruning"]["enabled"] = False
cfg["evaluation"]["enabled"] = True
cfg["evaluation"]["run_ood_test"] = True
cfg["evaluation"]["ood_test_dataset"] = "bioasq"
cfg["analysis"]["enabled"] = False
dump("exp6_general_ood", cfg)
PY
```

---

## 1) Experiment 1 — Teacher Entropy Analysis

```bash
python scripts/run_pipeline.py --config configs/experiments/exp1_teacher_entropy.yaml
```

결과:
- `runs/dapd/experiments/exp1_teacher_entropy/analysis/teacher_distribution.json`
- `runs/dapd/experiments/exp1_teacher_entropy/analysis/teacher_entropy_histogram.png`
- `runs/dapd/experiments/exp1_teacher_entropy/analysis/teacher_confidence_histogram.png`

---

## 2) Experiment 2 — Soft Label Information Gain

아래 스크립트는
- `KL(P_teacher || uniform)` (general/domain 각각)
- `KL(P_domain || P_general)` 토큰 분포
- KL heatmap
을 생성합니다.

```bash
python - <<'PY'
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from dapd.config import PipelineConfig
from dapd.data import build_unified_dataset, prepare_datasets_from_unified
from dapd.utils import infer_device

cfg = PipelineConfig.from_yaml("configs/experiments/exp1_teacher_entropy.yaml")
summary_path = Path("runs/dapd/experiments/exp1_teacher_entropy/pipeline_summary.json")
summary = json.loads(summary_path.read_text(encoding="utf-8"))

general_teacher_path = cfg.adaptation.teacher_model_name_or_path
domain_teacher_path = summary["adaptation"]["teacher_path"]

device = infer_device(cfg.runtime.device)
g_tok = AutoTokenizer.from_pretrained(general_teacher_path, use_fast=True)
d_tok = AutoTokenizer.from_pretrained(domain_teacher_path, use_fast=True)
if g_tok.vocab_size != d_tok.vocab_size:
    raise ValueError("Tokenizer/vocab mismatch. KL 비교를 위해 동일 tokenizer family가 필요합니다.")

unified = build_unified_dataset(cfg.data)
ds = prepare_datasets_from_unified(unified=unified, config=cfg.data, tokenizer=g_tok).validation_lm

g_model = AutoModelForCausalLM.from_pretrained(general_teacher_path, trust_remote_code=True).to(device).eval()
d_model = AutoModelForCausalLM.from_pretrained(domain_teacher_path, trust_remote_code=True).to(device).eval()

entropy_g = []
entropy_d = []
kl_d_to_g = []
heat = []
num_samples = min(64, len(ds))
max_tokens = 128

with torch.no_grad():
    for i in range(num_samples):
        row = ds[i]
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.tensor(row["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)
        labels = torch.tensor(row["labels"], dtype=torch.long, device=device).unsqueeze(0)
        valid = labels != -100

        g_logits = g_model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        d_logits = d_model(input_ids=input_ids, attention_mask=attention_mask).logits.float()

        g_probs = torch.softmax(g_logits, dim=-1)
        d_probs = torch.softmax(d_logits, dim=-1)
        g_ent = -(g_probs * g_probs.clamp_min(1e-12).log()).sum(dim=-1)
        d_ent = -(d_probs * d_probs.clamp_min(1e-12).log()).sum(dim=-1)
        token_kl = F.kl_div(torch.log_softmax(g_logits, dim=-1), d_probs, reduction="none").sum(dim=-1)

        entropy_g.extend(g_ent[valid].detach().cpu().tolist())
        entropy_d.extend(d_ent[valid].detach().cpu().tolist())
        kl_d_to_g.extend(token_kl[valid].detach().cpu().tolist())

        v = token_kl[0, :max_tokens].detach().cpu()
        if v.numel() < max_tokens:
            pad = torch.full((max_tokens - v.numel(),), float("nan"))
            v = torch.cat([v, pad], dim=0)
        heat.append(v)

vocab = float(g_tok.vocab_size)
kl_uniform_general = math.log(vocab) - (sum(entropy_g) / max(1, len(entropy_g)))
kl_uniform_domain = math.log(vocab) - (sum(entropy_d) / max(1, len(entropy_d)))

out_dir = Path("runs/dapd/experiments/exp2_soft_label_info")
out_dir.mkdir(parents=True, exist_ok=True)

result = {
    "general_teacher_path": general_teacher_path,
    "domain_teacher_path": domain_teacher_path,
    "num_samples": num_samples,
    "kl_teacher_to_uniform": {
        "general": kl_uniform_general,
        "domain": kl_uniform_domain,
    },
    "kl_domain_to_general_mean": float(sum(kl_d_to_g) / max(1, len(kl_d_to_g))),
}
(out_dir / "soft_label_information_gain.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
)

try:
    import matplotlib.pyplot as plt

    mat = torch.stack(heat, dim=0).numpy() if heat else None
    if mat is not None:
        plt.figure(figsize=(10, 5))
        plt.imshow(mat, aspect="auto", interpolation="nearest")
        plt.colorbar()
        plt.title("KL(P_domain || P_general) Heatmap")
        plt.xlabel("Token index")
        plt.ylabel("Sample index")
        plt.tight_layout()
        plt.savefig(out_dir / "kl_divergence_heatmap.png", dpi=150)
        plt.close()
except Exception as e:
    print("heatmap skipped:", e)

print("saved:", out_dir)
PY
```

---

## 3) Experiment 3 — Distillation Efficiency

```bash
python scripts/run_pipeline.py --config configs/experiments/exp3_domain.yaml
python scripts/run_pipeline.py --config configs/experiments/exp3_general.yaml
```

비교 플롯(학습 loss curve + validation accuracy):

```bash
python - <<'PY'
import json
from pathlib import Path

def load_series(trainer_state_path):
    s = json.loads(Path(trainer_state_path).read_text(encoding="utf-8"))
    xs, ys = [], []
    for row in s.get("log_history", []):
        if "loss" in row and "step" in row:
            xs.append(row["step"])
            ys.append(row["loss"])
    return xs, ys

domain_state = "runs/dapd/experiments/exp3_domain/distilled_student/trainer_state.json"
general_state = "runs/dapd/experiments/exp3_general/distilled_student/trainer_state.json"
dx, dy = load_series(domain_state)
gx, gy = load_series(general_state)

domain_eval = json.loads(Path("runs/dapd/experiments/exp3_domain/eval_metrics.json").read_text(encoding="utf-8"))
general_eval = json.loads(Path("runs/dapd/experiments/exp3_general/eval_metrics.json").read_text(encoding="utf-8"))

print("domain_teacher accuracy:", domain_eval.get("accuracy"), "f1:", domain_eval.get("f1"))
print("general_teacher accuracy:", general_eval.get("accuracy"), "f1:", general_eval.get("f1"))

try:
    import matplotlib.pyplot as plt
    out = Path("runs/dapd/experiments/exp3_efficiency")
    out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,4.8))
    if dx: plt.plot(dx, dy, label="domain teacher -> student")
    if gx: plt.plot(gx, gy, label="general teacher -> student")
    plt.xlabel("step")
    plt.ylabel("training loss")
    plt.title("Distillation efficiency comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "distillation_efficiency_curve.png", dpi=150)
    plt.close()
    print("saved:", out / "distillation_efficiency_curve.png")
except Exception as e:
    print("plot skipped:", e)
PY
```

---

## 4) Experiment 4 — Temperature Curriculum Analysis

```bash
python scripts/run_pipeline.py --config configs/experiments/exp4_temperature.yaml
```

결과:
- `runs/dapd/experiments/exp4_temperature/analysis/temperature_analysis.json`
- `runs/dapd/experiments/exp4_temperature/analysis/temperature_loss_vs_step.png`
- `runs/dapd/experiments/exp4_temperature/analysis/temperature_vs_step.png`

---

## 5) Experiment 5 — Domain-Specific Head Analysis

```bash
python scripts/run_pipeline.py --config configs/experiments/exp5_domain_heads.yaml
python scripts/run_pipeline.py --config configs/experiments/exp5_general_heads.yaml
```

domain vs general 비교 heatmap:

```bash
python - <<'PY'
import json
from pathlib import Path
import torch

def to_matrix(report_path):
    rep = json.loads(Path(report_path).read_text(encoding="utf-8"))
    rows = sorted(rep.get("attention_patterns", []), key=lambda x: (x.get("layer_index", -1), x.get("module", "")))
    if not rows:
        return torch.empty(0, 0)
    m = max(len(r.get("head_scores", [])) for r in rows)
    mat = torch.full((len(rows), m), float("nan"))
    for i, r in enumerate(rows):
        s = r.get("head_scores", [])
        if s:
            mat[i, :len(s)] = torch.tensor(s, dtype=torch.float32)
    return mat

domain_rep = "runs/dapd/experiments/exp5_domain_heads/pruned_student/pruning_report.json"
general_rep = "runs/dapd/experiments/exp5_general_heads/pruned_student/pruning_report.json"
dm = to_matrix(domain_rep)
gm = to_matrix(general_rep)

out = Path("runs/dapd/experiments/exp5_head_compare")
out.mkdir(parents=True, exist_ok=True)

try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if dm.numel() > 0:
        im0 = axes[0].imshow(dm.numpy(), aspect="auto", interpolation="nearest")
        axes[0].set_title("Domain teacher distillation")
        fig.colorbar(im0, ax=axes[0])
    if gm.numel() > 0:
        im1 = axes[1].imshow(gm.numpy(), aspect="auto", interpolation="nearest")
        axes[1].set_title("General teacher distillation")
        fig.colorbar(im1, ax=axes[1])
    for ax in axes:
        ax.set_xlabel("head index")
        ax.set_ylabel("layer index")
    fig.tight_layout()
    fig.savefig(out / "domain_vs_general_head_importance.png", dpi=150)
    plt.close(fig)
    print("saved:", out / "domain_vs_general_head_importance.png")
except Exception as e:
    print("plot skipped:", e)
PY
```

---

## 6) Experiment 6 — Out-of-Domain Distillation (PubMedQA -> BioASQ)

```bash
python scripts/run_pipeline.py --config configs/experiments/exp6_domain_ood.yaml
python scripts/run_pipeline.py --config configs/experiments/exp6_general_ood.yaml
```

OOD 성능 저하 비교:

```bash
python - <<'PY'
import json
from pathlib import Path

def load(prefix):
    in_domain = json.loads(Path(f"runs/dapd/experiments/{prefix}/eval_metrics.json").read_text(encoding="utf-8"))
    ood = json.loads(Path(f"runs/dapd/experiments/{prefix}/eval_metrics_ood.json").read_text(encoding="utf-8"))
    return in_domain, ood

dom_in, dom_ood = load("exp6_domain_ood")
gen_in, gen_ood = load("exp6_general_ood")

def row(name, i, o):
    acc_drop = i.get("accuracy", 0.0) - o.get("accuracy", 0.0)
    f1_drop = i.get("f1", 0.0) - o.get("f1", 0.0)
    ppl_increase = o.get("perplexity", 0.0) - i.get("perplexity", 0.0)
    return {
        "run": name,
        "in_domain_acc": i.get("accuracy"),
        "ood_acc": o.get("accuracy"),
        "accuracy_drop": acc_drop,
        "f1_drop": f1_drop,
        "perplexity_increase": ppl_increase,
    }

rows = [
    row("domain teacher -> student", dom_in, dom_ood),
    row("general teacher -> student", gen_in, gen_ood),
]
for r in rows:
    print(r)

out = Path("runs/dapd/experiments/exp6_ood_compare.json")
out.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
print("saved:", out)
PY
```

---

## 최종 산출물 위치 요약

- Exp1: `runs/dapd/experiments/exp1_teacher_entropy/analysis/*`
- Exp2: `runs/dapd/experiments/exp2_soft_label_info/*`
- Exp3: `runs/dapd/experiments/exp3_*/*`
- Exp4: `runs/dapd/experiments/exp4_temperature/analysis/*`
- Exp5: `runs/dapd/experiments/exp5_*/*`, `runs/dapd/experiments/exp5_head_compare/*`
- Exp6: `runs/dapd/experiments/exp6_*/*`, `runs/dapd/experiments/exp6_ood_compare.json`
