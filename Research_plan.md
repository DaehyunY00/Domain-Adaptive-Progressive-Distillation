# Domain-Adaptive Progressive Distillation for Efficient Small Language Models

## 1. Introduction

Large Language Models (LLMs) have achieved remarkable performance across a wide range of tasks. However, their computational cost makes deployment difficult in resource-constrained environments. Model compression techniques such as **knowledge distillation** and **structured pruning** have therefore become essential tools for efficient LLM deployment.

Despite their effectiveness, most existing distillation approaches assume that the teacher and student models operate within the same domain. In practice, many applications involve domain-specific tasks such as biomedical question answering, legal reasoning, or scientific knowledge extraction. In such cases, distillation from a general-domain teacher can produce suboptimal supervision signals due to domain mismatch.

This research proposes **Domain-Adaptive Progressive Distillation (DAPD)**, a multi-stage framework that integrates domain adaptation, progressive distillation, and structured pruning to efficiently train domain-specialized small language models.

Unlike conventional distillation pipelines, DAPD introduces an intermediate **domain-adapted teacher model**, which produces more informative soft-label distributions aligned with the target task domain. These improved teacher distributions enable more efficient knowledge transfer during student training.

---

# 2. Core Theoretical Insight

The key theoretical insight behind DAPD is:

**Domain-adapted teachers produce lower-entropy and more task-aligned soft-label distributions, which improve distillation efficiency.**

In standard knowledge distillation, the student model learns from the teacher probability distribution:

[
L_{KD} = KL(P_{teacher} || P_{student})
]

However, when the teacher model is trained on general-domain data while the task belongs to a specialized domain, the teacher distribution may not align well with the target task distribution.

Let

* (P_{task}): task distribution
* (P_{teacher}^{general}): general teacher
* (P_{teacher}^{domain}): domain-adapted teacher

DAPD hypothesizes:

[
KL(P_{teacher}^{domain} || P_{task}) < KL(P_{teacher}^{general} || P_{task})
]

This implies that the domain-adapted teacher provides a more informative supervision signal for the student model.

From an information-theoretic perspective, domain adaptation reduces prediction entropy:

[
H(P_{teacher}^{domain}) < H(P_{teacher}^{general})
]

Lower entropy indicates higher prediction confidence and clearer decision boundaries, resulting in stronger gradients during student optimization.

DAPD therefore improves distillation efficiency by aligning the teacher distribution with the target task distribution before distillation.

---

# 3. Proposed Method

## 3.1 Overview

DAPD consists of three main stages:

1. **Domain Adaptation**
2. **Progressive Knowledge Distillation**
3. **Structured Model Compression**

Pipeline:

```
General Teacher Model
        │
        ▼
Domain Adaptation (LoRA / QLoRA)
        │
        ▼
Domain-Specialized Teacher
        │
        ▼
Progressive Knowledge Distillation
        │
        ▼
Student Model
        │
        ▼
Structured Pruning
        │
        ▼
Efficient Domain-Specialized LLM
```

---

# 3.2 Stage 1: Domain Adaptation

A general teacher model is adapted to the target domain using parameter-efficient fine-tuning.

[
L_{adapt} = L_{task}(M_T(D_{domain}))
]

The resulting model becomes a **domain teacher**, which provides improved soft labels for distillation.

---

# 3.3 Stage 2: Progressive Knowledge Distillation

Instead of directly distilling from the general teacher, the student learns from the domain teacher.

The distillation loss:

[
L_{distill} =
\alpha \cdot CE(y, P_{student}) +
(1-\alpha) \cdot KL(P_{teacher}, P_{student})
]

DAPD introduces **progressive temperature scheduling** during distillation.

Early training:

```
high temperature → knowledge exploration
```

Later training:

```
low temperature → knowledge consolidation
```

This mechanism acts as a **curriculum distillation strategy**.

---

# 3.4 Stage 3: Structured Model Compression

After distillation, structured pruning reduces model complexity.

Importance score:

[
Importance_i = \beta |w_i| + (1-\beta)A_i
]

Where

* (w_i): weight magnitude
* (A_i): activation importance

Pruning is applied to:

```
attention heads
MLP neurons
```

to produce the final compressed student model.

---

# 4. Experimental Setup

## 4.1 Models

Teacher:

```
Qwen2.5-1.5B
```

Students:

```
Qwen2.5-0.5B
TinyLlama
Phi-2
```

---

## 4.2 Datasets

Biomedical QA benchmarks:

```
MedQA (USMLE)
PubMedQA
MedMCQA
BioASQ
```

These datasets provide diverse evaluation environments for medical reasoning tasks.

---

# 5. Evaluation Protocol

## Performance Metrics

```
Accuracy
F1 Score
Perplexity
```

## Efficiency Metrics

```
Compression ratio
Inference latency
Throughput
Memory usage
```

## Calibration Metrics

```
Expected Calibration Error (ECE)
Brier Score
```

## Statistical Significance

All experiments are repeated with **three random seeds**, and results are reported as:

```
mean ± standard deviation
```

---

# 6. Analysis Experiments (Reviewer-Focused)

To understand *why* DAPD works, we conduct six analysis experiments.

---

## Experiment 1: Teacher Entropy Analysis

Goal:

Measure how domain adaptation changes teacher prediction entropy.

Metrics:

```
H(P_teacher_general)
H(P_teacher_domain)
```

Expected observation:

```
Domain teacher → lower entropy
```

---

## Experiment 2: Soft Label Information Gain

Goal:

Measure the information provided by teacher soft labels.

Metrics:

```
KL(P_teacher || uniform)
KL(P_teacher_general || P_teacher_domain)
```

Visualization:

```
soft label distribution histograms
```

---

## Experiment 3: Distillation Efficiency

Goal:

Evaluate whether domain teachers improve student learning efficiency.

Method:

Compare student training curves under two teachers.

```
general teacher → student
domain teacher → student
```

Metrics:

```
validation accuracy
training loss convergence
```

---

## Experiment 4: Temperature Curriculum Analysis

Goal:

Analyze the effect of progressive temperature scheduling.

Compare:

```
constant temperature
linear schedule
cosine schedule
```

Metrics:

```
training stability
student accuracy
loss variance
```

---

## Experiment 5: Domain-Specific Head Analysis

Goal:

Identify domain-specific transformer components.

Method:

Analyze attention head importance scores during pruning.

Visualization:

```
attention head importance heatmap
layer-wise pruning patterns
```

---

## Experiment 6: Out-of-Domain Distillation

Goal:

Evaluate domain generalization.

Train on:

```
PubMedQA
```

Test on:

```
BioASQ
```

Compare:

```
general teacher distillation
domain teacher distillation
```

---

# 7. Baselines

DAPD will be compared with:

```
Direct Knowledge Distillation
LoRA Fine-tuning
SparseGPT
LLM-Pruner
```

---

# 8. Ablation Studies

We evaluate the contribution of each component:

```
DAPD (full pipeline)
no adaptation
no distillation
no pruning
constant temperature
```

---

# 9. Expected Contributions

This research contributes:

1. A theoretical insight linking **teacher distribution alignment and distillation efficiency**.
2. A progressive distillation framework for domain-specific LLM compression.
3. An empirical study of **teacher distribution shift in domain adaptation**.
4. Analysis of **domain-specific pruning patterns in LLMs**.
5. A reproducible open-source pipeline for small domain-specialized language models.

---

# 10. Timeline

| Phase                            | Duration |
| -------------------------------- | -------- |
| Literature Review                | 2 weeks  |
| Dataset Preparation              | 1 week   |
| Domain Adaptation Implementation | 2 weeks  |
| Distillation Implementation      | 2 weeks  |
| Pruning Optimization             | 2 weeks  |
| Experiments & Analysis           | 3 weeks  |
| Paper Writing                    | 3 weeks  |

Total duration:

```
15 weeks
```

---

# 11. Reproducibility

The entire pipeline will be implemented using open-source frameworks:

```
PyTorch
HuggingFace Transformers
PEFT (LoRA / QLoRA)
Accelerate
```

All experiments will be reproducible through configuration files and public datasets.

