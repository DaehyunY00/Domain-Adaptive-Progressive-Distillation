# Domain-Adaptive Progressive Distillation for Efficient Small Language Models

## 1. Introduction

Large Language Models (LLMs) have demonstrated strong performance across diverse tasks. However, their large computational requirements limit practical deployment in resource-constrained environments. To address this issue, model compression techniques such as knowledge distillation and pruning have been widely explored.

Despite their effectiveness, most existing distillation approaches assume that both teacher and student models operate within the same domain. In real-world scenarios, however, many applications require domain adaptation, such as biomedical question answering, legal reasoning, and scientific knowledge tasks. Direct distillation from a general-domain teacher model often leads to performance degradation due to domain shift.

This research proposes **Domain-Adaptive Progressive Distillation (DAPD)**, a multi-stage framework that integrates domain adaptation and model compression for small language models (≤5B parameters). Instead of directly distilling a general-domain teacher into a compressed student model, the proposed method introduces an intermediate domain-adapted model that progressively transfers domain knowledge before compression.

The proposed framework aims to improve the efficiency-performance trade-off of small LLMs while maintaining strong domain-specific performance.

---

## 2. Research Objectives

The objectives of this study are:

1. To develop a progressive distillation framework that integrates domain adaptation and model compression.
2. To improve domain-specific performance of small LLMs while maintaining computational efficiency.
3. To analyze the trade-off between model size, compression ratio, and domain adaptation performance.

---

## 3. Proposed Method

### 3.1 Overview

The proposed **Domain-Adaptive Progressive Distillation (DAPD)** framework consists of three stages:

1. **Domain Adaptation Stage**
2. **Progressive Knowledge Distillation Stage**
3. **Model Compression Stage**

The overall pipeline is illustrated below:


General Teacher Model
│
▼
Domain Adaptation (LoRA / QLoRA)
│
▼
Domain-Specialized Intermediate Model
│
▼
Knowledge Distillation
│
▼
Compressed Student Model
│
▼
Structured Pruning
│
▼
Efficient Domain-Specialized Small LLM


---

### 3.2 Stage 1: Domain Adaptation

A general-domain language model is first adapted to a specific domain using parameter-efficient fine-tuning methods such as **LoRA** or **QLoRA**.

Let:

- \( M_T \) : teacher model
- \( D_{domain} \) : domain dataset

The adapted model is obtained by minimizing:

\[
L_{adapt} = L_{task}(M_T(D_{domain}))
\]

This stage produces a **domain-specialized intermediate model**.

---

### 3.3 Stage 2: Progressive Knowledge Distillation

Instead of distilling directly from the teacher model, the student model learns from the domain-adapted intermediate model.

The distillation loss is defined as:

\[
L_{distill} = \alpha \cdot KL(p_T || p_S) + (1-\alpha) \cdot L_{task}
\]

Where:

- \(p_T\) : teacher output distribution
- \(p_S\) : student output distribution

This stage allows the student model to inherit domain-specific knowledge more effectively.

---

### 3.4 Stage 3: Model Compression via Structured Pruning

After distillation, structured pruning is applied to reduce model size.

Pruning is based on **activation importance**:

\[
Importance_i = \beta |w_i| + (1-\beta) A_i
\]

Where:

- \(w_i\) : weight magnitude
- \(A_i\) : domain activation score

This step produces the final compressed model.

---

## 4. Experimental Setup

### 4.1 Models

Small language models (≤5B parameters):

- Phi-2 (2.7B)
- Qwen 2.5 3B
- TinyLlama (1.1B)

---

### 4.2 Datasets

Public datasets are used to ensure reproducibility.

Biomedical domain:

- PubMedQA

Scientific reasoning:

- SciQ

Medical QA:

- MedMCQA

---

### 4.3 Baselines

The proposed method will be compared against:

1. Direct knowledge distillation
2. LoRA fine-tuning without distillation
3. Distillation without domain adaptation
4. Standard pruning methods

---

### 4.4 Evaluation Metrics

Performance metrics:

- Accuracy
- F1 Score
- Perplexity

Efficiency metrics:

- Model size
- Inference latency
- Memory usage

---

## 5. Expected Contributions

This research is expected to contribute:

1. A novel **Domain-Adaptive Progressive Distillation framework**.
2. A systematic study of **domain adaptation combined with model compression**.
3. An efficient training pipeline for small language models.
4. Open-source reproducible implementation.

---

## 6. Implementation Plan

The full training pipeline includes:

1. Dataset preprocessing
2. Domain adaptation training
3. Knowledge distillation
4. Structured pruning
5. Evaluation and benchmarking

All experiments will be conducted on a MacBook Pro M4 (16GB RAM) using parameter-efficient methods.

---

## 7. Timeline

| Phase | Duration |
|------|---------|
| Literature Review | ** weeks |
| Dataset Preparation | * week |
| Domain Adaptation Implementation | * weeks |
| Distillation Implementation | * weeks |
| Pruning & Optimization | * weeks |
| Experiments & Analysis | * weeks |
| Paper Writing | * weeks |

Total expected duration: ** * weeks**

---

## 8. Reproducibility

The entire pipeline will be implemented using open-source tools including:

- HuggingFace Transformers
- PEFT (LoRA / QLoRA)
- PyTorch
- Accelerate

All code and experiment configurations will be released publicly.