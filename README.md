Research Plan
Title

Domain-Adaptive Progressive Distillation:
Analyzing Teacher Distribution Shift for Efficient Small Language Models

1. Research Motivation

Large Language Models require substantial computational resources. Knowledge distillation and pruning are widely used to compress models.

However, most distillation methods assume that the teacher model and the target task share the same domain.

In many real-world scenarios (e.g., biomedical QA), the teacher model is trained on general-domain data, which leads to domain mismatch between teacher knowledge and student tasks.

This research investigates the following hypothesis:

Domain-adapted teachers produce more informative soft-label distributions for distillation.

This hypothesis is grounded in the idea that domain adaptation reduces teacher prediction entropy and improves knowledge transfer.

Recent work on domain-aware distillation also suggests that domain-specific knowledge alignment can significantly improve student models.

2. Key Research Questions

RQ1
How does the soft-label distribution of a domain-adapted teacher differ from a general teacher?

RQ2
Does progressive temperature scheduling improve distillation stability?

RQ3
Do domain-adapted teachers lead to better compressed student models?

RQ4
Which transformer components are pruned when compressing domain-specific LLMs?

3. Proposed Method
Overview

The proposed pipeline:

General Teacher
      │
      ▼
Domain Adaptation
      │
      ▼
Domain Teacher
      │
      ▼
Progressive Distillation
      │
      ▼
Student Model
      │
      ▼
Structured Pruning
      │
      ▼
Compressed Student
4. Key Novel Contributions
Contribution 1

Teacher Distribution Analysis

We analyze the soft-label distributions of teachers before and after domain adaptation.

Metrics:

entropy
confidence distribution
KL divergence
calibration error

Goal:

measure information gain from domain adaptation
Contribution 2

Progressive Temperature Distillation

We interpret temperature scheduling as a curriculum learning mechanism.

Early training:

high temperature
→ knowledge exploration

Late training:

low temperature
→ knowledge consolidation

We empirically analyze the effect of temperature schedules.

Contribution 3

Domain-Specific Pruning Analysis

We analyze which transformer components are pruned during compression.

Focus:

attention heads
MLP neurons

We visualize pruning patterns and test whether domain-specific heads exist.

5. Experimental Setup
Models

Teacher

Qwen2.5-1.5B

Student

Qwen2.5-0.5B
TinyLlama
Phi-2
Datasets

Medical QA benchmarks

MedQA (USMLE)
PubMedQA
MedMCQA
BioASQ

MedQA is a widely used benchmark derived from US medical licensing exam questions and is commonly used to evaluate LLM medical reasoning ability.

6. Evaluation Protocol
Performance Metrics
Accuracy
F1
Perplexity
Efficiency Metrics
Compression ratio
Latency
Throughput
Memory
Calibration Metrics
Expected Calibration Error (ECE)
Brier Score
Generalization Tests
train: PubMedQA
test: BioASQ
Statistical Testing
3 random seeds
mean ± std
paired t-test
7. Baselines

We compare with:

Direct KD
LoRA-only fine-tuning
SparseGPT pruning
LLM-Pruner
8. Ablation Studies

Experiments:

full DAPD
no adaptation
no distillation
no pruning
constant temperature
9. Expected Contributions
1. empirical analysis of domain-aware distillation
2. progressive distillation strategy
3. pruning analysis for domain-specific LLMs
4. efficient small-domain LLM pipeline