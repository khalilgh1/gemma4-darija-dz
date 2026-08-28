# Algerian Darija LLM Adaptation & Translation
## Experimental Sequence & Research Plan

---

## Table of Contents
1. [1. Project Overview & Research Questions](#1-project-overview--research-questions)
2. [2. Multi-Stage Architectural Flow](#2-multi-stage-architectural-flow)
3. [3. Experimental Model Taxonomy](#3-experimental-model-taxonomy)
4. [4. Detailed Research Questions](#4-detailed-research-questions)
5. [5. Data Ingestion & Token Analysis](#5-data-ingestion--token-analysis)
6. [6. Monolingual Data Cleaning Pipeline](#6-monolingual-data-cleaning-pipeline)
7. [7. Tokenizer Empirical Analysis](#7-tokenizer-empirical-analysis)
8. [8. Stage 1: Data Preparation](#8-stage-1-data-preparation)
9. [9. Stage 1: Sequence Length & Memory Optimization](#9-stage-1-sequence-length--memory-optimization)
10. [10. Stage 1: Causal LM Training Objective](#10-stage-1-causal-lm-training-objective)
11. [11. Stage 1: QLoRA Fine-Tuning Strategy](#11-stage-1-qlora-fine-tuning-strategy)
12. [12. Stage 1: Smoke Testing & Validation](#12-stage-1-smoke-testing--validation)
13. [13. Stage 1: Full-Scale Training Pipeline](#13-stage-1-full-scale-training-pipeline)
14. [14. Stage 1: Evaluation Protocol](#14-stage-1-evaluation-protocol)
15. [15. Stage 2: Parallel Translation Dataset](#15-stage-2-parallel-translation-dataset)
16. [16. Stage 2: Parallel Quality Assurance](#16-stage-2-parallel-quality-assurance)
17. [17. Stage 2: Data Splitting Protocol](#17-stage-2-data-splitting-protocol)
18. [18. Stage 2: Translation Instruction Formatting](#18-stage-2-translation-instruction-formatting)
19. [19. Stage 2: Supervised Fine-Tuning (SFT)](#19-stage-2-supervised-fine-tuning-sft)
20. [20. Translation Baselines & Checkpoint Benchmarking](#20-translation-baselines--checkpoint-benchmarking)
21. [21. Stage 2: Translation Evaluation Metrics](#21-stage-2-translation-evaluation-metrics)
22. [22. Core Ablation Matrix](#22-core-ablation-matrix)
23. [23. Parallel Data Sample-Efficiency Experiments](#23-parallel-data-sample-efficiency-experiments)
24. [24. Error Taxonomy & Qualitative Analysis](#24-error-taxonomy--qualitative-analysis)
25. [25. Experiment Tracking & Reproducibility](#25-experiment-tracking--reproducibility)
26. [26. Model Checkpoint Packaging & Versioning](#26-model-checkpoint-packaging--versioning)
27. [27. Dual-Direction Inference Pipeline](#27-dual-direction-inference-pipeline)
28. [28. Inference Benchmarking & Hardware Profiling](#28-inference-benchmarking--hardware-profiling)
29. [29. Serving & API Architecture](#29-serving--api-architecture)
30. [30. Recommended Project Directory Hierarchy](#30-recommended-project-directory-hierarchy)
31. [31. End-to-End Execution Pipeline](#31-end-to-end-execution-pipeline)
32. [32. Expected Outcomes & Research Significance](#32-expected-outcomes--research-significance)

---

## 1. Project Overview & Research Questions

### 1.1 Objective
The goal of this project is to empirically investigate whether a pretrained decoder-only Large Language Model (e.g. Gemma) can be effectively adapted to **Algerian Darija (`ary` / `ar-DZ`)** primarily through large-scale unlabeled monolingual data, and whether this unsupervised adaptation significantly boosts performance and sample efficiency on downstream **English $\leftrightarrow$ Algerian Darija translation** when subsequently trained on a small amount of high-quality parallel data.

---

## 2. Multi-Stage Architectural Flow

```text
                 PRETRAINED BASE GEMMA
                          │
                          ▼
         ┌─────────────────────────────────┐
         │ STAGE 1: Language Adaptation    │
         │ ~160k-row Monolingual Corpus    │
         │ Causal Language Modeling (CLM)  │
         └────────────────┬────────────────┘
                          │
                          ▼
               DARIJA-ADAPTED MODEL
                          │
                          ▼
         ┌─────────────────────────────────┐
         │ STAGE 2: Translation SFT        │
         │ Curated English ↔ Darija Pairs  │
         │ Supervised Fine-Tuning          │
         └────────────────┬────────────────┘
                          │
                          ▼
           FINAL TRANSLATION MODEL
```

---

## 3. Experimental Model Taxonomy

| Model ID | Identifier | Description | Training Data |
| :--- | :--- | :--- | :--- |
| **Model A** | `Base` | Pretrained Gemma foundation model | None (Baseline) |
| **Model B** | `Stage 1` | Continued pre-trained model for dialect acquisition | Monolingual Algerian Darija corpus |
| **Model C** | `Stage 2` | Fully aligned translation model | Stage 1 + Parallel Translation SFT |
| **Model D** | `Direct SFT` | Ablation model (Translation without Stage 1) | Base Gemma + Parallel Translation SFT |

---

## 4. Detailed Research Questions

### Q1: Monolingual Dialect Acquisition
> *Does continued causal language modeling on raw Algerian Darija text measurably improve dialect perplexity, vocabulary representation, and natural generation quality over the base foundation model?*

### Q2: Unsupervised Cross-Lingual Transfer
> *Does monolingual adaptation yield any zero-shot or implicit English $\leftrightarrow$ Darija translation gains prior to supervised translation fine-tuning?*

### Q3: Two-Stage Synergy & Sample Efficiency
> *Does Stage 1 pre-adaptation make supervised translation fine-tuning (Stage 2) significantly more data-efficient compared to fine-tuning the base model directly on parallel pairs?*

### Q4: Parallel Supervision Scaling Laws
> *How does translation accuracy scale as parallel data size increases ($N \in \{100, 500, 1000, 5000\}$ pairs) between the adapted model and non-adapted model?*

### Q5: Catastrophic Forgetting & General Ability Retention
> *Does dialect adaptation degrade general reasoning, English fluency, or instruction-following capabilities?*

---

## 5. Data Ingestion & Token Analysis

### Monolingual Corpus Specifications
- **Raw Volume:** $\approx 160,000$ lines.
- **Critical Measure:** Total token count under the base model's tokenizer (rather than simple raw line count).

### Key Corpus Diagnostics
- [x] Total and unique token counts
- [x] Length distribution (mean, median, 95th/99th percentiles)
- [x] Script distribution (Arabic script, Latin script, Arabizi alphanumeric)
- [x] Code-switching density (French / MSA / English borrowings)

---

## 6. Monolingual Data Cleaning Pipeline

```text
Raw Corpus ───► [Empty/Malformed Filter] ───► [Near/Exact Deduplication] ───► [Script & Language Verification] ───► Cleaned Splits
```

### Preprocessing Guidelines:
1. **Deduplication:** Remove exact and near duplicates to prevent verbatim memorization.
2. **Preserve Dialect Nuance:** Avoid aggressive text normalization that collapses authentic Algerian Darija phonetic variations or switches into Modern Standard Arabic.
3. **Multi-Script Integrity:** Preserve legitimate Arabizi and French code-switching common in spoken Algerian conversation.

---

## 7. Tokenizer Empirical Analysis

Evaluate how Gemma's SentencePiece/Byte-level BPE tokenizer handles Algerian Darija text across script variations:

| Script Modality | Metric | Target Assessment |
| :--- | :--- | :--- |
| **Arabic Script** | Tokens / Word | Measure subword fragmentation rate |
| **Latin Script** | Tokens / Character | Check compression efficiency |
| **Arabizi** (e.g. `3`, `7`, `9`) | Out-of-Vocabulary / Fallback | Ensure numbers are not stripped or fragmented |

---

## 8. Stage 1: Data Preparation

- **Task Objective:** Autoregressive Causal Language Modeling.
- **Supervision:** Unsupervised / Self-supervised (shifted labels $x_t \to x_{t+1}$).
- **Data Serialization:**
  ```json
  {"text": "الطقس اليوم في الجزائر شباب بزاف والعشية راح نخرجو..."}
  ```

---

## 9. Stage 1: Sequence Length & Memory Optimization

- **Target Context Window:** $L = 2048$ tokens.
- **Consumer Hardware Fallback (8GB VRAM / RTX 3070):** $L = 1024$ tokens.
- **Sequence Packing:** Concatenate short sentences separated by `<eos>` tokens to maximize GPU compute efficiency.

---

## 10. Stage 1: Causal LM Training Objective

The objective minimizes standard cross-entropy over token sequence $X = (x_1, \dots, x_n)$:

$$\mathcal{L}_{\text{CLM}}(\theta) = -\sum_{t=1}^{n} \log P_\theta(x_t \mid x_{<t})$$

---

## 11. Stage 1: QLoRA Fine-Tuning Strategy

To enable training on resource-constrained GPUs (e.g. 8GB VRAM):
- **Base Model Precision:** 4-bit NormalFloat (`NF4`) quantization via `bitsandbytes`.
- **LoRA Targets:** Attention projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`) + MLP layers (`gate_proj`, `up_proj`, `down_proj`).
- **Compute Precision:** `bfloat16` / `fp16` mixed precision with gradient checkpointing.

---

## 12. Stage 1: Smoke Testing & Validation

Execute a micro-training validation before launching long training jobs:
```text
Load Quantized Base Model ──► Check LoRA Init ──► Forward Pass ──► Loss Computation ──► Backward Pass ──► Adapter Save/Reload
```

---

## 13. Stage 1: Full-Scale Training Pipeline

### Training Telemetry
- Training & validation loss curves
- Perplexity on held-out Darija validation split: $\text{PPL} = \exp(\mathcal{L})$
- Hardware stats: VRAM consumption, tokens/second throughput

---

## 14. Stage 1: Evaluation Protocol

### 1. Quantitative Evaluation
- Perplexity on held-out test split.
- Zero-shot English $\leftrightarrow$ Darija translation loss.

### 2. Qualitative Evaluation
- Prompt completion & conversational fluency.
- Grammar, dialect authenticity, and vocabulary richness.
- Repetition rates and hallucination checks.

---

## 15. Stage 2: Parallel Translation Dataset

- **Source Pairs:** English $\leftrightarrow$ Algerian Darija.
- **Quality-First Paradigm:** High-accuracy human-verified pairs prioritized over large, noisy datasets.
- **Sample Tiers for Scaling Experiments:** $N \in \{100, 500, 1000, 5000\}$ pairs.

---

## 16. Stage 2: Parallel Quality Assurance

Audit translation pairs against the following failure modes:

- [ ] Source/target semantic mismatch
- [ ] Translation direction inversions
- [ ] Literal machine translation artifacts (unnatural syntactic phrasing)
- [ ] Omission or hallucinated additions
- [ ] Modern Standard Arabic substitutions for authentic Darija expressions

---

## 17. Stage 2: Data Splitting Protocol

- **Train Split:** 80%
- **Validation Split:** 10%
- **Test Split:** 10% *(Held-out strictly for final benchmarking)*

---

## 18. Stage 2: Translation Instruction Formatting

### English to Algerian Darija
```text
User: Translate this English sentence to Algerian Darija:
I am going to school.

Assistant:
راني رايح للمسيد.
```

### Algerian Darija to English
```text
User: Translate this Algerian Darija sentence to English:
راني رايح للمسيد.

Assistant:
I am going to school.
```

---

## 19. Stage 2: Supervised Fine-Tuning (SFT)

- **Initialization:** Stage 1 Darija-adapted LoRA weights.
- **Loss Masking:** Compute loss exclusively over the target translation tokens (`Assistant:` turn), masking out prompt tokens.
- **Overfitting Safeguards:** Early stopping based on validation loss / validation chrF score.

---

## 20. Translation Baselines & Checkpoint Benchmarking

Evaluate translations across 3 checkpoint stages on the exact same test split:
1. **Base Model** (Zero-shot baseline)
2. **Stage 1 Adapted Model** (Zero-shot bilingual baseline)
3. **Stage 2 SFT Model** (Supervised translation model)

---

## 21. Stage 2: Translation Evaluation Metrics

1. **BLEU (`sacrebleu`):** Standard n-gram precision metric.
2. **chrF++:** Character n-gram F-score (essential for morphologically rich and code-switched dialects).
3. **COMET:** Neural cross-lingual metric scoring semantic similarity.
4. **Human Evaluation:** Blind pairwise comparison assessing naturalness and semantic fidelity.

---

## 22. Core Ablation Matrix

```text
                 ┌───────────────────┐
                 │ Pretrained Gemma  │
                 └─────────┬─────────┘
           ┌───────────────┴───────────────┐
           ▼                               ▼
    [No Adaptation]               [Stage 1: CLM Darija]
           │                               │
     ┌─────┴─────┐                   ┌─────┴─────┐
     ▼           ▼                   ▼           ▼
[Exp A: Base] [Exp C: SFT Only] [Exp B: Stage 1] [Exp D: Two-Stage]
```

### Summary Comparison Table
| Experiment | Monolingual Pretraining (Stage 1) | Parallel Fine-Tuning (Stage 2) | Target Investigation |
| :--- | :---: | :---: | :--- |
| **Exp A** | ❌ | ❌ | Zero-shot baseline capabilities |
| **Exp B** | ✅ | ❌ | Unsupervised cross-lingual transfer |
| **Exp C** | ❌ | ✅ | Standard direct fine-tuning performance |
| **Exp D** | ✅ | ✅ | Full two-stage hypothesis verification |

---

## 23. Parallel Data Sample-Efficiency Experiments

Train separate Stage 2 heads using subsets of parallel data:
- **Condition 1:** 100 pairs
- **Condition 2:** 500 pairs
- **Condition 3:** 1,000 pairs
- **Condition 4:** 5,000 pairs

*Compare Exp C (Direct SFT) vs. Exp D (Two-Stage) at every sample size to measure sample-efficiency gains.*

---

## 24. Error Taxonomy & Qualitative Analysis

Classify test prediction errors into structured categories:
- **Grammar & Morphology:** Incorrect verb conjugations, pronoun mismatches.
- **Lexical Accuracy:** Modern Standard Arabic substitution vs. authentic Darija idiom.
- **Script Handling:** Arabic script vs. Arabizi transliteration errors.
- **Code-Switching:** Mishandled French/Arabic blended sentences.
- **Semantic Fidelity:** Omission, hallucination, or over-literal translation.

---

## 25. Experiment Tracking & Reproducibility

Each experiment run must be cataloged in `experiments/exp-XXX/` containing:
- `config.json` (Hyperparameters, base model hash, dataset commit)
- `metrics.json` (Loss curves, BLEU, chrF, COMET scores)
- `generations.json` (Sample predictions across test prompts)
- `notes.md` (Observations and hardware telemetry)

---

## 26. Model Checkpoint Packaging & Versioning

```text
models/
├── stage1/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── training_args.bin
└── stage2/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── training_args.bin
```

---

## 27. Dual-Direction Inference Pipeline

The inference engine should support:
- `en -> ar-DZ` (English to Algerian Darija translation)
- `ar-DZ -> en` (Algerian Darija to English translation)
- `ar-DZ -> ar-DZ` (Darija causal text continuation)

---

## 28. Inference Benchmarking & Hardware Profiling

Benchmark production metrics:
- Time-to-First-Token (TTFT)
- Generation throughput (tokens/sec)
- GPU memory peak during KV-cache allocation (batch sizes 1, 4, 8)

---

## 29. Serving & API Architecture

### FastAPI Endpoint Architecture:
- `GET /health` - Service and GPU memory status
- `POST /translate` - Bidirectional translation endpoint
- `POST /generate` - Dialectal text continuation endpoint

---

## 30. Recommended Project Directory Hierarchy

```text
darija-llm-project/
├── data/
│   ├── raw/                  # Original raw datasets
│   ├── cleaned/              # Cleaned & filtered data
│   ├── processed/            # Tokenized & formatted data
│   └── splits/               # Train / Val / Test splits
├── scripts/
│   ├── clean_darija.py
│   ├── analyze_darija.py
│   ├── analyze_tokenization.py
│   ├── split_darija.py
│   ├── prepare_causal_lm.py
│   ├── train_stage1.py
│   ├── evaluate_darija.py
│   ├── clean_parallel.py
│   ├── split_parallel.py
│   ├── format_translation.py
│   ├── train_stage2.py
│   ├── evaluate_translation.py
│   ├── inference.py
│   └── benchmark.py
├── models/
│   ├── stage1/               # Stage 1 adapter checkpoints
│   └── stage2/               # Stage 2 adapter checkpoints
├── experiments/              # Structured experiment logs
├── evaluation/               # Prediction dumps & eval sets
├── api/                      # FastAPI deployment service
├── pyproject.toml            # Project dependencies & environment config
├── README.md                 # Primary project overview
└── .gitignore
```

---

## 31. End-to-End Execution Pipeline

```text
1. Environment Setup & Dependency Resolution (pyproject.toml / uv)
                     │
                     ▼
2. Ingest 160k Monolingual Darija Corpus & Tokenizer Diagnostics
                     │
                     ▼
3. Clean, Deduplicate, Split (90/5/5) & Benchmark Base Gemma
                     │
                     ▼
4. [STAGE 1] QLoRA Causal LM Training (Dialect Adaptation)
                     │
                     ▼
5. Evaluate Stage 1 Perplexity & Zero-Shot Translation Metrics
                     │
                     ▼
6. Ingest, Clean, and Format Parallel Translation Dataset (80/10/10)
                     │
                     ▼
7. [STAGE 2] Supervised Fine-Tuning on Translation Pairs
                     │
                     ▼
8. Evaluate Final Models (BLEU / chrF / COMET / Human Eval)
                     │
                     ▼
9. Execute Ablation Grid (Exp A, B, C, D) & Sample Scaling Tests
                     │
                     ▼
10. Package Adapters, Benchmark Inference & Deploy FastAPI Service
```

---

## 32. Expected Outcomes & Research Significance

This empirical sequence delivers a rigorous, repeatable evaluation answering whether unsupervised monolingual adaptation provides an efficient inductive bias for low-resource dialect translation, minimizing the necessity for expensive bilingual supervision.