# Gemma 4 Algerian Darija (DZ) Adaptation & Bidirectional Translation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6+-EE4C2C.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Transformers%20%7C%20PEFT%20%7C%20TRL-yellow)](https://huggingface.co/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

Fine-tuning **Google Gemma 4 (`google/gemma-4-E2B`)** on **Algerian Arabic (Darija-DZ, `ar-DZ` / `ary`)** through an empirical comparison between a **Two-Stage Adaptation Pipeline** (Monolingual Causal LM $\rightarrow$ Translation SFT) and a **Direct Standalone SFT Baseline**.

---

## 📌 Table of Contents
1. [Project Overview & Research Questions](#-project-overview--research-questions)
2. [End-to-End Pipeline & Methodology](#-end-to-end-pipeline--methodology)
3. [Benchmark Results & Comparative Scores](#-benchmark-results--comparative-scores)
4. [Key Scientific & Empirical Findings](#-key-scientific--empirical-findings)
5. [Hardware & Engineering Lessons (T4 Post-Mortem)](#-hardware--engineering-lessons-t4-post-mortem)
6. [Repository Structure & Organization](#-repository-structure--organization)
7. [Repository Size Audit & Cleanup Guide](#-repository-size-audit--cleanup-guide)
8. [Quick Start & Inference Guide](#-quick-start--inference-guide)

---

## 🔬 Project Overview & Research Questions

Algerian Darija (`ary` / `ar-DZ`) is a low-resource Maghrebi Arabic dialect characterized by rich morphology, dialectal vocabulary, and frequent code-switching. Standard foundation LLMs struggle with Darija due to under-representation in massive multilingual pre-training corpora and lack of standardized orthography.

### Core Research Hypothesis
> **Hypothesis:** Continuing pre-training of a foundation LLM (Gemma 4) on a large, unlabeled monolingual Darija corpus (Stage 1) unlocks superior representation, vocabulary grounding, and sample efficiency when subsequently fine-tuned on bidirectional English $\leftrightarrow$ Darija translation pairs (Stage 2), compared to fine-tuning directly on translation pairs without dialect adaptation.

```text
                  PRETRAINED BASE GEMMA 4 (E2B)
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
┌─────────────────────────────────┐         ┌─────────────────────────────────┐
│ STAGE 1: Language Adaptation    │         │ [ABLATION BASELINE]             │
│ ~137k Cleaned Monolingual Rows  │         │ Direct Supervised Fine-Tuning   │
│ Causal Language Modeling (CLM)  │         │ No Monolingual Adaptation       │
└────────────────┬────────────────┘         └────────────────┬────────────────┘
                 │                                           │
                 ▼                                           │
┌─────────────────────────────────┐                          │
│ STAGE 2: Translation SFT        │                          │
│ ~47.5k Curated Parallel Pairs   │                          │
│ Bidirectional Instruction SFT   │                          │
└────────────────┬────────────────┘                          │
                 │                                           │
                 ▼                                           ▼
      [TWO-STAGE MODEL (S1+S2)]                   [STANDALONE MODEL (S2)]
```

---

## 🛠️ End-to-End Pipeline & Methodology

The project follows a rigorous 6-step experimental workflow:

### Step 1: Monolingual Corpus Curation & Script Filtering
- **Input:** `data/raw/v1.csv` (~168,653 raw scraped comments and conversational snippets).
- **Processing:** Script classification, normalization, regex-based deduplication, and removal of spam/bot repetitions (`notebooks/corpus-data-cleaning.ipynb`).
- **Output:**
  - `data/processed/arabic_v1.csv` (~137,000 cleaned rows of Arabic-script Darija).
  - `data/processed/latin_v1.csv` (~23,000 rows of Arabizi/Latin-script Darija).

### Step 2: Stage 1 Dialect Adaptation (Continued Pre-training)
- **Objective:** Causal Language Modeling (CLM) with cross-entropy loss on next-token prediction.
- **Base Model:** `google/gemma-4-E2B` quantized to 4-bit NormalFloat4 (NF4) with double quantization.
- **Adapter Configuration:** QLoRA rank $r=16$, $\alpha=32$, dropout $0.05$, targeting all linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Optimization:** Sequence length $L=512$, cosine learning rate schedule ($2\times 10^{-4}$), 8-bit AdamW.
- **Output Adapter:** `models/gemma4-darija-qlora` (`notebooks/stage1.ipynb`).

### Step 3: Parallel Dataset Cleaning & Validation
- **Input:** `data/raw/algerian_translation_50k.csv` (50,000 English $\leftrightarrow$ Algerian Darija sentence pairs).
- **Processing:** Filtered out empty translations, identical source-target copies, non-standard punctuation artifacts, and extreme length mismatches (`notebooks/translation-data-cleaning.ipynb`).
- **Output:** `data/processed/algerian_translation_50k_cleaned.csv` (~47,500 high-quality parallel pairs).
- **Splits:** 95% Train (~45,125 pairs), 5% Held-Out Test (~2,375 pairs). An identical 500-pair subset (random seed 42) was locked for rigorous comparative benchmarking.

### Step 4: Stage 2 Bidirectional Translation Fine-Tuning
- Continued training of the Stage 1 adapter on bidirectional instruction pairs (`notebooks/stage2.ipynb`):
  - **English $\rightarrow$ Darija**:
    ```text
    ترجم الجملة التالية من الإنجليزية إلى الدارجة الجزائرية:
    
    {sentence}
    
    الترجمة: {target}
    ```
  - **Darija $\rightarrow$ English**:
    ```text
    Translate the following Algerian Darija sentence to English:
    
    {sentence}
    
    Translation: {target}
    ```
- **Output Adapter:** `models/gemma4-darija-en-translation-qlora`.

### Step 5: Standalone SFT Baseline Training (Ablation)
- Fine-tuned base `google/gemma-4-E2B` **directly** on the exact same bidirectional dataset using identical hyper-parameters, random seeds, and prompt templates, bypassing Stage 1.
- **Output Adapter:** `models/gemma4-darija-en-translation-standalone-qlora`.

### Step 6: Blind Evaluation & Benchmarking
- Automated batched evaluation (`scripts/run_standalone_evaluation.py` and `notebooks/model-evaluation.ipynb`) computing SacreBLEU, chrF++, BERTScore, brevity penalties, and sentence-level deltas over the held-out 500 test sentences.

---

## 📊 Benchmark Results & Comparative Scores

All metrics were computed on the exact same held-out test split of 500 sentence pairs across both translation directions:

### 1. Overall Performance Metrics

| Translation Direction | Metric | Two-Stage (S1 + S2) | Standalone (Direct SFT) | Delta ($\Delta$) | Superior Variant |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **English $\rightarrow$ Algerian Darija** | **BLEU (SacreBLEU)** | 25.64 | **25.99** | +0.35 | **Standalone** |
| | **Brevity Penalty (BP)** | 0.7122 | **0.7134** | +0.0012 | **Standalone** |
| | **chrF++** | 45.33 | **45.70** | +0.37 | **Standalone** |
| | **BERTScore F1** | — | **84.56** | — | High semantic match |
| | **Length Ratio** | 0.747 | 0.748 | — | Balanced |
| **Algerian Darija $\rightarrow$ English** | **BLEU (SacreBLEU)** | 41.91 | **42.25** | +0.34 | **Standalone** |
| | **Brevity Penalty (BP)** | **0.7456** | 0.7395 | -0.0061 | Two-Stage |
| | **chrF++** | 54.82 | **55.18** | +0.36 | **Standalone** |
| | **BERTScore F1** | — | **100.00** | — | Near-perfect alignment |
| | **Length Ratio** | 0.773 | 0.768 | — | Balanced |

### 2. Sentence-Level chrF++ Win Rates (EN $\rightarrow$ Darija)

| Outcome | Sentence Count | Percentage |
| :--- | :---: | :---: |
| **Standalone Model Wins** | **200** | **40.0%** |
| **Two-Stage Model Wins** | 184 | 36.8% |
| **Ties** | 116 | 23.2% |

### 3. Qualitative Sample Comparison

| English Source | Reference (Darija) | Two-Stage Output | Standalone Output | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| *Mohamed's Center* | مركز محمد | ڨاع مركز محمد | ڨاع | Two-Stage kept colloquial prefix; Standalone under-generated. |
| *Thank you very much* | شكراً جزيلاً | ڨاع شكرا بزاف | **شكراً جزيلاً** | Standalone produced an exact 100% match; Two-Stage added "ڨاع". |
| *We will play* | رانا رايحين نلعبوا | **ڨاع رانا رايحين نلعبو** | ڨاع راح نلعبو | Two-Stage generated more natural Algerian verb morphology. |
| *Algeria is my mother.* | الجـزائر هي أمي. | الجزائر هي يما. | الجزائر هي يما. | Both correctly translated English "mother" into Darija "يما". |

---

## 💡 Key Scientific & Empirical Findings

### 1. The Dialect Adaptation Paradox (Negative/Neutral Transfer)
- **Finding:** Continued pre-training on unlabeled monolingual social-media text **did not provide a measurable boost** in downstream translation quality over direct SFT (+0.35 BLEU for Standalone).
- **Root Cause:**
  1. **Domain Noise & Artifact Injection:** Unfiltered social-media data introduced heavy conversational particles. In particular, the Two-Stage model learned to frequently prepend colloquial particles like *"ڨاع"* ("all" / "entirely") or filler phrases to translation outputs.
  2. **Sample Efficiency Threshold:** With ~47.5k high-quality parallel pairs (~95k instruction examples), direct SFT provides sufficient dialectal supervision for Gemma 4's strong base representation without needing noisy unsupervised pre-training.

### 2. Directional Performance Asymmetry
- Darija $\rightarrow$ English achieves significantly higher scores (**BLEU ~42.2 vs. ~26.0**) than English $\rightarrow$ Darija.
- **Explanation:** In Darija $\rightarrow$ English, the model decodes into standard English, leveraging Gemma's vast English pre-training. In English $\rightarrow$ Darija, the model must decode into a non-standardized dialect where multiple valid spellings, phonological variations, and dialectal synonyms compete.

### 3. chrF++ vs. BLEU for Dialectal Arabic
- Word-level BLEU penalizes valid dialectal morphological agglutination (e.g., prefixing particles like *بـ* "with", *لـ* "for", *ما...ش* negative circumfix).
- **chrF++** (character n-gram F-score with word order) correlates far better with human judgment for Algerian Darija because it rewards matching roots and subwords regardless of orthographic variations.

---

## ⚡ Hardware & Engineering Lessons (T4 Post-Mortem)

During Stage 2 fine-tuning on Kaggle NVIDIA Tesla T4 GPUs, critical hardware-level bottlenecks and bugs were analyzed and resolved:

### 1. Turing T4 Architecture & BF16 Trap
- **The Problem:** Setting `bf16=True` on Tesla T4 (Turing `sm_75`) projected **~49 hours** for 3,000 steps (~0.02 it/s).
- **Root Cause:** Turing GPUs **lack native BF16 tensor cores**. Enabling `bf16=True` forced PyTorch to emulate operations in software on CUDA cores, causing a 50-100x slowdown.
- **The Solution:** Use `fp16=True` with `compute_dtype=torch.float16`. This engages native FP16 tensor cores, reducing training time to ~1.2 hours.

### 2. PyTorch AMP GradScaler Crash
- **The Problem:** Training crashed with:
  `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`.
- **Root Cause:** Stage 1 LoRA adapter weights had been serialized in BF16. When loaded into an FP16 Trainer, the trainable parameters produced BF16 gradients during backpropagation. PyTorch's FP16 `GradScaler` does not support unscaling BF16 gradients.
- **The Solution:** Cast all trainable parameters to `torch.float32` immediately after PEFT initialization and enforce FP16 autocast via environment variable `ACCELERATE_MIXED_PRECISION="fp16"`.

### 3. Attention Cross-Contamination in Packed Sequences
- **The Problem:** Training with `packing=True` produced degraded translation coherence.
- **Root Cause:** In TRL's `SFTTrainer`, sequence packing without FlashAttention (which is unsupported on T4) relies on SDPA where attention masks can bleed across concatenated sample boundaries.
- **The Solution:** Disable packing (`packing=False`) and enable `group_by_length=True` to minimize padding waste without risking sample cross-contamination.

---

## 📁 Repository Structure & Organization

The repository is organized following clean, modular machine learning engineering standards:

```text
gemma4-darija-dz/
├── data/
│   ├── raw/                                      # Unprocessed raw datasets
│   │   ├── v1.csv                                # Raw scraped Darija text (~168k rows)
│   │   └── algerian_translation_50k.csv          # Raw parallel translation pairs (~50k rows)
│   └── processed/                                # Cleaned datasets used in training & testing
│       ├── arabic_v1.csv                         # Cleaned Arabic-script Darija (~137k rows)
│       ├── latin_v1.csv                          # Cleaned Arabizi/Latin-script Darija (~23k rows)
│       └── algerian_translation_50k_cleaned.csv  # Validated bidirectional pairs (~47.5k rows)
│
├── notebooks/                                    # Jupyter notebooks for interactive development
│   ├── corpus-data-cleaning.ipynb                # Raw text ingestion, script separation & cleaning
│   ├── stage1.ipynb                              # Stage 1 Continued Causal LM adaptation
│   ├── stage1-inference.ipynb                    # Qualitative evaluation of Stage 1 model
│   ├── translation-data-cleaning.ipynb           # Parallel dataset curation & cleaning
│   ├── stage2.ipynb                              # Stage 2 Bidirectional Translation fine-tuning
│   ├── stage2-inference.ipynb                    # Interactive translation generation & testing
│   └── model-evaluation.ipynb                    # Full comparative evaluation, win-rates & plots
│
├── scripts/                                      # Standalone execution scripts
│   ├── run_standalone_evaluation.py              # Automated batch evaluation pipeline
│   └── set_full_training.py                      # Training configuration helper
│
├── evaluation/                                   # Evaluation results & benchmark outputs
│   ├── comparative_evaluation_summary.json       # Top-level BLEU, chrF++, BERTScore metrics
│   └── comparison_evaluation_results.csv         # Complete 500-sample predictions and deltas
│
├── models/                                       # Trained LoRA adapters (lightweight weights)
│   ├── gemma4-darija-qlora/                      # Stage 1 Darija CLM adapter (~77 MB)
│   ├── gemma4-darija-en-translation-qlora/       # Stage 2 Two-Stage adapter (~123 MB)
│   └── gemma4-darija-en-translation-standalone-qlora/ # Standalone direct SFT adapter (~123 MB)
│
├── docs/                                         # Technical specifications & research notes
│   ├── sequence_plan.md                          # Research roadmap & experimental design
│   ├── command_cheatsheet.md                     # CLI cheatsheet & training commands
│   └── stage2-debugging-summary.md               # Hardware profiling & PyTorch AMP post-mortem
│
├── .gitignore                                    # Hardened gitignore (.venv, checkpoints, caches)
├── pyproject.toml                                # Project metadata & dependencies
└── README.md                                     # Project documentation & research summary
```

---

## 🚀 Quick Start & Inference Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/khalilgh1/gemma4-darija-dz.git
cd gemma4-darija-dz

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # On Windows
# source .venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -e .
```

### 2. Running Inference with the Trained Model

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "google/gemma-4-E2B"
# Select either the Standalone or Two-Stage adapter:
adapter_path = "models/gemma4-darija-en-translation-standalone-qlora"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=quant_config,
    device_map={"": 0} if torch.cuda.is_available() else None,
    torch_dtype=torch.float16,
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# English -> Algerian Darija
prompt = """ترجم الجملة التالية من الإنجليزية إلى الدارجة الجزائرية:

Welcome to Algeria, my friend!

الترجمة: """

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
translation = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
print("Darija Translation:", translation)
```

### 3. Running Automated Evaluation

To reproduce the benchmark metrics on the held-out test split:

```bash
python scripts/run_standalone_evaluation.py
```

Results will be automatically updated in `evaluation/comparison_evaluation_results.csv` and `evaluation/comparative_evaluation_summary.json`.

---

## 📜 Acknowledgements & Datasets

We gratefully acknowledge the creators of the open-source datasets utilized in this research:
- **Monolingual Darija Corpus**: [`ayoubkirouane/Algerian-Darija`](https://huggingface.co/datasets/ayoubkirouane/Algerian-Darija) on Hugging Face (crawled text filtered, script-separated, and cleaned into `data/processed/arabic_v1.csv`).
- **Parallel Translation Corpus**: [`touati-kamel/algerian-darja-corpus`](https://huggingface.co/datasets/touati-kamel/algerian-darja-corpus) on Hugging Face (English $\leftrightarrow$ Darija pairs validated, deduplicated, and cleaned into `data/processed/algerian_translation_50k_cleaned.csv`).

Both datasets underwent extensive preprocessing, normalization, and quality validation pipelines before training and evaluation.

---

## 📜 Citation & License

This project is licensed under the [Apache 2.0 License](LICENSE).  
For inquiries, contributions, or citations regarding Algerian Darija NLP adaptation, please submit an issue or pull request to this repository.

