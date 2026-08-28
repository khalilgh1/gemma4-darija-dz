# Gemma 4 Algerian Darija (DZ) Adaptation & Translation

Fine-tuning Google Gemma on **Algerian Arabic (Darija-DZ)** via a two-stage adaptation strategy: **Monolingual Causal Language Modeling** followed by **Supervised Translation Fine-Tuning**.

---

## 📌 Project Overview

Algerian Darija (`ary` / `ar-DZ`) is a low-resource Maghrebi Arabic dialect characterized by distinct phonetic features, rich morphology, and frequent code-switching with French and Berber. 

This repository evaluates whether adapting a foundation LLM (Gemma) on a large unlabeled monolingual Darija corpus unlocks high sample efficiency when subsequently fine-tuning on a small set of high-quality English $\leftrightarrow$ Darija translation pairs.

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

## 📚 Documentation & Research Roadmap

- **[Sequence Plan](file:///c:/Users/omen/Desktop/projects/gemma4-darija-dz/sequence_plan.md)**: Detailed experimental design, research hypotheses, mathematical formulation, ablation studies, and evaluation protocols.
- **[Command Cheatsheet](file:///c:/Users/omen/Desktop/projects/gemma4-darija-dz/command_cheatsheet.md)**: Comprehensive CLI reference, training commands, data preparation workflows, evaluation scripts, and deployment guides.

---

## ⚡ Quick Start

### 1. Prerequisites & Environment Setup

This project uses modern Python packaging via `uv` or standard virtual environments:

```bash
# Clone the repository
git clone https://github.com/khalilgh1/gemma4-darija-dz.git
cd gemma4-darija-dz

# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate

# Install dependencies (configured in pyproject.toml)
pip install -e .
```

### 2. Verify GPU & Environment

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 🔬 Experimental Matrix & Ablations

| Model Variant | Stage 1 (Monolingual CLM) | Stage 2 (Translation SFT) | Purpose |
| :--- | :---: | :---: | :--- |
| **Model A (Base)** | ❌ | ❌ | Zero-shot baseline |
| **Model B (Stage 1)** | ✅ | ❌ | Evaluate dialect perplexity & unsupervised transfer |
| **Model C (Direct SFT)** | ❌ | ✅ | Baseline supervised fine-tuning |
| **Model D (Two-Stage)** | ✅ | ✅ | Full two-stage proposed methodology |

---

## 📊 Evaluation Metrics

- **Perplexity ($\text{PPL}$):** Language modeling capability on held-out Darija test splits.
- **SacreBLEU:** Precision-based translation evaluation.
- **chrF++:** Character n-gram F-score suited for morphologically complex dialects.
- **COMET:** Neural cross-lingual semantic evaluation.
- **Human Evaluation:** Blind pairwise review on grammar, naturalness, and meaning preservation.

---

## 📁 Repository Structure

```text
gemma4-darija-dz/
├── data/                         # Monolingual & parallel datasets
├── scripts/                      # Data processing, training & evaluation scripts
├── models/                       # LoRA adapter weights & checkpoints
├── experiments/                  # Experiment configs, metrics & generation logs
├── evaluation/                   # Evaluation splits & test predictions
├── api/                          # FastAPI serving layer
├── pyproject.toml                # Project metadata & dependencies
├── sequence_plan.md              # In-depth research & sequence plan
├── command_cheatsheet.md         # Command & execution cheatsheet
└── README.md                     # Project overview & quickstart
```

---

## 📜 Citation & License

Under active development for Algerian Darija NLP research.
