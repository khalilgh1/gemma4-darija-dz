# Algerian Darija LLM Adaptation & Translation
## Commands Cheatsheet

---

## Table of Contents
1. [Project Workflow](#project-workflow)
2. [0. Project Setup](#0-project-setup)
3. [1. Install Dependencies](#1-install-dependencies)
4. [2. Verify the Environment](#2-verify-the-environment)
5. [3. Hugging Face Authentication](#3-hugging-face-authentication)
6. [4. Recommended Project Directory](#4-recommended-project-directory)
7. [5. Download / Prepare the Darija Dataset](#5-download--prepare-the-darija-dataset)
8. [6. Analyze the Darija Dataset](#6-analyze-the-darija-dataset)
9. [7. Analyze Gemma Tokenization](#7-analyze-gemma-tokenization)
10. [8. Clean the Darija Corpus](#8-clean-the-darija-corpus)
11. [9. Split the Darija Data](#9-split-the-darija-data)
12. [10. Establish the Base Darija Baseline](#10-establish-the-base-darija-baseline)
13. [11. Prepare Stage 1 Causal-LM Data](#11-prepare-stage-1-causal-lm-data)
14. [12. Stage 1 Sequence Length](#12-stage-1-sequence-length)
15. [13. Stage 1 QLoRA Configuration](#13-stage-1-qlora-configuration)
16. [14. Stage 1 Tiny Training Test](#14-stage-1-tiny-training-test)
17. [15. Configure Accelerate](#15-configure-accelerate)
18. [16. Stage 1 Full Training](#16-stage-1-full-training)
19. [17. Stage 1 Checkpoints](#17-stage-1-checkpoints)
20. [18. Evaluate Stage 1](#18-evaluate-stage-1)
21. [19. Prepare Stage 2 Parallel Data](#19-prepare-stage-2-parallel-data)
22. [20. Clean the Parallel Data](#20-clean-the-parallel-data)
23. [21. Split the Parallel Data](#21-split-the-parallel-data)
24. [22. Translation Baseline](#22-translation-baseline)
25. [23. Format Stage 2 Data](#23-format-stage-2-data)
26. [24. Stage 2 Tiny Training Test](#24-stage-2-tiny-training-test)
27. [25. Stage 2 Training](#25-stage-2-training)
28. [26. Stage 2 Evaluation](#26-stage-2-evaluation)
29. [27. BLEU Evaluation](#27-bleu-evaluation)
30. [28. chrF Evaluation](#28-chrf-evaluation)
31. [29. COMET Evaluation](#29-comet-evaluation)
32. [30. Human Evaluation](#30-human-evaluation)
33. [31. Ablation Experiments](#31-ablation-experiments)
34. [32. Parallel Data Size Experiment](#32-parallel-data-size-experiment)
35. [33. Error Analysis](#33-error-analysis)
36. [34. Experiment Tracking](#34-experiment-tracking)
37. [35. Save Model Artifacts](#35-save-model-artifacts)
38. [36. Single Inference](#36-single-inference)
39. [37. Batch Inference](#37-batch-inference)
40. [38. Inference Benchmark](#38-inference-benchmark)
41. [39. FastAPI Deployment](#39-fastapi-deployment)
42. [40. Docker](#40-docker)
43. [41. Version Control (Git)](#41-version-control-git)
44. [42. Final Evaluation Table](#42-final-evaluation-table)
45. [43. Complete Command Workflow Flowchart](#43-complete-command-workflow-flowchart)
46. [44. Stage 1 vs. Stage 2 Distinction](#44-stage-1-vs-stage-2-distinction)
47. [45. Final Research Goal & Verification](#45-final-research-goal--verification)

---

## Project Workflow

```text
BASE GEMMA
     │
     ▼
STAGE 1: Monolingual Algerian Darija Causal LM
     │
     ▼
DARIJA-ADAPTED MODEL
     │
     ▼
STAGE 2: High-Quality English ↔ Darija SFT
     │
     ▼
FINAL TRANSLATION MODEL
```

---

## 0. Project Setup

### Create Project Directory
```bash
mkdir darija-llm-project
cd darija-llm-project
```

### Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\activate

# Linux / WSL / macOS:
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

---

## 1. Install Dependencies

Install the core machine learning stack:
```bash
pip install torch transformers datasets peft trl accelerate bitsandbytes evaluate sacrebleu
```

*Optional packages:*
```bash
pip install sentencepiece
```

> **Note:** Install COMET separately (`unbabel-comet`) if utilized for the final translation evaluation.

---

## 2. Verify the Environment

```bash
# Check Python version
python --version

# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Check CUDA availability (Expected: True)
python -c "import torch; print(torch.cuda.is_available())"

# Check active GPU name
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Check CUDA version used by PyTorch
python -c "import torch; print(torch.version.cuda)"

# Monitor GPU (Single check / Continuous)
nvidia-smi
nvidia-smi -l 1
```

---

## 3. Hugging Face Authentication

If accessing gated models or datasets:
```bash
hf auth login
# Or on older setups:
huggingface-cli login
```

> **Security Reminder:** Never commit or hard-code access tokens inside source files.

---

## 4. Recommended Project Directory

```text
darija-llm-project/
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── processed/
│   └── splits/
│
├── scripts/
│   ├── clean_darija.py
│   ├── analyze_darija.py
│   ├── analyze_tokenization.py
│   ├── split_darija.py
│   ├── prepare_causal_lm.py
│   ├── train_stage1.py
│   ├── evaluate_darija.py
│   │
│   ├── clean_parallel.py
│   ├── split_parallel.py
│   ├── format_translation.py
│   ├── train_stage2.py
│   ├── evaluate_translation.py
│   │
│   ├── inference.py
│   └── benchmark.py
│
├── models/
│   ├── stage1/
│   └── stage2/
│
├── experiments/
├── evaluation/
├── api/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 5. Download / Prepare the Darija Dataset

The primary Stage 1 corpus contains approximately **160,000 rows**. First objective: determine actual token volume.

### Download via CLI or Python
```bash
hf download DATASET_REPOSITORY
```

```python
from datasets import load_dataset

dataset = load_dataset("DATASET_NAME")
print(dataset)
print(dataset["train"][0])
```

---

## 6. Analyze the Darija Dataset

```bash
python scripts/analyze_darija.py
```

**Key Metrics to Report:**
- Number of rows
- Total tokens (using target model's tokenizer)
- Average / Median tokens per row
- Min / Max length & percentiles
- Duplicate count
- Script distribution (Arabic script, Latin script, Arabizi)

---

## 7. Analyze Gemma Tokenization

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("MODEL_NAME")
text = "DARIJA_TEXT"
tokens = tokenizer.tokenize(text)
print(tokens)
print(f"Token count: {len(tokens)}")
```

Run full tokenization analysis:
```bash
python scripts/analyze_tokenization.py
```

**Metrics Evaluated:**
- Tokens per word / Tokens per character
- Sequence length & word fragmentation
- Script breakdown: Arabic script, Latin script, Arabizi

---

## 8. Clean the Darija Corpus

```bash
python scripts/clean_darija.py
```

### Pipeline Checks:
- [x] Empty rows
- [x] Exact & near duplicates
- [x] Corrupted text & extreme outliers
- [x] Irrelevant content & language inconsistencies

### Preserve Legitimate Dialect Variation:
- French code-switching
- Multi-script support (Arabic script, Latin script, Arabizi)
- Natural spelling variations

**Data Paths:**
- Raw data: `data/raw/`
- Cleaned data: `data/cleaned/`

---

## 9. Split the Darija Data

```bash
python scripts/split_darija.py
```

**Recommended Split:**
- **90%** Train
- **5%** Validation
- **5%** Test *(Must remain untouched)*

**Output Directory:** `data/splits/{train, validation, test}`

---

## 10. Establish the Base Darija Baseline

Evaluate the raw pretrained Gemma model:
```bash
python scripts/evaluate_darija.py --model BASE_MODEL
```

**Record Metrics:**
- Validation loss & Perplexity
- Generation quality samples
- Latency & Tokens/sec
- Peak VRAM footprint

---

## 11. Prepare Stage 1 Causal-LM Data

```bash
python scripts/prepare_causal_lm.py
```

**Sample format:**
```json
{
  "text": "Algerian Darija text..."
}
```

*Objective:* Next-token prediction ($t_1 \to t_2 \to t_3$). No translation labels required.

---

## 12. Stage 1 Sequence Length

- **Standard Target:** `max_length = 2048`
- **Fallback for 8GB VRAM (e.g. RTX 3070):** `max_length = 1024`
- Use sequence packing to avoid padding waste.

---

## 13. Stage 1 QLoRA Configuration

Using 4-bit quantized base model + LoRA adapters + mixed precision:
```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig
```
*(Base model weights remain frozen while adapter weights train).*

---

## 14. Stage 1 Tiny Training Test

Smoke test before long runs:
```bash
accelerate launch train_stage1.py --max_steps 20
```

**Verification Checklist:**
- [ ] Model & Tokenizer load properly
- [ ] CUDA allocation & forward pass succeed
- [ ] Loss computed & backward pass completes
- [ ] Checkpoint saves and reloads correctly
- [ ] Generation functions as expected

---

## 15. Configure Accelerate

```bash
# Configure local environment
accelerate config

# Verify environment details
accelerate env
```

---

## 16. Stage 1 Full Training

```bash
accelerate launch train_stage1.py
```

**Monitor:**
```bash
nvidia-smi -l 1
```

**Track:**
- Training/Validation Loss & Perplexity
- Tokens/sec & Peak VRAM
- Training Duration & Checkpoint Sizes

---

## 17. Stage 1 Checkpoints

```text
models/
└── stage1/
    ├── checkpoint-100/
    ├── checkpoint-200/
    ├── checkpoint-300/
    └── ...
```

---

## 18. Evaluate Stage 1

### Darija Generation & Perplexity
```bash
python scripts/evaluate_darija.py --checkpoint models/stage1/CHECKPOINT
```

### Pre-Stage 2 Translation Check
```bash
python scripts/evaluate_translation.py --checkpoint models/stage1/CHECKPOINT
```
*(Measures if monolingual adaptation yields zero-shot cross-lingual transfer).*

---

## 19. Prepare Stage 2 Parallel Data

- **Target:** High-quality English ↔ Algerian Darija pairs
- **Strategy:** Rigorous filtering over raw size (start with ~1,000 clean pairs; experiment with 100, 500, 1,000, 5,000).

---

## 20. Clean the Parallel Data

```bash
python scripts/clean_parallel.py
```

**Filtering Checklist:**
- Empty source/target or incorrect translation direction
- Untranslated text or language mismatches
- Length ratio anomalies & MT hallucinations
- Unnatural or distorted Darija

---

## 21. Split the Parallel Data

```bash
python scripts/split_parallel.py
```

**Split Ratio:**
- **80%** Train
- **10%** Validation
- **10%** Test *(Strictly held-out)*

---

## 22. Translation Baseline

```bash
# Base Gemma baseline
python scripts/evaluate_translation.py --model BASE_MODEL

# Stage 1 baseline
python scripts/evaluate_translation.py --checkpoint models/stage1/CHECKPOINT
```

---

## 23. Format Stage 2 Data

```bash
python scripts/format_translation.py
```

### Instruction Formats:

**English → Darija:**
```text
User: Translate this English sentence to Algerian Darija:
I am going to school.

Assistant:
[DARIJA TRANSLATION]
```

**Darija → English:**
```text
User: Translate this Algerian Darija sentence to English:
[DARIJA SENTENCE]

Assistant:
[ENGLISH TRANSLATION]
```

---

## 24. Stage 2 Tiny Training Test

```bash
accelerate launch train_stage2.py --max_steps 20
```

---

## 25. Stage 2 Training

Fine-tune starting from the **Stage 1 model**:
```bash
accelerate launch train_stage2.py
```
*(Watch for early signs of overfitting on small dataset sizes).*

---

## 26. Stage 2 Evaluation

```bash
python scripts/evaluate_translation.py --checkpoint models/stage2/CHECKPOINT
```

---

## 27. BLEU Evaluation

```bash
sacrebleu reference.txt -i prediction.txt
```

---

## 28. chrF Evaluation

```bash
sacrebleu reference.txt -i prediction.txt --chrf
```

---

## 29. COMET Evaluation

```bash
comet-score \
    -s source.txt \
    -t prediction.txt \
    -r reference.txt
```

---

## 30. Human Evaluation

Prepare comparison logs with schema:
`source` | `reference` | `base_prediction` | `stage1_prediction` | `stage2_prediction`

---

## 31. Ablation Experiments

| ID | Experiment Variant | Description |
| :--- | :--- | :--- |
| **A** | Base | Pretrained Gemma baseline |
| **B** | Monolingual adaptation | Base Gemma + Darija Causal LM (Stage 1) |
| **C** | Translation-only | Base Gemma + Parallel Translation SFT |
| **D** | Two-stage (Proposed) | Base Gemma + Stage 1 + Stage 2 SFT |

---

## 32. Parallel Data Size Experiment

Re-train Stage 2 with varying sample sizes (100, 500, 1,000, 5,000 pairs):
```bash
accelerate launch train_stage2.py --data_size 1000
python scripts/evaluate_translation.py --checkpoint CHECKPOINT
```

---

## 33. Error Analysis

```bash
python scripts/inference.py --input evaluation/test.txt --output evaluation/predictions.txt
```

**Error Taxonomy:** Grammar, Vocabulary, Spelling, Code-switching, Hallucinations, Literal translations, Direction confusion, Long sentence failures.

---

## 34. Experiment Tracking

```text
experiments/
└── exp-001/
    ├── config.json
    ├── metrics.json
    ├── generations.json
    └── notes.md
```

---

## 35. Save Model Artifacts

```text
models/
├── stage1/
│   ├── adapter/
│   └── config/
└── stage2/
    ├── adapter/
    └── config/
```

---

## 36. Single Inference

```bash
# Darija text generation (Stage 1)
python scripts/inference.py \
    --checkpoint models/stage1/adapter \
    --text "DARIJA_TEXT"

# English <-> Darija translation (Stage 2)
python scripts/inference.py \
    --checkpoint models/stage2/adapter \
    --task translation \
    --text "YOUR_TEXT"
```

---

## 37. Batch Inference

```bash
python scripts/inference.py --input input.txt --output predictions.txt
```

---

## 38. Inference Benchmark

```bash
python scripts/benchmark.py
```
*Measures first-token latency, tokens/sec throughput, and VRAM utilization across Base, Stage 1, and Stage 2.*

---

## 39. FastAPI Deployment

```bash
# Launch API server
uvicorn api:app --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health

# Translation request
curl -X POST http://localhost:8000/translate \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, how are you?", "direction": "en2dz"}'
```

---

## 40. Docker

```bash
# Build image
docker build -t darija-llm .

# Run container with GPU acceleration
docker run --gpus all -p 8000:8000 darija-llm
```

---

## 41. Version Control (Git)

```bash
git init
git status
git add .
git commit -m "Build Algerian Darija LLM adaptation pipeline"
```

> **Ignore Rules:** Exclude `.venv/`, `.parquet`, `.csv`, raw checkpoints, and API keys via `.gitignore`.

---

## 42. Final Evaluation Table

| Model | Darija PPL | BLEU | chrF | COMET |
| :--- | :---: | :---: | :---: | :---: |
| **Base Gemma** | - | - | - | - |
| **Stage 1 (Adapted)** | - | - | - | - |
| **Parallel-only (Direct SFT)** | - | - | - | - |
| **Stage 1 + Stage 2 (Full)** | - | - | - | - |

---

## 43. Complete Command Workflow Flowchart

```text
[Project Init & Env Setup]
         │
         ▼
[Acquire & Clean 160k Monolingual Corpus]
         │
         ▼
[Tokenizer Analysis & Base Darija Baseline]
         │
         ▼
[STAGE 1: QLoRA Causal LM Training]
         │
         ▼
[Evaluate Stage 1 (PPL & Zero-Shot Translation)]
         │
         ▼
[Filter & Prepare Parallel Translation Pairs]
         │
         ▼
[STAGE 2: Instruction SFT Training]
         │
         ▼
[Evaluate (BLEU, chrF, COMET, Human Eval)]
         │
         ▼
[Ablations, Benchmarking & FastAPI / Docker Deployment]
```

---

## 44. Stage 1 vs. Stage 2 Distinction

| Dimension | Stage 1 (Monolingual Adaptation) | Stage 2 (Translation SFT) |
| :--- | :--- | :--- |
| **Input** | Unlabeled Algerian Darija text | English ↔ Algerian Darija pairs |
| **Objective** | Causal next-token prediction | Supervised Instruction Fine-Tuning |
| **Dataset** | ~160k monolingual Darija corpus | High-quality filtered parallel pairs |
| **Primary Goal**| Domain & dialect language acquisition | Cross-lingual translation alignment |

---

## 45. Final Research Goal & Verification

The final experiment answers:
1. **Does monolingual Algerian Darija adaptation improve the model's ability to model and generate Algerian Darija?**
2. **Does prior monolingual adaptation make a small amount of English ↔ Algerian Darija parallel data significantly more sample-efficient?**