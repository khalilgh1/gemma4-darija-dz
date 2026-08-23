# Translation Fine-Tuning Project — Command Cheatsheet

**Target setup:** TranslateGemma 4B + QLoRA on an RTX 3070 8 GB.

This cheatsheet follows the same order and titles as the sequence plan.

---

## 0. Define the project and success criteria

There are no mandatory shell commands here.

Before running anything, define:

```text
source language → target language
dataset/domain
success criteria
baseline metrics
VRAM/latency constraints
```

---

## 1. Set up the ML environment

### Create the project

```bash
mkdir translation-project
cd translation-project
```

**Does:** Creates and enters the project directory.

**Use:** At the beginning of the project.

### Create a virtual environment

```bash
python -m venv .venv
```

**Does:** Creates an isolated Python environment.

**Use:** Once when initializing the project.

### Activate — Windows PowerShell

```powershell
.venv\Scripts\activate
```

**Use:** Every time you open a new terminal for the project.

### Activate — Linux/macOS

```bash
source .venv/bin/activate
```

### Upgrade pip

```bash
python -m pip install --upgrade pip
```

**Does:** Updates Python's package manager.

**Use:** After creating the environment.

---

## 2. Acquire and inspect the dataset

### Hugging Face authentication

```bash
hf auth login
```

Alternative on older Hugging Face CLI versions:

```bash
huggingface-cli login
```

**Does:** Authenticates your Hugging Face account.

**Use:** For gated/private models or datasets, and when uploading artifacts.

### Download a Hugging Face repository

```bash
hf download REPOSITORY_NAME
```

**Does:** Downloads files from a Hugging Face repository.

**Use:** When you need a model/dataset locally.

Do not download a dataset blindly. First inspect its structure and licensing.

---

## 3. Build a reproducible data-cleaning pipeline

### Run the cleaning pipeline

```bash
python scripts/clean_dataset.py
```

**Does:** Runs your project's cleaning/filtering logic.

**Use:** After inspecting the raw dataset.

Typical output:

```text
data/
├── raw/
├── cleaned/
└── processed/
```

Keep `raw/` unchanged so your preprocessing remains reproducible.

---

## 4. Perform exploratory data analysis

### Run dataset analysis

```bash
python scripts/analyze_dataset.py
```

**Does:** Produces dataset statistics such as:

- sample count
- length distributions
- duplicates
- filtering statistics
- vocabulary information

**Use:** Before deciding your final preprocessing strategy.

### Inspect installed packages

```bash
pip list
```

**Does:** Lists installed Python packages and versions.

**Use:** Mainly for environment/debugging.

---

## 5. Split the dataset correctly

### Run your splitting pipeline

```bash
python scripts/split_dataset.py
```

**Does:** Creates train/validation/test datasets.

**Use:** After cleaning and before model training.

Keep the test set untouched after this point.

---

## 6. Establish the baseline

### Run baseline evaluation

```bash
python scripts/evaluate.py --model BASE_MODEL
```

**Does:** Runs the base TranslateGemma model against your evaluation data.

**Use:** Before fine-tuning.

Record:

```text
BLEU
chrF
COMET
latency
tokens/sec
VRAM
```

The exact `--model` value depends on the model identifier you choose.

---

## 7. Understand the model before training

### Inspect the tokenizer

```bash
python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('MODEL'); print(t)"
```

**Does:** Loads and prints tokenizer information.

**Use:** When investigating tokenization behavior.

### Verify model loading

Your project can expose a small test script such as:

```bash
python scripts/test_model.py
```

**Does:** Confirms that the model and tokenizer load successfully.

**Use:** Before configuring training.

---

## 8. Format the dataset for instruction tuning

### Run dataset formatting

```bash
python scripts/format_dataset.py
```

**Does:** Converts cleaned parallel data into the exact training format expected by the model.

**Use:** After cleaning/splitting and before tokenization/training.

The important requirement is that the training prompt format and inference prompt format remain consistent.

---

## 9. Understand QLoRA before using it

There is no single command that "turns on QLoRA."

Your training code will configure concepts such as:

```text
BitsAndBytesConfig
LoraConfig
PEFT
4-bit quantization
```

The main package responsible for LoRA/PEFT is:

```bash
pip install peft bitsandbytes
```

**Use:** During environment setup if not already installed.

---

## 10. Configure the training experiment

### Configure Hugging Face Accelerate

```bash
accelerate config
```

**Does:** Configures how Accelerate launches your training.

**Use:** Before your first Accelerate-based training run.

For your machine, the important characteristic is:

```text
1 GPU
```

### Know the main training components

Your training script will generally use:

```text
BitsAndBytesConfig
LoraConfig
TrainingArguments
SFTTrainer
AutoModelForCausalLM
AutoTokenizer
```

Learn the role of each instead of memorizing the syntax.

---

## 11. Run a tiny training experiment first

### Launch a short test run

```bash
accelerate launch train.py --max_steps 20
```

**Does:** Launches training through Accelerate for a very small number of steps.

**Use:** Before committing to a long training run.

Check:

```text
CUDA OOM?
Loss calculated?
Backward pass works?
Checkpoint created?
Validation works?
Model reloads?
```

---

## 12. Monitor training

### Show GPU status

```bash
nvidia-smi
```

**Does:** Shows GPU utilization, VRAM usage, temperature, processes, driver information, etc.

**Use:** Anytime you want a snapshot of GPU usage.

### Continuously monitor GPU

```bash
nvidia-smi -l 1
```

**Does:** Refreshes GPU information every second.

**Use:** In a second terminal while training or running inference.

### Check whether PyTorch sees CUDA

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected:

```text
True
```

### Show GPU name

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Show the CUDA version used by PyTorch

```bash
python -c "import torch; print(torch.version.cuda)"
```

---

## 13. Experiment systematically

### Launch a training experiment

```bash
accelerate launch train.py
```

**Does:** Starts your configured training run.

**Use:** Once the tiny test succeeds.

Keep experiment configurations reproducible.

A useful naming pattern is:

```text
experiments/
├── exp-001/
├── exp-002/
└── exp-003/
```

Record:

```text
model
dataset version
LoRA parameters
learning rate
batch size
gradient accumulation
epochs
metrics
```

---

## 14. Evaluate the fine-tuned model

### Evaluate a checkpoint

```bash
python evaluate.py --checkpoint PATH
```

**Does:** Evaluates the fine-tuned checkpoint.

**Use:** After training and whenever you want to compare checkpoints.

### BLEU with SacreBLEU

```bash
sacrebleu reference.txt -i prediction.txt
```

**Does:** Calculates BLEU using SacreBLEU.

**Use:** When you have a reference translation file and model predictions.

### chrF

```bash
sacrebleu reference.txt -i prediction.txt --chrf
```

**Does:** Calculates chrF.

**Use:** Alongside BLEU for complementary MT evaluation.

---

## 15. Perform error analysis

### Generate predictions for inspection

```bash
python inference.py --input test.txt --output predictions.txt
```

**Does:** Produces model translations for a collection of inputs.

**Use:** When manually inspecting errors and preparing evaluation reports.

Create an analysis dataset containing:

```text
source
reference
baseline prediction
fine-tuned prediction
error category
notes
```

There is no universal shell command for human error analysis; this is an analysis workflow.

---

## 16. Compare against the baseline

### Run your evaluation pipeline

```bash
python evaluate.py --model BASE_MODEL
python evaluate.py --checkpoint FINE_TUNED_CHECKPOINT
```

**Does:** Produces directly comparable baseline and fine-tuned results.

**Use:** To determine whether fine-tuning actually helped.

Compare:

```text
BLEU
chrF
COMET
latency
tokens/sec
VRAM
```

Use the same test set and evaluation procedure for both.

---

## 17. Save and package the model

### Inspect Git status

```bash
git status
```

**Does:** Shows modified/untracked files.

**Use:** Before committing project changes.

### Initialize Git

```bash
git init
```

**Does:** Creates a Git repository.

**Use:** Once when starting version control.

### Stage files

```bash
git add .
```

**Does:** Stages changes for a commit.

**Use:** After reviewing what should be committed.

### Commit

```bash
git commit -m "Initial translation fine-tuning pipeline"
```

**Does:** Creates a versioned snapshot of your source code/configuration.

**Use:** After meaningful project changes.

Do not commit:

```text
.venv/
large datasets
model weights
checkpoints
API tokens/secrets
```

unless you intentionally use an appropriate artifact-storage strategy.

---

## 18. Build an inference pipeline

### Single-text inference

```bash
python inference.py --text "YOUR TEXT"
```

**Does:** Generates a translation for one input.

**Use:** Quick manual testing.

### Batch inference

```bash
python inference.py --input input.txt --output translations.txt
```

**Does:** Translates a file of inputs.

**Use:** Evaluation, testing, or batch processing.

---

## 19. Optimize inference

### Benchmark inference

```bash
python benchmark.py
```

**Does:** Measures your inference system.

Track:

```text
model loading time
first-token latency
total latency
tokens/sec
VRAM
```

### Monitor GPU while benchmarking

```bash
nvidia-smi -l 1
```

**Use:** Determine the actual GPU/VRAM cost of inference.

---

## 20. Serve the model

### Start a FastAPI server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Does:** Starts your translation API.

**Use:** Once local inference works reliably.

### Test the health endpoint

```bash
curl http://localhost:8000/health
```

**Does:** Checks whether the service is reachable.

**Use:** Basic service testing and later health checks.

### Test translation endpoint

```bash
curl -X POST http://localhost:8000/translate
```

**Does:** Sends a POST request to your translation endpoint.

**Use:** API testing.

The exact request body depends on your API implementation.

---

# Additional GPU/Environment Commands

These are useful throughout the project.

## Check NVIDIA driver/GPU

```bash
nvidia-smi
```

Use when:

- CUDA isn't working
- VRAM is unexpectedly full
- training is slow
- you suspect another process is using the GPU

## Continuously monitor GPU

```bash
nvidia-smi -l 1
```

Use during:

- training
- inference
- benchmarking

## Check PyTorch CUDA

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Use when debugging CUDA/PyTorch.

## Check PyTorch version

```bash
python -c "import torch; print(torch.__version__)"
```

## Check Transformers version

```bash
python -c "import transformers; print(transformers.__version__)"
```

## Check PEFT version

```bash
python -c "import peft; print(peft.__version__)"
```

These are useful when reproducing experiments or diagnosing dependency conflicts.

---

# Final Project Flow

The entire command-oriented workflow is:

```text
mkdir translation-project
        ↓
python -m venv .venv
        ↓
activate environment
        ↓
pip install ...
        ↓
nvidia-smi
        ↓
verify PyTorch + CUDA
        ↓
download/prepare dataset
        ↓
analyze_dataset.py
        ↓
clean_dataset.py
        ↓
split_dataset.py
        ↓
format_dataset.py
        ↓
evaluate.py --model BASE_MODEL
        ↓
accelerate config
        ↓
tiny QLoRA training
        ↓
nvidia-smi -l 1
        ↓
full training
        ↓
evaluate.py --checkpoint ...
        ↓
BLEU / chrF / COMET
        ↓
error analysis
        ↓
compare with baseline
        ↓
package adapter + configuration
        ↓
inference.py
        ↓
benchmark.py
        ↓
FastAPI
        ↓
Docker
```

The key principle is:

**Don't optimize for getting the model trained as quickly as possible. Optimize for understanding every stage of the pipeline and being able to explain why you made each engineering decision.**
