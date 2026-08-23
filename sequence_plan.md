# Translation Fine-Tuning Project — Sequence Plan

**Target setup:** TranslateGemma 4B + QLoRA on an RTX 3070 8 GB.

The goal is not just to fine-tune a model. The project is structured to practice real AI engineering concepts: data engineering, reproducible experiments, GPU/quantization, PEFT, evaluation, error analysis, inference optimization, and deployment.

---

## 0. Define the project and success criteria

Define:

- Source language → target language
- Dataset/domain
- What constitutes a good translation
- Maximum acceptable inference latency
- GPU/VRAM constraints
- Evaluation metrics

Establish a baseline:

```text
Baseline:
TranslateGemma 4B without fine-tuning

Fine-tuned:
TranslateGemma 4B + QLoRA

Goal:
Fine-tuned model should outperform the base model on held-out data.
```

### Why this matters

You need a baseline before claiming your model improved.

---

## 1. Set up the ML environment

Create an isolated Python environment and install:

- PyTorch with CUDA
- Hugging Face Transformers
- Datasets
- PEFT
- TRL
- Accelerate
- BitsAndBytes
- Evaluate
- SacreBLEU
- optionally COMET

Verify:

```text
Python
↓
PyTorch
↓
CUDA
↓
RTX 3070
```

### What you're learning

- CUDA
- GPU memory
- PyTorch CUDA tensors
- mixed precision
- quantization
- dependency management

---

## 2. Acquire and inspect the dataset

Use a parallel corpus:

```text
source sentence          target sentence
------------------------------------------------
Bonjour                  Hello
Je suis étudiant         I am a student
...
```

Investigate:

- number of examples
- languages
- duplicate sentences
- missing translations
- extremely long examples
- corrupted examples
- wrong language
- inconsistent punctuation
- HTML/noisy text
- train/test leakage

### Key lesson

**Data quality is often more important than model choice.**

---

## 3. Build a reproducible data-cleaning pipeline

Implement:

### Filtering

Remove:

- empty examples
- extremely short examples
- extremely long examples
- obviously corrupted examples

### Deduplication

Remove identical or near-identical examples.

### Language validation

Verify source and target language.

### Length-ratio filtering

Investigate suspicious alignments, e.g.:

```text
source: 10 tokens
target: 200 tokens
```

### Normalization

Normalize carefully without destroying translation-relevant information.

---

## 4. Perform exploratory data analysis

Calculate:

- number of samples
- source/target token lengths
- length distributions
- vocabulary statistics
- duplicate percentage
- language distribution
- percentage removed during cleaning

Inspect random examples manually.

### Key question

> What exactly is my model being trained on?

---

## 5. Split the dataset correctly

Create:

```text
train
validation
test
```

A starting split can be:

```text
80% train
10% validation
10% test
```

Avoid leakage. If examples come from documents, prefer document-level splitting:

```text
Document A → train
Document B → validation
Document C → test
```

Do not tune the model using the test set.

---

## 6. Establish the baseline

Run TranslateGemma before fine-tuning.

Record:

```text
BLEU
chrF
COMET
inference latency
tokens/sec
VRAM usage
```

Save actual:

```text
Input
Reference
Model output
```

This gives you a meaningful baseline for later comparison.

---

## 7. Understand the model before training

Study:

- architecture
- tokenizer
- context length
- parameter count
- attention
- embeddings
- transformer blocks
- precision
- quantization

Understand the high-level pipeline:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embeddings
 ↓
Transformer layers
 ↓
Logits
 ↓
Next-token probabilities
 ↓
Generated translation
```

---

## 8. Format the dataset for instruction tuning

Format examples according to the model's expected conversational/instruction format.

Conceptually:

```text
User:
Translate the following Arabic text to French:

...

Assistant:
...
```

### Key lesson

The training format affects model behavior.

Your inference format should be consistent with the format used during training.

---

## 9. Understand QLoRA before using it

With an RTX 3070 8 GB, use QLoRA rather than attempting an unrestricted full fine-tune.

Conceptually:

```text
4-bit quantized base model
+
LoRA adapters
+
mixed precision
```

Architecture:

```text
                 ┌─────────────────────┐
                 │ Quantized base model│
                 │      frozen         │
                 └──────────┬──────────┘
                            │
                            ▼
                     Transformer
                            │
                 ┌──────────┴──────────┐
                 │    LoRA adapters     │
                 │     trainable        │
                 └──────────────────────┘
```

Learn:

- quantization
- 4-bit weights
- LoRA
- rank `r`
- alpha
- dropout
- target modules
- PEFT
- adapter weights

Do not treat the configuration as magic numbers.

---

## 10. Configure the training experiment

Specify:

```text
quantization
precision
LoRA configuration
learning rate
batch size
gradient accumulation
number of epochs
sequence length
optimizer
scheduler
checkpoint strategy
evaluation frequency
```

Understand effective batch size:

```text
effective batch size =
batch size × gradient accumulation × number of GPUs
```

For your single-GPU setup:

```text
effective batch size =
batch size × gradient accumulation
```

---

## 11. Run a tiny training experiment first

Do not immediately run a multi-hour training job.

Run:

```text
small dataset
few steps
```

Validate:

```text
Dataset
 ↓
Tokenizer
 ↓
Collator
 ↓
Model
 ↓
QLoRA
 ↓
Forward pass
 ↓
Backward pass
 ↓
Optimizer
 ↓
Checkpoint
```

Check:

- no CUDA OOM
- loss is calculated
- loss starts behaving sensibly
- checkpoints work
- validation works
- model can be loaded afterward

### Key lesson

**Validate the pipeline before spending compute.**

---

## 12. Monitor training

Track:

### Training loss

Should generally decrease.

### Validation loss

Useful for detecting overfitting.

### GPU utilization

Check whether the GPU is being used efficiently.

### VRAM

Important because your GPU has only 8 GB.

### Training throughput

Measure:

```text
samples/sec
tokens/sec
```

### Checkpoints

Save periodically so interrupted training can be resumed.

---

## 13. Experiment systematically

Change one or a small number of variables at a time.

Example:

```text
Experiment A:
LoRA r=8
LR=2e-4

Experiment B:
LoRA r=16
LR=2e-4

Experiment C:
LoRA r=16
LR=1e-4
```

Maintain an experiment table:

| Experiment | LoRA r | LR | Epochs | BLEU | chrF | COMET |
|---|---:|---:|---:|---:|---:|---:|
| baseline | — | — | — | X | X | X |
| exp-1 | 8 | 2e-4 | 1 | X | X | X |
| exp-2 | 16 | 2e-4 | 1 | X | X | X |
| exp-3 | 16 | 1e-4 | 2 | X | X | X |

### Key lesson

Do not change ten things simultaneously. You want experiments whose results you can interpret.

---

## 14. Evaluate the fine-tuned model

Evaluate on the same held-out test set used for the baseline.

Use multiple metrics.

### BLEU

A traditional machine translation metric.

### chrF

Character-level metric that can provide a useful complementary view, especially for morphologically rich languages.

### COMET

A learned MT evaluation metric that provides a different perspective from surface-overlap metrics.

### Human evaluation

Inspect samples manually:

```text
Reference
Baseline
Fine-tuned
```

Look for:

- meaning preservation
- grammar
- fluency
- terminology
- hallucinations
- omissions
- unnecessary additions

### Key lesson

**A metric is not the same thing as quality.**

---

## 15. Perform error analysis

Categorize failures:

```text
1. terminology
2. grammar
3. missing information
4. hallucination
5. named entities
6. long sentences
7. ambiguous expressions
8. domain-specific language
```

Analyze performance by category.

For example:

```text
Short sentences       → 91%
Long sentences        → 72%
Technical terminology → 68%
General text          → 94%
```

Now you know where the model actually fails.

---

## 16. Compare against the baseline

Answer:

```text
Did fine-tuning improve translation?

By how much?

On what type of data?

Did it hurt anything?

Did inference become slower?

How much GPU memory does it use?

How large is the adapter?
```

This is the core of experimental AI engineering.

---

## 17. Save and package the model

Create a clean artifact:

```text
model/
├── adapter/
├── tokenizer/
├── config/
├── README
└── evaluation_results
```

Document:

- base model
- dataset
- preprocessing
- training configuration
- evaluation metrics
- limitations

### Key lesson

Treat models as reproducible artifacts, not just folders of checkpoints.

---

## 18. Build an inference pipeline

Create:

```text
Input text
     ↓
Tokenizer
     ↓
Model + LoRA adapter
     ↓
Generation
     ↓
Decoded translation
```

Test:

- short sentences
- long sentences
- unknown terminology
- different domains
- malformed input
- empty input

Measure:

```text
latency
tokens/sec
VRAM
```

---

## 19. Optimize inference

Experiment with:

- 4-bit loading
- generation parameters
- batch inference
- caching
- maximum sequence length
- prompt formatting

Understand the tradeoff:

```text
quality
     ↕
latency
     ↕
memory
```

---

## 20. Serve the model

Turn the model into an actual service:

```text
Client
  │
  ▼
API
  │
  ▼
Translation service
  │
  ▼
Model
  │
  ▼
Translation
```

For example:

```text
POST /translate
```

with a request such as:

```json
{
  "text": "..."
}
```

Explore:

- FastAPI
- Docker
- GPU containers
- batching
- health checks
- logging
- latency monitoring

At this point the project has evolved from:

> "I fine-tuned a model"

into:

> "I built an AI inference service."
