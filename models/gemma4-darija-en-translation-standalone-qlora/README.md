---
base_model: google/gemma-4-E2B
library_name: peft
model_name: gemma4-darija-en-translation-standalone-qlora
tags:
- base_model:adapter:google/gemma-4-E2B
- lora
- sft
- transformers
- trl
licence: license
pipeline_tag: text-generation
---

# Model Card for gemma4-darija-en-translation-standalone-qlora

This model is a fine-tuned version of [google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 



This model was trained with SFT.

### Framework versions

- PEFT 0.19.1
- TRL: 1.13.0
- Transformers: 5.16.1
- Pytorch: 2.10.0+cu128
- Datasets: 5.0.0
- Tokenizers: 0.23.2

## Citations



Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```