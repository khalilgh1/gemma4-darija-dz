from fastapi import FastAPI
from typing import Literal
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

model_id = "google/gemma-4-E2B"
# we will use the standalone model as the default one
adapter_path = "./models/gemma4-darija-en-translation-standalone-qlora"
compute_dtype = torch.float16

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map={"":0},
    dtype=compute_dtype,
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_path
)
model.eval()




from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

EN_TO_AR_TEMPLATE = """ترجم الجملة التالية من الإنجليزية إلى الدارجة الجزائرية:

{sentence}

الترجمة: """

AR_TO_EN_TEMPLATE = """Translate the following Algerian Darija sentence to English:

{sentence}

Translation: """

app = FastAPI(title="Gemma 4 Algerian Darija Translation API")

# Enable CORS for easy consumption from web interfaces (e.g. localhost, frontend dev servers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    direction: Literal["en_dz", "dz_en"] = Field(
        "en_dz",
        description="'en_dz' (English to Darija) or 'dz_en' (Darija to English)",
    )


class TranslationResponse(BaseModel):
    original: str
    direction: str
    translation: str


def run_model_inference(prompt: str, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens
    gen_tokens = outputs[0][prompt_len:]
    translation = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # Clean up any trailing repetitions or extra blank sections if present
    if "\n\n" in translation:
        translation = translation.split("\n\n")[0].strip()

    return translation


def clean_darija_translation(text: str) -> str:
    """
    Remove irrelevant leading 'ڨاع' prefix often added by conversational fine-tuning artifacts.
    If the response sentence is only one word 'ڨاع', do not remove it.
    """
    cleaned = text.strip()
    words = cleaned.split()
    if len(words) > 1 and words[0] == "ڨاع":
        return " ".join(words[1:]).strip()
    return cleaned


def translate_en_dz(text: str) -> str:
    """Translate text from English to Algerian Darija."""
    prompt = EN_TO_AR_TEMPLATE.format(sentence=text.strip())
    raw_translation = run_model_inference(prompt)
    return clean_darija_translation(raw_translation)


import re

def has_arabic(text: str) -> bool:
    """Return True if text contains Arabic/Darija script characters."""
    return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", text))


def translate_dz_en(text: str) -> str:
    """
    Translate text from Algerian Darija to English.
    If the base generation occasionally loops or mirrors in Arabic/Darija,
    it automatically conditions generation on an English start token (`<em>`)
    which reliably steers Gemma 4 into English output.
    """
    prompt = AR_TO_EN_TEMPLATE.format(sentence=text.strip())
    raw_translation = run_model_inference(prompt)

    # If output still contains Arabic script, steer the generation with English prefix conditioning
    if has_arabic(raw_translation):
        steered_prompt = (
            f"Translate the following Algerian Darija sentence to English:\n\n"
            f"{text.strip()}\n\n"
            f"Translation: <em>"
        )
        retry_translation = run_model_inference(steered_prompt)
        if retry_translation and not has_arabic(retry_translation):
            raw_translation = retry_translation

    # Strip conversational tags (<em>, <strong>, etc.) and clean leading artifacts
    raw_translation = re.sub(r"</?[a-zA-Z0-9]+>", "", raw_translation).strip()
    return raw_translation





from fastapi.responses import FileResponse

@app.get("/")
def root():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Gemma 4 Algerian Darija Translation Backend is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model": "google/gemma-4-E2B", "adapter": adapter_path}



@app.get("/translate", response_model=TranslationResponse)
def translate_get(text: str, direction: Literal["en_dz", "dz_en"] = "en_dz"):
    """
    Translate text from English to Algerian Darija or vice versa (GET endpoint).
    :param text: Text to translate
    :param direction: "en_dz" for English to Algerian Darija, "dz_en" for Algerian Darija to English
    """
    if direction == "en_dz":
        translated = translate_en_dz(text)
    else:
        translated = translate_dz_en(text)

    return TranslationResponse(
        original=text,
        direction=direction,
        translation=translated,
    )


@app.post("/translate", response_model=TranslationResponse)
def translate_post(payload: TranslationRequest):
    """
    Translate text from English to Algerian Darija or vice versa (POST endpoint).
    Ideal for web clients and longer text submissions.
    """
    if payload.direction == "en_dz":
        translated = translate_en_dz(payload.text)
    else:
        translated = translate_dz_en(payload.text)

    return TranslationResponse(
        original=payload.text,
        direction=payload.direction,
        translation=translated,
    )