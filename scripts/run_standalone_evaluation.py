"""
Standalone Gemma 4 translation evaluation.
Runs batched inference (batch_size=16) on the exact same 500 test set pairs,
computes BLEU, chrF++, BERTScore, brevity penalty, and outputs comparative summaries.
"""
import os
import sys
import gc
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import sacrebleu

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.model_selection import train_test_split

USE_BERTSCORE = True
try:
    import bert_score
    print("✓ bert_score is available.")
except ImportError:
    USE_BERTSCORE = False

model_id = "google/gemma-4-E2B"
# Detect paths whether run from root or from scripts/ directory
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) if os.path.basename(os.getcwd()) == "scripts" else os.getcwd()
standalone_adapter_path = os.path.join(REPO_ROOT, "models", "gemma4-darija-en-translation-standalone-qlora")
if not os.path.exists(standalone_adapter_path):
    standalone_adapter_path = os.path.join(REPO_ROOT, "gemma4-darija-en-translation-standalone-qlora")

compute_dtype = torch.float16

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print(f"Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print(f"Loading 4-bit base model {model_id}...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map={"": 0} if torch.cuda.is_available() else None,
    dtype=compute_dtype,
)

print(f"Loading Standalone LoRA adapter from {standalone_adapter_path}...")
model = PeftModel.from_pretrained(base_model, standalone_adapter_path)
model.eval()
print("✓ Standalone Model ready for evaluation.")

EN_TO_AR_TEMPLATE = """ترجم الجملة التالية من الإنجليزية إلى الدارجة الجزائرية:

{sentence}

الترجمة: """

AR_TO_EN_TEMPLATE = """Translate the following Algerian Darija sentence to English:

{sentence}

Translation: """

def generate_translations_batch(sentences, direction="en_to_ar", batch_size=16, max_new_tokens=128):
    if direction == "en_to_ar":
        prompts = [EN_TO_AR_TEMPLATE.format(sentence=s) for s in sentences]
    elif direction == "ar_to_en":
        prompts = [AR_TO_EN_TEMPLATE.format(sentence=s) for s in sentences]
    else:
        raise ValueError("Invalid direction.")

    translations = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    
    for b_idx in range(total_batches):
        batch_prompts = prompts[b_idx * batch_size : (b_idx + 1) * batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        for out in outputs:
            gen_tokens = out[inputs["input_ids"].shape[1]:]
            decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            decoded = decoded.split("\n\n")[0].strip()
            translations.append(decoded)
            
        if (b_idx + 1) % 5 == 0 or (b_idx + 1) == total_batches:
            print(f"[{direction}] Completed batch {b_idx + 1}/{total_batches} ({len(translations)}/{len(prompts)})")
            
    return translations

# Load dataset and extract identical 500 samples
csv_path = os.path.join(REPO_ROOT, "data", "processed", "algerian_translation_50k_cleaned.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join(REPO_ROOT, "algerian_translation_50k_cleaned.csv")
df_raw = pd.read_csv(csv_path).dropna(subset=["Arabic", "English"])
df_raw["Arabic"] = df_raw["Arabic"].astype(str).str.strip()
df_raw["English"] = df_raw["English"].astype(str).str.strip()
df_clean = df_raw[(df_raw["Arabic"] != "") & (df_raw["English"] != "")].reset_index(drop=True)

_, df_test = train_test_split(df_clean, test_size=0.05, random_state=42)
eval_df = df_test.sample(n=500, random_state=42).reset_index(drop=True)
print(f"Prepared test set: {len(eval_df)} sentence pairs.")

# Direction 1: English -> Algerian Darija (en_to_ar)
print("\n--- Starting Generation: English -> Algerian Darija ---")
sources_en = eval_df["English"].tolist()
preds_ar_standalone = generate_translations_batch(sources_en, direction="en_to_ar", batch_size=16, max_new_tokens=128)
eval_df["pred_ar_standalone"] = preds_ar_standalone

# Direction 2: Algerian Darija -> English (ar_to_en)
print("\n--- Starting Generation: Algerian Darija -> English ---")
sources_ar = eval_df["Arabic"].tolist()
preds_en_standalone = generate_translations_batch(sources_ar, direction="ar_to_en", batch_size=16, max_new_tokens=128)
eval_df["pred_en_standalone"] = preds_en_standalone

# Compute Metrics
def compute_translation_metrics(hypotheses, references, sources=None, lang_pair="en-ar"):
    results = {}
    tokenize_opt = "intl" if "ar" in lang_pair else "13a"
    bleu_calc = sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tokenize_opt)
    results["BLEU"] = round(bleu_calc.score, 2)
    results["BLEU_bp"] = round(bleu_calc.bp, 4)
    results["sys_len"] = bleu_calc.sys_len
    results["ref_len"] = bleu_calc.ref_len
    results["len_ratio"] = round(bleu_calc.sys_len / max(1, bleu_calc.ref_len), 3)
    
    chrf_calc = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    results["chrF++"] = round(chrf_calc.score, 2)
    
    sent_chrf = [
        sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score
        for hyp, ref in zip(hypotheses, references)
    ]
    results["sentence_chrF++"] = sent_chrf
    
    if USE_BERTSCORE:
        try:
            from bert_score import score as bert_scorer
            target_lang = "ar" if lang_pair.endswith("ar") else "en"
            P, R, F1 = bert_scorer(
                hypotheses,
                references,
                lang=target_lang,
                verbose=False,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            results["BERTScore_Precision"] = round(P.mean().item() * 100, 2)
            results["BERTScore_Recall"] = round(R.mean().item() * 100, 2)
            results["BERTScore_F1"] = round(F1.mean().item() * 100, 2)
            results["sentence_BERTScore_F1"] = [score.item() * 100 for score in F1]
        except Exception as e:
            print(f"Notice: BERTScore skipped: {e}")
            
    return results

print("\n=== Computing Metrics: English -> Algerian Darija (Standalone) ===")
metrics_en_to_ar = compute_translation_metrics(
    hypotheses=eval_df["pred_ar_standalone"].tolist(),
    references=eval_df["Arabic"].tolist(),
    sources=eval_df["English"].tolist(),
    lang_pair="en-ar"
)

print("\n=== Computing Metrics: Algerian Darija -> English (Standalone) ===")
metrics_ar_to_en = compute_translation_metrics(
    hypotheses=eval_df["pred_en_standalone"].tolist(),
    references=eval_df["English"].tolist(),
    sources=eval_df["Arabic"].tolist(),
    lang_pair="ar-en"
)

eval_df["chrf_en_to_ar_standalone"] = metrics_en_to_ar["sentence_chrF++"]
eval_df["chrf_ar_to_en_standalone"] = metrics_ar_to_en["sentence_chrF++"]

# Save Standalone summary
eval_dir = os.path.join(REPO_ROOT, "evaluation")
os.makedirs(eval_dir, exist_ok=True)

standalone_summary = {
    "sample_size": len(eval_df),
    "english_to_darija": {k: v for k, v in metrics_en_to_ar.items() if not k.startswith("sentence_")},
    "darija_to_english": {k: v for k, v in metrics_ar_to_en.items() if not k.startswith("sentence_")},
}

summary_out_path = os.path.join(eval_dir, "standalone_evaluation_summary.json")
with open(summary_out_path, "w", encoding="utf-8") as f:
    json.dump(standalone_summary, f, indent=4, ensure_ascii=False)
print(f"✓ Saved {summary_out_path}")

# If stage 2 results are present, merge for a complete side-by-side comparison
stage2_res_path = os.path.join(eval_dir, "stage2_evaluation_results.csv")
if not os.path.exists(stage2_res_path):
    stage2_res_path = os.path.join(REPO_ROOT, "stage2_evaluation_results.csv")

if os.path.exists(stage2_res_path):
    df_st2 = pd.read_csv(stage2_res_path)
    eval_df["pred_ar_stage2"] = df_st2["pred_ar"]
    eval_df["pred_en_stage2"] = df_st2["pred_en"]
    eval_df["chrf_en_to_ar_stage2"] = df_st2["chrf_en_to_ar"]
    eval_df["chrf_ar_to_en_stage2"] = df_st2["chrf_ar_to_en"]
    
    # Delta
    eval_df["delta_chrf_en_to_ar"] = eval_df["chrf_en_to_ar_stage2"] - eval_df["chrf_en_to_ar_standalone"]
    eval_df["delta_chrf_ar_to_en"] = eval_df["chrf_ar_to_en_stage2"] - eval_df["chrf_ar_to_en_standalone"]

comp_out_path = os.path.join(eval_dir, "comparison_evaluation_results.csv")
eval_df.to_csv(comp_out_path, index=False, encoding="utf-8-sig")
print(f"✓ Saved {comp_out_path}")

print("\n" + "="*70)
print("             STANDALONE MODEL EVALUATION SUMMARY                  ")
print("="*70)
print(f"English -> Darija: BLEU={metrics_en_to_ar.get('BLEU')}, chrF++={metrics_en_to_ar.get('chrF++')}, BERTScore_F1={metrics_en_to_ar.get('BERTScore_F1', 'N/A')}")
print(f"Darija -> English: BLEU={metrics_ar_to_en.get('BLEU')}, chrF++={metrics_ar_to_en.get('chrF++')}, BERTScore_F1={metrics_ar_to_en.get('BERTScore_F1', 'N/A')}")
print("="*70)
