import json

def configure_full_training():
    with open('stage1.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Modify cell 12 (SFTConfig) for full real training (Kaggle/Cloud/Local)
    nb['cells'][12]['source'] = [
        "from trl import SFTConfig\n",
        "\n",
        "training_args = SFTConfig(\n",
        "    output_dir=\"./gemma4-darija-qlora\",\n",
        "    dataset_text_field=\"text\",\n",
        "    max_length=512,\n",
        "    packing=True,                   # Dense 512-token packing (cuts 137k rows down to ~7.3k dense sequences)\n",
        "    dataset_num_proc=4,             # Parallel preprocessing\n",
        "    per_device_train_batch_size=2,\n",
        "    gradient_accumulation_steps=8,  # Effective batch size = 16 packed chunks (~350-400 sentences/step)\n",
        "    learning_rate=2e-4,\n",
        "    lr_scheduler_type=\"cosine\",\n",
        "    warmup_ratio=0.03,\n",
        "    num_train_epochs=1,             # Full real training for 1 complete epoch over the entire dataset\n",
        "    logging_steps=10,\n",
        "    eval_strategy=\"steps\",\n",
        "    eval_steps=50,                  # Periodic evaluation on validation split\n",
        "    save_steps=50,                  # Save LoRA checkpoints regularly\n",
        "    save_total_limit=3,\n",
        "    fp16=not torch.cuda.is_bf16_supported(),\n",
        "    bf16=torch.cuda.is_bf16_supported(),\n",
        "    optim=\"paged_adamw_8bit\",\n",
        "    dataloader_pin_memory=True,\n",
        "    gradient_checkpointing=True,\n",
        "    gradient_checkpointing_kwargs={\"use_reentrant\": False},\n",
        "    report_to=\"none\"\n",
        ")\n"
    ]

    # Clear outputs so the notebook is clean and ready for Kaggle/Cloud
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None

    with open('stage1.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("stage1.ipynb successfully configured for FULL training (Kaggle/Cloud ready)!")

if __name__ == '__main__':
    configure_full_training()
