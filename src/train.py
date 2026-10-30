import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.common import MAX_LEN, MODEL_ID, OUT_DIR, load_model

BATCH = int(os.environ.get("BATCH", 4))
ACCUM = int(os.environ.get("ACCUM", 4))

TRAIN_FILE = os.environ.get("TRAIN_FILE", "data/train_sft.jsonl")
ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
print(f"학습 데이터: {TRAIN_FILE}, {len(ds)}건")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = load_model(device_map={"": 0})

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
cfg = SFTConfig(
    output_dir=OUT_DIR,
    num_train_epochs=float(os.environ.get("EPOCHS", 2)),
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=ACCUM,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=MAX_LEN,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
)
trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=ds,
    processing_class=tok,
    peft_config=peft_config,
)
trainer.train()
trainer.save_model(f"{OUT_DIR}/final")
print(f"저장 완료: {OUT_DIR}/final")
print(f"VRAM 최대 {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")