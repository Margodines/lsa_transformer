
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import torch

# ======================
# CONFIGURACIÓN GENERAL
# ======================

MODEL_NAME = "google/flan-t5-base"  # mejor que t5-small para instrucciones
OUTPUT_DIR = "./lsa_model"

MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5

# ======================
# TOKENIZER Y MODELO
# ======================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ======================
# DATASET
# ======================

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/train.json",
        "validation": "data/validation.json"
    }
)

# ======================
# TOKENIZACIÓN
# ======================

def preprocess(batch):
    inputs = [
        "LSA-GLOSS: " + text
        for text in batch["source"]
    ]

    model_inputs = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    labels = tokenizer(
        text_target=batch["target"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ======================
# DATA COLLATOR
# ======================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# ======================
# TRAINING ARGUMENTS
# ======================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,
    weight_decay=0.01,

    fp16=torch.cuda.is_available(),
    logging_steps=50,

    save_total_limit=2,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    report_to="none"
)

# ======================
# TRAINER
# ======================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ======================
# ENTRENAMIENTO
# ======================


trainer.train()

# ======================
# GUARDADO FINAL
# ======================

BEST_MODEL_DIR = f"{OUTPUT_DIR}/best_model"

trainer.save_model(BEST_MODEL_DIR)
tokenizer.save_pretrained(BEST_MODEL_DIR)

print(f"\n✅ Modelo guardado en: {BEST_MODEL_DIR}")
