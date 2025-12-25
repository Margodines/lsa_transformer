from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

MODEL_NAME = "google/flan-t5-base"
OUTPUT_DIR = "./lsa_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/train.json",
        "validation": "data/validation.json"
    }
)

def tokenize_function(batch):
    model_inputs = tokenizer(
        batch["source"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target"],
            padding="max_length",
            truncation=True,
            max_length=64
        )["input_ids"]

    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels
    ]

    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["source", "target"]
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
    fp16=True  # si usás GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer
)

trainer.train()

# Guardado FINAL explícito
trainer.save_model(f"{OUTPUT_DIR}/final_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
