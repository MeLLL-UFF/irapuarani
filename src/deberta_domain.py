import ast
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from datasets import Dataset
import numpy as np
import sklearn
from huggingface_hub import login
import os

HUGGING_FACE_TOKEN = os.environ['HUGGING_FACE_TOKEN']
HUB_MODEL_ID_BASE = ""
INPUT_FILE_PATH = "./data"
OUTPUT_DIR = ""
NUM_FOLDS = 5
SEED = 42
MODEL_NAME = "microsoft/deberta-v3-base"
NUM_EPOCHS = 10
BATCH_SIZE = 16

login(token=HUGGING_FACE_TOKEN)

df = pd.read_csv(INPUT_FILE_PATH)
texts = df["content"].tolist()
df["label"] = df["domain"].apply(lambda x: 0 if x == "URW" else 1 if x == "CC" else 2)
labels = df["label"].tolist()
NUM_LABELS = 3

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
all_fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), start=1):
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    
    train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "labels": val_labels})
    
    train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=BATCH_SIZE)
    val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=BATCH_SIZE)
    train_dataset = train_dataset.remove_columns(["text"])
    val_dataset = val_dataset.remove_columns(["text"])
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/fold{fold}",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        push_to_hub=True,
        hub_model_id=f"{HUB_MODEL_ID_BASE}_fold{fold}",
        seed=SEED,
        save_total_limit=1,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        precision = sklearn.metrics.precision_score(labels, predictions, average='macro', zero_division=1)
        recall = sklearn.metrics.recall_score(labels, predictions, average='macro', zero_division=1)
        f1 = sklearn.metrics.f1_score(labels, predictions, average='macro', zero_division=1)
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    trainer.push_to_hub()
    
    predictions = trainer.predict(val_dataset).predictions
    predicted_labels = np.argmax(predictions, axis=-1)
    val_df = df.iloc[val_idx].copy()
    val_df["p_domains"] = predicted_labels
    val_df.to_csv(f"{OUTPUT_DIR}/fold{fold}_predictions.csv", index=False)
    
    fold_metrics = trainer.evaluate()
    all_fold_metrics.append(fold_metrics)

metrics_log_path = f"{OUTPUT_DIR}/cross_validation_metrics_log.txt"
with open(metrics_log_path, "w") as log_file:
    for fold, metrics in enumerate(all_fold_metrics, start=1):
        log_file.write(f"Fold {fold} Metrics:\n")
        for metric_name, value in metrics.items():
            log_file.write(f"{metric_name}: {value:.4f}\n")
        log_file.write("\n")
    avg_metrics = {key: np.mean([m[key] for m in all_fold_metrics]) for key in all_fold_metrics[0]}
    log_file.write("Average Metrics Across All Folds:\n")
    for metric_name, value in avg_metrics.items():
        log_file.write(f"{metric_name}: {value:.4f}\n")

print(f"Metrics log saved to {metrics_log_path}")
