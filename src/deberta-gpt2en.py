import ast
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from datasets import Dataset
from huggingface_hub import login
import os

HUGGING_FACE_TOKEN = os.environ['HUGGING_FACE_TOKEN']
HUB_MODEL_ID = ""
OUTPUT_DIR = ""
SEED = 42
MODEL_NAME = "microsoft/deberta-v3-base"
NUM_EPOCHS = 10
BATCH_SIZE = 16

TRAIN_FILE_PATH = ""

login(token=HUGGING_FACE_TOKEN)

train_df = pd.read_csv(TRAIN_FILE_PATH)
print(len(train_df), "training dataset")

train_texts = train_df["en_content_gpt"].tolist()
train_labels = train_df["label"].apply(ast.literal_eval).tolist()

mlb = MultiLabelBinarizer()
train_binarized_labels = mlb.fit_transform(train_labels).astype(float)

NUM_LABELS = len(mlb.classes_)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_binarized_labels})
train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=BATCH_SIZE)
train_dataset = train_dataset.remove_columns(["text"])

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
    seed=SEED,
    save_total_limit=1,
    evaluation_strategy="no",
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
trainer.push_to_hub()

print("Model training complete and pushed to Hugging Face Hub.")
