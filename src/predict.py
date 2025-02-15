import ast
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset
from huggingface_hub import login
import os

HUGGING_FACE_TOKEN = os.environ['HUGGING_FACE_TOKEN']
login(token=HUGGING_FACE_TOKEN)

SEED = 42
NUM_EPOCHS = 10
OUTPUT_DIR = ""
#model in {g-assismoraes/deberta-semeval25task10-gpt2en, g-assismoraes/deberta-semeval25task10-aya2en, g-assismoraes/mdeberta-semeval25task10}
HUB_MODEL_ID = "" 
MODEL_NAME = "" #microsoft/deberta-v3-base or microsoft/mdeberta-v3-base --> to get tokenizer
BATCH_SIZE = 16

TRAIN_FILE_PATH = ""
VALIDATION_FILE_PATH = ""

#en_content_gpt → translated by gpt4o-mini
#en_content → translated by aya
#content → original text

TRAIN_TEXT_COLUMN = "en_content"
VALIDATION_TEXT_COLUMN = "en_content"
VALIDATION_LANGUAGE_FILTER = "BG" #or EN PT BG RU


train_df = pd.read_csv(TRAIN_FILE_PATH)
train_texts = train_df[TRAIN_TEXT_COLUMN].tolist()
train_labels = train_df["label"].apply(ast.literal_eval).tolist()

validation_df = pd.read_csv(VALIDATION_FILE_PATH)
if VALIDATION_LANGUAGE_FILTER:
    validation_df = validation_df[validation_df["language"] == VALIDATION_LANGUAGE_FILTER]
validation_texts = validation_df[VALIDATION_TEXT_COLUMN].tolist()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(HUB_MODEL_ID)

validation_dataset = Dataset.from_dict({"text": validation_texts})
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
validation_dataset = validation_dataset.map(tokenize_function, batched=True, batch_size=BATCH_SIZE)
validation_dataset = validation_dataset.remove_columns(["text"])

training_args = TrainingArguments( #just for load model
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
)

predictions = trainer.predict(validation_dataset).predictions
binarized_predictions = (predictions > 0.8).astype(int)

mlb = MultiLabelBinarizer()
mlb.fit(train_labels)
predicted_labels = mlb.inverse_transform(binarized_predictions)

validation_df["predicted_labels"] = [",".join(labels) for labels in predicted_labels]
validation_output_path = ""
validation_df.to_csv(validation_output_path, index=False)

print(f"Validation predictions saved to {validation_output_path}")
