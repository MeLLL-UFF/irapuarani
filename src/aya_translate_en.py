from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import pandas as pd
import gc
from huggingface_hub import login
import os

HUGGING_FACE_TOKEN = os.environ['HUGGING_FACE_TOKEN']
login(token=HUGGING_FACE_TOKEN)

INPUT_FILE_PATH = ""
OUTPUT_FILE_PATH = "./data"
INTERMEDIATE_SAVE_PATH = ""
SAVE_INTERVAL = 10
RESUME_INDEX = 0

df = pd.read_csv(INPUT_FILE_PATH)
texts = df["content"].tolist()
languages = df["language"].tolist()

model_id = "CohereForAI/aya-expanse-8b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
pipeline = transformers.pipeline(
    task="text-generation",
    trust_remote_code=True,
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

def get_prompt(text):
    return f'''Translate the following text into English. Be as precise as possible in retaining the information conveyed.

{text}'''

translated_texts = []
if RESUME_INDEX > 0 and pd.read_csv(INTERMEDIATE_SAVE_PATH).shape[0] > 0:
    df_intermediate = pd.read_csv(INTERMEDIATE_SAVE_PATH)
    translated_texts = df_intermediate["en_content"].tolist()
    print(f"Loaded {len(translated_texts)} already translated rows.")

translated_texts = ["" for _ in range(len(texts))]

for idx in range(RESUME_INDEX, len(texts)):
    text = texts[idx]
    lang = languages[idx]
    print(lang)

    if lang == "EN":
        translated_texts[idx] = text
        
    else:
        prompt = get_prompt(text)
        messages = [{"role": "user", "content": prompt}]
        outputs = pipeline(
            messages,
            max_new_tokens=4000,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
        )
        translated_text = outputs[0]["generated_text"][1]['content']
        translated_texts[idx] = translated_text

    torch.cuda.empty_cache()
    gc.collect()

    if (idx + 1) % SAVE_INTERVAL == 0:
        df["en_content"] = translated_texts
        df.to_csv(INTERMEDIATE_SAVE_PATH, index=False)
        print(f"Saved intermediate progress at row {idx + 1}")

df["en_content"] = translated_texts
df.to_csv(OUTPUT_FILE_PATH, index=False)

print(f"Translation completed. The updated CSV is saved at {OUTPUT_FILE_PATH}.")
