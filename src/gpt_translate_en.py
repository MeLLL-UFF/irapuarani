from dotenv import load_dotenv
import openai
import pandas as pd
import os

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
client = openai.OpenAI()

INPUT_FILE_PATH = ""
OUTPUT_FILE_PATH = "./data"
INTERMEDIATE_SAVE_PATH = ""
SAVE_INTERVAL = 10
RESUME_INDEX = 0

df = pd.read_csv(INPUT_FILE_PATH)
texts = df["content"].tolist()
languages = df["language"].tolist()

print(len(df))
df.head()

def get_prompt(text):
    return f"""Translate the following text into English. Be as precise as possible in retaining the information conveyed.

{text}"""

translated_texts = []
if RESUME_INDEX > 0 and pd.read_csv(INTERMEDIATE_SAVE_PATH).shape[0] > 0:
    df_intermediate = pd.read_csv(INTERMEDIATE_SAVE_PATH)
    translated_texts = df_intermediate["en_content_gpt"].tolist()
    print(f"Loaded {len(translated_texts)} already translated rows.")

translated_texts = ["" for _ in range(len(texts))]

for idx in range(RESUME_INDEX, len(texts)):
    text = texts[idx]
    lang = languages[idx]

    if lang == "EN":
        translated_texts[idx] = text
    else:
        prompt = get_prompt(text)
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.8,
                top_p=0.95
            )
            translated_text = response.choices[0].message.content
            translated_texts[idx] = translated_text
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            translated_texts[idx] = "Error: Translation Failed"

    if (idx + 1) % SAVE_INTERVAL == 0:
        df["en_content_gpt"] = translated_texts
        df.to_csv(INTERMEDIATE_SAVE_PATH, index=False)
        print(f"Saved intermediate progress at row {idx + 1}")

df["en_content_gpt"] = translated_texts
df.to_csv(OUTPUT_FILE_PATH, index=False)

print(f"Translation completed. The updated CSV is saved at {OUTPUT_FILE_PATH}.")
