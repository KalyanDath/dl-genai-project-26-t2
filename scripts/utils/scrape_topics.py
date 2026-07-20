import pandas as pd
import os 
import time
import json
from tqdm import tqdm

from google import genai

api_key = os.environ["GOOGLE_API_KEY"]
client = genai.Client(api_key = api_key)


PROMPT = """
You are an expert scientific knowledge extraction assistant.

You will receive multiple multiple-choice science questions.

For EACH question, identify exactly THREE official Wikipedia article titles needed to answer it.

Rules:
- Return EXACTLY 3 article titles.
- Prefer official Wikipedia article names.
- Use both question and options.
- No explanations.
- No markdown.
- Return ONLY valid JSON.

Output format:

[
  {{
    "id": 1,
    "topics": [
      "Topic A",
      "Topic B",
      "Topic C"
    ]
  }}
]

Questions:

{questions}
"""

test_df = pd.read_csv('test.csv')

def build_prompt(batch_df):
    text = ""
    for index, row in batch_df.iterrows():
        text += f"""
            ID: {index}

            Question:
            {row['prompt']}

            Options:
            A. {row['A']}
            B. {row['B']}
            C. {row['C']}
            D. {row['D']}
            E. {row['E']}
            --------------------------------
            """

    return PROMPT.format(questions=text)

# -------------------------------
# Gemini call
# -------------------------------
def extract_batch(batch_df):

    prompt = build_prompt(batch_df)

    for attempt in range(3):

        try:

            start = time.time()

            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt
            )

            print(f"Batch completed in {time.time()-start:.2f} sec")

            text = response.text.strip()

            # remove markdown if Gemini adds it
            text = text.replace("```json", "").replace("```", "").strip()

            return json.loads(text)

        except Exception as e:

            print(f"Retry {attempt+1}: {e}")

            time.sleep(3)

    return []

# -------------------------------
# Batch inference
# -------------------------------
BATCH_SIZE = 50

results = []

for start in tqdm(range(0, len(test_df), BATCH_SIZE), desc="Processing batches"):

    batch = test_df.iloc[start:start+BATCH_SIZE]

    batch_result = extract_batch(batch)

    results.extend(batch_result)

# -------------------------------
# Save
# -------------------------------
output_df = pd.DataFrame(results)

output_df.to_csv(
    "wiki_topics.csv",
    index=False,
    encoding="utf-8"
)

print(output_df.head())