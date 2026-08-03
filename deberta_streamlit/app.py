import torch
import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice
)

MODEL_PATH = "KalyanDath18/deberta-finetuned"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForMultipleChoice.from_pretrained(
    MODEL_PATH
)

model.eval()


def predict(prompt, A, B, C, D, E):

    choices = [A, B, C, D, E]

    encoding = tokenizer(
        [prompt] * 5,
        choices,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

    # Shape: (1, 5, seq_len)
    input_ids = encoding["input_ids"].unsqueeze(0)
    attention_mask = encoding["attention_mask"].unsqueeze(0)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probs = torch.softmax(outputs.logits, dim=1)

        top3 = torch.argsort(
            probs,
            descending=True
        )[0, :3]

    labels = ["A", "B", "C", "D", "E"]

    return " ".join(labels[i] for i in top3)


st.title("DeBERTa MCQ Solver")

prompt = st.text_area("Question")

A = st.text_input("Option A")
B = st.text_input("Option B")
C = st.text_input("Option C")
D = st.text_input("Option D")
E = st.text_input("Option E")


if st.button("Predict"):

    prediction = predict(
        prompt,
        A,
        B,
        C,
        D,
        E
    )

    st.success(f"Top 3 Predictions: {prediction}")