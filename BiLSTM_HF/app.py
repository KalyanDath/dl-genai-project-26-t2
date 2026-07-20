import torch
import streamlit as st
from pathlib import Path

st.write("Current working directory:", os.getcwd())
st.write("App file:", __file__)
st.write("Files in app directory:", os.listdir(Path(__file__).parent))

from transformers import AutoTokenizer
from model import BiLSTMAttention

tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-uncased"
)

model = BiLSTMAttention(tokenizer.vocab_size)

model.load_state_dict(
    torch.load(
        "bilstm_attention.pth",
               map_location="cpu"
    )
)

model.eval()

def predict(prompt,A,B,C,D,E):

    text = f"{prompt} [SEP] A:{A} B:{B} C:{C} D:{D} E:{E}"

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():

        logits = model(
            encoding["input_ids"],
            encoding["attention_mask"]
        )

        probs = torch.softmax(logits,dim=1)

        top3 = torch.argsort(
            probs,
            descending=True
        )[0,:3]

    labels=["A","B","C","D","E"]

    return " ".join(labels[i] for i in top3)


st.title("BiLSTM Attention QA")

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

    st.success(prediction)