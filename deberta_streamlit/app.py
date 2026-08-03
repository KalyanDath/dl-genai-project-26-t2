import torch
import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice
)

MODEL_PATH = "KalyanDath18/deberta-finetuned"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForMultipleChoice.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


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

    results = []

    for idx in top3:
        results.append(
            (
                labels[idx],
                probs[0][idx].item() * 100
            )
        )

    return results


st.title("DeBERTa MCQ Solver")
st.caption(
    "Enter a multiple-choice question and five answer options. "
    "The model returns the three most likely answers."
)

st.sidebar.title("Model Information")

st.sidebar.markdown(f"""
**Model:** DeBERTa-v3 Base

**Task:** Multiple Choice Question Answering

**Max Sequence Length:** {tokenizer.model_max_length}

**Classes:** A, B, C, D, E

**Framework:** Hugging Face Transformers + PyTorch
""")



prompt = st.text_area("Question")

A = st.text_input("Option A")
B = st.text_input("Option B")
C = st.text_input("Option C")
D = st.text_input("Option D")
E = st.text_input("Option E")

with st.expander("Model Details"):
    st.write(f"Model: {MODEL_PATH}")
    st.write(f"Tokenizer: {tokenizer.__class__.__name__}")
    st.write(f"Vocabulary Size: {tokenizer.vocab_size:,}")
    st.write(f"Maximum Sequence Length: {tokenizer.model_max_length}")
    
    if prompt:
        num_tokens = len(tokenizer.tokenize(prompt))
        st.write(f"Question Tokens: {num_tokens}")


if st.button("Predict"):

    if not all([
        prompt.strip(),
        A.strip(),
        B.strip(),
        C.strip(),
        D.strip(),
        E.strip()
    ]):
        st.error("Please enter the question and all five options.")
    else:
        with st.spinner("Predicting..."):
            prediction = predict(
                prompt,
                A,
                B,
                C,
                D,
                E
            )

            st.success("Top 3 Predictions")

            for i, (option, confidence) in enumerate(prediction, 1):
                st.write(f"{i}. **{option}** — {confidence:.2f}%")