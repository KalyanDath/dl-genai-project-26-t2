import pandas as pd
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForMultipleChoice, Trainer
from scripts.deberta_model.preprocess import LABEL_MAP
from scripts.deberta_model.train import DataCollatorForMultipleChoice

INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

def generate_predictions():
    print("Loading test dataset and saved model...")

    # Load the test dataset
    tokenized_test_dataset = load_from_disk("./preprocessed_data/deberta/test")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./models/deberta_model")
    model = AutoModelForMultipleChoice.from_pretrained("./models/deberta_model")
    
    data_collator = DataCollatorForMultipleChoice(tokenizer = tokenizer)

    # Define the trainer for prediction
    trainer = Trainer(
        model = model,
        data_collator = data_collator
    )

    print("Run the inference for the test dataset...")
    test_predictions = trainer.predict(tokenized_test_dataset)
    logits = test_predictions.predictions

    print("Processing Top 3 predictions....")
    top_3_indices = np.argsort(logits, axis=1)[:, -1:-4:-1]
    

    predicted_strings = []
    for row in top_3_indices:
        letters = [INDEX_TO_LETTER[idx] for idx in row]
        predicted_strings.append(" ".join(letters))


    # Load the test.csv to create the submission.csv file
    test_df = pd.read_csv("./data/test.csv")

    submission_df = pd.DataFrame({
        "id": test_df["id"],
        "prediction": predicted_strings
    })

    submission_df.to_csv("./outputs/submission.csv", index=False)
    print("Submission csv file created...")
    print(submission_df.head())


if __name__ == "__main__":
    generate_predictions()
