import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset

DATA_PATH = "../../data/"

MODEL = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def preprocess(data):
    output = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    label_map = {"A":0, "B":1, "C":2, "D":3, "E":4}

    prompts = []
    options = []
    answer  = []

    for i in range(len(data["prompt"])):
        # populate the prompts , options and answer list.
        # prompts list will have one prompt repeated 5 times for each of the option.
        # options list will have all the options for all the rows of data.
        # answer  list will have a single answer as a number corresponding to the
        # label_map for each prompt.
        prompts += [data["prompt"][i] * 5]
        options += [data[x][i] for x in label_map.keys()]
        answer.append(label_map[data["answer"][i]])

    tokenized_input = tokenizer(prompts, options, truncation = True, max_length = 256)

    for i in range(0,len(tokenized_input["input_ids"]),5):
        tokenized_inputs = {
            "input_ids": [],
            "attention_mask": []
        }

        output["input_ids"].append(tokenized_input["input_ids"][i:i+5])
        output["attention_mask"].append(tokenized_input["attention_mask"][i:i+5])
        output["labels"].append(answer[i//5])

    print(output["input_ids"][0])

def setup_data():
    print("Loading the datasets....")
    train_df = pd.read_csv(DATA_PATH + "train.csv")
    test_df  = pd.read_csv(DATA_PATH + "test.csv")

    # Add an answer column to the test_df with a default answer
    test_df["answer"] = "A"
    
    # Convert pandas to Datasets type
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset  = Dataset.from_pandas(test_df)

    # Split the training dataset into train and validation.
    split_dataset = train_dataset.train_test_split(test_size = 0.2, seed = 42)

    print("Tokenizing the datasets...")
    tokenized_train = split_dataset.map(preprocess, batched = True)
    tokenized_test  = split_dataset.map(preprocess, batched = True)

    print("Example train tokens:" , tokenized_train["input_ids"][0])
    print("Example test tokens:"  , tokenized_test["input_ids"][0])

    print("Save the processed datasets to the disk..")
    tokenized_train.save_to_disk("./preprocessed_data/train")
    tokenized_test.save_to_disk("./preprocess_data/test")

    print("Preprocessing done....")

if __name__ == "__main__":
    setup_data()