import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset

DATA_PATH = "./data/"
MAX_LENGTH = 256
LABEL_MAP = {"A":0, "B":1, "C":2, "D":3, "E":4}

MODEL = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def preprocess(data):
    output = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    prompts = []
    options = []
    answer  = []

    for i in range(len(data["prompt"])):
        # populate the prompts , options and answer list.
        # prompts list will have one prompt repeated 5 times for each of the option.
        # options list will have all the options for all the rows of data.
        # answer  list will have a single answer as a number corresponding to the
        # label_map for each prompt.
        prompts += [data["prompt"][i]] * 5
        options += [data[x][i] for x in LABEL_MAP.keys()]
        answer.append(LABEL_MAP[data["answer"][i]])

    tokenized_input = tokenizer(prompts, options, truncation = True, MAX_LENGTH = 256)

    for i in range(0,len(tokenized_input["input_ids"]),5):

        output["input_ids"].append(tokenized_input["input_ids"][i:i+5])
        output["attention_mask"].append(tokenized_input["attention_mask"][i:i+5])
        output["labels"].append(answer[i//5])

    return output

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
    tokenized_test  = test_dataset.map(preprocess, batched = True)

    print("Train Dataset:" , tokenized_train)
    print("Test Dataset:"  , tokenized_test)

    # Verify the sizes
    sample = tokenized_train["train"][0]
    print("Size of input_ids:", len(sample["input_ids"]))
    print("Size of attention_mask:", len(sample["attention_mask"]))
    print("Label:", sample["labels"])

    print("Save the processed datasets to the disk..")
    tokenized_train["train"].save_to_disk("./preprocessed_data/deberta/train")
    tokenized_train["test"].save_to_disk("./preprocessed_data/deberta/validation")
    tokenized_test.save_to_disk("./preprocessed_data/deberta/test")

    print("Preprocessing done....")

if __name__ == "__main__":
    setup_data()