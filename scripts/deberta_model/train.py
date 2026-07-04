import torch
import wandb
from datasets import load_from_disk
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForMultipleChoice, Trainer, TrainingArguments
from dotenv import load_dotenv
from pathlib import Path
from transformers import PreTrainedTokenizerBase
from scripts.map3_metric import compute_metrics
import os

load_dotenv()
WB_KEY  = os.getenv("WANDB_API_KEY")
MODEL = "microsoft/deberta-v3-base"

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    
    def __call__(self, features):
        flat_input = {
            "input_ids": [],
            "attention_mask": [],
        }

        labels = []

        for item in features:
            flat_input["input_ids"] += item["input_ids"]
            flat_input["attention_mask"] += item["attention_mask"]
            labels.append(item["labels"])

        batch = self.tokenizer.pad(
            flat_input,
            return_tensors = "pt"
        )

        batch["input_ids"] = batch["input_ids"].view(len(features),5, -1)
        batch["attention_mask"] = batch["attention_mask"].view(len(features),5, -1)
        batch["labels"] = torch.tensor(labels)

        return batch
    
def train_model():
    config = {
        "model" : "microsoft/deberta-v3-base",
        "max_sequence_length": 256,
        "random_seed": 42,
    }

    wandb.login(key=WB_KEY)

    run = wandb.init(
        project = "test", #"21f2000763-t22026",
        name = "deberta-v3-finetune-run4",
        tags = ["deberta", "transformer"],
        config = config,
    )

    print("Loading the preprocessed dataset...")

    tokenized_train_dataset = load_from_disk("./preprocessed_data/deberta/train")
    tokenized_val_dataset = load_from_disk("./preprocessed_data/deberta/validation")

    tokenizer  = AutoTokenizer.from_pretrained(MODEL)
    data_collator = DataCollatorForMultipleChoice(tokenizer = tokenizer)

    model = AutoModelForMultipleChoice.from_pretrained(MODEL, torch_dtype=torch.float32)

    training_args = TrainingArguments(
        output_dir = "./results",
        save_strategy = "epoch",
        eval_strategy= "epoch",
        learning_rate = 1e-5,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size=4,
        num_train_epochs = 3,
        weight_decay = 0.01,
        report_to = "wandb",
        logging_steps = 50,
        load_best_model_at_end=True,
        metric_for_best_model="map@3",
        seed=42,
        run_name = "deberta-v3-finetune-run4"
    )

    # Initialize the trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        compute_metrics = compute_metrics,
        train_dataset = tokenized_train_dataset,
        eval_dataset= tokenized_val_dataset,
        data_collator = data_collator
    )

    print("Starting training...")
    trainer.train()
    run.finish()

    print("Saving final model....")
    Path("./models/deberta_model").mkdir(
        parents=True,
        exist_ok=True
    )
    trainer.save_model("./models/deberta_model")
    tokenizer.save_pretrained("./models/deberta_model")

if __name__ == "__main__":
    train_model()
