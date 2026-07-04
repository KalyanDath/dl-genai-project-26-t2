# Smart MCQ Solver Challenge - DLGenAI Project
**Kalyan Dath** **Roll no: 21f200763**

## Models Implemented
1. **Finetuned DeBerta-v3** : A transformer model fine tuned using the hugging face `Trainer` to evaluate semantic relationship between the prompts and options.

## Project Structure
* ```/scripts/deberta_model/preprocess.py``` : Loads dataset, and does tokenization.
* ```/scripts/deberta_model/train.py```: Training for the deberta model
* ```/scripts/map3_metric.py```: Evalution logic to compute the Mean Average Precision @ 3 during the training phase.
* ```/notebooks/``` : This contains my experiment codes where I do the actual training and try out other things. Also contains notebook ran in Kaggle.

## Evaluation Metric
Models are evaluated using the **Mean Average Precision @ 3 (MAP@3)** metric. This scores model based on whether the correct answer was ranked as 1st (1.0 points), 2nd (0.5 points), or 3rd (0.33 points) guess.

## Experiment Tracking
All training runs, hyperparameter configs, and validation metrics are tracked and visualized using wandb.

## Inference
The notebook [pretrained_deberta.ipynb](https://github.com/KalyanDath/dl-genai-project-26-t2/blob/main/notebooks/pretrained_deberta.ipynb) is designed for inference on the Kaggle platform.It loads the pre-trained DeBERTa weights, processes the `test.csv` dataset using `DataCollatorForMultipleChoice`, and exports formatted `submission.csv` containing the top 3 predictions per question.


