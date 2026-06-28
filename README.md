# Smart MCQ Solver Challenge - DLGenAI Project
**Kalyan Dath** **Roll no: 21f200763**

## Models Implemented
1. **Finetuned DeBerta-v3** : A transformer model fine tuned using the hugging face `Trainer` to evaluate semantic relationship between the prompts and options.


## Evaluation Metric
Models are evaluated using the **Mean Average Precision @ 3 (MAP@3)** metric. This scores model based on whether the correct answer was ranked as 1st (1.0 points), 2nd (0.5 points), or 3rd (0.33 points) guess.

## Inference
The notebook [pretrained_deberta.ipynb](https://github.com/KalyanDath/dl-genai-project-26-t2/blob/main/notebooks/pretrained_deberta.ipynb) is designed for inference on the Kaggle platform.It loads the pre-trained DeBERTa weights, processes the `test.csv` dataset using `DataCollatorForMultipleChoice`, and exports formatted `submission.csv` containing the top 3 predictions per question.


