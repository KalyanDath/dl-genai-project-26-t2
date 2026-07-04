import numpy as np

def map_at_3(predictions, labels):

    top3 = np.argsort(-predictions, axis=1)[:, :3]

    scores = []

    for pred, label in zip(top3, labels):
        if label == pred[0]:
            scores.append(1.0)
        elif label == pred[1]:
            scores.append(0.5)
        elif label == pred[2]:
            scores.append(1.0/3.0)
        else:
            scores.append(0.0)

    return np.mean(scores)

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    score = map_at_3(logits, labels)

    return {
        "map@3": score,
    }