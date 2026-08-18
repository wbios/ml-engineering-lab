# Exercise 06 - Confusion Matrix & Classification Metrics

## Goal

The goal of this exercise is to train a binary classification model, generate predictions, and analyze the results using a confusion matrix and common classification metrics.

The exercise uses Pandas and scikit-learn.

## Dataset

The dataset contains customer information and a binary target indicating whether a customer purchased a product.

| Column | Description |
| ------ | ----------- |
| `age` | Customer age |
| `income` | Customer income |
| `purchased` | Purchase target (`0` or `1`) |

## Train/Test Split

The dataset is divided into training and test sets using `train_test_split()`.

The split uses:

- 80% training data
- 20% test data
- `random_state=42` for reproducibility

The input data is divided into:

- Features: `age` and `income`
- Target: `purchased`

```python
X = df[["age", "income"]]
y = df["purchased"]
```

## Model

The exercise uses `LogisticRegression` as a binary classification model.

## Prediction

The `predict()` function uses the trained model to generate predictions for the test set

## Confusion Matrix

The confusion matrix shows how the model's predictions compare with the actual target values.

### Metrics

The classification metrics are calculated from the confusion matrix.

Accuracy measures the proportion of correct predictions among all predictions.
(TP + TN) / (TP + TN + FP + FN)

Precision measures how many of the samples predicted as positive are actually positive.
TP / (TP + FP)

Recall measures how many of the actual positive samples are correctly identified by the model.
TP / (TP + FN)

F1-score measures how well a classification model finds the positive cases while avoiding false alarms.
2 * (Precision * Recall) / (Precision + Recall)

## Usage
From the project root, activate the virtual environment:

.\.venv\Scripts\Activate.ps1

Install the dependencies:

python -m pip install -r .\02_machine_learning\exercise_03_confusion_matrix\requirements.txt

Run the exercise:

python .\02_machine_learning\exercise_03_confusion_matrix\exercise.py