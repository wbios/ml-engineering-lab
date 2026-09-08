# Exercise 07 - Class Imbalance

## Goal

The goal of this exercise is to understand class imbalance in a binary classification problem and why accuracy alone can be misleading when one class is much more frequent than the other.

The exercise uses Pandas and scikit-learn.

## Dataset

The dataset contains customer information and a binary target indicating whether a customer purchased a product.

The target is intentionally imbalanced:
    9 samples with purchased = 0
    1 sample with purchased = 1

| Column | Description |
| ------ | ----------- |
| `age` | Customer age |
| `income` | Customer income |
| `purchased` | Purchase target (0 o 1) |

## Class Distribution

The target variable is analyzed using `value_counts()`.

```python
class_counts = df["purchased"].value_counts()
```
The expected distribution is:

`0` `9`
`1` `1`

The majority class represents 90% of the dataset.

The majority class ratio is calculated as:

```python
majority_ratio = class_counts.max() / len(df)
```

The expected result is:

`0.9`

## Naive Classifier

A naive classifier is created that always predicts the majority class.

```python
predictions = [0] * len(df)
```

The classifier therefore predicts 0 for every sample.

Although this classifier does not learn any useful pattern from the data, it can still achieve high accyracy because the majority class represents most of the dataset.

## Confusion Matrix

The predictions are compared with the actual target values using a confusion matrix.

The expected values are:

    True Negative (TN) = 9
    False Positive (FP) = 0
    False Negative (FN) = 1
    True Positive (TP) = 0

The confusion matrix is:

```python
[[9 0]
 [1 0]]
```

## Classification Metrics

The exercise calculates common classification metrics using scikit-learn.

Accuracy measures the proportion of correct predictions among all predictions.

    (TP + TN) / (TP + TN + FP + FN)

The expected accuracy is:

    0.9

Precision measures how many of the samples predicted as positive are actually positive.

    TP / (TP + FP)

The expected precision is:

    0.0

Recall measures how many of the actual positives samples are correctly identified by the model.

    TP / (TP + FN)

The expected recall is:

    0.0

F1-score combines precision and recall into a single metric.

    2* (Precision * Recall) / (Precision + Recall)

The expected F1-score is:

    0.0

## Conclusion

The exercise demonstrates that a classifier can achieve high accuracy while completely failing to identify the minority class.

In this example:

    Accuracy = 0.9
    Precision = 0.0
    Recall = 0.0
    F1 = 0.0

This shows why accuracy should not be used as the only evalution metric when dealing with imbalanced classes.

## Usage

From the project root, activate the virtual environment:

    .\.venv\Scritps\Activate.ps1

Install the dependencies:

    python -m pip install -r .\02_machine_learning\exercise_04_class_imbalance_\requirements.txt

Run the exercise:

    python .\02_machine_learning\exercise_04_class_imbalance\exercise.py