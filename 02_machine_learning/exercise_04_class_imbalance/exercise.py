import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

df = pd.DataFrame({
    "age": [22, 25, 28, 32, 35, 40, 45, 50, 55, 60],
    "income": [22000, 25000, 28000, 35000, 38000, 45000, 52000, 60000, 70000, 80000,],
    "purchased": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
})

# Class distribution

class_counts = df["purchased"].value_counts()

assert class_counts[0] == 9
assert class_counts[1] == 1

print(class_counts)

# Majority class ratio

majority_ratio = class_counts.max() / len(df)

assert majority_ratio == 0.9

print("Majority ratio:", majority_ratio)

# A naive classifier
# The classifier always predicts the majority class: 0

predictions = [0] * len(df)

assert len(predictions) == len(df)
assert all(prediction == 0 for prediction in predictions)

# Confusion matrix

y_true = df["purchased"]

cm = confusion_matrix(y_true, predictions)

assert cm.shape == (2,2)
assert cm.sum() == len(df)

tn, fp, fn, tp = cm.ravel()

assert tn ==9
assert fp == 0
assert fn == 1
assert tp == 0

# Classification metrics

accuracy = accuracy_score(y_true, predictions)

precision = precision_score(
    y_true,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_true,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_true,
    predictions,
    zero_division=0
)

# Verify the results

assert accuracy == 0.9
assert precision == 0.0
assert recall == 0.0
assert f1 == 0.0

# Results

print("\nConfusion matrix:")
print(cm)

print("\nMetrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)