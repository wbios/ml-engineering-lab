import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

np.random.seed(42)

class_0 = np.random.normal(
    loc=0,
    scale=1,
    size=90
)

class_1 = np.random.normal(
    loc=1,
    scale=1,
    size=10
)
y = pd.Series([0]*90 + [1]*10)
X = pd.DataFrame({
    "score": np.concatenate([class_0, class_1])
})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

assert len(X_train) == 80
assert len(X_test) == 20

assert y_train.value_counts()[0] == 72
assert y_train.value_counts()[1] == 8

assert y_test.value_counts()[0] == 18
assert y_test.value_counts()[1] == 2

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print(cm)

assert cm[1, 1] == 0
assert cm[1, 0] == 2


precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Without class_weight='balanced'")
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

assert recall == 0.0

model_balanced = LogisticRegression(
    class_weight="balanced"
)

model_balanced.fit(X_train, y_train)
y_pred_balanced = model_balanced.predict(X_test)
cm_balanced = confusion_matrix(y_test, y_pred_balanced)

print(cm_balanced)

assert cm_balanced[1, 1] == 2
assert cm_balanced[1, 0] == 0
assert cm_balanced[0, 1] == 5

precision_balanced = precision_score(y_test, y_pred_balanced)
recall_balanced = recall_score(y_test, y_pred_balanced)
f1_balanced = f1_score(y_test, y_pred_balanced)

print("With class_weight='balanced'")
print("Precision:", precision_balanced)
print("Recall:", recall_balanced)
print("F1:", f1_balanced)

assert recall_balanced == 1.0