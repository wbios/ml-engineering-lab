import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

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

y_proba = model.predict_proba(X_test)

y_proba_class_1 = y_proba[:, 1]

precisions, recalls, thresholds = precision_recall_curve(
    y_test,
    y_proba_class_1,
)

print("Probabilities of class 1")
print(y_proba_class_1)

print("Precision:", precisions)
print("Recalls:", recalls)
print("Thresholds:", thresholds)

plt.plot(recalls, precisions)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.show()