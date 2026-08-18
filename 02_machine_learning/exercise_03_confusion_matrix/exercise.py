import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


df = pd.DataFrame({
    "age": [22, 25, 28, 32, 35, 40, 45, 50, 55, 60],
    "income": [22000, 25000, 28000, 35000, 38000, 45000, 52000, 60000, 70000, 80000],
    "purchased": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
})

X = df[["age", "income"]]
y = df["purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

assert len(X_train) == 8
assert len(X_test) == 2
assert len(y_train) == 8
assert len(y_test) == 2


assert len(X_train) + len(X_test) == len(df)
assert len(y_train) + len(y_test) == len(df)

train_indices = set(X_train.index)
test_indices = set(X_test.index)

assert train_indices.isdisjoint(test_indices)
assert train_indices.union(test_indices) == set(df.index)


def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)

model = train_model(X_train, y_train)
predictions = predict(model, X_test)

assert len(predictions) == len(y_test)

cm = confusion_matrix(y_test, predictions)
assert cm.shape == (2,2)
assert cm.sum() == len(y_test)

tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

assert accuracy == accuracy_score(y_test, predictions)
assert precision == precision_score(y_test, predictions)
assert recall == recall_score(y_test, predictions)
assert f1 == f1_score(y_test, predictions)

print("All tests passed!")
