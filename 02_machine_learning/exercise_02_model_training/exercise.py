import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)

def evaluate_model(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

model = train_model(X_train, y_train)
predictions = predict(model, X_test)
metrics = evaluate_model(y_test, predictions)

# One prediction for each test sample
assert len(predictions) == len(y_test)

# Accuracy: proportion of correct predictions
assert 0 <= metrics["accuracy"] <= 1

# Precision: among predicted positives, how many are correct?
assert 0 <= metrics["precision"] <= 1 

# Recall: among actual positives, how many were identified?
assert 0 <= metrics["recall"] <= 1 

# F1: balance between precision and recall
assert 0 <= metrics["f1"] <= 1 

print("All tests passed!")
 
