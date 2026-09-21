import pandas as pd
import numpy as np

from sklearn.model_selection import (
    StratifiedKFold, 
    cross_val_score
)
from sklearn.linear_model import LogisticRegression

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

kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model = LogisticRegression()
scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="accuracy"
)

print("Cross-validation scores:", scores)

mean_accuracy = scores.mean()

print("Mean accuracy:", mean_accuracy)