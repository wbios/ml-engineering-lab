import pandas as pd
from sklearn.model_selection import train_test_split

y = pd.Series([0]*90 + [1]*10)
age = pd.Series(range(20, 120))
X = pd.DataFrame({
    "age":age
})

X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

assert y_train.value_counts()[0] / len(y_train) ==0.9
assert y_train.value_counts()[1] / len(y_train) ==0.1

assert y_test.value_counts()[0] / len(y_test) ==0.9
assert y_test.value_counts()[1] / len(y_test) ==0.1

