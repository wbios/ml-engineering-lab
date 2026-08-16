import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({
    "age": [22, 25, 28, 32, 35, 40, 45, 50, 55, 60],
    "income": [22000, 25000, 28000, 35000, 38000, 45000, 52000, 60000, 70000, 80000],
    "purchased": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
})

def split_data(df):
    X = df[["age", "income"]]
    y = df["purchased"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(df)

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



def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

assert abs(X_train_scaled.mean(axis=0)).max() < 1e-10
assert X_train_scaled.shape == X_train.shape
assert X_test_scaled.shape == X_test.shape

print("All tests passed!")