import pandas as pd
import numpy as np
df = pd.DataFrame({
    "customer_id": [1, 2, 3, 4, 5, 6],
    "age": [25, 34, None, 45, 34, 52],
    "country": ["IT", "IT", "FR", "IT", None, "FR"],
    "purchase": [100, 250, 80, 300, 150, 400],
    "is_fraud": [0, 0, 1, 0, 1, 0]
})



def preprocess(df):
    df = df.copy()
    median_age = df["age"].median()
    df["age"] = df["age"].fillna(median_age)
    df["country"] = df["country"].fillna("UNKNOWN")
    df["purchase_log"] = np.log(df["purchase"])
    return df

result = preprocess(df)

assert result["age"].isna().sum() == 0
assert result["country"].isna().sum == 0
assert "purchase_log" in result.columns

print(result)

