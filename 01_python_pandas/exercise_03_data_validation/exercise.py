import pandas as pd

df = pd.DataFrame({
    "customer_id": [1, 2, 3, 4, 5, 6],
    "age": [25, 34, -5 ,45, None, 200],
    "country": ["IT", "IT", "FR", "IT", None, "FR"],
    "amount": [100, 250, -80, 300, 150, None]
})

def validate_data(df):
    missing_values = df.isna().sum().to_dict()
    invalid_age = ((df["age"] < 0) | (df["age"] > 120)).sum()
    invalid_amount = (df["amount"] <0).sum()
    duplicate_customer_ids = df["customer_id"].duplicated().sum()
    return {
        "missing_values": missing_values,
        "invalid_age": invalid_age,
        "invalid_amount": invalid_amount,
        "duplicate_customer_ids": duplicate_customer_ids
    }

result = validate_data(df)

assert result["missing_values"] == {
    "customer_id": 0,
    "age": 1,
    "country": 1,
    "amount": 1
}

assert result["invalid_age"] == 2
assert result["invalid_amount"] == 1
assert result["duplicate_customer_ids"] == 0

print("All tests passed!")