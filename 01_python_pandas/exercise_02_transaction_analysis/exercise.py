import pandas as pd

df = pd.DataFrame({
    "customer_id": [1, 2, 1, 3, 2, 4, 3, 1],
    "country": ["IT", "IT", "IT", "FR", "IT", "DE", "FR", "IT"],
    "category": [
        "electronics",
        "clothing",
        "electronics",
        "food",
        "food",
        "electronics",
        "clothing",
        "food",
        ],
    "amount": [120, 50, 80, 30, 70, 200, 40, 90]
})

def analyze_transactions(df):
    total_sales = df["amount"].sum()
    average_sale = df["amount"].mean()
    
    customer_sales = df.groupby("customer_id")["amount"].sum()
    top_customer = customer_sales.idxmax()
    category_sales = df.groupby("category")["amount"].sum()
    top_category = category_sales.idxmax()

    country_sales = df.groupby("country")["amount"].sum().to_dict()
    transactions_per_customer = df.groupby("customer_id").size().to_dict()
    result = {
        "total_sales": total_sales,
        "average_sale": average_sale,
        "top_customer": top_customer,
        "top_category": top_category,
        "sales_by_country": country_sales,
        "transactions_per_customer": transactions_per_customer
    }
    return result

res = analyze_transactions(df)

assert res["total_sales"] == 680
assert res["average_sale"] == 85.0
assert res["top_customer"] == 1
assert res["top_category"] == "electronics"
assert res["sales_by_country"] == {
    "DE": 200,
    "FR": 70,
    "IT": 410
}
assert res["transactions_per_customer"] == {
    1: 3,
    2: 2,
    3: 2,
    4: 1
}

print("All tests passed!")
