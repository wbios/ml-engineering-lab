\# Exercise 03 - Data Quality \& Validation

\## Goal

The goal of this exercise is to pratice basic data quality checks using Pandas.

The validation function identifies missing values, invalid values, and duplicate customer IDs before the data can be used in a Machine Learning pipeline.

\## Dataset

The dataset contains customer information and transaction data.

| Column | Description |
| ------ | ----------- |
| 'customer\_id' | Unique customer identifier |
| 'age' | Customer age |
| 'country' | 'Customer country' |
| 'amount' | Transaction amount |

The dataset intentionally contains invalid and missing values.

\##Data Quality Checks

The 'validate\_data()' function performs the following checks:

1. Count missing values for each column.
2. Identify invalid ages.
3. Identify negative transaction amounts.
4. Identify duplicate customer IDs.

## Validation Rules

### Missing values

Missing values are counted for every column.

### 'age'

Valid ages must satisfy:

0 <= age <= 120

## Usage

From the project root, activate the virtual environment:

'''powershell

.\.venv\Scripts\Activate.ps1

python -m pip install -r .\01_python_pandas\exercise_03_data_validation\requirements.txt

python .\01_python_pandas\exercise_03_data_validation\exercise.py













