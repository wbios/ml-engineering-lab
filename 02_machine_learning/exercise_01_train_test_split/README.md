# Exercise 04 - Train/Test Split & Data Leakage

## Goal

The goal of this exercise is to practice splitting a dataset into training and test sets and applying feature scaling without introducing data leakage.

The exercise uses Pandas and scikit-learn.

## Dataset

The dataset contains customer information and a binary target indicating whether a customer purchased a product.

| Column | Description |
| ------ | ----------- |
| 'age' | Customer age |
| 'income' | Customer income |
| 'purchased' | Purchase target ('0' or '1') |

## Train/Test Split

The 'split_data()' function separates the dataset into:

- 'X' - input features
- 'y' - target variable

The dataset is then divided into:

- 80% training data
- 20% test data

The split uses 'random_state=42' to make the result reproducible.

## Data Validation

The exercise validates the train/test split using assertions.

The tests verify that:

- the training set contains 8 rows;
- the test set contains 2 rows;
- the feature and target sets have matching sizes;
- no rows are shared between training and test sets;
- no rows are lost during the split.

## Feature Scaling

The 'scale_data()' function uses 'StandardScaler' to standardize the features.

The scaler is fitted only on the training data:

X_train_scaled = scaler.fit_transform(X_train)

The same scaler is then used to transform the test data:

X_test_scaled = scaler.transform(X_test)

#Data Leakage: 

Data leakage occurs when information from the test set is used during the training process.


## Usage

From the project root, activate the virtual environment:

.\.venv\Scripts\Activate.ps1

Install the dependencies:

python -m pip install -r .\02_machine_learning\exercise_01_train_test_split\requirements.txt

Run the exercise:

python .\02_machine_learning\exercise_01_train_test_split\exercise.py

