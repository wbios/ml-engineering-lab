# Exercise 05 - Model Training & Evaluation

## Goal
The goal of this exercise is to train a classification model and evaluate its predictions using common machine learning metrics.

The exercise uses Pandas and scikit-learn.

## Dataset 

The dataset contains customer information and a binary target indicating whether a customer purchased a product.

| Column | Description |
| ------ | ----------- |
| 'age' | Customer age |
| 'income' | Customer income |
| 'purchased' | Purchased target ('0' or '1') |

## Train/Test Split

The dataset is divided into training and test sets using `train_test_split()`.

The split uses:

- 80% training data
- 20% test data
- `ramdom_state=42` for reproducibility

The input data is divided into:

- Features: `age` and `income`
- Target: `purchased`

```python
X = df[["age", "income"]]
y = df["purchased"]
```

## Model

The exercise uses `LogisticRegression` as a binary classification model.

## Prediction

The `predict()` function uses the trained model to generate predictions for the test set

## Evaluation

The `evaluate_model()` function compares the predictions with the actual target values from the test set.

### Metrics

Accuracy measures the proportion of correct predictions among all predictions.

Precision measures how many of the samples predicted as positive are actually positive.

Recall measures how many of the actual positive samples are correctly identified by the model.

F1-score combines precision and recall into a single metric and provides a balance between the two.

## Usage
From the project root, activate the virtual environment:

.\.venv\Scripts\Activate.ps1

Install the dependencies:

python -m pip install -r .\02_machine_learning\exercise_02_model_training\requirements.txt

