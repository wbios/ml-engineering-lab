# Exercise 09 - Precision-Recall Curve

## Goal

The goal of this exercise is to introduce the Precision-Recall curve using the predicted probabilities produced by a Logistic Regression model.

The exercise builds on the predict_proba() method introduced in the classification threshold exercise.

Instead of evaluating the model at manually selected classification thresholds, the Precision-Recall curve shows how Precision and Recall change across different thresholds.

## Dataset

The dataset contains 100 samples:
    90 samples belong to class 0.
    10 samples belong to class 1.

The dataset contains one feature:
    Score

The feature values are generated using Numpy from two normal distributions.

A fixed random seed is used:

```python
    np.random.seed(42)
```
This makes the generated dataset reproducible.

## Stratified Split

The dataset is split into training and test set using train_test_split():

```python
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
```
The resulting split contains:
    Training set: 80 samples
        Class 0: 72
        Class 1: 8
    Test set: 20 samples
        Class 0: 18
        Class 1: 2

Assertions are used to verify the expected split.

## Logistic Regression

A Logistic Regression model is trained on the training data.

```python
    model = LogisticRegression()
    model.fit(X_train, y_train)
```

## Predicted Probabilities

The model probabilities are obtained using predict_proba():

```python
    y_proba = model.predict_proba(X_test)
```

The probabilities for class 1 are selected using:

```python
    y_proba_class_1 = y_proba[:, 1]
```

These probabilities are used as the input for the Precision-Recall curve calculation.

## Precision-Recall Curve

The Precision-Recall curve is calculated using precision_recall_curve():

```python
    precisions, recalls, thresholds = precision_recall_curve(
        y_test,
        y_proba_class_1
    )
```

The function returns three arrays:
    precicions: Precision values
    recalls: Recall values
    thresholds: classification thresholds associated with the curve

The Precision-Recall curve is plotted using Recall on the x-axis and Precision on the y-axis:

```python
    plt.plot(recalls, precisions)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
```
The curve represents the relationship between Precision and Recall as the classification threshold changes.

The returned precisions and recalls arrays contain one more value than the thresholds array. The additional Precision-Recall point represents the case where no samples are predicted as positive.

## Results

For this dataset, the model produces the following probabilities for class 1.

The precision_recall_curve() funtion calculates Precision and Recall for the different thresholds derived from the predicted probabilities.

The resulting values are used to generate the Precision-Recall curve.

## Usage

From the project root, activate the virtual environment:

    .\.venv\Scripts\Activate.ps1

Install the dependencies:

    python -m pip install -r .\02_machine_learning\exercise_09_precision_recall_curve\requirements.txt

Run the exercise:

    python .\02_machine_learning\exercise_09_precision_recall_curve\requirements.txt