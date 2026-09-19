# Exercise 08 - ROC Curve and AUC

## Goal

The goal of this exercise is to introduce the ROC curve and AUC using the predicted probabilities produced by a Logistic Regression model.

The exercise builds on the predict_proba() method introduced in the classification threshold exercise.

Instead of evaluating the model at a single classification threshold, the ROC curve shows how the False Positive Rate (FPR) and True Positive Rate (TPR) change across different thresholds.

AUC (Area Under the Curve) summarizes the area under the ROC curve with a single value.

## Dataset

The dataset contains 100 samples:
    Class 0: 90 samples
    Class 1: 10 samples

The dataset contains one feature:
    Score

The feature values are generated using Numpy from two normal distributions.

A fixed random seed is used:

```python
    np.random.seed(42)
```

This makes the generated dataset reproducible,

## Stratified Split

The dataset is split into training and test sets using train_test_split():

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

A Logistic Regression model is trained using the training data:

```python
    model = LogisticRegression()

    model.fit(X_train, y_train)
```

The standard predictions are generated using:

```python
    y_pred = model.predict(X_test)
```

A confusion matrix is calculated:

```python
    cm_default = confusion_matrix(y_test, y_pred)
```

The result is:

```python
    [[18 0]
    [2 0]]
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

These probabilities are used as the input for the ROC curve calculation.

## ROC Curve

The ROC curve is calculated using roc_curve():

```python
    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_proba_class_1
    )
```
The function returns three arrays:
    fpr: False Positive Rate values
    tpr: True Positive Rate values
    thresholds: classification thresholds associated with the ROC points

Each position in the three arrays represents the same point of the ROC curve.

The ROC curve is plotted using FPR on the x-axis and TPR on the y-axis:

```python
    plt.plot(fpr, tpr)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
```

For this dataset, the calculated values are:

```python
    FPR: [0. 0.05555556 0.16666667 0.16666667 1.]
    TPR: [0. 0. 0. 1. 1.]
    Thresholds: [inf 0.27412577 0.25992288 0.1953024 0.0058086]
```

Each pair (FPR, TPR) represents the model behavior at a different threshold.

## AUC

The Area Under the ROC Curve is calculated using roc_auc_score():

```python
    auc = roc_auc_score(
        y_test,
        y_proba_class_1
    )
```

The result for this exercise is:

```python
    AUC: 0.8333333333333334
```

AUC represents the area under the ROC curve.

An AUC of 0.5 corresponds to the diagonal reference line of the ROC plot, while an AUC of 1.0 corresponds to a perfect ROC curve.

The AUC value is not a probability that an individual prediction is correct. It is a summary measure based on the ROC curve.

## Results

The standard-threshold confusion matrix is:

```python
    [[18 0]
    [2 0]]
```

The ROC curve produces the following values:

```python
    FPR: [0. 0.0555556 0.16666667 0.16666667 1.]
    TPR: [0. 0. 0. 1. 1.]
```

The AUC is:

    0.8333333333333334

## Usage

From the project root, activate the virtual environment:

    .\.venv\Scripts\Activate.ps1

Install the dependencies:

    python -m pip install -r .\02_machine_learning\exercise_08_roc_auc\requirements.txt

Run the exercise:

    python .\02_machine_learning\exercise_08_roc_auc\exercise.py
