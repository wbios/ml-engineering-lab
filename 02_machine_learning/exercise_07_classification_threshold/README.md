# Exercise 07 - Classification Threshold

## Goal

The goal of this exercise is to understand how changing the classification threshold affects the predictions of a Logistic Regression model.

We first use the standard classification threshold and then apply a lower threshold of 0.2.

The comparison shows how changing the threshold can affect the trade-off betweem false positives and false negatives, and can change precision and recall.

## Dataset

The dataset contains 100 samples with one feature called score.

The target variable is purchased:

    0 --> 90 samples
    1 --> 10 samples

The feature values are generated using Numpy from two normal distributions.
The two classes have different average values but can overlap.

A random seed is used to make the generated dataset reproducible.

## Stratified Split

The dataset is split into training and test sets using train_test_split.

A test size of 0.2 is used, together with random_state=42 and stratify=y.

This preserves the class distribution in both sets:

    Training set: 72 samples of class 0 and 8 samples of class 1 
    Test set: 18 samples of class 0 and 2 samples of class 1

The stratified split ensures that the minority class is represented in both the training and test sets.

## Logistic Regression

A Logistic Regression model is trained using the training data.

The model is trained without specifying class weights.

After training, the model is used to make predictions on the test set.

Using the standard classification threshold produces the following confusion matrix:

```python
    [[18 0]
    [2 0]]
```

The model correctly classifies all 18 samples of class 0, but it does not correctly identify either of the 2 samples belonging to class 1.

## Classification Threshold

Instead of using the predctions returned by model.predict(), the exercise uses predict_proba() to obtain the probability assigned to class 1.

The probability of class 1 is selected using:

```python
    y_proba_class_1 = y_proba[:, 1]
```

A threshold of 0.2 is then applied:

```python
    y_pred_threshold = (y_proba_class_1 >= threshold).astype(int)
```

This means that samples with a predicted probability of at least 0.2 for class 1 are classified as class 1.

The resulting confusion matrix is:

```python
    [[15 3]
    [1 1]]
```

Comprared with the standard threshold, the model identifies one sample from the minority class.

However, it also produces three false positives.

## Results

The two thresholds produces different results on the same test set.

### Standard threshold

```python
    [[18 0]
    [2 0]]
```

The model does not identify any samples from the minority class.

### Threshold 0.2

```python
    [[15 3]
    [1 1]]
```

The model identifies one sample from the minority class.

The resulting metrics are:

    Precision: 0.25
    Recall: 0.5

Compared with the standard threshold, the lower threshold increases the number of positives predictions.

This allos the model to identify a sample from the minority class, but it also creates false positives.

The exercise demonstrates that changing the classification threshold can change the model's predictions and the resulting precision and recall.

## Usage

From the project root, activate the virtual environment:

    .\.venv\Scripts\Activate.ps1

Install the dependencies:

    python -m pip install -r .\02_machine_learning\exercise_07_classification_threshold\requirements.txt

Run the exercise:

    python .\02_machine_learning\exercise_07_classification_threshold\exercise.py

