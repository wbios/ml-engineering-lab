# Exercise 06 - Class Weight

## Goal

The goal of this exercise is to understand how class_weight="balanced" affects Logistic Regression when working with an imbalanced dataset.

We compare two Logistic Regression models:

    one without class weight;
    one using class_weight="balanced".

The comparison shows how giving more importance to the minority class can improve its recall, while potentially increasing false positivies for the majority class.

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

The stratified split ensures that the minority class is represented in both  the training and test sets.

## Logistic Regression Without Class Weight

First, a Logistic Regression model is trained without specifying class weights.

The model is trained on an imbalanced dataset containing more samples from class 0 than from class 1.

The resulting confusion matrix is:

```python
    [[18 0]
    [2 0]]
```

The model correctly classifies all 18 samples of class 0, but it does not correctly identify either of the 2 samples belonging to class 1.

This results in:

    Precision: 0.0
    Recall: 0.0
    F1-score: 0.0

The result shows that a model can perform well on the majority class while failing to identify the minority class.

## Logistic Regression With Class Weight

A second Logistic Regression model is trained using:

```python
    class_weight="balanced"
```

This gives more importances to the minority class during model training.

The resulting confusion matrix is:

```python
    [[13 5]
    [0 2]]
```

Compared with the model without class weights, the balanced model correctly identifies both samples of the minority class.

However, it also produces 5 false positives for the majority class.

This shows that giving more importance to the minority class can improve its recognition, but it can also change the trade-off between false positives and false negatives.

## Results

The two models produce different results on the same test set.

### Without class weights

```python
    [[18 0]
    [2 0]]
```

The model does not identify any samples from the minority class.

    True Positives: 0
    False Negatives: 2
    Recall: 0.0
    F1-score: 0.0

### With class_weight="balanced"

```python
    [[13 5]
    [0 2]]
```

The model identifies both samples from the minority class.

    True Positives: 2
    False Negatives: 0
    False Positives: 5
    Recall: 1.0
    F1-score: 0.4444

The balanced model improves the recognition of the minority class, but this comes with more false positives for the majority class.

The exercise demonstrates that class_weight="balanced" can change the model's behavior when the dataset is imbalanced. It does not guarantee better performance on every metric.

## Usage

From the project root, activate the virtual environment:

    .\.venv\Scripts\Activate.ps1

Install the dependencies:

    python -m pip install -r .\02_machine_learning\exercise_06_class_weight\requirements.txt

Run the exercise:

    python .\02_machine_learning\exercise_06_class_weight\exercise.py