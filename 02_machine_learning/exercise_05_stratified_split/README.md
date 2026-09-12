# Exercise 05 - Stratified Split

## Goal

Understand why stratified train/test splitting is useful when working with an imbalanced dataset.

The dataset contains two classes:

    0 --> 90 samples
    1 --> 10 samples

The goal is to split the dataset into training and test sets while preserving the original class distribution.

## Dataset

The exercise uses a simple dataset with 100 samples.

The feature is:

    age --> values from 20 to 119

The target is:

    purchased --> 90 samples with value 0 and 10 samples with value 1

The target is stored in a Pandas Series, while the feature is stores in a Pandas DataFrame.

## Random Train/Test split

First, the dataset is split using train_test_split() without  stratification:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This produces:

    Training:
    0 --> 71
    1 --> 9

    Test:
    0 --> 19
    1 --> 1

The training and test sets have the expected number of samples, but the class distribution is different from the original 90% / 10% distribution.

## Stratified Train/Test Split

The dataset is then split using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Using stratify=y preserves the class distribution in the training and test sets.

The result is:

    Training:
    0 --> 72
    1 --> 8

    Test:
    0 --> 18
    1 --> 2

Both sets therefore maintain the original 90% / 10% class distribution.

## Assertions

The exercise uses assert statements to verify that the strtified split produces the expected number of samples for each class.

## Requirements

    Python
    pandas
    scikit-learn

## Usage

From the project root, activate the virtual environment:

    .\venv\Scripts\Activate.ps1

Install the dependencies:

    python -m pip install -r .\02_machine_learning\exercise_05_stratified_split\requirements.txt

Run the exercise:

    python .\02_machine_learning\exercise_05_stratified_Split\exercise.py

