# Exercise 10 - Cross-Validation

## Goal

The goal of this exercise is to introduce cross-validation using StratifiedKFold and cross_val_score().

The exercise builds on the train/test split and stratification concepts introduced in previous exercises.

Instead of evaluting a model using a single train/test division, cross-validation evaluates the model across multiple training and validation splits.

## Dataset

The dataset contains 100 samples:
    90 samples belong to class 0
    10 samples belong to class 1

The dataset contains one feature:
    Score

The feature values are generated using Numpy from two normal distributions.

A fixed random seed is used:

```python
    np.random.seed(42)
```

This makes the generated dataset reproducible.

## Stratified K-Fold

The dataset is divided into 5 folds using StratifiedKFold():

```python
    kf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
```

StratifiedKFold keeps the class distribution consistent across the folds.

With 100 samples and 5 folds, each validation fold contains 20 samples and the remaining 80 samples are used for training.

Because the dataset contains 90 samples from class 0 and 10 samples from class 1, each validation fold contains:
    Class 0: 18 samples
    Class 1: 2 samples

## Logistic Regression

A Logistic Regression model is used for the cross-validation:

```python
    model = LogisticRegression()
```

The model is trained separately on the training portion of each fold.

It is then evaluated on the corresponding validation portion.

## Cross-Validation

The cross-validation is performed using cross_val_score():

```python
    scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="accuracy"
    )
```

The cv parameter receives the StratifiedKFold object.

This means that cross_val_score() uses the splitting strategy defined by StratifiedKFold.

For each fold, the process is:
    Training data --> Model training --> Validation data --> Prediction --> Accuracy

The process is repeated for all 5 folds.

## Results

The cross-validation produces one accuracy value for each fold:

```python
    Cross-validation scores: [0.9 0.9 0.9 0.9 0.9]
```

The mean accuracy is calculated using:

```python
    mean_accuracy = scores.mean()
```

The result is:
    Mean accuracy: 0.9

The individual scores show the model performance on each validation fold, while the mean provides a single summary of the results across the five folds.

## Usage

From the project root, activate the virtual environment:

    .\.venv\Scripts\Activate.ps1

Install the dependencies:

    python -m pip install -r .\02_machine_learning\exercise_10_cross_validation\requirements.txt

Run the exercise:
    
    python .\02_machine_learning\exercise_10_cross_validation\exercise.py