"""
Custom evaluation metrics for thyroid disease classification.

Defines a custom metric that averages the recall of the two disease classes
(hyperthyroid and hypothyroid) to prioritize detection of sick patients
regardless of class imbalance.
"""

import numpy as np
from sklearn.metrics import make_scorer, recall_score


def thyroid_mean_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the mean recall for hyperthyroid and hypothyroid classes.
    
    This metric focuses exclusively on the minority "sick" classes, ignoring
    the dominant "negative" class. It provides a fair assessment of how well
    the model detects actual thyroid conditions.
    
    Args:
        y_true: Ground truth labels (string or integer encoded).
        y_pred: Predicted labels (same encoding as y_true).
        
    Returns:
        Mean of hyperthyroid recall and hypothyroid recall (0.0 to 1.0).
    """
    # Convert to numpy arrays to handle pandas Series with non-sequential indices
    # (which occur during cross-validation splits where the original index is preserved)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle both string labels and integer-encoded labels
    if isinstance(y_true[0], (int, np.integer)):
        # Integer labels: assume standard LabelEncoder ordering
        # alphabetical: hyperthyroid=0, hypothyroid=1, negative=2
        hyper_label = 0
        hypo_label = 1
    else:
        # String labels
        hyper_label = 'hyperthyroid'
        hypo_label = 'hypothyroid'
    
    hyper_recall = recall_score(
        y_true, y_pred, labels=[hyper_label], average=None, zero_division=0
    )[0]
    hypo_recall = recall_score(
        y_true, y_pred, labels=[hypo_label], average=None, zero_division=0
    )[0]
    
    return float(np.mean([hyper_recall, hypo_recall]))


def thyroid_recall_scorer(estimator, X, y) -> float:
    """
    Scorer function compatible with scikit-learn's cross_val_score and GridSearchCV.
    
    This wrapper allows using thyroid_mean_recall directly with scikit-learn's
    cross-validation utilities:
    
        scores = cross_val_score(model, X, y, scoring=thyroid_recall_scorer)
    
    Args:
        estimator: Fitted scikit-learn estimator with predict method.
        X: Feature matrix.
        y: True labels.
        
    Returns:
        Thyroid mean recall score.
    """
    y_pred = estimator.predict(X)
    return thyroid_mean_recall(y, y_pred)


# Pre-built scorer for convenience with cross_val_score, GridSearchCV, etc.
thyroid_scorer = make_scorer(thyroid_mean_recall)
