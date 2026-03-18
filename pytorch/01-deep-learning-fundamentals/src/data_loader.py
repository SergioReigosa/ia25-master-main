"""
Data loading utilities for the UCI Thyroid Disease dataset.

This module handles downloading the dataset from Kaggle and applying the target
variable simplification (from 30+ original categories to 3: hyperthyroid, 
hypothyroid, negative).
"""

import os
from pathlib import Path

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split


def download_kaggle_dataset() -> Path:
    """
    Download the thyroid disease dataset from Kaggle.
    
    Uses kagglehub to download and cache the dataset locally.
    
    Returns:
        Path to the downloaded dataset directory.
    """
    path = kagglehub.dataset_download("emmanuelfwerr/thyroid-disease-data")
    return Path(path)


def simplify_thyroid_class(diagnosis_string: str) -> str:
    """
    Map original thyroid diagnosis codes to simplified 3-class target.
    
    The original UCI dataset has 30+ diagnosis categories. This function
    simplifies them into three clinically meaningful groups:
    
    - 'hyperthyroid': Codes A, B, C, D (overactive thyroid)
    - 'hypothyroid': Codes E, F, G, H (underactive thyroid)  
    - 'negative': All other codes (no thyroid dysfunction)
    
    This simplification facilitates model interpretation and reduces
    the severe class imbalance present in the original fine-grained labels.
    
    Args:
        diagnosis_string: Original diagnosis code from the dataset.
        
    Returns:
        One of 'hyperthyroid', 'hypothyroid', or 'negative'.
    """
    if any(code in diagnosis_string for code in ['A', 'B', 'C', 'D']):
        return 'hyperthyroid'
    elif any(code in diagnosis_string for code in ['E', 'F', 'G', 'H']):
        return 'hypothyroid'
    else:
        return 'negative'


def load_thyroid_data_3_classes(
    test_size: float | None = None,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the thyroid dataset with simplified 3-class target variable.
    
    Downloads the dataset from Kaggle (if not cached), applies the target
    simplification, and removes non-predictive columns (patient_id, 
    referral_source).
    
    Args:
        test_size: If provided, split data into train/test sets with this 
                   proportion for test. If None, return full dataset.
        random_state: Random seed for reproducible train/test splits.
        
    Returns:
        If test_size is None: (X, y) tuple with features and target.
        If test_size is provided: (X_train, X_test, y_train, y_test) tuple.
    """
    dataset_path = download_kaggle_dataset()
    df = pd.read_csv(dataset_path / "thyroidDF.csv")

    df['target'] = df['target'].apply(simplify_thyroid_class)
    df = df.dropna(subset=['target'])

    columns_to_drop = ['patient_id', 'referral_source', 'target']
    X = df.drop(columns=columns_to_drop)
    y = df['target']

    if test_size is not None:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    return X, y


if __name__ == "__main__":
    X, y = load_thyroid_data_3_classes()
    print("Dataset shape:", X.shape)
    print("\nClass distribution:")
    print(y.value_counts())
    print("\nClass proportions:")
    print(y.value_counts(normalize=True).round(3))
