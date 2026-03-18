"""
Preprocessing pipelines for thyroid disease classification.

This module implements three distinct preprocessing strategies, each designed
for a specific type of model. The strategies differ in how they handle missing
values, which is a critical consideration for this dataset.

UNDERSTANDING MISSINGNESS IN THE THYROID DATASET
------------------------------------------------
Not all missing values are created equal:

1. "Missing by Design" (TBG column):
   - TBG (Thyroxine-Binding Globulin) is measured in only ~10% of patients
   - This is NOT random: doctors order TBG only when clinically indicated
   - The binary flag TBG_measured is informative: "Doctor thought TBG was relevant"
   
2. "Missing by Chance" (T3, TSH, etc.):
   - These common thyroid panel tests are missing for various reasons
   - May be random (lab error, sample issues) or systematic (test ordering patterns)

STRATEGY OVERVIEW
-----------------
- Simple Imputation: Global median/mode imputation with scaling (for linear models)
- Native NaN: Pass NaN directly to tree models that handle them natively
- Zero with Flags: Fill with zeros but keep _measured columns as features (for NNs)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Feature groups
NUMERIC_FEATURES = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
BINARY_FEATURES = [
    'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment',
    'query_hypothyroid', 'query_hyperthyroid', 'lithium',
    'goitre', 'tumor', 'hypopituitary', 'psych'
]
CATEGORICAL_FEATURES = ['sex']
MEASURED_FLAGS = ['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured']


class OutlierToNanTransformer(BaseEstimator, TransformerMixin):
    """
    Convert extreme outliers to NaN for subsequent imputation.
    
    The age column in this dataset contains biologically impossible values
    (e.g., age > 150). Rather than dropping rows, we convert these to NaN
    and let the imputer handle them, preserving other feature information.
    """
    
    def __init__(self, threshold: float = 150.0):
        self.threshold = threshold
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        X_transformed = X.copy().astype(float) # Ensure float for NaN support
        X_transformed[X_transformed > self.threshold] = np.nan
        return X_transformed


class TSHLogTransformer(BaseEstimator, TransformerMixin):
    """
    Apply log transformation to TSH values.
    
    TSH (Thyroid Stimulating Hormone) has a highly right-skewed distribution
    with values ranging from near-zero to hundreds. Log transformation 
    makes the distribution more symmetric and helps linear models.
    
    Uses log1p (log(1 + x)) to safely handle zero values.
    """
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.log1p(X)


def get_simple_imputation_pipeline() -> ColumnTransformer:
    """
    Create preprocessing pipeline with global median/mode imputation.
    
    This strategy is designed for linear models like Logistic Regression that:
    - Cannot handle NaN values
    - Benefit from scaled features (StandardScaler)
    - Need explicit imputation for all missing values
    
    Key decisions:
    - Age: Outliers converted to NaN, then median imputed
    - TSH: Median imputed, then log-transformed (highly skewed)
    - Other numerics: Median imputed (robust to outliers)
    - TBG: Dropped (90%+ missing), but TBG_measured kept as informative flag
    - Categorical: Mode imputed, then one-hot encoded
    - Binary t/f columns: One-hot encoded with drop='if_binary'
    
    Returns:
        Fitted ColumnTransformer ready for pipeline use.
    """
    age_pipeline = Pipeline([
        ('outlier_to_nan', OutlierToNanTransformer(threshold=150)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    tsh_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log_transform', TSHLogTransformer()),
        ('scaler', StandardScaler())
    ])
    
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    binary_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
    ])
    
    # Features to use (excluding TBG numeric, keeping TBG_measured)
    standard_numeric = ['T3', 'TT4', 'T4U', 'FTI']
    
    return ColumnTransformer(
        transformers=[
            ('age', age_pipeline, ['age']),
            ('tsh', tsh_pipeline, ['TSH']),
            ('numeric', numeric_pipeline, standard_numeric),
            ('binary', binary_pipeline, BINARY_FEATURES),
            ('categorical', categorical_pipeline, CATEGORICAL_FEATURES),
            ('measured_flags', binary_pipeline, ['TBG_measured']),
        ],
        remainder='drop'
    )


def get_native_nan_pipeline() -> ColumnTransformer:
    """
    Create preprocessing pipeline that preserves NaN for tree-based models.
    
    This strategy is designed for tree models (Random Forest, XGBoost, CatBoost) 
    that can handle NaN values natively:
    - scikit-learn >= 1.4: RandomForest supports NaN
    - XGBoost/CatBoost: Native missing value handling by learning optimal splits
    
    Key decisions:
    - Numeric features: Pass through with NaN preserved (no imputation)
    - _measured columns: DROPPED (tree models learn splits on NaN, making flags redundant)
    - Categorical: One-hot encoded
    - Binary: One-hot encoded
    
    Note: Requires scikit-learn >= 1.4 for Random Forest NaN support.
    
    Returns:
        Fitted ColumnTransformer ready for pipeline use.
    """
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    binary_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
    ])
    
    return ColumnTransformer(
        transformers=[
            ('numeric', 'passthrough', NUMERIC_FEATURES),
            ('binary', binary_pipeline, BINARY_FEATURES),
            ('categorical', categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder='drop'  # Drops all _measured columns
    )


def get_zero_imputation_with_flags_pipeline() -> ColumnTransformer:
    """
    Create preprocessing pipeline with zero imputation and missingness flags.
    
    This strategy is designed for neural networks that:
    - Cannot handle NaN values (require numeric tensors)
    - Can learn missingness patterns through explicit flags
    - Benefit from the "missingness indicator" pattern
    
    Key decisions:
    - All numeric features: Impute with 0.0 (neutral for neural nets with batch norm)
    - ALL _measured flags: KEPT (network learns to weight the 0s appropriately)
    - Features scaled after imputation
    - Categorical/Binary: One-hot encoded as usual
    
    The intuition: when TSH is missing (TSH=0, TSH_measured=False), the network
    learns that 0 doesn't mean "zero TSH" but rather "TSH not available".
    
    Returns:
        Fitted ColumnTransformer ready for pipeline use.
    """
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    binary_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
    ])
    
    return ColumnTransformer(
        transformers=[
            ('numeric', numeric_pipeline, NUMERIC_FEATURES),
            ('binary', binary_pipeline, BINARY_FEATURES),
            ('categorical', categorical_pipeline, CATEGORICAL_FEATURES),
            ('measured_flags', binary_pipeline, MEASURED_FLAGS),
        ],
        remainder='drop'
    )
