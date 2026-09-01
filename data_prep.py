"""
Data Preparation and Preprocessing Module for AI Fraud Detection System.

This module handles:
1. Loading transaction data (creditcard.csv) with automated cleaning and deduplication.
2. Automated detection of numerical and categorical features.
3. Robust handling of missing values (imputation).
4. Feature scaling and one-hot encoding without data leakage.
5. Stratified train/test splitting (80/20).
6. SMOTE balancing applied strictly on the training partition.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE


@dataclass
class PreparedData:
    """Dataclass holding preprocessed data splits, transformations, and metadata."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    feature_names: List[str]
    numerical_cols: List[str]
    categorical_cols: List[str]
    raw_train_shape: Tuple[int, int]
    resampled_train_shape: Tuple[int, int]


def generate_synthetic_transactions(
    n_samples: int = 5000,
    random_state: int = 42,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate a realistic synthetic transaction dataset matching credit card fraud PCA schema.
    
    Features include:
    - Time: Seconds elapsed
    - V1 through V28: PCA component features
    - Amount: Transaction monetary amount
    - Class: Binary target (0 = Legitimate, 1 = Fraudulent)
    
    Args:
        n_samples: Total number of records to generate.
        random_state: Seed for reproducibility.
        output_path: Optional file path to persist the CSV.
        
    Returns:
        pd.DataFrame containing synthetic transactions.
    """
    rng = np.random.default_rng(random_state)
    
    time_val = np.sort(rng.integers(0, 172800, size=n_samples)).astype(float)
    amount = rng.lognormal(mean=3.5, sigma=1.2, size=n_samples)
    
    # 28 PCA features standard normal
    pca_features = {f"V{i}": rng.normal(0, 1, size=n_samples) for i in range(1, 29)}
    
    # Imbalance ~0.5% fraud
    fraud_prob_base = np.full(n_samples, 0.005)
    # Correlation with specific V features and high amounts
    fraud_prob_base += np.where(pca_features["V14"] < -2.0, 0.25, 0.0)
    fraud_prob_base += np.where(pca_features["V10"] < -2.0, 0.20, 0.0)
    fraud_prob_base += np.where(amount > np.percentile(amount, 95), 0.05, 0.0)
    
    fraud_prob = np.clip(fraud_prob_base, 0.001, 0.90)
    is_fraud = (rng.random(size=n_samples) < fraud_prob).astype(int)
    
    data_dict = {"Time": time_val}
    data_dict.update(pca_features)
    data_dict["Amount"] = np.round(amount, 2)
    data_dict["Class"] = is_fraud
    
    df = pd.DataFrame(data_dict)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[DataPrep] Synthetic dataset generated and saved to '{output_path}' ({len(df)} rows, fraud rate: {df['Class'].mean():.2%}).")
        
    return df


def load_data(
    file_path: str = "data/creditcard.csv",
    generate_if_missing: bool = True,
    drop_duplicates: bool = True,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load dataset from CSV, handling fallback paths and cleaning duplicate records.
    
    Args:
        file_path: Path to dataset CSV (e.g. data/creditcard.csv).
        generate_if_missing: If True, generate synthetic data if no file is found.
        drop_duplicates: If True, remove duplicate records.
        random_state: Seed if synthetic data is generated.
        
    Returns:
        pd.DataFrame containing transaction data.
    """
    if not os.path.exists(file_path):
        # Check alternative common locations
        alt_paths = [
            "fraud_model_comparison/data/creditcard.csv",
            "data/creditcard.csv",
            "fraud_model_comparison/data/transactions.csv",
            "data/transactions.csv"
        ]
        found_path = None
        for p in alt_paths:
            if os.path.exists(p):
                found_path = p
                break
                
        if found_path:
            file_path = found_path
        elif generate_if_missing:
            print(f"[DataPrep] Dataset not found at '{file_path}'. Auto-generating synthetic dataset for validation...")
            return generate_synthetic_transactions(n_samples=5000, random_state=random_state, output_path=file_path)
        else:
            raise FileNotFoundError(f"Dataset not found at '{file_path}' and generate_if_missing is False.")
    
    df = pd.read_csv(file_path)
    initial_rows = len(df)
    print(f"[DataPrep] Successfully loaded dataset from '{file_path}' ({initial_rows:,} rows, {df.shape[1]} columns).")
    
    # Check for missing values
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls > 0:
        print(f"[DataPrep] Found {total_nulls} missing values across {(null_counts > 0).sum()} columns.")
    else:
        print("[DataPrep] Data integrity check: 0 missing values detected.")
        
    # Check and handle duplicates
    num_duplicates = int(df.duplicated().sum())
    if num_duplicates > 0:
        if drop_duplicates:
            df = df.drop_duplicates().reset_index(drop=True)
            print(f"[DataPrep] Data cleaning: Removed {num_duplicates:,} duplicate rows ({initial_rows:,} -> {len(df):,} rows).")
        else:
            print(f"[DataPrep] Warning: Dataset contains {num_duplicates:,} duplicate rows (retained).")
            
    return df


def detect_feature_types(
    df: pd.DataFrame,
    target_col: str = "Class",
    id_cols: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    Automatically identify numerical and categorical features, excluding target and ID columns.
    
    Args:
        df: Input dataframe.
        target_col: Target column name.
        id_cols: List of column names or patterns to ignore (identifiers).
        
    Returns:
        Tuple of (numerical_columns, categorical_columns).
    """
    if id_cols is None:
        id_cols = ['transaction_id', 'id', 'txn_id', 'user_id', 'customer_id', 'index']
        
    candidate_cols = [
        col for col in df.columns 
        if col != target_col and col.lower() not in [i.lower() for i in id_cols]
    ]
    
    numerical_cols = []
    categorical_cols = []
    
    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
            
    return numerical_cols, categorical_cols


def build_preprocessor(
    numerical_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer for imputing, scaling, and one-hot encoding.
    
    - Numerical Pipeline: SimpleImputer(median) -> StandardScaler()
    - Categorical Pipeline: SimpleImputer(most_frequent) -> OneHotEncoder(ignore unknown)
    
    Args:
        numerical_cols: List of numerical feature names.
        categorical_cols: List of categorical feature names.
        
    Returns:
        Configured ColumnTransformer instance.
    """
    from sklearn.pipeline import Pipeline
    
    transformers = []
    
    if numerical_cols:
        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_pipeline, numerical_cols))
        
    if categorical_cols:
        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipeline, categorical_cols))
        
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    return preprocessor


def prepare_fraud_data(
    file_path: str = "data/creditcard.csv",
    target_col: Optional[str] = None,
    test_size: float = 0.20,
    random_state: int = 42
) -> PreparedData:
    """
    Full data preparation pipeline:
    1. Load transaction dataset with automated cleaning.
    2. Detect numerical & categorical features.
    3. Perform stratified train/test split.
    4. Fit preprocessor STRICTLY on train data, transform train & test (preventing leakage).
    5. Balance training set using SMOTE (test set remains untouched).
    
    Args:
        file_path: CSV path (defaults to data/creditcard.csv).
        target_col: Name of fraud target column. If None, auto-detects 'Class' or 'is_fraud'.
        test_size: Proportion of dataset for testing (default 0.20 for 80/20 split).
        random_state: Random seed for reproducibility.
        
    Returns:
        PreparedData dataclass containing splits, transformer, and metadata.
    """
    df = load_data(file_path=file_path, generate_if_missing=True, drop_duplicates=True, random_state=random_state)
    
    # Auto-detect target column if not specified
    if target_col is None:
        if "Class" in df.columns:
            target_col = "Class"
        elif "is_fraud" in df.columns:
            target_col = "is_fraud"
        elif "fraud" in df.columns:
            target_col = "fraud"
        else:
            raise ValueError(f"Could not automatically identify target column in: {list(df.columns)}")
            
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")
    
    # Drop rows where target is missing if any
    if df[target_col].isnull().any():
        df = df.dropna(subset=[target_col])
        
    # Auto-detect feature types
    numerical_cols, categorical_cols = detect_feature_types(df, target_col=target_col)
    
    feature_cols = numerical_cols + categorical_cols
    X = df[feature_cols]
    y = df[target_col].astype(int).values
    
    print(f"[DataPrep] Feature detection: {len(numerical_cols)} numerical features, {len(categorical_cols)} categorical features.")
    print(f"[DataPrep] Class distribution before split: Legitimate (0) = {np.sum(y == 0):,}, Fraud (1) = {np.sum(y == 1):,} (Fraud rate: {np.mean(y):.3%})")
    
    # Stratified 80/20 train/test split
    X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Build and fit preprocessing pipeline strictly on training split
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)
    
    # Extract feature names after one-hot encoding
    encoded_cat_names = []
    if categorical_cols and 'cat' in preprocessor.named_transformers_:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        encoded_cat_names = list(cat_encoder.get_feature_names_out(categorical_cols))
    feature_names = numerical_cols + encoded_cat_names
    
    raw_train_shape = X_train_proc.shape
    
    # Apply SMOTE strictly on training data
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_proc, y_train_raw)
    resampled_train_shape = X_train_resampled.shape
    
    print(f"[DataPrep] Training split pre-SMOTE: {raw_train_shape[0]:,} samples (Fraud: {np.sum(y_train_raw == 1):,}, Legitimate: {np.sum(y_train_raw == 0):,})")
    print(f"[DataPrep] Training split post-SMOTE: {resampled_train_shape[0]:,} samples (Fraud: {np.sum(y_train_resampled == 1):,}, Legitimate: {np.sum(y_train_resampled == 0):,})")
    print(f"[DataPrep] Test split untouched: {X_test_proc.shape[0]:,} samples (Fraud: {np.sum(y_test == 1):,}, Legitimate: {np.sum(y_test == 0):,})")
    
    return PreparedData(
        X_train=X_train_resampled,
        y_train=y_train_resampled,
        X_test=X_test_proc,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        raw_train_shape=raw_train_shape,
        resampled_train_shape=resampled_train_shape
    )


if __name__ == "__main__":
    data = prepare_fraud_data()
    print(f"[DataPrep] Verification complete! Feature count: {len(data.feature_names)}")
