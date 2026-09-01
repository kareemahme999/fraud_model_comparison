"""
Machine Learning Model Training Module for AI Fraud Detection System (Phase 5).

Provides distinct model training functions and hyperparameter configurations for:
1. Logistic Regression (Linear baseline with calibrated probabilities)
2. Random Forest (Ensemble bagging tree classifier)
3. XGBoost (Extreme Gradient Boosting classifier)
"""

from __future__ import annotations

import time
from typing import Dict, Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    C: float = 1.0,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    **kwargs: Any
) -> LogisticRegression:
    """
    Train a Logistic Regression classifier on preprocessed & SMOTE-balanced training data.
    
    Args:
        X_train: Preprocessed training feature matrix.
        y_train: Training target labels.
        random_state: Random seed for reproducibility.
        C: Inverse of regularization strength.
        max_iter: Maximum solver iterations for convergence.
        solver: Optimization algorithm.
        **kwargs: Extra parameters forwarded to LogisticRegression.
        
    Returns:
        Fitted LogisticRegression model.
    """
    print("[Models] Training Logistic Regression...")
    start_time = time.perf_counter()
    
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        random_state=random_state,
        **kwargs
    )
    model.fit(X_train, y_train)
    
    elapsed = time.perf_counter() - start_time
    print(f"[Models] Logistic Regression trained in {elapsed:.3f}s.")
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    n_estimators: int = 150,
    max_depth: Optional[int] = 12,
    min_samples_split: int = 4,
    min_samples_leaf: int = 2,
    n_jobs: int = -1,
    **kwargs: Any
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier with sensible non-overfitting hyperparameters.
    
    Args:
        X_train: Preprocessed training feature matrix.
        y_train: Training target labels.
        random_state: Random seed for reproducibility.
        n_estimators: Number of decision trees in the ensemble.
        max_depth: Maximum tree depth to constrain overfitting.
        min_samples_split: Minimum samples required to split an internal node.
        min_samples_leaf: Minimum samples required at a leaf node.
        n_jobs: Number of CPU threads (-1 for all).
        **kwargs: Extra parameters forwarded to RandomForestClassifier.
        
    Returns:
        Fitted RandomForestClassifier model.
    """
    print("[Models] Training Random Forest Classifier...")
    start_time = time.perf_counter()
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )
    model.fit(X_train, y_train)
    
    elapsed = time.perf_counter() - start_time
    print(f"[Models] Random Forest trained in {elapsed:.3f}s.")
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    n_estimators: int = 150,
    max_depth: int = 5,
    learning_rate: float = 0.08,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    eval_metric: str = "logloss",
    n_jobs: int = -1,
    **kwargs: Any
) -> XGBClassifier:
    """
    Train an XGBoost (Extreme Gradient Boosting) classifier for non-linear fraud boundary detection.
    
    Args:
        X_train: Preprocessed training feature matrix.
        y_train: Training target labels.
        random_state: Random seed for reproducibility.
        n_estimators: Number of boosting stages.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate (shrinkage).
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of columns when constructing each tree.
        eval_metric: Loss metric used for optimization.
        n_jobs: Number of CPU threads.
        **kwargs: Extra parameters forwarded to XGBClassifier.
        
    Returns:
        Fitted XGBClassifier model.
    """
    print("[Models] Training XGBoost Classifier...")
    start_time = time.perf_counter()
    
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric=eval_metric,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )
    model.fit(X_train, y_train)
    
    elapsed = time.perf_counter() - start_time
    print(f"[Models] XGBoost trained in {elapsed:.3f}s.")
    return model


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train all benchmark models sequentially and return them in a structured dictionary.
    
    Args:
        X_train: Preprocessed training features.
        y_train: Training labels.
        random_state: Random seed.
        
    Returns:
        Dictionary mapping model names to fitted model instances.
    """
    models = {
        "Logistic Regression": train_logistic_regression(X_train, y_train, random_state=random_state),
        "Random Forest": train_random_forest(X_train, y_train, random_state=random_state),
        "XGBoost": train_xgboost(X_train, y_train, random_state=random_state),
    }
    return models


if __name__ == "__main__":
    from data_prep import prepare_fraud_data
    prepared = prepare_fraud_data()
    models = train_all_models(prepared.X_train, prepared.y_train)
    print(f"[Models] All {len(models)} models trained successfully!")
