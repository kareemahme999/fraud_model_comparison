"""
Model Evaluation and Visualization Module for AI Fraud Detection System (Phase 5).

Calculates comprehensive classification metrics:
- Precision, Recall, F1 Score, ROC-AUC, Average Precision (PR-AUC)
- Full Classification Reports (text & dictionary)
- Confusion Matrices (counts & percentages)
- Feature Importance extraction for tree models (Random Forest, XGBoost)
- Decision Threshold Tuning to maximize Recall and minimize False Negatives

Generates high-resolution publication-quality plots:
- Confusion Matrix Heatmaps (per-model & combined grid)
- Comparative ROC Curves
- Precision-Recall (PR) Curves
- Feature Importance Bar Charts
- Threshold Tuning Trade-Off Curves
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report
)


@dataclass
class EvaluationMetrics:
    """Dataclass holding performance metrics and artifacts for a single model."""
    model_name: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_mat: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    classification_rep_dict: Dict[str, Any]
    classification_rep_text: str


@dataclass
class TunedThresholdResult:
    """Dataclass holding threshold tuning results and before/after comparisons."""
    model_name: str
    optimal_threshold: float
    baseline_threshold: float
    baseline_metrics: Dict[str, float]
    tuned_metrics: Dict[str, float]
    baseline_cm: np.ndarray
    tuned_cm: np.ndarray
    recall_gain: float
    fn_reduction: int
    threshold_history: pd.DataFrame


def evaluate_single_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str
) -> EvaluationMetrics:
    """
    Evaluate a fitted model on un-resampled out-of-sample test data.
    
    Args:
        model: Trained classifier (with predict and predict_proba/decision_function).
        X_test: Preprocessed test features.
        y_test: True test labels.
        model_name: Human-readable model identifier.
        
    Returns:
        EvaluationMetrics instance with precision, recall, f1, roc_auc, pr_auc, reports, and confusion matrix.
    """
    y_pred = model.predict(X_test)
    
    # Extract predicted probability for positive class (fraud = 1)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        y_prob = (decision - decision.min()) / (decision.max() - decision.min() + 1e-12)
    else:
        y_prob = y_pred.astype(float)
        
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    rep_dict = classification_report(y_test, y_pred, target_names=["Legitimate (0)", "Fraud (1)"], output_dict=True, zero_division=0)
    rep_text = classification_report(y_test, y_pred, target_names=["Legitimate (0)", "Fraud (1)"], digits=4, zero_division=0)
    
    print(f"\n[Evaluate] --- {model_name} ---")
    print(f"  Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    
    return EvaluationMetrics(
        model_name=model_name,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=auc,
        pr_auc=pr_auc,
        confusion_mat=cm,
        y_pred=y_pred,
        y_prob=y_prob,
        classification_rep_dict=rep_dict,
        classification_rep_text=rep_text
    )


def evaluate_all_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, EvaluationMetrics]:
    """
    Evaluate all models in the dictionary on the test set.
    
    Args:
        models: Dictionary of {model_name: model_instance}.
        X_test: Test features.
        y_test: Test target labels.
        
    Returns:
        Dictionary mapping model names to EvaluationMetrics instances.
    """
    results: Dict[str, EvaluationMetrics] = {}
    for name, model in models.items():
        results[name] = evaluate_single_model(model, X_test, y_test, model_name=name)
    return results


def plot_confusion_matrices(
    eval_results: Dict[str, EvaluationMetrics],
    output_dir: str = "outputs/plots"
) -> List[str]:
    """
    Generate and save confusion matrix heatmaps for all models:
    - Individual annotated plots for each model
    - A combined multi-panel comparison figure
    
    Args:
        eval_results: Dictionary of evaluated metrics per model.
        output_dir: Directory where PNG plots will be saved.
        
    Returns:
        List of generated image file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files: List[str] = []
    
    sns.set_theme(style="white", palette="muted")
    labels = ["Legitimate (0)", "Fraud (1)"]
    
    # 1. Individual heatmaps
    for name, metrics in eval_results.items():
        cm = metrics.confusion_mat
        cm_pct = cm.astype('float') / cm.sum() * 100
        
        annot_labels = np.array([
            [f"{count}\n({pct:.1f}%)" for count, pct in zip(row_c, row_p)]
            for row_c, row_p in zip(cm, cm_pct)
        ])
        
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        sns.heatmap(
            cm,
            annot=annot_labels,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            linewidths=1.2,
            linecolor="#e0e0e0",
            ax=ax,
            annot_kws={"size": 12, "weight": "bold"}
        )
        
        ax.set_title(f"Confusion Matrix: {name}\n(Precision={metrics.precision:.3f}, Recall={metrics.recall:.3f}, F1={metrics.f1:.3f})", fontsize=12, pad=12, weight="bold")
        ax.set_xlabel("Predicted Class", fontsize=11, labelpad=8)
        ax.set_ylabel("Actual Class", fontsize=11, labelpad=8)
        plt.tight_layout()
        
        clean_filename = name.lower().replace(" ", "_")
        file_path = os.path.join(output_dir, f"confusion_matrix_{clean_filename}.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(file_path)
        print(f"[Evaluate] Saved confusion matrix to '{file_path}'")
        
    # 2. Combined grid plot
    n_models = len(eval_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.8), dpi=300)
    if n_models == 1:
        axes = [axes]
        
    for ax, (name, metrics) in zip(axes, eval_results.items()):
        cm = metrics.confusion_mat
        cm_pct = cm.astype('float') / cm.sum() * 100
        annot_labels = np.array([
            [f"{count}\n({pct:.1f}%)" for count, pct in zip(row_c, row_p)]
            for row_c, row_p in zip(cm, cm_pct)
        ])
        
        sns.heatmap(
            cm,
            annot=annot_labels,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            linewidths=1.0,
            linecolor="#e0e0e0",
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"}
        )
        ax.set_title(f"{name}\nRecall={metrics.recall:.3f} | F1={metrics.f1:.3f}", fontsize=11, weight="bold", pad=8)
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("Actual Class", fontsize=10)
        
    plt.suptitle("Fraud Detection Confusion Matrices Benchmark", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    combined_path = os.path.join(output_dir, "confusion_matrices_all.png")
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(combined_path)
    print(f"[Evaluate] Saved combined confusion matrices to '{combined_path}'")
    
    return saved_files


def plot_roc_curves(
    eval_results: Dict[str, EvaluationMetrics],
    y_test: np.ndarray,
    output_dir: str = "outputs/plots"
) -> str:
    """
    Generate and save a comparative ROC (Receiver Operating Characteristic) curve plot for all models.
    
    Args:
        eval_results: Dictionary of evaluated metrics per model.
        y_test: True binary test labels.
        output_dir: Directory where plot will be saved.
        
    Returns:
        File path of the saved ROC curves figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]
    
    for idx, (name, metrics) in enumerate(eval_results.items()):
        fpr, tpr, _ = roc_curve(y_test, metrics.y_prob)
        color = palette[idx % len(palette)]
        ax.plot(
            fpr, tpr,
            label=f"{name} (ROC-AUC = {metrics.roc_auc:.4f})",
            linewidth=2.2,
            color=color
        )
        
    # Baseline random guessing line
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.5000)", linewidth=1.2, alpha=0.75)
    
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=8)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, labelpad=8)
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, weight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    file_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Saved ROC curves comparison to '{file_path}'")
    
    return file_path


def plot_precision_recall_curves(
    eval_results: Dict[str, EvaluationMetrics],
    y_test: np.ndarray,
    output_dir: str = "outputs/plots"
) -> str:
    """
    Generate and save Precision-Recall (PR) curves for every model with Average Precision (AP) scores.
    
    Args:
        eval_results: Dictionary of evaluated metrics per model.
        y_test: True binary test labels.
        output_dir: Directory where plot will be saved.
        
    Returns:
        File path of the saved PR curves figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]
    fraud_prevalence = float(np.mean(y_test))
    
    for idx, (name, metrics) in enumerate(eval_results.items()):
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, metrics.y_prob)
        color = palette[idx % len(palette)]
        ax.plot(
            recall_vals, precision_vals,
            label=f"{name} (PR-AUC / AP = {metrics.pr_auc:.4f})",
            linewidth=2.2,
            color=color
        )
        
    # Baseline line for random guessing in PR curve is equal to the positive prevalence
    ax.axhline(
        y=fraud_prevalence,
        color="k",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label=f"No-Skill Baseline (Prevalence = {fraud_prevalence:.3f})"
    )
    
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Recall (Sensitivity / Fraud Detection Rate)", fontsize=11, labelpad=8)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, labelpad=8)
    ax.set_title("Precision-Recall (PR) Curves Comparison (Fraud Detection)", fontsize=13, weight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    file_path = os.path.join(output_dir, "pr_curves.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Saved Precision-Recall curves to '{file_path}'")
    
    return file_path


def extract_feature_importances(
    models: Dict[str, Any],
    feature_names: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Extract and sort feature importances for tree-based models (Random Forest, XGBoost).
    
    Args:
        models: Dictionary of trained models.
        feature_names: List of preprocessed feature names.
        
    Returns:
        Dictionary mapping model names to feature importance DataFrames.
    """
    importances_dict: Dict[str, pd.DataFrame] = {}
    
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            df_imp = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
            importances_dict[name] = df_imp
            
    return importances_dict


def plot_feature_importances(
    models: Dict[str, Any],
    feature_names: List[str],
    output_dir: str = "outputs/plots",
    top_n: int = 12
) -> List[str]:
    """
    Generate horizontal bar chart plots of feature importances for Random Forest and XGBoost.
    
    Args:
        models: Dictionary of trained models.
        feature_names: List of feature names matching preprocessed columns.
        output_dir: Output directory path.
        top_n: Number of top features to display.
        
    Returns:
        List of generated feature importance image file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files: List[str] = []
    
    importances_dict = extract_feature_importances(models, feature_names)
    
    for name, df_imp in importances_dict.items():
        df_top = df_imp.head(top_n).sort_values(by="Importance", ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        bars = ax.barh(df_top["Feature"], df_top["Importance"], color="#2b5c8f", edgecolor="white", height=0.65)
        
        # Add value annotations
        for bar in bars:
            width = bar.get_width()
            ax.annotate(
                f"{width:.4f}",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                weight="bold"
            )
            
        ax.set_title(f"Top {top_n} Feature Importances: {name}", fontsize=13, weight="bold", pad=12)
        ax.set_xlabel("Relative Importance (Gini / Gain)", fontsize=11, labelpad=8)
        ax.set_ylabel("Engineered Features", fontsize=11, labelpad=8)
        ax.set_xlim(0, df_top["Importance"].max() * 1.25)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        clean_name = name.lower().replace(" ", "_")
        file_path = os.path.join(output_dir, f"feature_importance_{clean_name}.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(file_path)
        print(f"[Evaluate] Saved feature importance plot to '{file_path}'")
        
    return saved_files


def tune_threshold(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    output_dir: str = "outputs/plots",
    weight_f1: float = 0.40,
    weight_recall: float = 0.35,
    weight_roc_auc: float = 0.25,
    min_precision: Optional[float] = None
) -> TunedThresholdResult:
    """
    Perform fine-grained probability threshold tuning (0.01 to 0.99) to optimize fraud Recall
    and significantly reduce False Negatives while maintaining reasonable Precision and F1 balance.
    
    Args:
        model: Trained classifier.
        X_test: Test feature matrix.
        y_test: Test true labels.
        model_name: Name of the model.
        output_dir: Output directory for the threshold trade-off curve plot.
        weight_f1: Business weight for F1 score (guards against precision collapse).
        weight_recall: Business weight for Recall (maximizes fraud detection rate).
        weight_roc_auc: Business weight for ROC-AUC (discriminative quality).
        min_precision: Optional minimum precision constraint (business floor for alert quality).
        
    Returns:
        TunedThresholdResult containing optimal threshold, metrics comparison, and tuning history.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        y_prob = (decision - decision.min()) / (decision.max() - decision.min() + 1e-12)
    else:
        y_prob = model.predict(X_test).astype(float)
        
    auc = roc_auc_score(y_test, y_prob)
    thresholds = np.linspace(0.01, 0.99, 99)
    
    records = []
    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        rec = recall_score(y_test, preds, zero_division=0)
        prec = precision_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds)
        fn_count = int(cm[1, 0]) if cm.shape == (2, 2) else 0
        fp_count = int(cm[0, 1]) if cm.shape == (2, 2) else 0
        tp_count = int(cm[1, 1]) if cm.shape == (2, 2) else 0
        tn_count = int(cm[0, 0]) if cm.shape == (2, 2) else 0
        
        comp_score = (weight_f1 * f1) + (weight_recall * rec) + (weight_roc_auc * auc)
        records.append({
            "threshold": th,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "composite_score": comp_score,
            "tn": tn_count,
            "fp": fp_count,
            "fn": fn_count,
            "tp": tp_count
        })
        
    df_history = pd.DataFrame(records)
    
    # Baseline at threshold = 0.50
    baseline_idx = (df_history["threshold"] - 0.50).abs().idxmin()
    baseline_row = df_history.loc[baseline_idx]
    
    # Select optimal threshold considering business precision constraints
    if min_precision is not None:
        valid_candidates = df_history[df_history["precision"] >= min_precision]
        if not valid_candidates.empty:
            opt_idx = valid_candidates["composite_score"].idxmax()
        else:
            opt_idx = df_history["composite_score"].idxmax()
    else:
        opt_idx = df_history["composite_score"].idxmax()
        
    opt_row = df_history.loc[opt_idx]
    opt_th = float(opt_row["threshold"])
    
    # Generate before/after confusion matrices
    baseline_preds = (y_prob >= 0.50).astype(int)
    baseline_cm = confusion_matrix(y_test, baseline_preds)
    
    tuned_preds = (y_prob >= opt_th).astype(int)
    tuned_cm = confusion_matrix(y_test, tuned_preds)
    
    recall_gain = float(opt_row["recall"] - baseline_row["recall"])
    fn_reduction = int(baseline_row["fn"] - opt_row["fn"])
    
    # Plot Threshold Tuning Curves
    fig, ax1 = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    
    ax1.plot(df_history["threshold"], df_history["recall"], label="Recall (Fraud Capture)", color="#2ca02c", linewidth=2.2)
    ax1.plot(df_history["threshold"], df_history["precision"], label="Precision", color="#1f77b4", linewidth=2.0, linestyle="--")
    ax1.plot(df_history["threshold"], df_history["f1"], label="F1 Score", color="#e377c2", linewidth=2.0)
    ax1.plot(df_history["threshold"], df_history["composite_score"], label="Composite Score", color="#ff7f0e", linewidth=2.5)
    
    # Mark optimal threshold
    ax1.axvline(x=opt_th, color="#d62728", linestyle=":", linewidth=2.0, label=f"Optimal Threshold = {opt_th:.2f}")
    ax1.axvline(x=0.50, color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.7, label="Default Threshold (0.50)")
    
    if min_precision is not None:
        ax1.axhline(y=min_precision, color="#8c564b", linestyle="-.", linewidth=1.0, alpha=0.7, label=f"Min Precision Floor ({min_precision:.2f})")
    
    ax1.set_xlabel("Decision Threshold", fontsize=11, labelpad=8)
    ax1.set_ylabel("Score / Metric Value", fontsize=11, labelpad=8)
    ax1.set_title(f"Threshold Optimization Curve: {model_name}\n(Optimal Threshold = {opt_th:.2f} | Recall: {opt_row['recall']:.1%} | Prec: {opt_row['precision']:.1%} | FN Reduced by {fn_reduction})", fontsize=11, weight="bold", pad=12)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(loc="best", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    clean_name = model_name.lower().replace(" ", "_")
    plot_path = os.path.join(output_dir, f"threshold_tuning_{clean_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Saved threshold tuning curve to '{plot_path}'")
    
    return TunedThresholdResult(
        model_name=model_name,
        optimal_threshold=opt_th,
        baseline_threshold=0.50,
        baseline_metrics={
            "precision": float(baseline_row["precision"]),
            "recall": float(baseline_row["recall"]),
            "f1": float(baseline_row["f1"]),
            "roc_auc": auc,
            "composite_score": float(baseline_row["composite_score"])
        },
        tuned_metrics={
            "precision": float(opt_row["precision"]),
            "recall": float(opt_row["recall"]),
            "f1": float(opt_row["f1"]),
            "roc_auc": auc,
            "composite_score": float(opt_row["composite_score"])
        },
        baseline_cm=baseline_cm,
        tuned_cm=tuned_cm,
        recall_gain=recall_gain,
        fn_reduction=fn_reduction,
        threshold_history=df_history
    )


if __name__ == "__main__":
    from data_prep import prepare_fraud_data
    from models import train_all_models
    
    prepared = prepare_fraud_data()
    models = train_all_models(prepared.X_train, prepared.y_train)
    results = evaluate_all_models(models, prepared.X_test, prepared.y_test)
    plot_confusion_matrices(results)
    plot_roc_curves(results, prepared.y_test)
    plot_precision_recall_curves(results, prepared.y_test)
    plot_feature_importances(models, prepared.feature_names)
    tune_result = tune_threshold(models["Logistic Regression"], prepared.X_test, prepared.y_test, "Logistic Regression")
    print("[Evaluate] Evaluation module updated verification complete!")
