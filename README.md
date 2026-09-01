# AI Fraud Detection System — Model Comparison & Benchmark

A production-grade machine learning benchmarking pipeline and interactive web dashboard for financial transaction fraud detection on the Credit Card Fraud dataset (`creditcard.csv`). This project systematically evaluates and compares **Logistic Regression**, **Random Forest**, and **XGBoost** on class-imbalanced transaction data using leak-free preprocessing, SMOTE oversampling, decision threshold tuning, and business-cost-aligned evaluation metrics.

---

## 1. Project Directory Structure

```text
fraud_model_comparison/
├── data/
│   └── creditcard.csv                        # Credit card transaction dataset (284,807 rows, 31 columns)
├── outputs/
│   ├── best_model.joblib                     # Serialized winning model bundle (model, preprocessor, metadata)
│   ├── comparison_table.csv                  # Exported metrics comparison table
│   ├── final_report.md                       # Comprehensive executive & technical report
│   └── plots/
│       ├── comparison_bar.png                # Grouped Precision/Recall/F1 bar chart
│       ├── confusion_matrices_all.png         # Combined side-by-side confusion matrices
│       ├── confusion_matrix_logistic_regression.png
│       ├── confusion_matrix_random_forest.png
│       ├── confusion_matrix_xgboost.png
│       ├── feature_importance_random_forest.png # Tree feature importances
│       ├── feature_importance_xgboost.png       # XGBoost feature importances
│       ├── pr_curves.png                     # Precision-Recall curves with AP scores
│       ├── roc_curves.png                    # Overlay ROC-AUC curves with baseline
│       ├── threshold_tuning_logistic_regression.png # Threshold trade-off curve
│       ├── threshold_tuning_random_forest.png
│       └── threshold_tuning_xgboost.png
├── app.py                                    # Interactive Streamlit Web Dashboard & Inference Server
├── data_prep.py                              # Data ingestion, cleaning, leakage-free pipeline & SMOTE
├── models.py                                 # Individual model training routines & hyperparameters
├── evaluate.py                               # Metrics computation, PR curves, feature importance & threshold tuning
├── compare.py                                # Pipeline orchestrator, threshold optimizer & report generator
├── requirements.txt                          # Pinned dependencies
└── README.md                                 # Technical documentation
```

---

## 2. Dataset Overview (`creditcard.csv`)

- **Total Records:** 284,807 transactions (283,726 unique transactions after deduplication)
- **Feature Set:**
  - `Time`: Number of seconds elapsed between this transaction and the first transaction in the dataset.
  - `V1` – `V28`: 28 principal component features obtained via PCA transformation.
  - `Amount`: Transaction monetary value.
- **Target Label:** `Class` (0 = Legitimate [99.83%], 1 = Fraudulent [0.17%]).
- **Missing Values:** 0 missing values.

---

## 3. Core Architecture & Engineering Highlights

### Leak-Free Preprocessing Pipeline (`data_prep.py`)
- **Automated Data Cleaning**: Detects and removes duplicate transactions while validating schema integrity.
- **Feature Detection**: Dynamically distinguishes numerical features from categorical features.
- **Missing Value Imputation**: Imputes median values for numerical columns (`SimpleImputer(strategy='median')`).
- **Feature Scaling**: Standardizes features (`StandardScaler()`).
- **Stratified Train/Test Split**: 80/20 train/test partition stratified by `Class` target to preserve the natural fraud distribution.
- **Strict Data Leakage Prevention**: Transformers are fitted **strictly on the training partition** (`X_train`) and applied downstream to `X_test`.
- **Targeted SMOTE Resampling**: `SMOTE(random_state=42)` is applied **exclusively to the training split**. The test split (`X_test`, `y_test`) remains untouched in its natural class distribution to ensure genuine out-of-sample evaluation.

---

## 4. Models Implemented (`models.py`)

1. **Logistic Regression**: Linear baseline optimized via `lbfgs` solver (`max_iter=1000`, `C=1.0`). Provides calibrated linear decision boundaries.
2. **Random Forest Classifier**: Non-linear ensemble bagging model (`n_estimators=150`, `max_depth=12`, `min_samples_split=4`, `min_samples_leaf=2`, `n_jobs=-1`).
3. **XGBoost Classifier**: Extreme Gradient Boosting with regularization and shrinkage (`n_estimators=150`, `max_depth=5`, `learning_rate=0.08`, `subsample=0.85`, `colsample_bytree=0.85`, `eval_metric='logloss'`, `n_jobs=-1`).

---

## 5. Quick Start & Execution

### 1. Launch Interactive Web Server & Dashboard

Start the live Streamlit dashboard:

```bash
streamlit run fraud_model_comparison/app.py
```
Open your browser at **http://localhost:8501** to view the interactive dashboard, test real-time transaction scoring with threshold adjustment, and view metric curves.

### 2. Run Full Comparison Benchmark (CLI)

Execute the complete end-to-end pipeline:

```bash
python fraud_model_comparison/compare.py
```

### 3. Running Individual Modules

```bash
# Verify data preprocessing and SMOTE pipeline
python fraud_model_comparison/data_prep.py

# Verify model training
python fraud_model_comparison/models.py

# Verify evaluation metrics, PR curves, and plots
python fraud_model_comparison/evaluate.py
```

---

## 6. Generated Artifacts & Deliverables

Upon running `compare.py`, the following outputs are produced:

- `outputs/best_model.joblib`: Serialized dictionary containing the winning trained model, fitted `ColumnTransformer`, optimal decision threshold, and training metadata.
- `outputs/comparison_table.csv`: Comprehensive metrics matrix across all models (baseline & threshold-tuned).
- `outputs/final_report.md`: Executive and technical summary report with tables, feature importance insights, classification reports, and production recommendations.
- `outputs/plots/comparison_bar.png`: Grouped bar chart comparing Precision, Recall, and F1.
- `outputs/plots/pr_curves.png`: Precision-Recall curves with Average Precision scores.
- `outputs/plots/roc_curves.png`: Multi-model ROC curves with baseline.
- `outputs/plots/confusion_matrix_{model}.png`: Heatmaps showing true counts and distribution percentages.
- `outputs/plots/confusion_matrices_all.png`: Side-by-side comparative confusion matrices.
- `outputs/plots/feature_importance_{model}.png`: Feature importance bar charts for Random Forest and XGBoost.
- `outputs/plots/threshold_tuning_{model}.png`: Threshold optimization trade-off curves.
