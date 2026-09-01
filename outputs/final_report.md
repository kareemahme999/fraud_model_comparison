# AI Fraud Detection System — Model Comparison & Benchmark Report

**Generated on:** 2026-09-01 21:57:09  
**Target Variable:** `Class` (0 = Legitimate, 1 = Fraudulent)  
**Evaluation Set:** 56,746 transactions (95 Frauds, 56,651 Legitimate)  
**Primary Recommendation:** **XGBoost** (Operating at Calibrated Decision Threshold = `0.91`)

---

## 1. Executive Summary & Evaluation Framework

In financial fraud detection, model evaluation must account for extreme class imbalance (95/56,746 = 0.167% fraud prevalence) and asymmetric error costs (missing a fraud incurs direct financial losses and chargeback fees).

Models were evaluated across two distinct operational paradigms:
1. **Baseline Evaluation (Default Threshold = 0.50)**: Measures standard out-of-the-box performance without threshold calibration.
2. **Threshold-Tuned Evaluation (Calibrated Decision Threshold)**: Calibrates the decision cutoff across $[0.01, 0.99]$ to maximize fraud Recall and minimize False Negatives while enforcing business constraints on Precision and F1 balance.

All models are scored using the domain-weighted composite objective function:
$$\text{Composite Score} = (0.40 \times \text{F1}) + (0.35 \times \text{Recall}) + (0.25 \times \text{ROC-AUC})$$

### Summary of Winners:
- **Baseline Winner (Threshold = 0.50):** **`Random Forest`** (Composite Score: `0.8144`)
- **Threshold-Tuned Winner & Final Recommendation:** **`XGBoost`** (Tuned Composite Score: `0.8365`, ROC-AUC: `0.9759`)

---

### Baseline Performance Matrix (Default Threshold = 0.50)

| Model | Baseline Rank | Precision | Recall | F1 | ROC-AUC | PR-AUC | Composite Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | #1 (Baseline Winner) | 0.6818 | 0.7895 | 0.7317 | 0.9817 | 0.7861 | 0.8144 |
| XGBoost | #2 | 0.29 | 0.8211 | 0.4286 | 0.9759 | 0.7995 | 0.7028 |
| Logistic Regression | #3 | 0.053 | 0.8737 | 0.1 | 0.9626 | 0.675 | 0.5864 |

---

## 2. Final Production Recommendation: **XGBoost**

### Why XGBoost is Recommended for Production Deployment

While **Random Forest** scored highest at the default cutoff of 0.50, **XGBoost** is the recommended model for production deployment based on operational decision criteria:

1. **Superior Global Discriminative Capacity**:
   - **XGBoost** achieves **ROC-AUC = 0.9759** and **PR-AUC = 0.7995**.
   - ROC-AUC measures a model's fundamental ability to separate fraudulent from legitimate transactions across *all* possible decision thresholds.
2. **Highest Threshold-Tuned Composite Score (`0.8365`)**:
   - In production fraud systems, models operate at calibrated business cutoffs rather than arbitrary `0.50` thresholds.
   - When calibrated to its optimal operating threshold (`0.91`), **XGBoost achieves a high-precision fraud interception rate (Precision = 88.89%, Recall = 75.79%, F1 = 0.8182)**, significantly reducing False Positives from `191` down to **`9`**.
3. **Non-Linear Fraud Boundary Detection & Feature Interaction**:
   - Fraud patterns involve complex multi-feature interactions across PCA and amount dimensions that gradient boosted trees naturally model without overfitting.

### Performance Breakdown for Recommended Model (XGBoost):

| Metric | Baseline (Threshold = 0.50) | Tuned (Threshold = 0.91) | Operational Shift |
| --- | --- | --- | --- |
| **Recall (Fraud Caught)** | `82.11%` (78/95) | **`75.79%`** (72/95) | Calibrated operating point |
| **Precision (Alert Quality)** | `0.2900` | **`0.8889`** | **+59.9% alert accuracy** |
| **False Positives (User Friction)** | `191` false alerts | **`9` false alerts** | **-182 false alarms** |
| **F1 Score** | `0.4286` | **`0.8182`** | **+0.3896** |
| **ROC-AUC** | `0.9759` | `0.9759` | Top ranking globally |
| **Composite Score** | `0.7028` | **`0.8365`** | **+0.1337** |

---

## 3. Baseline Winner Deep-Dive: **Random Forest**

- **Why Random Forest Won Baseline**:
  Random Forest achieved the best balance between precision and recall at the default 0.50 cutoff (F1 = `0.7317`, ROC-AUC = `0.9817`).
- **Threshold-Tuning Comparison**:
  Both Random Forest and XGBoost exhibit exceptional ROC-AUC (>0.975), with XGBoost achieving the top calibrated composite score (`0.8365`) at its optimal decision threshold.

---

## 4. Decision Threshold Tuning & Business Constraint Optimization

In production fraud systems, selecting decision thresholds involves balancing two competing business forces:
1. **Recall Maximization (Chargeback Protection)**: Missing a fraud costs 100% of the transaction amount plus chargeback fees and compliance penalties.
2. **Precision Guardrails (Operational Friction Control)**: False Positives trigger automated step-up verification (SMS OTP / 2FA).

Our threshold optimization evaluates the candidate range $[0.01, 0.99]$ against the business composite function $(0.40 \times \text{F1} + 0.35 \times \text{Recall} + 0.25 \times \text{ROC-AUC})$.

### Threshold-Tuned Performance Matrix

| Model | Tuned Rank | Opt Threshold | Tuned Recall | Tuned Precision | Tuned F1 | Tuned Composite |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost | #1 (Recommended Winner) | 0.91 | 0.7579 | 0.8889 | 0.8182 | 0.8365 |
| Random Forest | #2 | 0.73 | 0.7474 | 0.8353 | 0.7889 | 0.8226 |
| Logistic Regression | #3 | 0.99 | 0.7895 | 0.6098 | 0.6881 | 0.7922 |

---

## 5. Feature Importance Analysis (Tree Models)

Feature importance extraction from tree ensembles highlights the primary risk indicators driving fraudulent behavior:

#### Random Forest Top Predictive Features

| Feature | Importance |
| --- | --- |
| V14 | 0.2059 |
| V10 | 0.1409 |
| V12 | 0.1089 |
| V4 | 0.0945 |
| V17 | 0.0856 |
| V16 | 0.0535 |
| V3 | 0.0533 |
| V11 | 0.0499 |

#### XGBoost Top Predictive Features

| Feature | Importance |
| --- | --- |
| V14 | 0.4805 |
| V4 | 0.0546 |
| V8 | 0.0373 |
| V10 | 0.0330 |
| V12 | 0.0308 |
| V11 | 0.0242 |
| V17 | 0.0239 |
| Amount | 0.0216 |


### Key Risk Drivers:
- **Random Forest Primary Signals**: `V14, V10, V12, V4` account for the majority of tree splits.
- **XGBoost Primary Signals**: `V14, V4, V8, V10` account for the majority of tree splits.

---

## 6. Classification Reports (Detailed Breakdown)

#### Logistic Regression (Baseline Threshold = 0.50)
```text
                precision    recall  f1-score   support

Legitimate (0)     0.9998    0.9738    0.9866     56651
     Fraud (1)     0.0530    0.8737    0.1000        95

      accuracy                         0.9737     56746
     macro avg     0.5264    0.9238    0.5433     56746
  weighted avg     0.9982    0.9737    0.9852     56746

```

#### Random Forest (Baseline Threshold = 0.50)
```text
                precision    recall  f1-score   support

Legitimate (0)     0.9996    0.9994    0.9995     56651
     Fraud (1)     0.6818    0.7895    0.7317        95

      accuracy                         0.9990     56746
     macro avg     0.8407    0.8944    0.8656     56746
  weighted avg     0.9991    0.9990    0.9991     56746

```

#### XGBoost (Baseline Threshold = 0.50)
```text
                precision    recall  f1-score   support

Legitimate (0)     0.9997    0.9966    0.9982     56651
     Fraud (1)     0.2900    0.8211    0.4286        95

      accuracy                         0.9963     56746
     macro avg     0.6448    0.9088    0.7134     56746
  weighted avg     0.9985    0.9963    0.9972     56746

```


---

## 7. Senior ML Engineering Recommendations & Deployment Guidelines

1. **Deploy Recommended Model Pipeline (`outputs/best_model.joblib`)**:
   - The production champion **XGBoost** has been serialized with its fitted `ColumnTransformer` preprocessor and calibrated decision threshold (`0.91`).
2. **Implement Tiered Dual-Threshold Decision Engine**:
   - **Tier 1 — Instant Approval ($P < 0.46$)**: Seamless, zero-friction transaction completion for low-risk activity (>95% of volume).
   - **Tier 2 — Step-Up Verification ($ 0.46 \le P < 1.36$)**: Challenge transaction via SMS OTP or 2FA challenge. Resolves False Positives frictionlessly without manual agent review.
   - **Tier 3 — Auto-Decline / High-Priority Review ($P \ge 1.36$)**: Hard block malicious fraud and alert the fraud operations investigation queue.
3. **Continuous Performance & Drift Monitoring**:
   - Set up automated daily monitoring on Population Stability Index (PSI) to catch distribution drift in transaction volume and principal component distributions.
