"""
Interactive Streamlit Dashboard & Inference Server for AI Fraud Detection System.
Provides:
1. Executive Benchmark Overview & Leaderboard
2. Interactive Visualizations (ROC, PR Curves, Confusion Matrices, Feature Importance)
3. Real-Time Transaction Fraud Prediction & Risk Scoring Engine
4. Full Technical Report Viewer
"""

from __future__ import annotations

import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="AI Fraud Detection System — Benchmark & Inference Portal",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.joblib")
TABLE_PATH = os.path.join(OUTPUT_DIR, "comparison_table.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "final_report.md")


@st.cache_resource
def load_champion_bundle():
    """Load the trained winning model artifact bundle."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_sample_dataset():
    """Load a sample of creditcard transactions for the interactive demo."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, nrows=5000)
        return df
    return None


# Sidebar Navigation & Branding
st.sidebar.title("💳 AI Fraud Guard")
st.sidebar.markdown("**Dataset:** `creditcard.csv` (284,807 transactions)")

nav_selection = st.sidebar.radio(
    "Navigation",
    ["📊 Benchmark Dashboard", "⚡ Real-Time Fraud Predictor", "📈 Visualizations Gallery", "📋 Final Technical Report"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 Production Champion")
bundle = load_champion_bundle()
if bundle:
    st.sidebar.success(f"**Model:** {bundle.get('model_name', 'XGBoost')}")
    st.sidebar.info(f"**Calibrated Threshold:** `{bundle.get('optimal_threshold', 0.91):.2f}`")
else:
    st.sidebar.warning("Model bundle not found. Please run compare.py first.")

st.sidebar.markdown("---")
st.sidebar.caption("Google DeepMind Pair Programming • Phase 5")

# -------------------------------------------------------------
# TAB 1: BENCHMARK DASHBOARD
# -------------------------------------------------------------
if nav_selection == "📊 Benchmark Dashboard":
    st.title("💳 AI Fraud Detection System — Model Benchmark Dashboard")
    st.markdown("End-to-end performance benchmarking of **Logistic Regression**, **Random Forest**, and **XGBoost** on class-imbalanced transaction data.")

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", "284,807", "283,726 unique")
    with col2:
        st.metric("Fraud Prevalence", "0.167%", "473 actual frauds")
    with col3:
        st.metric("Top ROC-AUC", "0.9817", "Random Forest")
    with col4:
        st.metric("Champion Tuned F1", "0.8182", "XGBoost (Th = 0.91)")

    st.markdown("---")
    
    st.subheader("📋 Model Comparison Leaderboard")
    if os.path.exists(TABLE_PATH):
        df_table = pd.read_csv(TABLE_PATH)
        st.dataframe(df_table, use_container_width=True)
    else:
        st.info("Comparison table not found. Run compare.py to generate metrics.")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("📊 Precision vs. Recall vs. F1")
        bar_plot_path = os.path.join(PLOTS_DIR, "comparison_bar.png")
        if os.path.exists(bar_plot_path):
            st.image(bar_plot_path, use_container_width=True)
            
    with col_right:
        st.subheader("🎯 Side-by-Side Confusion Matrices")
        cm_plot_path = os.path.join(PLOTS_DIR, "confusion_matrices_all.png")
        if os.path.exists(cm_plot_path):
            st.image(cm_plot_path, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: REAL-TIME FRAUD PREDICTOR
# -------------------------------------------------------------
elif nav_selection == "⚡ Real-Time Fraud Predictor":
    st.title("⚡ Real-Time Transaction Scoring & Fraud Simulation")
    st.markdown("Test the production-ready serialized model (`outputs/best_model.joblib`) on transaction samples with interactive threshold tuning.")

    if bundle is None:
        st.error("Winning model artifact bundle is not available. Please run `python compare.py` first.")
    else:
        model = bundle["model"]
        preprocessor = bundle["preprocessor"]
        feature_names = bundle["feature_names"]
        opt_th = float(bundle.get("optimal_threshold", 0.91))
        
        df_sample = load_sample_dataset()
        
        st.markdown("### 1. Select or Simulate a Transaction")
        sim_mode = st.radio("Input Source:", ["Sample from Test Dataset", "Random Known Fraud Sample", "Random Legitimate Sample"], horizontal=True)
        
        selected_row = None
        if df_sample is not None:
            if sim_mode == "Random Known Fraud Sample":
                fraud_rows = df_sample[df_sample["Class"] == 1]
                if not fraud_rows.empty:
                    selected_row = fraud_rows.sample(1, random_state=int(time.time()) % 1000).iloc[0]
            elif sim_mode == "Random Legitimate Sample":
                legit_rows = df_sample[df_sample["Class"] == 0]
                if not legit_rows.empty:
                    selected_row = legit_rows.sample(1, random_state=int(time.time()) % 1000).iloc[0]
            else:
                idx = st.slider("Transaction Index", 0, min(1000, len(df_sample) - 1), 0)
                selected_row = df_sample.iloc[idx]
                
        if selected_row is not None:
            actual_class = int(selected_row.get("Class", 0))
            st.write(f"**Actual Ground Truth:** {'🚨 Fraud (1)' if actual_class == 1 else '✅ Legitimate (0)'}")
            
            with st.expander("🔍 View Transaction Feature Values"):
                st.json(selected_row.to_dict())
                
            st.markdown("### 2. Decision Threshold Setting")
            threshold = st.slider("Decision Threshold (Cutoff):", min_value=0.01, max_value=0.99, value=opt_th, step=0.01)
            
            # Predict
            df_input = pd.DataFrame([selected_row[feature_names]])
            transformed_input = preprocessor.transform(df_input)
            
            if hasattr(model, "predict_proba"):
                fraud_prob = float(model.predict_proba(transformed_input)[0, 1])
            else:
                fraud_prob = float(model.predict(transformed_input)[0])
                
            is_flagged = fraud_prob >= threshold
            
            st.markdown("### 3. Real-Time Risk Assessment")
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Predicted Fraud Probability", f"{fraud_prob:.2%}")
            with res_col2:
                risk_status = "🚨 HIGH RISK (FRAUD)" if is_flagged else "✅ APPROVED (LEGITIMATE)"
                st.metric("Decision Verdict", risk_status)
            with res_col3:
                st.metric("Calibrated Threshold", f"{threshold:.2f}")
                
            st.progress(fraud_prob)
            
            if is_flagged:
                st.error(f"🚨 **ALERT:** Transaction flagged as FRAUDULENT (Probability: {fraud_prob:.2%} ≥ Threshold: {threshold:.2f}). Triggering automated 2FA / Manual Investigation.")
            else:
                st.success(f"✅ **APPROVED:** Transaction classified as LEGITIMATE (Probability: {fraud_prob:.2%} < Threshold: {threshold:.2f}). Instant settlement authorized.")

# -------------------------------------------------------------
# TAB 3: VISUALIZATIONS GALLERY
# -------------------------------------------------------------
elif nav_selection == "📈 Visualizations Gallery":
    st.title("📈 Performance Curves & Feature Importance Gallery")
    
    tab_roc, tab_pr, tab_fi, tab_th = st.tabs(["ROC Curves", "Precision-Recall Curves", "Feature Importance", "Threshold Tuning"])
    
    with tab_roc:
        roc_path = os.path.join(PLOTS_DIR, "roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
            
    with tab_pr:
        pr_path = os.path.join(PLOTS_DIR, "pr_curves.png")
        if os.path.exists(pr_path):
            st.image(pr_path, use_container_width=True)
            
    with tab_fi:
        col_fi1, col_fi2 = st.columns(2)
        with col_fi1:
            fi_rf = os.path.join(PLOTS_DIR, "feature_importance_random_forest.png")
            if os.path.exists(fi_rf):
                st.image(fi_rf, use_container_width=True)
        with col_fi2:
            fi_xgb = os.path.join(PLOTS_DIR, "feature_importance_xgboost.png")
            if os.path.exists(fi_xgb):
                st.image(fi_xgb, use_container_width=True)
                
    with tab_th:
        col_th1, col_th2 = st.columns(2)
        with col_th1:
            th_xgb = os.path.join(PLOTS_DIR, "threshold_tuning_xgboost.png")
            if os.path.exists(th_xgb):
                st.image(th_xgb, use_container_width=True)
        with col_th2:
            th_rf = os.path.join(PLOTS_DIR, "threshold_tuning_random_forest.png")
            if os.path.exists(th_rf):
                st.image(th_rf, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: FINAL TECHNICAL REPORT
# -------------------------------------------------------------
elif nav_selection == "📋 Final Technical Report":
    st.title("📋 Executive & Technical Evaluation Report")
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report_text = f.read()
        st.markdown(report_text)
    else:
        st.info("Report file not found. Run compare.py to generate.")
