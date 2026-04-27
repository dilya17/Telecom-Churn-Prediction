import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Churn Prediction App", layout="wide")

st.title("📊 Telecom Churn Prediction App")
st.write("Predict customer churn probability and explain predictions using SHAP")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# -----------------------------
# OPTIONAL: LOAD HOLDOUT DATA
# -----------------------------
@st.cache_data
def load_holdout():
    try:
        return pd.read_csv("cell2cellholdout.csv")
    except:
        return None

holdout_df = load_holdout()

# -----------------------------
# USER INPUT
# -----------------------------
st.sidebar.header("🧾 Customer Input")

def user_input():
    data = {
        "MonthlyMinutes": st.sidebar.number_input("Monthly Minutes", 0.0, 5000.0, 500.0),
        "MonthlyRevenue": st.sidebar.number_input("Monthly Revenue", 0.0, 500.0, 50.0),
        "OverageMinutes": st.sidebar.number_input("Overage Minutes", 0.0, 1000.0, 10.0),
        "CustomerCareCalls": st.sidebar.number_input("Customer Care Calls", 0, 20, 1),
        "DroppedCalls": st.sidebar.number_input("Dropped Calls", 0, 50, 2),
        "BlockedCalls": st.sidebar.number_input("Blocked Calls", 0, 50, 1),
        "MonthsInService": st.sidebar.number_input("Months In Service", 1, 100, 12),
        "CurrentEquipmentDays": st.sidebar.number_input("Equipment Days", 0, 2000, 300),
        "AgeHH1": st.sidebar.number_input("Age", 18, 100, 35),
        "HandsetPrice": st.sidebar.number_input("Handset Price", 0.0, 2000.0, 200.0),
        "IncomeGroup": st.sidebar.number_input("Income Group", 0, 10, 5),
        "RetentionCalls": st.sidebar.number_input("Retention Calls", 0, 10, 0),
    }
    return pd.DataFrame([data])

input_df = user_input()

# -----------------------------
# HOLDOUT SAMPLE OPTION
# -----------------------------
if holdout_df is not None:
    if st.checkbox("🎲 Use random customer from holdout dataset"):
        sample = holdout_df.sample(1)
        input_df = sample.copy()
        st.write("Sample from holdout dataset:")
        st.dataframe(input_df)

# -----------------------------
# FIX MISSING FEATURES
# -----------------------------
def align_features(df, model):
    expected_cols = model.feature_names_in_
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    return df[expected_cols]

input_df = align_features(input_df, model)

# -----------------------------
# PREDICTION
# -----------------------------
st.subheader("📥 Input Data")
st.dataframe(input_df)

threshold = st.slider("🎯 Classification Threshold", 0.0, 1.0, 0.5)

if st.button("🚀 Predict"):

    prob = model.predict_proba(input_df)[0][1]
    pred = int(prob > threshold)

    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{prob:.3f}")

    with col2:
        if pred == 1:
            st.error("⚠️ Likely to churn")
        else:
            st.success("✅ Likely to stay")

    # -----------------------------
    # PROFIT CALCULATION
    # -----------------------------
    st.subheader("💰 Profit Analysis")

    FN_COST = st.slider("Cost of missing a churner (FN)", 100, 1000, 500)
    FP_COST = st.slider("Cost of unnecessary offer (FP)", 10, 200, 50)

    expected_profit = prob * FN_COST - (1 - prob) * FP_COST

    st.write(f"**Expected Profit from targeting this customer:** ${expected_profit:.2f}")

    # -----------------------------
    # SHAP EXPLANATION
    # -----------------------------
    st.subheader("🔍 SHAP Explanation")

    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)

        st.write("Feature contribution (waterfall plot):")

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP visualization failed. Possibly due to model compatibility.")
        st.write(str(e))