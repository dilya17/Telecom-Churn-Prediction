import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

def engineer_all_features(X):
    X = X.copy()
    X["RevenuePerMinute"] = X["MonthlyRevenue"] / (X["MonthlyMinutes"] + 1)
    X["ServiceDistressIndex"] = (
        (X["CustomerCareCalls"] + X["DroppedCalls"] + X["BlockedCalls"])
        / (X["MonthsInService"] + 1)
    )
    X["OverageRatio"] = X["OverageMinutes"] / (X["MonthlyMinutes"] + 1)
    return X

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
st.write(model.named_steps)
# -----------------------------
# LOAD HOLDOUT DATA
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
# ALIGN FEATURES (CORRECT WAY)
# -----------------------------
def align_features(df, model):
    df = df.copy()

    expected_cols = model.named_steps["preprocessor"].feature_names_in_

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    return df[expected_cols]

input_df = align_features(input_df, model)

# -----------------------------
# DISPLAY INPUT
# -----------------------------
st.subheader("📥 Input Data")
st.dataframe(input_df)

threshold = st.slider("🎯 Classification Threshold", 0.0, 1.0, 0.5)

# -----------------------------
# SHAP EXPLAINER (CACHED)
# -----------------------------
@st.cache_resource
def get_explainer(_model):
    final_model = list(_model.named_steps.values())[-1]
    return shap.TreeExplainer(final_model)

explainer = get_explainer(model)

# -----------------------------
# PREDICTION
# -----------------------------
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
    # SHAP EXPLANATION (FIXED)
    # -----------------------------
    st.subheader("🔍 SHAP Explanation")

    try:
        # Transform input using pipeline (excluding final model)
        X_processed = model[:-1].transform(input_df)

        # Get feature names AFTER transformation
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()

        # Convert to DataFrame (IMPORTANT FIX)
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

        # Compute SHAP
        shap_values = explainer.shap_values(X_processed_df)

        st.write("Feature contribution:")

        fig, ax = plt.subplots()

        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_processed_df.iloc[0],
                feature_names=feature_names
            ),
            show=False
        )

        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP visualization failed.")
        st.write(str(e))
