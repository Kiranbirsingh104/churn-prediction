import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model & preprocessors
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")
ct = joblib.load("ct.pkl")

# Page config
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>📊 Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict whether a customer will leave the bank</p>", unsafe_allow_html=True)

st.divider()

# Inputs (simplified)
col1, col2 = st.columns(2)

with col1:
    credit = st.slider("Credit Score", 300, 900, 600)
    age = st.slider("Age", 18, 80, 30)
    balance = st.number_input("Balance", value=50000)

with col2:
    geo = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    products = st.selectbox("Products", [1, 2, 3, 4])

# Minimal extra inputs (hidden complexity reduced)
tenure = 5
card = 1
active = 1
salary = 50000

st.divider()

# Predict button
if st.button("🔍 Predict", use_container_width=True):

    gender_val = 1 if gender == "Male" else 0

    input_df = pd.DataFrame([{
        "Geography": geo,
        "CreditScore": credit,
        "Gender": gender_val,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": card,
        "IsActiveMember": active,
        "EstimatedSalary": salary
    }])

    # Transform
    input_transformed = ct.transform(input_df)
    input_scaled = scaler.transform(input_transformed)

    # Prediction
    prob = model.predict(input_scaled)[0][0]
    prediction = 1 if prob > 0.5 else 0

    st.divider()

    # Output
    if prediction == 1:
        st.error(f"⚠️ Customer likely to EXIT\n\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Customer likely to STAY\n\nProbability: {prob:.2f}")
