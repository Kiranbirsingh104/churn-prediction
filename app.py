import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

st.title("Bank Customer Churn Prediction (ANN)")

# Load saved files
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")
ct = joblib.load("ct.pkl")

st.header("Enter Customer Details")

credit = st.number_input("Credit Score")
geo = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age")
tenure = st.number_input("Tenure")
balance = st.number_input("Balance")
products = st.number_input("Number of Products")
card = st.selectbox("Has Credit Card", [0, 1])
active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary")

if st.button("Predict"):

    # Encode Gender
    gender_val = 1 if gender == "Male" else 0

    # Create DataFrame (IMPORTANT)
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

    # Apply transformations
    input_transformed = ct.transform(input_df)
    input_scaled = scaler.transform(input_transformed)

    # Predict
    prediction = (model.predict(input_scaled) > 0.5).astype(int)

    if prediction[0][0] == 1:
        st.error("Customer will EXIT ❌")
    else:
        st.success("Customer will STAY ✅")
