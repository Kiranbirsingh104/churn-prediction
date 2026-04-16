import streamlit as st
import numpy as np
from model import train_model

st.title("Bank Customer Churn Prediction (ANN)")

model, scaler, ct = train_model()

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

    # Prepare input
    sample = np.array([[geo, credit, gender_val, age, tenure, balance,
                        products, card, active, salary]])

    # Apply same encoding
    sample = ct.transform(sample)

    # Scale
    sample = scaler.transform(sample)

    prediction = (model.predict(sample) > 0.5).astype(int)

    if prediction[0][0] == 1:
        st.error("Customer will EXIT ❌")
    else:
        st.success("Customer will STAY ✅")