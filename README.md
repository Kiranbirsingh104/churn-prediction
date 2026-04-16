# 🏦 Bank Customer Churn Prediction (ANN + Streamlit)

## 📌 Project Overview

This project predicts whether a bank customer will **leave the bank (Exited)** or stay using an **Artificial Neural Network (ANN)**.
The model is deployed as an interactive web application using Streamlit.

---

## 🎯 Objective

To identify customers who are likely to churn so that businesses can take preventive actions and improve customer retention.

---

## 🧠 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* TensorFlow / Keras (ANN Model)
* Streamlit (Web App)

---

## 📊 Dataset

The dataset contains bank customer details such as:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Has Credit Card
* Is Active Member
* Estimated Salary

### 🎯 Target Variable:

* **Exited (0 = Stay, 1 = Leave)**

---

## ⚙️ Model Details

* Model Type: Artificial Neural Network (ANN)
* Hidden Layers: 2
* Activation Function: Sigmoid
* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Batch Size: 25
* Epochs: 10

---

## 🚀 Features

* Predict customer churn in real-time
* User-friendly web interface
* Handles categorical data using encoding
* Scaled input for better performance

---

## 📁 Project Structure

```
churn_project/
│── app.py              # Streamlit app
│── model.py            # ANN model training
│── data.csv            # Dataset
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
```

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Run the app

```
streamlit run app.py
```

---

## 🌐 Output

The app takes customer details as input and predicts:

* ❌ Customer will Exit
* ✅ Customer will Stay

---

## 📈 Future Improvements

* Improve model accuracy
* Add visualization (graphs & charts)
* Deploy on cloud platforms
* Add login system

---

## 🧠 Conclusion

This project demonstrates how machine learning and neural networks can be used to solve real-world business problems like customer churn prediction.

---

## 👨‍💻 Author

Kiranbir Singh
11222569
B.Tech CSE (Data Science)

---
