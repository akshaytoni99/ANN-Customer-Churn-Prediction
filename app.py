import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# Load model and preprocessors
model = tf.keras.models.load_model("model.h5")

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------- UI HEADER ----------
st.markdown(
    "<h1 style='text-align: center; color:#4CAF50;'>📊 Customer Churn Prediction App</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Predict whether a customer is likely to churn using Machine Learning</p>",
    unsafe_allow_html=True
)

st.divider()

# ---------- SIDEBAR ----------
st.sidebar.header("🧾 Enter Customer Details")

geography = st.sidebar.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox("👤 Gender", label_encoder_gender.classes_)
age = st.sidebar.slider("🎂 Age", 18, 92, 30)
tenure = st.sidebar.slider("⏳ Tenure (Years)", 0, 10, 3)
credit_score = st.sidebar.number_input("💳 Credit Score", 300, 900, 650)
balance = st.sidebar.number_input("💰 Balance", value=50000.0)
estimated_salary = st.sidebar.number_input("💵 Estimated Salary", value=60000.0)
num_of_products = st.sidebar.slider("📦 Number of Products", 1, 4, 1)
has_cr_card = st.sidebar.selectbox("💳 Has Credit Card", ["Yes", "No"])
is_active_member = st.sidebar.selectbox("⚡ Is Active Member", ["Yes", "No"])

has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0

# ---------- MAIN PANEL ----------
st.subheader("🔍 Prediction Result")

if st.button("🚀 Predict Churn"):

    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.write("### 📈 Churn Probability")
    st.progress(float(prediction_proba))

    st.write(f"**Probability:** {prediction_proba:.2f}")

    if prediction_proba > 0.5:
        st.error("⚠️ The customer is likely to churn!")
    else:
        st.success("✅ The customer is not likely to churn.")

# ---------- FOOTER ----------
st.divider()
st.markdown(
    "<p style='text-align:center;'>💡 Built with Machine Learning & Streamlit</p>",
    unsafe_allow_html=True
)
