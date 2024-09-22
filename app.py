import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model

# Load trained Pipeline
model = load_model("best_model")

# Streamlit app title
st.title("Insurance Prediction App")

# Input fields for user data
age = st.number_input("Age", min_value=0, value=36)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=0.0, value=27.5)
children = st.number_input("Children", min_value=0, value=3)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Create DataFrame from user input
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# Predict button
if st.button("Predict"):
    predictions = predict_model(model, data=input_data)
    st.success(f"Prediction: {predictions['prediction_label'].iloc[0]}")

