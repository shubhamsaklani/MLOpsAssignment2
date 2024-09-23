import pandas as pd
import streamlit as st
from pycaret.regression import predict_model, load_model
import requests
import os

# Function to download the model file from Hugging Face
def download_model(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# Streamlit app title
st.title("Insurance Prediction App")

# Define the model file path
model_file_path = "best_model.pkl"

# Check if the model file already exists, if not, download it.
if not os.path.exists(model_file_path):
    with st.spinner("Downloading model..."):
        download_model("https://huggingface.co/Suerz/MLOpsAssignment/resolve/main/best_model_v1.pkl", model_file_path)
        st.success("Model downloaded!")

# Load the downloaded model using PyCaret
model = load_model('best_model')

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
