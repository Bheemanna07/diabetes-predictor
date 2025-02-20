import os
import pandas as pd
import pickle as pkl
import streamlit as st
from sklearn.metrics import accuracy_score  # type: ignore

# Set page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🧑‍⚕")

# Define model path
diabetes_model_path = "C:/Users/Bheem/OneDrive/Desktop/predictor/diabetes-prediction-model.sav"

# Check if the model file exists before loading
if os.path.exists(diabetes_model_path):
    with open(diabetes_model_path, "rb") as file:
        diabetes_model = pkl.load(file)
    st.success("✅ Model loaded successfully.")
else:
    st.error("❌ Model file NOT found! Check the file path.")
    st.stop()  # Stop execution if model is missing

# Page title
st.title("Diabetes Prediction using Machine Learning")

# Getting user input
col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.text_input("Number of Pregnancies")

with col2:
    glucose = st.text_input("Glucose Level")

with col3:
    blood_pressure = st.text_input("Blood Pressure") 

with col1:
    skin_thickness = st.text_input("Skin Thickness")

with col2:
    insulin = st.text_input("Insulin Level")

with col3:
    bmi = st.text_input("BMI (Body Mass Index)")

with col1:
    diabetes_pedigree_function = st.text_input("Diabetes Pedigree Function")

with col2:
    age = st.text_input("Age")

# Variable for storing the result
diab_diagnosis = ""

# Creating a button to predict the output
if st.button("Diabetes Test Result"):
    try:
        # Convert user input to float
        user_input = [
            float(pregnancies),
            float(glucose),
            float(blood_pressure),
            float(skin_thickness),
            float(insulin),
            float(bmi),
            float(diabetes_pedigree_function),
            float(age),
        ]

        # Make the Prediction
        diab_prediction = diabetes_model.predict([user_input])

        # Display the result
        diab_diagnosis = (
            "The person has Diabetes 🩸" if diab_prediction[0] == 1 else "The person does not have Diabetes ✅"
        )

    except ValueError:
        diab_diagnosis = "⚠ Please enter valid numeric values."

# Show the prediction result
st.subheader("Prediction Result:")
st.write(diab_diagnosis)

# Show model accuracy
if st.button("Show Model Accuracy"):
    test_data_path = "C:/Users/Bheem/OneDrive/Desktop/predictor/diabetes.csv"

    if os.path.exists(test_data_path):
        test_data = pd.read_csv(test_data_path)
        
        X = test_data.drop("Outcome", axis=1)
        y = test_data["Outcome"]

        y_pred = diabetes_model.predict(X)

        accuracy = accuracy_score(y, y_pred)
        st.write(f"Model Accuracy: {accuracy*100:.2f}%")
    else:
        st.error("❌ Test data file NOT found! Check the file path.")
