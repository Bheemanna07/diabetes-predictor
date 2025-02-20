import os
import pandas as pd
import pickle as pkl
import streamlit as st
from sklearn.metrics import accuracy_score # type: ignore
# Set page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🧑‍⚕")

# Load the saved diabetes model
diabetes_model_path = r"diabetes-prediction-model.sav"
diabetes_model = pkl.load(open(diabetes_model_path, "rb"))

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
        # Convert the user input into float
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
        if diab_prediction[0] == 1:
            diab_diagnosis = "The person has Diabetes 🩸"
        else:
            diab_diagnosis = "The person does not have Diabetes ✅"

    except ValueError:
        diab_diagnosis = "⚠ Please enter valid numeric values."

# Show the prediction result
st.subheader("Prediction Result:")
st.write(diab_diagnosis)

if st.button("show model accuracy"):
    
    test_data =  pd.read_csv(r"diabetes.csv")

    X = test_data.drop("Outcome", axis=1)
    y = test_data["Outcome"]

    y_pred = diabetes_model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    
    st.write(f"Model Accuracy: {accuracy*100:.2f}%")
