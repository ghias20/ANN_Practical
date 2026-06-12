import streamlit as st
import pandas as pd
import pickle

from tensorflow.keras.models import load_model

# Load files
model = load_model("model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoders = pickle.load(f)

st.title("Employee Attrition Prediction Using ANN")

age = st.number_input("Age", 18, 80, 30)

length_of_service = st.number_input(
    "Length Of Service",
    0,
    40,
    5
)

city = st.selectbox(
    "City",
    encoders['city_name'].classes_
)

department = st.selectbox(
    "Department",
    encoders['department_name'].classes_
)

gender = st.selectbox(
    "Gender",
    encoders['gender_full'].classes_
)

business = st.selectbox(
    "Business Unit",
    encoders['BUSINESS_UNIT'].classes_
)

if st.button("Predict"):

    data = pd.DataFrame({
        'age': [age],
        'length_of_service': [length_of_service],
        'city_name': [
            encoders['city_name'].transform([city])[0]
        ],
        'department_name': [
            encoders['department_name'].transform([department])[0]
        ],
        'gender_full': [
            encoders['gender_full'].transform([gender])[0]
        ],
        'BUSINESS_UNIT': [
            encoders['BUSINESS_UNIT'].transform([business])[0]
        ]
    })

    data = scaler.transform(data)

    prediction = model.predict(data)

    probability = prediction[0][0]

    st.write(f"Attrition Probability: {probability:.2f}")

    if probability > 0.5:
        st.error("Employee likely to leave the company")
    else:
        st.success("Employee likely to stay in the company")