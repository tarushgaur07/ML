import streamlit as st
import pandas as pd
import joblib
import os

# Get directory of current app.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Load trained artifacts safely using absolute path
model = joblib.load(os.path.join(BASE_DIR, 'LR_HEART.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
expected_columns = joblib.load(os.path.join(BASE_DIR, 'columns.pkl'))
# 1. Load trained artifacts


st.title("Heart Disease Prediction")
st.markdown("This app predicts the likelihood of heart disease based on user input.")

# 2. User Interface Setup
age = st.slider("Age", 18, 100, 48)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain_type = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
rest_ecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
max_heart_rate = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_induced_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.slider("Oldpeak (ST depression induced by exercise relative to rest)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("Slope of the peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"])

# 3. Prediction Pipeline
if st.button("Predict"):
    # Create feature dictionary matching expected_columns exact names
    raw_data = {
        "Age": age,
        "RestingBP": resting_blood_pressure,
        "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_blood_sugar == "Yes" else 0,
        "MaxHR": max_heart_rate,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "Male" else 0,
        "ChestPainType_ATA": 1 if chest_pain_type == "Atypical Angina" else 0,
        "ChestPainType_NAP": 1 if chest_pain_type == "Non-anginal Pain" else 0,
        "ChestPainType_TA": 1 if chest_pain_type == "Typical Angina" else 0,
        "RestingECG_Normal": 1 if rest_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if rest_ecg == "ST-T Wave Abnormality" else 0,
        "ExerciseAngina_Y": 1 if exercise_induced_angina == "Yes" else 0,
        "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
        "ST_Slope_Up": 1 if st_slope == "Upsloping" else 0,
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_data])

    # Reindex columns to guarantee correct ordering without generating NaNs
    input_df = input_df.reindex(columns=expected_columns)

    # Scale inputs and predict
    scaled_features = scaler.transform(input_df)
    prediction = model.predict(scaled_features)

    # Display prediction
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error("⚠️ High likelihood of heart disease detected. Please consult a doctor.")
    else:
        st.success("✅ Low likelihood of heart disease detected.")
