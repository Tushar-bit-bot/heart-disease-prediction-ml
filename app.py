
import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Load trained model and preprocessing objects
# --------------------------------------------------

model = joblib.load("logistic_regression_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# --------------------------------------------------
# App title
# --------------------------------------------------

st.title("❤️ Heart Disease Prediction")
st.markdown(
    "Enter the patient's medical information below to predict "
    "the likelihood of heart disease."
)

st.divider()


# --------------------------------------------------
# User Inputs
# --------------------------------------------------

Age = st.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)

Sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_BP = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=200,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dl)",
    min_value=100,
    max_value=600,
    value=200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.slider(
    "Maximum Heart Rate",
    min_value=60,
    max_value=220,
    value=150
)

exercise_angina = st.selectbox(
    "Exercise-Induced Angina",
    ["Y", "N"]
)

oldpeak = st.slider(
    "Oldpeak (ST Depression)",
    min_value=0.0,
    max_value=6.0,
    value=1.0
)

st_slope = st.selectbox(
    "ST Slope",
    ["UP", "FLAT", "DOWN"]
)


st.divider()


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict Heart Disease", use_container_width=True):

    # Create dataframe from user input
    raw_input = {
        "Age": Age,
        "Sex": Sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_BP,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }

    input_df = pd.DataFrame([raw_input])


    # --------------------------------------------------
    # Encode categorical variables
    # Same approach used during model training
    # --------------------------------------------------

    input_df = pd.get_dummies(
        input_df,
        drop_first=True
    )


    # --------------------------------------------------
    # Make input columns identical to training columns
    # --------------------------------------------------

    for col in expected_columns:

        if col not in input_df.columns:
            input_df[col] = 0


    # Remove unexpected columns
    input_df = input_df[expected_columns]


    # --------------------------------------------------
    # Convert to integer
    # --------------------------------------------------

    input_df = input_df.astype(int)


    # --------------------------------------------------
    # Scale using the saved training scaler
    # --------------------------------------------------

    scaled_input = scaler.transform(input_df)


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    prediction = model.predict(scaled_input)[0]


    # --------------------------------------------------
    # Prediction probability
    # --------------------------------------------------

    probability = model.predict_proba(scaled_input)[0][1]


    # --------------------------------------------------
    # Display result
    # --------------------------------------------------

    if prediction == 1:

        st.error(
            "⚠️ Higher Likelihood of Heart Disease"
        )

        st.metric(
            "Estimated Risk Probability",
            f"{probability * 100:.2f}%"
        )

    else:

        st.success(
            "✅ Lower Likelihood of Heart Disease"
        )

        st.metric(
            "Estimated Risk Probability",
            f"{probability * 100:.2f}%"
        )


    # --------------------------------------------------
    # Disclaimer
    # --------------------------------------------------

    st.warning(
        "This application is for educational purposes only "
        "and should not be used as a medical diagnosis."
    )
