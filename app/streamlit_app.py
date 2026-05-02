import streamlit as st
import pandas as pd
import joblib

# ==============================
# Load trained XGBoost model
# ==============================
model = joblib.load("xgb_weighted_diabetes_model.pkl")

# ==============================
# Feature list used during training
# (MUST match training features exactly)
# ==============================
FEATURE_COLUMNS = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

# ==============================
# Streamlit UI
# ==============================
st.title("Diabetes Risk Prediction System")
st.write(
    "This application demonstrates the deployment of a trained XGBoost model "
    "for diabetes risk screening based on selected health indicators."
)

st.subheader("Input Health Indicators")

# --- User inputs (key features only) ---
BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=30.0)
Age = st.number_input("Age Category (1–13)", min_value=1, max_value=13, value=9)
HighBP = st.selectbox("High Blood Pressure (0 = No, 1 = Yes)", [0, 1])
HighChol = st.selectbox("High Cholesterol (0 = No, 1 = Yes)", [0, 1])
GenHlth = st.selectbox("General Health (1 = Excellent to 5 = Poor)", [1, 2, 3, 4, 5])

# ==============================
# Prediction
# ==============================
if st.button("Predict Diabetes Risk"):

    # Initialize all features with default value 0
    input_dict = {feature: 0 for feature in FEATURE_COLUMNS}

    # Update features provided by user
    input_dict["BMI"] = BMI
    input_dict["Age"] = Age
    input_dict["HighBP"] = HighBP
    input_dict["HighChol"] = HighChol
    input_dict["GenHlth"] = GenHlth

    # Create DataFrame with correct feature order
    input_data = pd.DataFrame([input_dict])[FEATURE_COLUMNS]

    # Predict probability
    risk_prob = model.predict_proba(input_data)[0, 1]

    # Screening threshold (selected in Chapter 5)
    threshold = 0.3
    decision = (
        "High Risk (Screening Recommended)"
        if risk_prob >= threshold
        else "Low Risk"
    )

    # ==============================
    # Display results
    # ==============================
    st.subheader("Prediction Result")
    st.write(f"**Predicted Diabetes Risk Probability:** {risk_prob:.3f}")
    st.write(f"**Screening Decision:** {decision}")
