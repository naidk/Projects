import streamlit as st
from model_loader import load_model_and_scaler
from predict import make_prediction

# Load the trained model and scaler
model, scaler = load_model_and_scaler()

# UI Title
st.set_page_config(page_title="Diabetes Prediction App")
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter patient health metrics below to predict diabetes likelihood.")

# User inputs
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=140, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Predict Button
if st.button("Predict"):
    input_data = [
        pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi,
        diabetes_pedigree, age
    ]
    
    try:
        # Prediction
        prediction, probabilities = make_prediction(input_data, model, scaler)
        diabetic_proba = probabilities[1]
        non_diabetic_proba = probabilities[0]

        # Display Result
        if diabetic_proba >= 0.5:
            st.error(f"⚠️ Patient is likely **diabetic** (Confidence: {diabetic_proba * 100:.2f}%)")
        else:
            st.success(f"✅ Patient is **not diabetic** (Confidence: {non_diabetic_proba * 100:.2f}%)")
        
        # Optional: Probability display
        st.markdown(f"""
        **Prediction Breakdown:**
        - 🟢 Non-Diabetic: `{non_diabetic_proba:.2f}`
        - 🔴 Diabetic: `{diabetic_proba:.2f}`
        """)

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
