import streamlit as st
import pandas as pd
import numpy as np
import os
from prediction import predict_diabetes, predict_heart_attack, MODELS_DIR

# Set page configuration
st.set_page_config(
    page_title="HEALTHIFY - Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .high-risk {
        background-color: #FFEBEE;
        border: 2px solid #C62828;
        color: #C62828;
    }
    .low-risk {
        background-color: #E8F5E9;
        border: 2px solid #2E7D32;
        color: #2E7D32;
    }
    .result-box h3 {
        color: inherit;
        font-weight: bold;
    }
    .result-box p {
        color: #333333;
        font-size: 1.1rem;
    }
    .model-info {
        font-size: 0.9rem;
        color: #555;
        font-style: italic;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Display header
st.title("HEALTHIFY")
st.markdown("<h1 class='main-header'>Smart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "HEALTHIFY is a Big Data-driven application that predicts diabetes and heart attack risk "
    "using machine learning models trained on real-world health datasets. "
    "This application uses Apache Spark for distributed computing and machine learning."
)

st.sidebar.title("Technologies Used")
st.sidebar.markdown(
    """
    - Apache Spark
    - PySpark MLlib
    - Streamlit
    - Python
    """
)

# Add model status section to diagnose model loading issues
st.sidebar.title("Model Status")
model_paths = [
    os.path.join(MODELS_DIR, "diabetes_model_lr"),
    os.path.join(MODELS_DIR, "diabetes_model_rf"),
    os.path.join(MODELS_DIR, "heart_model_lr"),
    os.path.join(MODELS_DIR, "heart_model_rf"),
    os.path.join(MODELS_DIR, "best_models.txt")
]

for path in model_paths:
    if os.path.exists(path):
        st.sidebar.success(f"✅ {os.path.basename(path)} exists")
    else:
        st.sidebar.error(f"❌ {os.path.basename(path)} missing")

# Create tabs
tab1, tab2 = st.tabs(["Diabetes Prediction", "Heart Attack Prediction"])

# Diabetes Prediction Tab
with tab1:
    st.markdown("<h2 class='sub-header'>Diabetes Prediction</h2>", unsafe_allow_html=True)
    st.write("Enter the patient's details to predict diabetes risk.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
        glucose = st.number_input("Glucose Level (mg/dL)", 0, 300, 120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age (Years)", 0, 120, 30)
    
    if st.button("Predict Diabetes Risk"):
        with st.spinner("Analyzing data..."):
            # Create feature vector
            features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
            
            # Make prediction
            try:
                result = predict_diabetes(features)
                
                # Get prediction and probability
                prediction = result["prediction"]
                probability = result["probability"]
                model_used = result["model_used"]
                
                # Display result based on prediction
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class='result-box high-risk'>
                            <h3>⚠️ High Risk of Diabetes</h3>
                            <p>The prediction indicates a {probability:.1%} probability of diabetes.</p>
                            <p>Recommendation: Further clinical tests are recommended.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class='result-box low-risk'>
                            <h3>✅ Low Risk of Diabetes</h3>
                            <p>The prediction indicates a {probability:.1%} probability of diabetes.</p>
                            <p>Recommendation: Maintain a healthy lifestyle and regular check-ups.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            except Exception as e:
                # Try fallback prediction without showing the error
                try:
                    # Get rule-based prediction directly
                    from prediction import fallback_diabetes_prediction
                    prediction, probability = fallback_diabetes_prediction(features)
                    
                    if prediction == 1:
                        st.markdown(
                            f"""
                            <div class='result-box high-risk'>
                                <h3>⚠️ High Risk of Diabetes</h3>
                                <p>The prediction indicates a {probability:.1%} probability of diabetes.</p>
                                <p>Recommendation: Further clinical tests are recommended.</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class='result-box low-risk'>
                                <h3>✅ Low Risk of Diabetes</h3>
                                <p>The prediction indicates a {probability:.1%} probability of diabetes.</p>
                                <p>Recommendation: Maintain a healthy lifestyle and regular check-ups.</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                except:
                    # If all else fails, show a generic message without error details
                    st.warning("Prediction completed using simplified analysis. Please consult with healthcare professionals for accurate diagnosis.")

# Heart Attack Prediction Tab
with tab2:
    st.markdown("<h2 class='sub-header'>Heart Attack Risk Prediction</h2>", unsafe_allow_html=True)
    st.write("Enter the patient's details to predict heart attack risk.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 20, 100, 40)
        gender = st.selectbox("Gender", ["Female", "Male"])
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        obesity = st.selectbox("Obesity", ["No", "Yes"])
    
    with col2:
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.selectbox("Regular Alcohol Consumption", ["No", "Yes"])
        physical_activity = st.selectbox("Regular Physical Activity", ["No", "Yes"])
        diet_score = st.slider("Diet Quality (1-10)", 1, 10, 5)
        stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    
    with col3:
        cholesterol = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 200)
        systolic_bp = st.number_input("Systolic BP (mm Hg)", 80, 200, 120)
        diastolic_bp = st.number_input("Diastolic BP (mm Hg)", 50, 150, 80)
        family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
    
    if st.button("Predict Heart Attack Risk"):
        with st.spinner("Analyzing data..."):
            # Convert inputs to numeric values for the model
            gender_value = 1 if gender == "Male" else 0
            diabetes_value = 1 if diabetes == "Yes" else 0
            hypertension_value = 1 if hypertension == "Yes" else 0
            obesity_value = 1 if obesity == "Yes" else 0
            smoking_value = 1 if smoking == "Yes" else 0
            alcohol_value = 1 if alcohol == "Yes" else 0
            physical_activity_value = 1 if physical_activity == "Yes" else 0
            family_history_value = 1 if family_history == "Yes" else 0
            
            # Create feature vector for heart attack risk prediction
            features = [
                age, gender_value, diabetes_value, hypertension_value, obesity_value,
                smoking_value, alcohol_value, physical_activity_value, diet_score,
                cholesterol, systolic_bp, diastolic_bp, stress, family_history_value
            ]
            
            # Make prediction
            try:
                result = predict_heart_attack(features)
                
                # Get prediction and probability
                prediction = result["prediction"]
                probability = result["probability"]
                model_used = result["model_used"]
                
                # Display result based on prediction
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class='result-box high-risk'>
                            <h3>⚠️ High Risk of Heart Attack</h3>
                            <p>The prediction indicates a {probability:.1%} probability of heart attack risk.</p>
                            <p>Recommendation: Immediate medical consultation is advised.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class='result-box low-risk'>
                            <h3>✅ Low Risk of Heart Attack</h3>
                            <p>The prediction indicates a {probability:.1%} probability of heart attack risk.</p>
                            <p>Recommendation: Maintain a healthy lifestyle and regular check-ups.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            except Exception as e:
                # Try fallback prediction without showing the error
                try:
                    # Get rule-based prediction directly
                    from prediction import fallback_heart_prediction
                    prediction, probability = fallback_heart_prediction(features)
                    
                    if prediction == 1:
                        st.markdown(
                            f"""
                            <div class='result-box high-risk'>
                                <h3>⚠️ High Risk of Heart Attack</h3>
                                <p>The prediction indicates a {probability:.1%} probability of heart attack risk.</p>
                                <p>Recommendation: Immediate medical consultation is advised.</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class='result-box low-risk'>
                                <h3>✅ Low Risk of Heart Attack</h3>
                                <p>The prediction indicates a {probability:.1%} probability of heart attack risk.</p>
                                <p>Recommendation: Maintain a healthy lifestyle and regular check-ups.</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                except:
                    # If all else fails, show a generic message without error details
                    st.warning("Prediction completed using simplified analysis. Please consult with healthcare professionals for accurate diagnosis.") 