import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model_rf.joblib")

st.title("Lung Cancer Risk Predictor (Numeric Inputs)")

# Example inputs
age = st.number_input("Age", min_value=0, max_value=120, value=45)
gender = st.selectbox("Gender", [1, 2])  # 1=Male, 2=Female
air_pollution = st.slider("Air Pollution (1-9)", 1, 9, 5)
alcohol_use = st.slider("Alcohol use (1-9)", 1, 9, 3)
dust_allergy = st.slider("Dust Allergy (1-9)", 1, 9, 3)
occupational_hazards = st.slider("Occupational Hazards (1-9)", 1, 9, 3)
genetic_risk = st.slider("Genetic Risk (1-9)", 1, 9, 3)
chronic_lung = st.slider("Chronic Lung Disease (1-9)", 1, 9, 3)
balanced_diet = st.slider("Balanced Diet (1-9)", 1, 9, 3)
obesity = st.slider("Obesity (1-9)", 1, 9, 3)
smoking = st.slider("Smoking (1-9)", 1, 9, 3)
passive_smoker = st.slider("Passive Smoker (1-9)", 1, 9, 3)
chest_pain = st.slider("Chest Pain (1-9)", 1, 9, 3)
coughing_blood = st.slider("Coughing of Blood (1-9)", 1, 9, 3)
fatigue = st.slider("Fatigue (1-9)", 1, 9, 3)
weight_loss = st.slider("Weight Loss (1-9)", 1, 9, 3)
shortness_breath = st.slider("Shortness of Breath (1-9)", 1, 9, 3)
wheezing = st.slider("Wheezing (1-9)", 1, 9, 3)
swallowing_diff = st.slider("Swallowing Difficulty (1-9)", 1, 9, 3)
clubbing = st.slider("Clubbing of Finger Nails (1-9)", 1, 9, 3)
frequent_cold = st.slider("Frequent Cold (1-9)", 1, 9, 3)
dry_cough = st.slider("Dry Cough (1-9)", 1, 9, 3)
snoring = st.slider("Snoring (1-9)", 1, 9, 3)

# Build input row
input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Air Pollution": air_pollution,
    "Alcohol use": alcohol_use,
    "Dust Allergy": dust_allergy,
    "OccuPational Hazards": occupational_hazards,
    "Genetic Risk": genetic_risk,
    "chronic Lung Disease": chronic_lung,
    "Balanced Diet": balanced_diet,
    "Obesity": obesity,
    "Smoking": smoking,
    "Passive Smoker": passive_smoker,
    "Chest Pain": chest_pain,
    "Coughing of Blood": coughing_blood,
    "Fatigue": fatigue,
    "Weight Loss": weight_loss,
    "Shortness of Breath": shortness_breath,
    "Wheezing": wheezing,
    "Swallowing Difficulty": swallowing_diff,
    "Clubbing of Finger Nails": clubbing,
    "Frequent Cold": frequent_cold,
    "Dry Cough": dry_cough,
    "Snoring": snoring
}])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    st.success(f"Predicted Level: {pred}")