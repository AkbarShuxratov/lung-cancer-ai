import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

def map_value(x):
    # 0 -> 1, 1 -> 5, 2 -> 9
    return {0: 1, 1: 5, 2: 9}[x]

# Modelni yuklash
@st.cache_resource
def load_model():
    return joblib.load("model_rf.joblib")

model = load_model()

st.set_page_config(page_title="Lung Cancer Risk Predictor", page_icon="🫁", layout="wide")

# -----------------------
# Language selection
# -----------------------
lang = st.sidebar.radio("🌐 Language / Til", ["English", "O‘zbekcha"])

# -----------------------
# Text dictionary
# -----------------------
TEXTS = {
    "English": {
        "title": "🫁 Lung Cancer Risk Predictor",
        "about": "**Lung Cancer Risk Predictor**\nBuilt with Streamlit + Random Forest.\nDataset: Kaggle (Lung Cancer Prediction).\n\n⚠️ *Disclaimer: This is a demo ML app, not medical advice.*",
        "demographics": "👤 Demographics",
        "lifestyle": "🏠 Lifestyle Factors",
        "medical": "🩺 Medical History",
        "symptoms": "⚠️ Symptoms",
        "predict_btn": "🔮 Diagnose",
        "result_low": "🟢 Risk Level: Low",
        "result_med": "🟠 Risk Level: Medium",
        "result_high": "🔴 Risk Level: High",
        "confidence": "Prediction Confidence (%)",
        "importance": "Top 10 Important Features"
    },
    "O‘zbekcha": {
        "title": "🫁 O‘pka saratoni xavfini aniqlash",
        "about": "**O‘pka saratoni xavfini bashoratlovchi dastur**\nStreamlit + Random Forest yordamida qurilgan.\nMa’lumotlar manbai: Kaggle (Lung Cancer Prediction).\n\n⚠️ *Eslatma: Bu dastur faqat o‘quv va demo maqsadida. Tibbiy maslahat emas.*",
        "demographics": "👤 Demografiya",
        "lifestyle": "🏠 Turmush tarzi",
        "medical": "🩺 Tibbiy tarix",
        "symptoms": "⚠️ Belgilar (simptomlar)",
        "predict_btn": "🔮 Tashxis qo'yish",
        "result_low": "🟢 xavf darajasi: Past",
        "result_med": "🟠 xavf darajasi: O‘rta",
        "result_high": "🔴 xavf darajasi: Yuqori",
        "confidence": "Bashorat ishonchliligi (%)",
        "importance": "Eng muhim 10 ta belgi"
    }
}

T = TEXTS[lang]

# Sidebar info
st.sidebar.title("ℹ️ Info / Ma’lumot")
st.sidebar.markdown(T["about"])

st.title(T["title"])

# -----------------------
# Inputs
# -----------------------
with st.expander(T["demographics"]):
    age = st.number_input("Age / Yosh", min_value=0, max_value=120, value=45)
    gender = st.selectbox("Gender / Jins", [1, 2], format_func=lambda x: "Male / Erkak" if x == 1 else "Female / Ayol")

with st.expander(T["lifestyle"]):
    air_pollution = st.slider("Air Pollution / Havo ifloslanishi (0-2)", 0, 2, 1)
    alcohol_use = st.slider("Alcohol use / Spirtli ichimlik (0-2)", 0, 2, 1)
    dust_allergy = st.slider("Dust Allergy / Chang allergiyasi (0-2)", 0, 2, 1)
    occupational_hazards = st.slider("Occupational Hazards / Kasbiy xavflar (0-2)", 0, 2, 1)
    balanced_diet = st.slider("Balanced Diet / Muvozanatli ovqatlanish (0-2)", 0, 2, 1)
    obesity = st.slider("Obesity / Semizlik (0-2)", 0, 2, 1)
    smoking = st.slider("Smoking / Chekish (0-2)", 0, 2, 1)
    snoring = st.slider("Snoring / Xurrak (0-2)", 0, 2, 1)

with st.expander(T["medical"]):
    genetic_risk = st.slider("Genetic Risk / Genetik xavf (0-2)", 0, 2, 1)
    chronic_lung = st.slider("Chronic Lung Disease / Surunkali o‘pka kasalligi (0-2)", 0, 2, 1)

with st.expander(T["symptoms"]):
    chest_pain = st.slider("Chest Pain / Ko‘krak og‘rig‘i (0-2)", 0, 2, 1)
    coughing_blood = st.slider("Coughing of Blood / Qonli yo‘tal (0-2)", 0, 2, 1)
    fatigue = st.slider("Fatigue / Holzizlik (0-2)", 0, 2, 1)
    weight_loss = st.slider("Weight Loss / Vazn yo‘qotish (0-2)", 0, 2, 1)
    shortness_breath = st.slider("Shortness of Breath / Nafas qisishi (0-2)", 0, 2, 1)
    wheezing = st.slider("Wheezing / Hirillash (0-2)", 0, 2, 1)
    swallowing_diff = st.slider("Swallowing Difficulty / Yutishda qiyinchilik (0-2)", 0, 2, 1)
    clubbing = st.slider("Clubbing of Finger Nails / Tirnoq qalinlashishi (0-2)", 0, 2, 1)
    frequent_cold = st.slider("Frequent Cold / Tez-tez shamollash (0-2)", 0, 2, 1)
    dry_cough = st.slider("Dry Cough / Quruq yo‘tal (0-2)", 0, 2, 1)

# -----------------------
# DataFrame
# -----------------------
input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Air Pollution": map_value(air_pollution),
    "Alcohol use": map_value(alcohol_use),
    "Dust Allergy": map_value(dust_allergy),
    "OccuPational Hazards": map_value(occupational_hazards),
    "Genetic Risk": map_value(genetic_risk),
    "chronic Lung Disease": map_value(chronic_lung),
    "Balanced Diet": map_value(balanced_diet),
    "Obesity": map_value(obesity),
    "Smoking": map_value(smoking),
    "Chest Pain": map_value(chest_pain),
    "Coughing of Blood": map_value(coughing_blood),
    "Fatigue": map_value(fatigue),
    "Weight Loss": map_value(weight_loss),
    "Shortness of Breath": map_value(shortness_breath),
    "Wheezing": map_value(wheezing),
    "Swallowing Difficulty": map_value(swallowing_diff),
    "Clubbing of Finger Nails": map_value(clubbing),
    "Frequent Cold": map_value(frequent_cold),
    "Dry Cough": map_value(dry_cough),
    "Snoring": map_value(snoring)
}])
# Passive Smoker ustuni model kutgani uchun dummy qiymat qo‘shamiz
input_df["Passive Smoker"] = 0

# -----------------------
# Prediction
# -----------------------
if st.button(T["predict_btn"]):
    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]

    if pred == "Low":
        st.success(T["result_low"])
    elif pred == "Medium":
        st.warning(T["result_med"])
    else:
        st.error(T["result_high"])