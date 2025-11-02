import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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
        "predict_btn": "🔮 Predict",
        "result_low": "🟢 Predicted Risk Level: Low",
        "result_med": "🟠 Predicted Risk Level: Medium",
        "result_high": "🔴 Predicted Risk Level: High",
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
        "predict_btn": "🔮 Bashorat qilish",
        "result_low": "🟢 Bashorat qilingan xavf darajasi: Past",
        "result_med": "🟠 Bashorat qilingan xavf darajasi: O‘rta",
        "result_high": "🔴 Bashorat qilingan xavf darajasi: Yuqori",
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
    air_pollution = st.slider("Air Pollution / Havo ifloslanishi (1-9)", 1, 9, 5)
    alcohol_use = st.slider("Alcohol use / Spirtli ichimlik (1-9)", 1, 9, 3)
    dust_allergy = st.slider("Dust Allergy / Chang allergiyasi (1-9)", 1, 9, 3)
    occupational_hazards = st.slider("Occupational Hazards / Kasbiy xavflar (1-9)", 1, 9, 3)
    balanced_diet = st.slider("Balanced Diet / Muvozanatli ovqatlanish (1-9)", 1, 9, 3)
    obesity = st.slider("Obesity / Semizlik (1-9)", 1, 9, 3)
    smoking = st.slider("Smoking / Chekish (1-9)", 1, 9, 3)
    passive_smoker = st.slider("Passive Smoker / Passiv chekuvchi (1-9)", 1, 9, 3)
    snoring = st.slider("Snoring / Xurrak (1-9)", 1, 9, 3)

with st.expander(T["medical"]):
    genetic_risk = st.slider("Genetic Risk / Genetik xavf (1-9)", 1, 9, 3)
    chronic_lung = st.slider("Chronic Lung Disease / Surunkali o‘pka kasalligi (1-9)", 1, 9, 3)

with st.expander(T["symptoms"]):
    chest_pain = st.slider("Chest Pain / Ko‘krak og‘rig‘i (1-9)", 1, 9, 3)
    coughing_blood = st.slider("Coughing of Blood / Qonli yo‘tal (1-9)", 1, 9, 3)
    fatigue = st.slider("Fatigue / Holzizlik (1-9)", 1, 9, 3)
    weight_loss = st.slider("Weight Loss / Vazn yo‘qotish (1-9)", 1, 9, 3)
    shortness_breath = st.slider("Shortness of Breath / Nafas qisishi (1-9)", 1, 9, 3)
    wheezing = st.slider("Wheezing / Hirillash (1-9)", 1, 9, 3)
    swallowing_diff = st.slider("Swallowing Difficulty / Yutishda qiyinchilik (1-9)", 1, 9, 3)
    clubbing = st.slider("Clubbing of Finger Nails / Tirnoq qalinlashishi (1-9)", 1, 9, 3)
    frequent_cold = st.slider("Frequent Cold / Tez-tez shamollash (1-9)", 1, 9, 3)
    dry_cough = st.slider("Dry Cough / Quruq yo‘tal (1-9)", 1, 9, 3)

# -----------------------
# DataFrame
# -----------------------
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

    st.subheader(T["confidence"])
    prob_df = pd.DataFrame({"Level": model.classes_, "Probability": probs})
    st.bar_chart(prob_df.set_index("Level"))

    st.subheader(T["importance"])
    clf = model.named_steps["clf"]
    feat_names = model.named_steps["prep"].get_feature_names_out()
    importances = clf.feature_importances_

    imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=False).head(10)

    fig, ax = plt.subplots()
    imp_df.plot(kind="barh", x="Feature", y="Importance", ax=ax, legend=False)
    ax.invert_yaxis()
    st.pyplot(fig)