import streamlit as st
import pandas as pd
import pickle

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="🩺 Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(to bottom,#EAF8FF,#FFFFFF);
}

h1{
    color:#0A6EBD;
    text-align:center;
}

.stButton>button{
    background:#00A86B;
    color:white;
    border-radius:12px;
    height:55px;
    width:100%;
    font-size:20px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#00895A;
    color:white;
}

div[data-testid="stNumberInput"]{
    background:white;
    padding:8px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🏥 Diabetes Prediction")

st.sidebar.success("Algorithm")
st.sidebar.write("Logistic Regression")

st.sidebar.success("Developer")
st.sidebar.write("Your Name")

st.sidebar.success("Technology")
st.sidebar.write("Python • Streamlit • Scikit-learn")

# ---------------- LOGO ----------------
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Blue_circle_for_diabetes.svg/512px-Blue_circle_for_diabetes.svg.png",
    width=120
)

# ---------------- TITLE ----------------
st.title("🩺 Diabetes Prediction System")

st.info("Enter the patient's health information below.")

# ---------------- INPUTS ----------------
glucose = st.number_input(
    "🩸 Random Glucose Level (mg/dL)",
    0.0,500.0,100.0
)

bp = st.number_input(
    "❤️ Diastolic Blood Pressure (mmHg)",
    0.0,200.0,70.0
)

insulin = st.number_input(
    "💉 2-Hour Serum Insulin",
    0.0,1000.0,80.0
)

bmi = st.number_input(
    "⚖️ BMI",
    0.0,70.0,25.0
)

age = st.number_input(
    "🎂 Age",
    1,120,30
)

# ---------------- PREDICTION ----------------
if st.button("➕ Predict Diabetes"):

    input_data = pd.DataFrame({
        "Random_Glucose_Level":[glucose],
        "Diastolic_Blood_Pressure":[bp],
        "2_Hour_Serum_Insulin":[insulin],
        "BMI":[bmi],
        "Age":[age]
    })

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    confidence = probability.max()*100

    st.subheader("📊 Prediction Confidence")
    st.progress(float(confidence/100))
    st.write(f"### {confidence:.2f}%")

    if prediction[0] == 1:
        st.error("🔴 Prediction: Diabetes")
    else:
        st.success("🟢 Prediction: No Diabetes")

# ---------------- FOOTER ----------------
st.markdown("---")


