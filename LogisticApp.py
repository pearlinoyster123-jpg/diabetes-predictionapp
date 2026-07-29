import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Blue_circle_for_diabetes.svg/512px-Blue_circle_for_diabetes.svg.png",
    width=150
)
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient's details below.")

glucose = st.number_input("Random Glucose Level", 0.0, 500.0, 100.0)
bp = st.number_input("Diastolic Blood Pressure", 0.0, 200.0, 70.0)
insulin = st.number_input("2-Hour Serum Insulin", 0.0, 1000.0, 80.0)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):
    input_data = pd.DataFrame({
        "Random_Glucose_Level": [glucose],
        "Diastolic_Blood_Pressure": [bp],
        "2_Hour_Serum_Insulin": [insulin],
        "BMI": [bmi],
        "Age": [age]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Prediction: Has Sugar")
    else:
        st.success("✅ Prediction: No Sugar")
st.markdown("""
<style>

.stApp{
    background-color:#f4fbff;
}

h1{
    color:#0d6efd;
    text-align:center;
}

h2,h3{
    color:#198754;
}

.stButton>button{
    background-color:#198754;
    color:white;
    font-size:18px;
    border-radius:12px;
    height:55px;
    width:100%;
    font-weight:bold;
}

.stButton>button:hover{
    background-color:#0f5132;
    color:white;
}

div[data-testid="stNumberInput"]{
    background-color:white;
    border-radius:10px;
    padding:8px;
}

</style>
""", unsafe_allow_html=True)
probability = model.predict_proba(input_data)

st.write(f"Confidence: {probability.max()*100:.2f}%")
st.progress(float(probability.max()))
