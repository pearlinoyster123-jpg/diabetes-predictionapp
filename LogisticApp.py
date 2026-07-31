import streamlit as st
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #

with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right,#eef7ff,#ffffff);
}

/* Main Title */

.title{
font-size:42px;
font-weight:bold;
color:#0A4FA8;
text-align:center;
}

.subtitle{
font-size:18px;
text-align:center;
color:#444444;
margin-bottom:30px;
}

/* Cards */

.card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 5px 20px rgba(0,0,0,0.15);
}

/* Button */

div.stButton > button:first-child{
background:#0A84FF;
color:white;
font-size:20px;
font-weight:bold;
border-radius:10px;
height:55px;
width:100%;
border:none;
}

div.stButton > button:first-child:hover{
background:#0070E0;
}

/* Success */

.success{
background:#D4F8D4;
padding:18px;
border-radius:10px;
font-size:22px;
font-weight:bold;
color:green;
text-align:center;
}

/* Danger */

.danger{
background:#FFDADA;
padding:18px;
border-radius:10px;
font-size:22px;
font-weight:bold;
color:red;
text-align:center;
}

/* Footer */

.footer{
text-align:center;
color:gray;
font-size:15px;
margin-top:30px;
}

label{
color:#0A4FA8 !important;
font-weight:bold !important;
font-size:16px !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Blue_circle_for_diabetes.svg/512px-Blue_circle_for_diabetes.svg.png",
    width=120,
)

st.sidebar.title("🩺 Model Information")

st.sidebar.success("Algorithm\n\nLogistic Regression")

st.sidebar.info("Dataset\n\nDiabetes")

st.sidebar.warning("Accuracy\n\n77.21%")

st.sidebar.markdown("---")

st.sidebar.write("### Developed By")
st.sidebar.write("👩‍💻 Umme Ahmad")

# ---------------- TITLE ---------------- #

st.markdown('<div class="title">🩺 AI Diabetes Prediction System</div>', unsafe_allow_html=True)

st.markdown(
'<div class="subtitle">Early diabetes screening using Artificial Intelligence and Machine Learning</div>',
unsafe_allow_html=True)

# ---------------- INPUTS ---------------- #

st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    glucose = st.number_input(
        "🍬 Random Glucose Level (mg/dL)",
        min_value=50,
        max_value=300,
        value=120
    )

    blood_pressure = st.number_input(
        "❤️ Diastolic Blood Pressure (mmHg)",
        min_value=30,
        max_value=150,
        value=80
    )

    insulin = st.number_input(
        "💉 2-Hour Serum Insulin",
        min_value=0,
        max_value=900,
        value=80
    )

with col2:

    bmi = st.number_input(
        "⚖ Body Mass Index (BMI)",
        min_value=10.0,
        max_value=70.0,
        value=25.0
    )

    age = st.number_input(
        "🎂 Age",
        min_value=1,
        max_value=120,
        value=30
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ---------------- #

if st.button("🔍 Predict Diabetes"):

    input_data = pd.DataFrame(
        [[glucose,
          blood_pressure,
          insulin,
          bmi,
          age]],
        columns=[
            "Random_Glucose_Level",
            "Diastolic_Blood_Pressure",
            "2_Hour_Serum_Insulin",
            "BMI",
            "Age"
        ]
    )

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)

    confidence = probability.max() * 100

    st.markdown("### Prediction Confidence")

    st.progress(int(confidence))

    st.write(f"**Confidence:** {confidence:.2f}%")

    st.markdown("---")

    if prediction == 0:

        st.markdown(
        '<div class="success">✅ Result: The person is NOT diabetic.</div>',
        unsafe_allow_html=True)

    else:

        st.markdown(
        '<div class="danger">⚠ Result: The person is likely DIABETIC. Please consult a healthcare professional.</div>',
        unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #

st.markdown("---")

st.markdown(
"""
<div class="footer">
AI Diabetes Prediction System<br>
Machine Learning Based Clinical Decision Support Tool
</div>
""",
unsafe_allow_html=True
)
