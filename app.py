import streamlit as st
import pandas as pd
import joblib

model = joblib.load('hiring_decision_model.pkl')

st.set_page_config(page_title="Hiring Decision Predictor", page_icon="💼", layout="centered")
st.title("💼 Hiring Decision Predictor")
st.markdown("Fill in the candidate's information to predict whether they will be hired.")

st.subheader("Candidate Profile")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=20, max_value=50, value=30)
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    education_level = st.selectbox(
        "Education Level",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "High School", 2: "Bachelor's", 3: "Master's", 4: "PhD"}[x]
    )
    experience_years = st.slider("Experience Years", min_value=0, max_value=15, value=3)
    previous_companies = st.selectbox("Previous Companies", options=[1, 2, 3, 4, 5])

with col2:
    distance = st.number_input("Distance from Company (km)", min_value=0.5, max_value=52.0, value=10.0, step=0.5)
    interview_score = st.slider("Interview Score", min_value=0, max_value=100, value=70)
    skill_score = st.slider("Skill Score", min_value=0, max_value=100, value=70)
    personality_score = st.slider("Personality Score", min_value=0, max_value=100, value=70)
    recruitment_strategy = st.selectbox(
        "Recruitment Strategy",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Strategy 1", 2: "Strategy 2", 3: "Strategy 3"}[x]
    )

st.divider()

if st.button("Predict Hiring Decision", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "EducationLevel": education_level,
        "ExperienceYears": experience_years,
        "PreviousCompanies": previous_companies,
        "DistanceFromCompany": distance,
        "InterviewScore": interview_score,
        "SkillScore": skill_score,
        "PersonalityScore": personality_score,
        "RecruitmentStrategy": recruitment_strategy
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success("### Hired!")
        st.metric("Confidence", f"{probability[1]*100:.1f}%")
    else:
        st.error("### Not Hired")
        st.metric("Confidence", f"{probability[0]*100:.1f}%")

    st.markdown("**Prediction Probabilities**")
    prob_df = pd.DataFrame({
        "Outcome": ["Not Hired", "Hired"],
        "Probability": [f"{probability[0]*100:.1f}%", f"{probability[1]*100:.1f}%"]
    })
    st.table(prob_df)
