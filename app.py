import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- LOAD MODEL ---
# Note: Ensure 'best_model.pkl' exists in the same directory
try:
    model = joblib.load('best_model.pkl')
except:
    st.error("Model file 'best_model.pkl' not found. Please run main.py first!")

st.set_page_config(page_title="Student AI Predictor", layout="wide")
st.title("🎓 Student Performance Predictor")

# --- INPUT SECTION ---
st.header("Academic Scores & Student Metrics")
st.info("Adjust the sliders below to simulate different student scenarios.")

col1, col2 = st.columns(2)

with col1:
    g1_score = st.slider("Assignment / Midterm 1 (G1)", 0, 20, 10)
    # Adding 'Failures' because it is a huge driver of 'Failure Scenarios' in your abstract
    past_failures = st.selectbox("Past Class Failures", [0, 1, 2, 3, 4], help="Number of past class failures")
    attendance_pct = st.slider("Attendance Rate (%)", 0, 100, 85)

with col2:
    g2_score = st.slider("Assignment / Midterm 2 (G2)", 0, 20, 10)
    study_time = st.selectbox("Weekly Study Time", options=[1, 2, 3, 4],
                              format_func=lambda x: ["<2 hours", "2-5 hours", "5-10 hours", ">10 hours"][x - 1])
    internet = st.radio("Has Internet Access?", ["Yes", "No"], index=0)

# --- PREDICTION LOGIC ---
if st.button("Predict Final Result"):
    # 1. Prepare Features (Matching the 32-feature UCI structure)
    features = np.zeros((1, 32))
    features[0, 30] = g1_score  # G1
    features[0, 31] = g2_score  # G2
    features[0, 13] = study_time  # Study time
    features[0, 14] = past_failures  # Failures
    features[0, 29] = (100 - attendance_pct) * 0.93  # Converting % to 'absences' count
    features[0, 21] = 1 if internet == "Yes" else 0  # Internet feature

    # 2. Run Prediction
    prediction = model.predict(features)[0]

    # 3. Handle Failure Scenario
    st.divider()

    if prediction < 13:
        st.error(f"### ❌ Result: FAIL")
        st.write(f"**Predicted Final Grade (G3): {prediction:.2f} / 20**")

        # --- Explainable AI Section (Simplified LIME/SHAP logic) ---
        st.subheader("🔍 Why did the student fail?")

        # We manually check the inputs to explain the 'Intervention' needed
        reasons = []
        if g2_score < 10: reasons.append("- Low performance in recent assignments (G2).")
        if past_failures > 0: reasons.append(f"- History of {past_failures} past class failures.")
        if attendance_pct < 75: reasons.append("- High number of absences (Attendance below 75%).")
        if study_time == 1: reasons.append("- Insufficient weekly study hours (<2 hours).")

        for r in reasons:
            st.write(r)

        st.warning("**Intervention Recommendation:** Immediate academic counseling and remedial classes for G2 topics.")

    else:
        st.success(f"### ✅ Result: PASS")
        st.write(f"**Predicted Final Grade (G3): {prediction:.2f} / 20**")
        st.balloons()