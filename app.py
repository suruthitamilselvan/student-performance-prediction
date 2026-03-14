import streamlit as st
import pickle
import numpy as np

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# ── Load Models ───────────────────────────────────────────────
with open('model_lr.pkl', 'rb') as f:
    lr = pickle.load(f)

with open('model_rf.pkl', 'rb') as f:
    rf = pickle.load(f)

with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

scaler = data['scaler']

# ── Header ────────────────────────────────────────────────────
st.title("🎓 Student Performance Predictor")
st.markdown("Enter the student details below to predict their **Final Score** and **Pass/Fail** result.")
st.markdown("---")

# ── Input Form ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    study_hours      = st.slider("📚 Study Hours per Day",
                                  min_value=1.0, max_value=10.0,
                                  value=5.0, step=0.5)
    attendance       = st.slider("🏫 Attendance (%)",
                                  min_value=50.0, max_value=100.0,
                                  value=75.0, step=1.0)
    previous_marks   = st.slider("📝 Previous Marks",
                                  min_value=30.0, max_value=100.0,
                                  value=65.0, step=1.0)

with col2:
    assignments_done = st.slider("✅ Assignments Completed (out of 10)",
                                  min_value=0, max_value=10,
                                  value=7, step=1)
    sleep_hours      = st.slider("😴 Sleep Hours per Day",
                                  min_value=4.0, max_value=10.0,
                                  value=7.0, step=0.5)
    internet_hours   = st.slider("🌐 Internet Usage Hours per Day",
                                  min_value=0.0, max_value=8.0,
                                  value=3.0, step=0.5)

st.markdown("---")

# ── Predict Button ────────────────────────────────────────────
if st.button("🔮 Predict Performance", use_container_width=True):

    # Prepare input
    input_data = np.array([[study_hours, attendance, previous_marks,
                             assignments_done, sleep_hours, internet_hours]])

    input_scaled = scaler.transform(input_data)

    # Predictions
    predicted_score  = lr.predict(input_scaled)[0]
    predicted_score  = round(np.clip(predicted_score, 0, 100), 2)
    pass_fail_result = rf.predict(input_scaled)[0]

    st.markdown("## 📊 Prediction Results")

    # Score result
    col3, col4 = st.columns(2)

    with col3:
        st.metric(label="Predicted Final Score", value=f"{predicted_score} / 100")

    with col4:
        if pass_fail_result == 1:
            st.success("## Result: PASS 🎉")
        else:
            st.error("## Result: FAIL ❌")

    # Score feedback
    st.markdown("### Feedback")
    if predicted_score >= 85:
        st.success("Excellent performance! Keep it up!")
    elif predicted_score >= 70:
        st.info("Good performance! A little more effort can make it great!")
    elif predicted_score >= 50:
        st.warning("Average performance. Focus more on studies!")
    else:
        st.error("Poor performance. Needs significant improvement!")

    # Input summary
    st.markdown("---")
    st.markdown("### Input Summary")
    summary = {
        "Study Hours"       : study_hours,
        "Attendance (%)"    : attendance,
        "Previous Marks"    : previous_marks,
        "Assignments Done"  : assignments_done,
        "Sleep Hours"       : sleep_hours,
        "Internet Hours"    : internet_hours
    }
    st.table(summary)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("Built with Python, Scikit-learn and Streamlit")