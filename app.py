import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Load models
# -----------------------------
import os
import joblib

BASE_DIR = os.path.dirname(__file__)

income_path = os.path.join(BASE_DIR, "models/income_model.pkl")
burnout_path = os.path.join(BASE_DIR, "models/burnout_model.pkl")

# If models not found → train automatically
if not os.path.exists(income_path) or not os.path.exists(burnout_path):
    import subprocess
    subprocess.run(["python", "train_model.py"])

income_model = joblib.load(income_path)
burnout_model = joblib.load(burnout_path)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI Lifestyle Analytics", layout="wide")

st.title("🚀 AI-Powered Lifestyle Analytics & Burnout Prediction System")
st.markdown("Understand your lifestyle patterns and predict future outcomes using AI")

st.markdown("---")

# -----------------------------
# Input Mode
# -----------------------------
st.sidebar.header("🧾 Input Mode")
mode = st.sidebar.radio("Choose Input Type:", ["Slider Input", "Manual Input"])

st.sidebar.markdown("---")
st.sidebar.header("Enter Your Details")

if mode == "Slider Input":
    age = st.sidebar.slider("Age", 18, 40, 22)
    sleep = st.sidebar.slider("Sleep Hours", 0, 12, 7)
    work = st.sidebar.slider("Work Hours", 0, 15, 8)
    exercise = st.sidebar.slider("Exercise Days", 0, 7, 3)
    learning = st.sidebar.slider("Learning Hours", 0, 20, 5)
    savings = st.sidebar.slider("Savings Rate (%)", 0, 100, 20)
    stress = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)

else:
    age = st.sidebar.number_input("Age", 18, 40, 22)
    sleep = st.sidebar.number_input("Sleep Hours", 0, 12, 7)
    work = st.sidebar.number_input("Work Hours", 0, 15, 8)
    exercise = st.sidebar.number_input("Exercise Days", 0, 7, 3)
    learning = st.sidebar.number_input("Learning Hours", 0, 20, 5)
    savings = st.sidebar.number_input("Savings Rate (%)", 0, 100, 20)
    stress = st.sidebar.number_input("Stress Level (1-10)", 1, 10, 5)

# -----------------------------
# Convert to DataFrame
# -----------------------------
features = pd.DataFrame([{
    "age": age,
    "sleep_hours": sleep,
    "work_hours": work,
    "exercise_days": exercise,
    "learning_hours": learning,
    "savings_rate": savings,
    "stress_level": stress
}])

# -----------------------------
# Predict
# -----------------------------
if st.button("🔮 Analyze My Lifestyle"):

    income = income_model.predict(features)[0]
    burnout = burnout_model.predict(features)[0]

    risk_map = {0: "Low", 1: "Medium", 2: "High"}
    risk_label = risk_map[burnout]

    # -----------------------------
    # Top Metrics
    # -----------------------------
    col1, col2 = st.columns(2)

    col1.metric("💰 Predicted Income", f"₹{int(income)}")
    col2.metric("🔥 Burnout Risk", risk_label)

    st.markdown("---")

    # -----------------------------
    # 🔥 PERFECT LIFE SCORE (NEW LOGIC)
    # -----------------------------
    score = 0

    # Sleep (ideal 7–8)
    if 7 <= sleep <= 8:
        score += 20
    elif 6 <= sleep < 7 or 8 < sleep <= 9:
        score += 15
    else:
        score += 5

    # Work (ideal 7–9)
    if 7 <= work <= 9:
        score += 20
    elif 6 <= work < 7 or 9 < work <= 10:
        score += 15
    else:
        score += 5

    # Exercise
    if exercise >= 4:
        score += 15
    elif exercise >= 2:
        score += 10
    else:
        score += 5

    # Learning
    if learning >= 6:
        score += 15
    elif learning >= 3:
        score += 10
    else:
        score += 5

    # Savings
    if savings >= 30:
        score += 15
    elif savings >= 15:
        score += 10
    else:
        score += 5

    # Stress (lower is better)
    if stress <= 3:
        score += 15
    elif stress <= 6:
        score += 10
    else:
        score += 2

    score = min(score, 100)

    st.subheader("🌟 Life Balance Score")
    st.progress(score / 100)
    st.write(f"Score: **{score}/100**")

    # -----------------------------
    # Personalized Suggestions
    # -----------------------------
    st.markdown("## 💡 Personalized Suggestions")

    suggestions = []

    if sleep < 6:
        suggestions.append("🛌 Increase sleep to 7–8 hours")
    if work > 10:
        suggestions.append("⚠ Reduce work hours to prevent burnout")
    if stress > 7:
        suggestions.append("🧘 Practice stress management techniques")
    if exercise < 3:
        suggestions.append("🏃 Exercise regularly (at least 3 days/week)")
    if learning < 4:
        suggestions.append("📚 Improve learning consistency")
    if savings < 20:
        suggestions.append("💰 Increase your savings rate")

    if len(suggestions) == 0:
        st.success("🎉 Excellent! Your lifestyle is well balanced")
    else:
        for s in suggestions:
            st.write("✔", s)

    # -----------------------------
    # AI Insights (SMART)
    # -----------------------------
    st.markdown("## 🧠 AI Insights")

    if burnout == 2:
        st.error("🔥 High burnout risk detected due to stress and workload imbalance")

    elif burnout == 1:
        st.warning("⚠ Moderate burnout risk — lifestyle adjustments recommended")

    else:
        st.success("✅ Low burnout risk — maintain your current habits")

    if sleep < 6 and stress > 7:
        st.warning("Low sleep + high stress is a critical risk factor")

    if learning > 6:
        st.info("Strong learning habits positively impact your future income")

    if exercise >= 3:
        st.info("Regular exercise supports mental and physical health")

    # -----------------------------
    # Chart
    # -----------------------------
    st.markdown("## 📊 Lifestyle Analysis")

    chart_data = pd.DataFrame({
        "Factor": ["Sleep","Work","Exercise","Learning","Stress"],
        "Value": [sleep, work, exercise, learning, stress]
    })

    st.bar_chart(chart_data.set_index("Factor"))

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("<center>🚀 Built with Machine Learning & Streamlit</center>", unsafe_allow_html=True)