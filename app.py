import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Load Models
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))

income_path = os.path.join(current_dir, "income_model.pkl")
burnout_path = os.path.join(current_dir, "burnout_model.pkl")

income_model = joblib.load(income_path)
burnout_model = joblib.load(burnout_path)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI Lifestyle Analytics", layout="wide")

st.title("🚀 AI-Powered Lifestyle Analytics & Burnout Prediction System")
st.markdown("Predict your future income, burnout risk, and life balance")

st.markdown("---")

# -----------------------------
# Sidebar Inputs
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
    stress = st.sidebar.slider("Stress Level", 1, 10, 5)
    current_income = st.sidebar.slider("Current Income (₹)", 10000, 200000, 40000)
    expenses = st.sidebar.slider("Monthly Expenses (₹)", 5000, 150000, 20000)

else:
    age = st.sidebar.number_input("Age", 18, 40, 22)
    sleep = st.sidebar.number_input("Sleep Hours", 0, 12, 7)
    work = st.sidebar.number_input("Work Hours", 0, 15, 8)
    exercise = st.sidebar.number_input("Exercise Days", 0, 7, 3)
    learning = st.sidebar.number_input("Learning Hours", 0, 20, 5)
    stress = st.sidebar.number_input("Stress Level", 1, 10, 5)
    current_income = st.sidebar.number_input("Current Income (₹)", 10000, 200000, 40000)
    expenses = st.sidebar.number_input("Monthly Expenses (₹)", 5000, 150000, 20000)

# -----------------------------
# Derived Features
# -----------------------------
savings_amount = current_income - expenses
savings_rate = (savings_amount / current_income) * 100 if current_income != 0 else 0

features = pd.DataFrame([{
    "age": age,
    "sleep_hours": sleep,
    "work_hours": work,
    "exercise_days": exercise,
    "learning_hours": learning,
    "stress_level": stress,
    "current_income": current_income,
    "monthly_expenses": expenses,
    "savings_rate": savings_rate,
    "savings_amount": savings_amount
}])

# -----------------------------
# 🔥 Hybrid Burnout Function
# -----------------------------
def calculate_burnout(age, sleep, work, exercise, learning, stress, income, expenses):

    score = 0

    score += stress * 2

    if work > 9:
        score += (work - 8) * 2

    if sleep < 6:
        score += (6 - sleep) * 2

    score -= exercise * 1

    if expenses > income * 0.7:
        score += 4

    if learning > 8:
        score += 2

    if score >= 22:
        return 2
    elif score >= 13:
        return 1
    else:
        return 0

# -----------------------------
# Button Action
# -----------------------------
if st.button("🔮 Analyze My Future"):

    # Income Prediction
    income = income_model.predict(features)[0]

    # Hybrid Burnout
    burnout_rule = calculate_burnout(
        age, sleep, work, exercise, learning, stress, current_income, expenses
    )

    burnout_ml = burnout_model.predict(features)[0]

    burnout = round((burnout_rule + burnout_ml) / 2)

    risk_map = {0: "Low", 1: "Medium", 2: "High"}
    risk = risk_map.get(burnout, "Unknown")

    col1, col2 = st.columns(2)
    col1.metric("💰 Predicted Future Income", f"₹{int(income)}")
    col2.metric("🔥 Burnout Risk", risk)

    st.markdown("---")

    # -----------------------------
    # 🌟 Life Score
    # -----------------------------
    score = 100

    if sleep < 5: score -= 20
    elif sleep < 6: score -= 10

    if work > 10: score -= 15

    score -= stress * 3

    if exercise < 2: score -= 10
    elif exercise >= 4: score += 5

    if learning < 3: score -= 10
    elif learning > 6: score += 5

    if savings_rate < 10: score -= 15
    elif savings_rate > 30: score += 5

    score = max(0, min(100, int(score)))

    st.subheader("🌟 Life Balance Score")
    st.progress(score / 100)
    st.write(f"Score: **{score}/100**")

    # -----------------------------
    
    # 💡 Personalized Suggestions
    # -----------------------------
    st.markdown("## 💡 Personalized Suggestions")

    suggestions = []

    if sleep < 6 and stress > 7:
        suggestions.append("⚠ Critical: Low sleep + high stress → burnout risk")

    if work > 10 and exercise < 2:
        suggestions.append("⚠ Long work hours + low exercise → health risk")

    if savings_rate < 10 and expenses > current_income * 0.7:
        suggestions.append("💸 High spending compared to income")

    if sleep < 6:
        suggestions.append("🛌 Improve sleep (7–8 hrs)")

    if stress > 7:
        suggestions.append("🧘 Reduce stress via meditation")

    if exercise < 3:
        suggestions.append("🏃 Exercise at least 3 days/week")

    if learning < 4:
        suggestions.append("📚 Increase learning for career growth")

    if savings_rate < 20:
        suggestions.append("💰 Improve savings habits")

    if not suggestions:
        st.success("🎉 Excellent! Your lifestyle is optimized")
    else:
        for s in suggestions:
            st.write("✔", s)

    # -----------------------------
    # 🧠 AI Insights
    # -----------------------------
    st.markdown("## 🧠 AI Insights")

    if burnout == 2:
        st.error("🔥 High burnout risk due to stress & workload imbalance")
    elif burnout == 1:
        st.warning("⚠ Moderate burnout risk detected")
    else:
        st.success("✅ Low burnout risk — stable lifestyle")

    if sleep < 6 and work > 9:
        st.warning("📉 Low sleep + high work reduces productivity")

    if learning > 6 and stress < 5:
        st.info("📈 Strong growth mindset detected")

    if savings_rate > 25:
        st.info("💡 Strong financial discipline")

    if exercise >= 3 and stress <= 5:
        st.info("💪 Healthy routine supports resilience")

    if stress > 8:
        st.error("🚨 Extreme stress detected — act immediately")

    if income > current_income * 1.5:
        st.success("🚀 Strong future financial growth predicted")
    elif income < current_income:
        st.warning("📉 Lifestyle may reduce future income")

    # -----------------------------
    # Chart
    # -----------------------------
    st.markdown("## 📊 Lifestyle Analysis")

    chart = pd.DataFrame({
        "Factor": ["Sleep","Work","Exercise","Learning","Stress"],
        "Value": [sleep, work, exercise, learning, stress]
    })

    st.bar_chart(chart.set_index("Factor"))

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("<center>🚀 Built with Machine Learning & Streamlit</center>", unsafe_allow_html=True)