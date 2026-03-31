import pandas as pd
import numpy as np

# -----------------------------
# Setup
# -----------------------------
np.random.seed(42)
rows = 1500

# -----------------------------
# Generate Features
# -----------------------------
age = np.random.randint(21, 35, rows)

sleep_hours = np.clip(np.random.normal(6.5, 1, rows), 4, 9)
work_hours = np.clip(np.random.normal(8.5, 2, rows), 4, 13)

exercise_days = np.random.choice(
    [0, 1, 2, 3, 4, 5],
    size=rows,
    p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.1]
)

learning_hours = np.clip(np.random.normal(5, 2, rows), 0, 12)

current_income = np.clip(np.random.normal(40000, 15000, rows), 10000, 150000)
expenses = np.clip(np.random.normal(25000, 12000, rows), 5000, 120000)

# -----------------------------
# Derived Features
# -----------------------------
savings_amount = current_income - expenses
savings_rate = (savings_amount / current_income) * 100

# Stress (depends on lifestyle)
stress_level = np.clip(
    10 - sleep_hours + (work_hours / 2) - (exercise_days / 2) + np.random.normal(0, 1, rows),
    1, 10
)

# -----------------------------
# Create DataFrame
# -----------------------------
df = pd.DataFrame({
    "age": age,
    "sleep_hours": sleep_hours,
    "work_hours": work_hours,
    "exercise_days": exercise_days,
    "learning_hours": learning_hours,
    "stress_level": stress_level,
    "current_income": current_income,
    "monthly_expenses": expenses,
    "savings_rate": savings_rate,
    "savings_amount": savings_amount
})

# -----------------------------
# 🔥 ADVANCED BURNOUT LOGIC
# -----------------------------
def burnout(row):

    score = 0

    # Stress (major factor)
    score += row["stress_level"] * 2

    # Work overload
    score += max(0, row["work_hours"] - 9) * 1.5

    # Sleep deficit
    score += max(0, 6 - row["sleep_hours"]) * 2

    # Work + sleep interaction
    if row["work_hours"] > 7 and row["sleep_hours"] < 5:
        score += 5

    # Exercise reduces burnout
    score -= row["exercise_days"] * 1

    # Financial stress
    expense_ratio = row["monthly_expenses"] / row["current_income"]
    if expense_ratio > 0.8:
        score += 6
    elif expense_ratio > 0.6:
        score += 3

    # Learning overload
    if row["learning_hours"] > 7:
        score += 2

    # Add randomness (real-world behavior)
    score += np.random.normal(0, 2)

    # Classification
    if score >= 22:
        return 2   # High
    elif score >= 13:
        return 1   # Medium
    else:
        return 0   # Low


df["burnout_risk"] = df.apply(burnout, axis=1)

# -----------------------------
# 💰 FUTURE INCOME LOGIC
# -----------------------------
df["future_income"] = (
    df["learning_hours"] * 4000 +
    df["savings_rate"] * 1500 +
    (35 - df["age"]) * 1000 -
    df["stress_level"] * 2000 +
    np.random.normal(0, 8000, rows)
)

df["future_income"] = df["future_income"].clip(20000, 200000)

# -----------------------------
# Save Dataset
# -----------------------------
df.to_csv("future_life_dataset_updated.csv", index=False)

# -----------------------------
# Check Distribution
# -----------------------------
print("✅ Dataset generated successfully!")
print("\nBurnout Distribution:")
print(df["burnout_risk"].value_counts())