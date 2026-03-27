import pandas as pd
import numpy as np

np.random.seed(42)
rows = 1200

# -----------------------------
# Generate features
# -----------------------------
age = np.random.randint(21, 35, rows)

sleep_hours = np.clip(np.random.normal(6.5, 1, rows), 4.5, 8.5)
work_hours = np.clip(np.random.normal(8.5, 1.5, rows), 5, 12)

exercise_days = np.random.choice([0,1,2,3,4,5], size=rows, p=[0.1,0.2,0.3,0.2,0.15,0.05])

learning_hours = np.clip(np.random.normal(5, 2, rows), 0, 12)
savings_rate = np.clip(np.random.normal(25, 10, rows), 5, 50)

# -----------------------------
# Stress calculation
# -----------------------------
stress_level = (
    8
    - sleep_hours
    + (work_hours * 0.6)
    - (exercise_days * 0.5)
    + np.random.normal(0, 1, rows)
)

stress_level = np.clip(stress_level, 1, 10).astype(int)

# -----------------------------
# DataFrame
# -----------------------------
df = pd.DataFrame({
    "age": age,
    "sleep_hours": sleep_hours.round(1),
    "work_hours": work_hours.round(1),
    "exercise_days": exercise_days,
    "learning_hours": learning_hours.round(1),
    "savings_rate": savings_rate.round(1),
    "stress_level": stress_level
})

# -----------------------------
# Income
# -----------------------------
df["future_income"] = (
    df["learning_hours"] * 5000 +
    df["savings_rate"] * 2000 +
    (35 - df["age"]) * 1200 -
    df["stress_level"] * 2500 -
    df["work_hours"] * 800 +
    np.random.normal(0, 7000, rows)
)

df["future_income"] = df["future_income"].clip(25000, 180000)

# -----------------------------
# Balanced Burnout Logic (FIXED)
# -----------------------------
def burnout(row):
    score = (
        row["stress_level"] * 2 +
        row["work_hours"] -
        row["sleep_hours"] * 1.8 -
        row["exercise_days"] * 1.2
    )

    # Adjusted thresholds for balance
    if score > 17:
        return 2   # High
    elif score > 11:
        return 1   # Medium
    else:
        return 0   # Low

df["burnout_risk"] = df.apply(burnout, axis=1)

# -----------------------------
# Save dataset
# -----------------------------
df.to_csv("data/future_life_dataset.csv", index=False)

print("✅ Balanced dataset generated!")