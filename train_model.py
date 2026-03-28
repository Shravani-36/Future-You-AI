import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import joblib

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("future_life_dataset_updated.csv")

# -----------------------------
# Select Required Columns ONLY
# -----------------------------
required_cols = [
    "age", "sleep_hours", "work_hours", "exercise_days",
    "learning_hours", "savings_rate", "stress_level",
    "future_income", "burnout_risk"
]

df = df[required_cols]

# -----------------------------
# Clean Data
# -----------------------------
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

# Safety check
if df.shape[0] < 10:
    raise ValueError("Dataset is too small or corrupted")

# -----------------------------
# Split Features & Targets
# -----------------------------
X = df.drop(["future_income", "burnout_risk"], axis=1)
y_income = df["future_income"]
y_burnout = df["burnout_risk"]

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_income, test_size=0.2, random_state=42
)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X, y_burnout, test_size=0.2, random_state=42
)

# -----------------------------
# Train Models
# -----------------------------
income_model = RandomForestRegressor(n_estimators=100, random_state=42)
income_model.fit(X_train, y_train)

burnout_model = RandomForestClassifier(n_estimators=120, random_state=42)
burnout_model.fit(X_train_b, y_train_b)

# -----------------------------
# Evaluate Models
# -----------------------------
income_pred = income_model.predict(X_test)
burnout_pred = burnout_model.predict(X_test_b)

r2 = r2_score(y_test, income_pred)
accuracy = accuracy_score(y_test_b, burnout_pred)

print("📈 Income Model R2 Score:", round(r2 * 100, 2), "%")
print("🔥 Burnout Model Accuracy:", round(accuracy * 100, 2), "%")

# -----------------------------
# Save Models
# -----------------------------
joblib.dump(income_model, "income_model.pkl")
joblib.dump(burnout_model, "burnout_model.pkl")

print("✅ Models trained and saved successfully!")
