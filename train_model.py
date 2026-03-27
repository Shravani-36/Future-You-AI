import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, classification_report
import joblib

df = pd.read_csv("data/future_life_dataset.csv")

X = df.drop(["future_income","burnout_risk"], axis=1)
y_income = df["future_income"]
y_burnout = df["burnout_risk"]

# -----------------------------
# Income Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_income, test_size=0.2, random_state=42)

income_model = RandomForestRegressor(n_estimators=120, random_state=42)
income_model.fit(X_train, y_train)

income_pred = income_model.predict(X_test)
r2 = r2_score(y_test, income_pred)

print("📈 Income Model R2 Score:", round(r2*100,2), "%")

# -----------------------------
# Burnout Model (FIXED)
# -----------------------------
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y_burnout, test_size=0.2, random_state=42)

burnout_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    class_weight="balanced",   # 🔥 IMPORTANT FIX
    random_state=42
)

burnout_model.fit(X_train_b, y_train_b)

burnout_pred = burnout_model.predict(X_test_b)
accuracy = accuracy_score(y_test_b, burnout_pred)

print("🔥 Burnout Accuracy:", round(accuracy*100,2), "%")

print("\n📊 Classification Report:\n")
print(classification_report(y_test_b, burnout_pred, zero_division=1))

# -----------------------------
# Save models
# -----------------------------
joblib.dump(income_model, "models/income_model.pkl")
joblib.dump(burnout_model, "models/burnout_model.pkl")

print("\n✅ Models saved successfully!")