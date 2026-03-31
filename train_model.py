import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import joblib

df = pd.read_csv("future_life_dataset_updated.csv")

X = df.drop(["future_income", "burnout_risk"], axis=1)

y_income = df["future_income"]
y_burnout = df["burnout_risk"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_income, test_size=0.2, random_state=42)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y_burnout, test_size=0.2, random_state=42)

# Models
income_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
income_model.fit(X_train, y_train)

burnout_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)
burnout_model.fit(X_train_b, y_train_b)

# Metrics
income_r2 = r2_score(y_test, income_model.predict(X_test))
burnout_acc = accuracy_score(y_test_b, burnout_model.predict(X_test_b))

print(f"💰 Income R2 Score: {income_r2 * 100:.2f}%")
print(f"🔥 Burnout Accuracy: {burnout_acc * 100:.2f}%")
# Save
joblib.dump(income_model, "income_model.pkl")
joblib.dump(burnout_model, "burnout_model.pkl")

print("✅ Models saved!")