@st.cache_resource
def train_models():
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    rows = 1000

    # Generate synthetic clean dataset
    df = pd.DataFrame({
        "age": np.random.randint(18, 40, rows),
        "sleep_hours": np.random.uniform(5, 9, rows),
        "work_hours": np.random.uniform(6, 12, rows),
        "exercise_days": np.random.randint(0, 7, rows),
        "learning_hours": np.random.uniform(1, 8, rows),
        "savings_rate": np.random.uniform(5, 40, rows),
        "stress_level": np.random.randint(1, 10, rows)
    })

    # Targets
    df["future_income"] = (
        df["learning_hours"] * 5000 +
        df["savings_rate"] * 1500 -
        df["stress_level"] * 2000 +
        np.random.normal(0, 3000, rows)
    )

    df["burnout_risk"] = (
        (df["stress_level"] + df["work_hours"]/2 - df["sleep_hours"]).apply(
            lambda x: 2 if x > 10 else (1 if x > 6 else 0)
        )
    )

    X = df.drop(["future_income", "burnout_risk"], axis=1)
    y_income = df["future_income"]
    y_burnout = df["burnout_risk"]

    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    income_model = RandomForestRegressor(n_estimators=100, random_state=42)
    income_model.fit(X, y_income)

    burnout_model = RandomForestClassifier(n_estimators=120, random_state=42)
    burnout_model.fit(X, y_burnout)

    return income_model, burnout_model
