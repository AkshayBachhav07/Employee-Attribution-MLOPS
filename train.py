import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Generate Synthetic HR Dataset
np.random.seed(42)
n_samples = 1500

age = np.random.randint(21, 60, size=n_samples)
monthly_income = np.random.randint(20000, 180000, size=n_samples)
job_satisfaction = np.random.randint(1, 5, size=n_samples)
work_life_balance = np.random.randint(1, 5, size=n_samples)
total_working_years = np.random.randint(1, 35, size=n_samples)
years_at_company = np.clip(np.random.randint(0, 20, size=n_samples), 0, total_working_years)
monthly_hours = np.random.randint(120, 280, size=n_samples)

data = pd.DataFrame({
    "age": age,
    "monthly_income": monthly_income,
    "job_satisfaction": job_satisfaction,
    "work_life_balance": work_life_balance,
    "total_working_years": total_working_years,
    "years_at_company": years_at_company,
    "monthly_hours": monthly_hours
})

logits = (
    -0.03 * data["age"]
    - 0.000025 * data["monthly_income"]
    - 0.75 * data["job_satisfaction"]
    - 0.65 * data["work_life_balance"]
    + 0.015 * data["monthly_hours"]
    - 0.04 * data["years_at_company"]
    + 1.8
)
prob = 1.0 / (1.0 + np.exp(-logits))
data["attrition"] = (prob > 0.45).astype(int)

data.to_csv(os.path.join("data", "dataset.csv"), index=False)
print("1. Dataset saved to data/dataset.csv")

X = data.drop(columns=["attrition"])
y = data["attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
])

pipeline.fit(X_train, y_train)
acc = accuracy_score(y_test, pipeline.predict(X_test))
print(f"2. Model trained with test accuracy: {acc * 100:.2f}%")

joblib.dump(pipeline, os.path.join("models", "model.pkl"))
print("3. Model artifact exported to models/model.pkl successfully!")