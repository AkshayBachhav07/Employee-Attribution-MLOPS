import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure necessary directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. Generate Synthetic HR Dataset with Realistic Domain Rules
np.random.seed(42)
n_samples = 1500

age = np.random.randint(21, 60, size=n_samples)
monthly_income = np.random.randint(20000, 180000, size=n_samples)
job_satisfaction = np.random.randint(1, 5, size=n_samples)  # 1 (Low) to 4 (Very High)
work_life_balance = np.random.randint(1, 5, size=n_samples) # 1 (Bad) to 4 (Best)
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

# Log-odds probability of leaving:
# High working hours, low income, and low satisfaction increase attrition risk
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

# Persist dataset
csv_path = os.path.join("data", "dataset.csv")
data.to_csv(csv_path, index=False)
print(f"[1/3] Dataset saved to {csv_path} ({len(data)} records)")

# 2. Train-Test Split
X = data.drop(columns=["attrition"])
y = data["attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Build & Train Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42))
])

print("[2/3] Training Random Forest model...")
pipeline.fit(X_train, y_train)

# 4. Evaluate
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Performance:\n- Test Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 5. Save Artifact
model_path = os.path.join("models", "model.pkl")
joblib.dump(pipeline, model_path)
print(f"[3/3] Trained model artifact exported to {model_path}\n")