import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="End-to-End MLOps REST API for employee attrition risk estimation.",
    version="1.0.0"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=500,
                detail="Model artifact missing. Run 'python train.py' first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model

class EmployeeFeatures(BaseModel):
    age: int = Field(..., ge=18, le=75, description="Employee age", example=32)
    monthly_income: float = Field(..., ge=10000.0, le=500000.0, description="Monthly gross income", example=45000.0)
    job_satisfaction: int = Field(..., ge=1, le=4, description="1 (Low) to 4 (Very High)", example=2)
    work_life_balance: int = Field(..., ge=1, le=4, description="1 (Bad) to 4 (Best)", example=2)
    total_working_years: int = Field(..., ge=0, le=50, description="Total career experience", example=8)
    years_at_company: int = Field(..., ge=0, le=40, description="Years spent at current firm", example=3)
    monthly_hours: int = Field(..., ge=60, le=350, description="Average monthly working hours", example=210)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Employee Attrition Prediction API",
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    model_loaded = os.path.exists(MODEL_PATH)
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_artifact_loaded": model_loaded
    }

@app.post("/predict")
def predict_attrition(payload: EmployeeFeatures):
    model = get_model()
    input_df = pd.DataFrame([payload.model_dump()])

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    return {
        "attrition_prediction": prediction,
        "prediction_label": "Likely to Leave" if prediction == 1 else "Likely to Stay",
        "attrition_risk_score": round(probability, 4),
        "status": "Success"
    }