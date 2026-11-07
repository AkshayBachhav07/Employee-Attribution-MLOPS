import os
import pytest
from fastapi.testclient import TestClient

# Ensure model is trained prior to running server tests
if not os.path.exists("models/model.pkl"):
    import train

from app import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_artifact_loaded"] is True

def test_model_artifact_existence():
    assert os.path.isfile("models/model.pkl"), "Trained model.pkl must exist."

def test_predict_endpoint_valid_payload():
    payload = {
        "age": 32,
        "monthly_income": 45000.0,
        "job_satisfaction": 2,
        "work_life_balance": 2,
        "total_working_years": 8,
        "years_at_company": 3,
        "monthly_hours": 210
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "attrition_prediction" in data
    assert data["attrition_prediction"] in [0, 1]
    assert 0.0 <= data["attrition_risk_score"] <= 1.0
    assert data["prediction_label"] in ["Likely to Stay", "Likely to Leave"]

def test_predict_endpoint_invalid_payload():
    # job_satisfaction cannot be 99 (must be 1-4)
    invalid_payload = {
        "age": 32,
        "monthly_income": 45000.0,
        "job_satisfaction": 99,
        "work_life_balance": 2,
        "total_working_years": 8,
        "years_at_company": 3,
        "monthly_hours": 210
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422