"""Integration tests for FastAPI REST endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["docs_url"] == "/docs"


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "champion_model" in data


def test_model_info_endpoint():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "champion_model" in data
    assert "test_roc_auc" in data
    assert data["test_roc_auc"] > 0.70
    assert "optimal_decision_threshold" in data
    assert "business_impact" in data


def test_predict_single_high_risk_customer():
    # Customer with multiple high risk flags: month-to-month, fiber optic, no tech support, high spend
    payload = {
        "customerID": "9999-RISKY",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 191.00,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "9999-RISKY"
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in [0, 1]
    assert data["risk_tier"] in ["Low", "Medium", "High"]
    assert len(data["key_risk_drivers"]) > 0
    assert len(data["actionable_recommendation"]) > 0


def test_predict_single_low_risk_loyal_customer():
    # Customer with strong retention characteristics: two year contract, high tenure, bundled services
    payload = {
        "customerID": "1111-LOYAL",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 68,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 64.0,
        "TotalCharges": 4352.0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "1111-LOYAL"
    # Probability should be low for loyal customer
    assert data["churn_probability"] < 0.35
    assert data["churn_prediction"] == 0
    assert data["risk_tier"] == "Low"


def test_predict_validation_error_invalid_input():
    # Negative tenure is rejected by Pydantic schema
    payload = {
        "tenure": -5,
        "Contract": "Month-to-month",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_batch():
    payload = {
        "customers": [
            {
                "customerID": "BATCH-001",
                "tenure": 1,
                "Contract": "Month-to-month",
                "MonthlyCharges": 85.0,
                "TotalCharges": 85.0,
            },
            {
                "customerID": "BATCH-002",
                "tenure": 60,
                "Contract": "Two year",
                "MonthlyCharges": 30.0,
                "TotalCharges": 1800.0,
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 2
    assert len(data["predictions"]) == 2
    assert 0 <= data["predicted_churners"] <= 2
    assert 0.0 <= data["predicted_churn_rate_percent"] <= 100.0
