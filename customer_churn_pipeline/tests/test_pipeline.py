"""Unit tests for the end-to-end model pipeline training, inference, and serialization."""

import os
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import ARTIFACTS_DIR
from src.preprocessing import build_pipeline


@pytest.fixture
def dummy_train_dataset():
    data = {
        "gender": ["Female", "Male", "Female", "Male", "Male", "Female"] * 10,
        "SeniorCitizen": [0, 0, 1, 0, 1, 0] * 10,
        "Partner": ["Yes", "No", "No", "Yes", "No", "Yes"] * 10,
        "Dependents": ["No", "No", "No", "Yes", "No", "Yes"] * 10,
        "tenure": [1, 34, 2, 45, 8, 62] * 10,
        "PhoneService": ["No", "Yes", "Yes", "No", "Yes", "Yes"] * 10,
        "MultipleLines": ["No phone service", "No", "No", "No phone service", "Yes", "Yes"] * 10,
        "InternetService": ["DSL", "DSL", "Fiber optic", "DSL", "Fiber optic", "DSL"] * 10,
        "OnlineSecurity": ["No", "Yes", "No", "Yes", "No", "Yes"] * 10,
        "OnlineBackup": ["Yes", "No", "No", "No", "No", "Yes"] * 10,
        "DeviceProtection": ["No", "Yes", "No", "Yes", "No", "Yes"] * 10,
        "TechSupport": ["No", "No", "No", "Yes", "No", "Yes"] * 10,
        "StreamingTV": ["No", "No", "No", "No", "Yes", "Yes"] * 10,
        "StreamingMovies": ["No", "No", "No", "No", "Yes", "Yes"] * 10,
        "Contract": ["Month-to-month", "One year", "Month-to-month", "One year", "Month-to-month", "Two year"] * 10,
        "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes", "No"] * 10,
        "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check", "Bank transfer (automatic)", "Electronic check", "Credit card (automatic)"] * 10,
        "MonthlyCharges": [29.85, 56.95, 75.35, 42.30, 99.65, 85.50] * 10,
        "TotalCharges": ["29.85", "1889.5", "150.7", "1840.75", "797.2", "5300.0"] * 10,
    }
    y = np.array([1, 0, 1, 0, 1, 0] * 10)
    return pd.DataFrame(data), y


def test_pipeline_fit_and_predict(dummy_train_dataset):
    X, y = dummy_train_dataset
    pipeline = build_pipeline(LogisticRegression(max_iter=200))

    # Fit pipeline
    pipeline.fit(X, y)

    # Predict probabilities
    probs = pipeline.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_pipeline_serialization(dummy_train_dataset, tmp_path):
    X, y = dummy_train_dataset
    pipeline = build_pipeline(LogisticRegression(max_iter=200))
    pipeline.fit(X, y)

    # Save to temp path
    test_model_file = tmp_path / "test_pipeline.joblib"
    joblib.dump(pipeline, test_model_file)

    # Load back
    loaded_pipeline = joblib.load(test_model_file)
    preds_original = pipeline.predict_proba(X)
    preds_loaded = loaded_pipeline.predict_proba(X)

    np.testing.assert_array_almost_equal(preds_original, preds_loaded)
