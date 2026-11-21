"""Unit tests for data cleaning and feature engineering transformers."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import ChurnFeatureEngineer, build_preprocessor


@pytest.fixture
def sample_raw_dataframe():
    return pd.DataFrame(
        [
            {
                "customerID": "1111-TESTA",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": "29.85",
            },
            {
                "customerID": "2222-TESTB",
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 0,  # Zero tenure edge case
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
                "MonthlyCharges": 95.0,
                "TotalCharges": " ",  # Whitespace edge case
            },
            {
                "customerID": "3333-TESTC",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 65,  # Loyal customer
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 80.0,
                "TotalCharges": "5200.0",
            },
        ]
    )


def test_feature_engineer_handles_missing_and_spaces(sample_raw_dataframe):
    transformer = ChurnFeatureEngineer()
    df_transformed = transformer.transform(sample_raw_dataframe)

    # TotalCharges should be converted to numeric without nulls
    assert pd.api.types.is_numeric_dtype(df_transformed["TotalCharges"])
    assert not df_transformed["TotalCharges"].isnull().any()
    # Customer with zero tenure and empty space TotalCharges should have 0.0 TotalCharges
    assert df_transformed.loc[1, "TotalCharges"] == 0.0


def test_engineered_features_present(sample_raw_dataframe):
    transformer = ChurnFeatureEngineer()
    df_transformed = transformer.transform(sample_raw_dataframe)

    expected_new_cols = [
        "tenure_group",
        "avg_monthly_spend",
        "monthly_spend_ratio",
        "total_services",
        "is_month_to_month",
        "has_support_security",
    ]
    for col in expected_new_cols:
        assert col in df_transformed.columns, f"Missing engineered column: {col}"

    # Verify tenure group bins
    assert df_transformed.loc[0, "tenure_group"] == "0-12m"
    assert df_transformed.loc[2, "tenure_group"] == "60+m"

    # Verify total services count
    # Customer 0: PhoneService (Yes), OnlineBackup (Yes), DSL (Internet) = 3 services
    assert df_transformed.loc[0, "total_services"] == 3
    # Customer 2: PhoneService, MultipleLines, DSL, OnlineSec, OnlineBack, DevProt, TechSupp, StreamTV, StreamMov = 9 services
    assert df_transformed.loc[2, "total_services"] == 9

    # Verify contract indicator
    assert df_transformed.loc[0, "is_month_to_month"] == 1
    assert df_transformed.loc[2, "is_month_to_month"] == 0


def test_preprocessor_transformation(sample_raw_dataframe):
    transformer = ChurnFeatureEngineer()
    df_feat = transformer.transform(sample_raw_dataframe)

    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(df_feat)

    # Check that output is a 2D numpy array without NaNs
    assert isinstance(X_processed, np.ndarray)
    assert X_processed.shape[0] == 3
    assert X_processed.shape[1] > 20  # Numerical + One-hot columns
    assert not np.isnan(X_processed).any()
