"""Data cleaning, feature engineering, and pipeline construction."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    SERVICE_COLUMNS,
    TENURE_BINS,
    TENURE_LABELS,
)


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer to engineer subscription domain features and clean inputs."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Clean raw columns and engineer business-driven churn features."""
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X).copy()

        # 1. Clean TotalCharges: handle spaces, missing values, string conversions
        if "TotalCharges" in df.columns:
            # Replace whitespace strings with NaN and parse as float
            cleaned_total = pd.to_numeric(
                df["TotalCharges"].astype(str).str.strip().replace("", np.nan),
                errors="coerce",
            )
            # If tenure is 0, TotalCharges is 0; otherwise impute with MonthlyCharges * tenure or median
            if "tenure" in df.columns and "MonthlyCharges" in df.columns:
                tenure_num = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
                monthly_num = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
                df["TotalCharges"] = cleaned_total.fillna(tenure_num * monthly_num)
            else:
                df["TotalCharges"] = cleaned_total.fillna(0.0)
        else:
            df["TotalCharges"] = 0.0

        # Ensure tenure and MonthlyCharges are numeric
        if "tenure" in df.columns:
            df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
        else:
            df["tenure"] = 0

        if "MonthlyCharges" in df.columns:
            df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
        else:
            df["MonthlyCharges"] = 0.0

        # 2. Feature 1: Tenure Group (Cohort binning)
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=TENURE_BINS,
            labels=TENURE_LABELS,
            include_lowest=True,
        ).astype(str)

        # 3. Feature 2: Average Monthly Spend across customer history
        safe_tenure = np.maximum(df["tenure"].values, 1)
        df["avg_monthly_spend"] = df["TotalCharges"].values / safe_tenure

        # 4. Feature 3: Monthly Spend Ratio (Current bill vs historical average)
        # Captures recent bill escalation / price shocks
        df["monthly_spend_ratio"] = df["MonthlyCharges"].values / np.maximum(
            df["avg_monthly_spend"].values, 0.01
        )

        # 5. Feature 4: Total Subscribed Services Count
        # Bundled services significantly reduce churn likelihood
        service_count = np.zeros(len(df), dtype=int)
        for col in SERVICE_COLUMNS:
            if col in df.columns:
                service_count += (df[col].astype(str) == "Yes").astype(int)

        if "InternetService" in df.columns:
            service_count += (
                df["InternetService"].astype(str).isin(["DSL", "Fiber optic"])
            ).astype(int)

        df["total_services"] = service_count

        # 6. Feature 5: Month-to-month contract indicator
        if "Contract" in df.columns:
            df["is_month_to_month"] = (df["Contract"].astype(str) == "Month-to-month").astype(int)
        else:
            df["is_month_to_month"] = 0

        # 7. Feature 6: Has Tech Support or Online Security
        sec = df["OnlineSecurity"].astype(str) == "Yes" if "OnlineSecurity" in df.columns else False
        tech = df["TechSupport"].astype(str) == "Yes" if "TechSupport" in df.columns else False
        df["has_support_security"] = (sec | tech).astype(int)

        return df


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer for scaling numerical and one-hot encoding categorical features."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "bin",
                "passthrough",
                BINARY_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def build_pipeline(classifier) -> Pipeline:
    """Construct an end-to-end Pipeline that takes raw inputs and outputs predictions."""
    return Pipeline(
        steps=[
            ("feature_engineer", ChurnFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
