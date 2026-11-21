"""Service layer for model inference, batch scoring, and retention insights."""

import json
import os
import uuid
from typing import List, Tuple
import joblib
import pandas as pd

from app.schemas import (
    BatchPredictionResponse,
    ChurnPredictionResponse,
    CustomerInput,
    ModelInfoResponse,
)
from src.config import METADATA_PATH, MODEL_PATH


class ChurnPredictionService:
    """Manages model loading and prediction business logic."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChurnPredictionService, cls).__new__(cls)
            cls._instance._load_model_and_metadata()
        return cls._instance

    def _load_model_and_metadata(self):
        """Load serialized pipeline and metadata from artifacts."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model artifact not found at {MODEL_PATH}. Run run_pipeline.py first."
            )

        self.pipeline = joblib.load(MODEL_PATH)

        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "champion_model": "Unknown",
                "optimal_decision_threshold": 0.5,
                "cv_roc_auc": 0.0,
                "test_roc_auc": 0.0,
                "test_pr_auc": 0.0,
                "business_impact": {},
                "best_hyperparameters": {},
            }

        self.threshold = float(self.metadata.get("optimal_decision_threshold", 0.50))
        self.model_name = self.metadata.get("champion_model", "Trained ML Model")

    def _derive_insights(self, row: dict, proba: float) -> Tuple[str, List[str], str]:
        """Derive risk tier, contributing drivers, and tailored retention recommendation."""
        drivers = []

        if row.get("Contract") == "Month-to-month":
            drivers.append("Month-to-month contract (flexible cancellation)")
        if row.get("tenure", 0) <= 12:
            drivers.append("Short tenure / Early customer lifecycle (< 1 year)")
        if row.get("InternetService") == "Fiber optic" and row.get("TechSupport") == "No":
            drivers.append("High-speed Fiber without dedicated Tech Support")
        if row.get("PaymentMethod") == "Electronic check":
            drivers.append("Manual Electronic check payment method")
        if row.get("OnlineSecurity") == "No":
            drivers.append("No Online Security add-on protection")
        if row.get("MonthlyCharges", 0) >= 75.0:
            drivers.append(f"High monthly recurring spend (${row.get('MonthlyCharges'):.2f})")

        # Risk tiering
        if proba >= 0.55:
            tier = "High"
        elif proba >= self.threshold:
            tier = "Medium"
        else:
            tier = "Low"

        # Strategic recommendations
        if tier == "High":
            rec = (
                "CRITICAL: Initiate high-priority retention outreach. Propose a 1-year contract "
                "with a 15% promotional discount, free TechSupport upgrade, and switch to automatic billing."
            )
        elif tier == "Medium":
            rec = (
                "MODERATE RISK: Offer account check-in with complimentary digital security trial "
                "or bundle loyalty incentive before next billing cycle."
            )
        else:
            rec = (
                "LOW RISK: Customer relationship is healthy and stable. Continue regular service "
                "and include in standard loyalty appreciation program."
            )

        return tier, drivers[:4], rec

    def predict_single(self, customer: CustomerInput) -> ChurnPredictionResponse:
        """Run real-time inference on a single customer payload."""
        cid = customer.customer_id or f"CUST-{uuid.uuid4().hex[:8].upper()}"
        data_dict = customer.model_dump(by_alias=True)
        # Remove customerID from features dataframe
        features_dict = {k: [v] for k, v in data_dict.items() if k not in ["customerID", "customer_id"]}

        df_input = pd.DataFrame(features_dict)
        churn_prob = float(self.pipeline.predict_proba(df_input)[0, 1])
        prediction = 1 if churn_prob >= self.threshold else 0

        risk_tier, drivers, recommendation = self._derive_insights(data_dict, churn_prob)

        return ChurnPredictionResponse(
            customer_id=cid,
            churn_prediction=prediction,
            churn_probability=round(churn_prob, 4),
            risk_tier=risk_tier,
            decision_threshold=round(self.threshold, 4),
            actionable_recommendation=recommendation,
            key_risk_drivers=drivers,
        )

    def predict_batch(self, customers: List[CustomerInput]) -> BatchPredictionResponse:
        """Run batch inference on multiple customer payloads."""
        records = []
        cids = []
        for cust in customers:
            cid = cust.customer_id or f"CUST-{uuid.uuid4().hex[:8].upper()}"
            cids.append(cid)
            d = cust.model_dump(by_alias=True)
            records.append(d)

        # Build feature DataFrame
        df_all = pd.DataFrame(records)
        df_features = df_all.drop(columns=[col for col in ["customerID", "customer_id"] if col in df_all.columns])

        probabilities = self.pipeline.predict_proba(df_features)[:, 1]

        predictions_list = []
        churners_count = 0

        for i, prob in enumerate(probabilities):
            prob_float = float(prob)
            is_churn = 1 if prob_float >= self.threshold else 0
            if is_churn == 1:
                churners_count += 1

            tier, drivers, rec = self._derive_insights(records[i], prob_float)

            predictions_list.append(
                ChurnPredictionResponse(
                    customer_id=cids[i],
                    churn_prediction=is_churn,
                    churn_probability=round(prob_float, 4),
                    risk_tier=tier,
                    decision_threshold=round(self.threshold, 4),
                    actionable_recommendation=rec,
                    key_risk_drivers=drivers,
                )
            )

        total = len(customers)
        churn_rate = round((churners_count / total) * 100.0, 2) if total > 0 else 0.0

        return BatchPredictionResponse(
            total_customers=total,
            predicted_churners=churners_count,
            predicted_churn_rate_percent=churn_rate,
            predictions=predictions_list,
        )

    def get_model_info(self) -> ModelInfoResponse:
        """Return model metadata, CV scores, and business parameters."""
        return ModelInfoResponse(
            champion_model=self.model_name,
            cv_roc_auc=round(self.metadata.get("cv_roc_auc", 0.0), 4),
            test_roc_auc=round(self.metadata.get("test_roc_auc", 0.0), 4),
            test_pr_auc=round(self.metadata.get("test_pr_auc", 0.0), 4),
            optimal_decision_threshold=round(self.threshold, 4),
            business_impact=self.metadata.get("business_impact", {}),
            best_hyperparameters=self.metadata.get("best_hyperparameters", {}),
        )


# Global service accessor
def get_service() -> ChurnPredictionService:
    return ChurnPredictionService()
