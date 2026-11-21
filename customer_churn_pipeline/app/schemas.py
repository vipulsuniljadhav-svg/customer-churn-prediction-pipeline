"""Pydantic data schemas for API requests, responses, and validation."""

from typing import List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class CustomerInput(BaseModel):
    """Customer subscription and demographic features for churn scoring."""

    customer_id: Optional[str] = Field(
        default=None,
        alias="customerID",
        description="Unique identifier for the customer (optional)",
        examples=["7590-VHVEG"],
    )
    gender: Literal["Male", "Female"] = Field(
        default="Female",
        description="Customer gender",
        examples=["Female"],
    )
    SeniorCitizen: Literal[0, 1] = Field(
        default=0,
        description="Whether the customer is a senior citizen (1) or not (0)",
        examples=[0],
    )
    Partner: Literal["Yes", "No"] = Field(
        default="No",
        description="Whether the customer has a partner",
        examples=["Yes"],
    )
    Dependents: Literal["Yes", "No"] = Field(
        default="No",
        description="Whether the customer has dependents",
        examples=["No"],
    )
    tenure: int = Field(
        default=1,
        ge=0,
        le=120,
        description="Number of months customer has stayed with the company",
        examples=[1],
    )
    PhoneService: Literal["Yes", "No"] = Field(
        default="Yes",
        description="Whether customer has phone service",
        examples=["Yes"],
    )
    MultipleLines: Literal["No", "Yes", "No phone service"] = Field(
        default="No",
        description="Whether customer has multiple lines",
        examples=["No"],
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        default="DSL",
        description="Customer's internet service provider type",
        examples=["DSL"],
    )
    OnlineSecurity: Literal["No", "Yes", "No internet service"] = Field(
        default="No",
        description="Whether customer has online security",
        examples=["No"],
    )
    OnlineBackup: Literal["No", "Yes", "No internet service"] = Field(
        default="No",
        description="Whether customer has online backup",
        examples=["Yes"],
    )
    DeviceProtection: Literal["No", "Yes", "No internet service"] = Field(
        default="No",
        description="Whether customer has device protection",
        examples=["No"],
    )
    TechSupport: Literal["No", "Yes", "No internet service"] = Field(
        default="No",
        description="Whether customer has technical support",
        examples=["No"],
    )
    StreamingTV: Literal["No", "Yes", "No internet service"] = Field(
        default="No",
        description="Whether customer has streaming TV",
        examples=["No"],
    )
    StreamingMovies: Literal["No", "Yes", "No internet service"] = Field(
        default="No",
        description="Whether customer has streaming movies",
        examples=["No"],
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        default="Month-to-month",
        description="Customer contract duration",
        examples=["Month-to-month"],
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        default="Yes",
        description="Whether customer has paperless billing",
        examples=["Yes"],
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(
        default="Electronic check",
        description="Customer payment method",
        examples=["Electronic check"],
    )
    MonthlyCharges: float = Field(
        default=29.85,
        ge=0.0,
        description="Monthly bill amount charged to customer in USD",
        examples=[29.85],
    )
    TotalCharges: Union[float, str] = Field(
        default=29.85,
        description="Total cumulative amount charged to customer in USD",
        examples=[29.85],
    )

    model_config = ConfigDict(populate_by_name=True)


class ChurnPredictionResponse(BaseModel):
    """Prediction outcome, risk evaluation, and actionable retention recommendations."""

    customer_id: str = Field(description="Customer identifier")
    churn_prediction: int = Field(description="Binary churn prediction: 1 = At Risk, 0 = Retained")
    churn_probability: float = Field(description="Calibrated probability of customer churn (0.0 to 1.0)")
    risk_tier: Literal["Low", "Medium", "High"] = Field(description="Categorical risk classification")
    decision_threshold: float = Field(description="Optimal business threshold applied for classification")
    actionable_recommendation: str = Field(description="Proactive retention strategy based on risk drivers")
    key_risk_drivers: List[str] = Field(description="Leading contributing indicators identified for this account")


class BatchPredictionRequest(BaseModel):
    """Batch scoring request payload."""

    customers: List[CustomerInput] = Field(
        description="List of customer records to score in batch",
        min_length=1,
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction summary and individual customer evaluations."""

    total_customers: int
    predicted_churners: int
    predicted_churn_rate_percent: float
    predictions: List[ChurnPredictionResponse]


class HealthCheckResponse(BaseModel):
    """Health check and model status response."""

    status: str
    model_loaded: bool
    champion_model: str
    api_version: str


class ModelInfoResponse(BaseModel):
    """Champion model metadata, CV scores, and business parameters."""

    champion_model: str
    cv_roc_auc: float
    test_roc_auc: float
    test_pr_auc: float
    optimal_decision_threshold: float
    business_impact: dict
    best_hyperparameters: dict
