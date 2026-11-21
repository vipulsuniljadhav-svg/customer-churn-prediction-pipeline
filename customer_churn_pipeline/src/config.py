"""Configuration module for the Customer Churn Prediction Pipeline."""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_churn_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

# Column definitions
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

# Raw numerical features
RAW_NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Engineered features
ENGINEERED_NUMERICAL_FEATURES = [
    "avg_monthly_spend",
    "monthly_spend_ratio",
    "total_services"
]
ENGINEERED_CATEGORICAL_FEATURES = [
    "tenure_group"
]
ENGINEERED_BINARY_FEATURES = [
    "is_month_to_month",
    "has_support_security"
]

# All numerical features to be standardized
NUMERICAL_FEATURES = RAW_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES

# All categorical features to be One-Hot Encoded
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group"
]

# Binary features passed through as numeric indicators (0 or 1)
BINARY_FEATURES = ENGINEERED_BINARY_FEATURES

# Service columns used to count total subscribed services
SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

# Tenure bins and labels
TENURE_BINS = [-1, 12, 24, 48, 60, 100]
TENURE_LABELS = ["0-12m", "13-24m", "25-48m", "49-60m", "60+m"]

# Split and CV parameters
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

# Business Impact Parameters
# CLV: Value saved if an at-risk customer is retained
CUSTOMER_LIFETIME_VALUE = 720.0  # Approx. $60/month * 12 months
# Retention campaign cost per customer reached (offer/discount/outreach)
RETENTION_OFFER_COST = 50.0
# Probability that an at-risk customer accepts the retention offer and stays
RETENTION_SUCCESS_RATE = 0.60
