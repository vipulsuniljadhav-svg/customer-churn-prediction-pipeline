"""FastAPI REST Service for Customer Churn Prediction."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ChurnPredictionResponse,
    CustomerInput,
    HealthCheckResponse,
    ModelInfoResponse,
)
from app.service import ChurnPredictionService, get_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly load model pipeline on server startup
    try:
        service = get_service()
        print(f"[API] Initialized with champion model: {service.model_name}")
        print(f"[API] Business-optimal decision threshold: {service.threshold:.2f}")
    except Exception as e:
        print(f"[API] Warning during startup: {e}")
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="""
## Production Machine Learning API for Customer Churn Prediction

Predict at-risk subscription customers, estimate churn probabilities, and trigger data-driven retention interventions.

### Key Capabilities:
* **Real-time Inference (`/predict`)**: Low-latency single customer churn scoring.
* **Batch Scoring (`/predict/batch`)**: Bulk predictions for cohort campaigns.
* **Model Insights (`/model/info`)**: Detailed model performance metrics, CV results, and business evaluation.
* **Health Check (`/health`)**: High-availability readiness and liveness probe.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    summary="Root Status",
    tags=["General"],
)
async def root():
    """Root endpoint welcoming users and directing to interactive documentation."""
    return {
        "message": "Welcome to the Customer Churn Prediction REST API",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health",
        "model_info": "/model/info",
    }


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Service Health Check",
    tags=["System"],
)
async def health_check():
    """Verify service status, uptime, and model loading state."""
    try:
        service = get_service()
        return HealthCheckResponse(
            status="healthy",
            model_loaded=True,
            champion_model=service.model_name,
            api_version="1.0.0",
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "champion_model": "None",
                "api_version": "1.0.0",
                "error": str(e),
            },
        )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model Metadata & Performance",
    tags=["Model Information"],
)
async def model_info():
    """Retrieve champion model parameters, cross-validation ROC-AUC, hold-out test metrics, and business economics."""
    try:
        service = get_service()
        return service.get_model_info()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model info: {str(e)}",
        )


@app.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    summary="Predict Customer Churn",
    tags=["Inference"],
)
async def predict_churn(customer: CustomerInput):
    """Predict churn probability and classification for an individual customer profile."""
    try:
        service = get_service()
        result = service.predict_single(customer)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction error: {str(e)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Churn Scoring",
    tags=["Inference"],
)
async def predict_churn_batch(payload: BatchPredictionRequest):
    """Score a batch of customer profiles and return cohort churn analytics."""
    try:
        service = get_service()
        result = service.predict_batch(payload.customers)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch prediction error: {str(e)}",
        )
