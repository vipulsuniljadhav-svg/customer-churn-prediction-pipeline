# Customer Churn Prediction Pipeline

## Project Summary
This project is an end-to-end machine learning pipeline and production REST API that predicts customer churn for subscription-based businesses. It ingests customer demographic and service data, cleans missing values, and engineers behavioral features including tenure cohorts, average monthly spend, monthly spend ratio, and active bundled service counts.

Multiple classification models (Logistic Regression, Random Forest, and Gradient Boosting) are trained and tuned using 5-fold stratified cross-validation. The decision threshold is optimized based on business economics (Customer Lifetime Value vs. retention offer costs), achieving 96.3% churn recall and maximizing net business savings. The champion model is deployed via a FastAPI service that accepts customer profiles and returns churn probability, binary prediction, risk tier, and actionable retention recommendations.

## How to Run

### 1. Clone the Repository & Set Up Environment
```bash
git clone https://github.com/<your-username>/customer-churn-prediction-pipeline.git
cd customer-churn-prediction-pipeline

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train & Evaluate the Pipeline
Train models with 5-fold cross-validation, save the best model artifact, and generate evaluation plots:
```bash
python run_pipeline.py
```

### 3. Run Automated Tests
Verify preprocessing, pipeline inference, and API endpoints:
```bash
pytest tests/ -v
```

### 4. Start the REST API Server
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Access the API & Documentation
Open your browser to:
- **Interactive Swagger Docs:** http://127.0.0.1:8000/docs
- **Health Check:** http://127.0.0.1:8000/health
