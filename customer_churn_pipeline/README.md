# End-to-End Customer Churn Prediction & Retention ML Pipeline

A production-grade, supervised machine learning pipeline and REST API that predicts subscription customer churn, identifies high-risk churn drivers, and provides automated, revenue-optimizing retention recommendations. Built with **Python 3.13**, **Scikit-learn**, and **FastAPI**.

---

## 📌 Executive Summary & Architecture

Customer churn is among the costliest business challenges for subscription services (telecommunications, SaaS, streaming). Standard models relying solely on raw accuracy or default 0.50 classification thresholds fail because **false negatives (lost customers) are substantially more expensive than false positives (cost of retention outreach)**.

This pipeline delivers:
1. **Robust Preprocessing & Feature Engineering**: Handles missing values, zero tenure edge cases, and constructs business-relevant interaction features within a zero-leakage Scikit-learn `Pipeline`.
2. **Multi-Model Benchmark & Hyperparameter Tuning**: 5-Fold Stratified Cross-Validation across **Logistic Regression**, **Random Forest**, and **Gradient Boosting**.
3. **Business-Driven Decision Optimization**: Evaluates expected revenue curves across probability thresholds to identify the financially optimal cutoff ($t_{\text{opt}} = 0.25$) maximizing Net Savings.
4. **Production REST API**: Low-latency FastAPI microservice supporting real-time single-customer scoring, batch cohort processing, risk tiering, and actionable interventions.

```
┌────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│   Raw Telco Dataset    │ ───► │  Feature Engineering    │ ───► │ 5-Fold Stratified CV    │
│  (7,043 subscriptions) │      │  • Tenure cohorts       │      │ • Logistic Regression   │
└────────────────────────┘      │  • Spend ratio          │      │ • Random Forest (Best)  │
                                │  • Service bundles      │      │ • Gradient Boosting     │
                                └─────────────────────────┘      └────────────┬────────────┘
                                                                              │
┌────────────────────────┐      ┌─────────────────────────┐                   │
│   FastAPI Service      │ ◄─── │  Serialized Artifacts   │ ◄─────────────────┘
│  • POST /predict       │      │  • best_churn_model     │
│  • POST /predict/batch │      │  • model_metadata.json  │
│  • GET  /health        │      │  • Evaluation Curves    │
│  • GET  /model/info    │      └─────────────────────────┘
└────────────────────────┘
```

---

## 🔬 Feature Engineering

Six domain-specific features were engineered to capture customer commitment, pricing pressure, and switching friction:

| Engineered Feature | Definition / Formula | Business Rationale |
| :--- | :--- | :--- |
| `tenure_group` | Binned tenure cohorts: `0-12m`, `13-24m`, `25-48m`, `49-60m`, `60+m` | Customers in the first year exhibit significantly higher churn volatility before habituation. |
| `avg_monthly_spend` | $\text{TotalCharges} / \max(\text{tenure}, 1)$ | Captures lifetime average spend per month. |
| `monthly_spend_ratio` | $\text{MonthlyCharges} / \max(\text{avg\_monthly\_spend}, 0.01)$ | Identifies bill shock or recent price increases relative to historical spending. |
| `total_services` | Count of active services (0 to 9) across Phone, Internet, Security, Backup, TechSupport, TV, Movies | Bundled accounts incur high switching friction; churn inversely correlates with bundle count. |
| `is_month_to_month` | Indicator if `Contract == 'Month-to-month'` | Month-to-month contracts have zero lock-in and represent the primary churn driver. |
| `has_support_security`| Indicator if `OnlineSecurity == 'Yes'` OR `TechSupport == 'Yes'` | Support and security add-ons foster sticky customer relationships. |

Numerical features are scaled using `StandardScaler` and categorical attributes are one-hot encoded using `OneHotEncoder(handle_unknown='ignore')` inside a unified `ColumnTransformer`.

---

## 📊 Model Training & Hyperparameter Tuning

All candidate pipelines were trained on an 80% stratified training split ($N = 5,634$) and tuned using 5-Fold Stratified Cross-Validation (`GridSearchCV`) optimizing for `roc_auc`:

1. **Logistic Regression (Class-Balanced)**:
   - Tuned: $C \in [0.05, 0.2, 1.0, 5.0]$, solvers: `liblinear`, `lbfgs`.
   - Best parameters: `{'C': 0.05, 'solver': 'liblinear'}`
   - CV ROC-AUC: **0.8461**
2. **Random Forest Classifier (Class-Balanced)**:
   - Tuned: `n_estimators` $\in [100, 200]$, `max_depth` $\in [6, 10, \text{None}]$, `min_samples_split` $\in [2, 5]$.
   - Best parameters: `{'max_depth': 6, 'min_samples_split': 2, 'n_estimators': 100}`
   - CV ROC-AUC: **0.8473**
3. **Gradient Boosting Classifier**:
   - Tuned: `n_estimators` $\in [100, 150]$, `learning_rate` $\in [0.03, 0.1]$, `max_depth` $\in [3, 4]$.
   - Best parameters: `{'learning_rate': 0.03, 'max_depth': 3, 'n_estimators': 150}`
   - CV ROC-AUC: **0.8488**

---

## 📈 Evaluation & Comparative Results (Hold-Out Test Set, N = 1,409)

| Model Name | Test ROC-AUC | Test PR-AUC | Default F1 ($t=0.50$) | Default Recall | Optimal Cutoff ($t_{\text{opt}}$) | Optimal Recall | Optimal Net Savings | Financial Lift Over Default |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **0.8421** | **0.6487** | **0.6280** | **81.02%** | **0.25** | **96.26%** | **\$110,320** | **+\$8,974** |
| **Logistic Regression** | 0.8416 | 0.6325 | 0.6123 | 78.34% | 0.29 | 93.58% | \$109,950 | +\$12,524 |
| **Gradient Boosting** | 0.8456 | 0.6599 | 0.5823 | 51.07% | 0.12 | 94.65% | \$110,278 | +\$41,866 |

### Financial Impact Model

* **Customer Lifetime Value (CLV)**: \$720.00 (estimated \$60/month $\times$ 12 months retained).
* **Retention Campaign Cost**: \$50.00 per customer contacted.
* **Retention Success Rate**: 60% of targeted churners accept the offer and stay.
* **Expected Value Equation**:
  $$\text{Net Savings} = \text{TP} \times (\text{CLV} \times \text{RetentionRate}) - (\text{TP} + \text{FP}) \times \text{Cost}$$
* **Decision Threshold Shift**: By lowering the decision cutoff from the arbitrary 0.50 threshold down to the business-optimal **0.25**, the model captures **96.26% of all churners** (up from 81.02%), generating **\$110,320 in net savings** on the test cohort.

---

## 🚀 REST API Reference

The deployed microservice exposes OpenAPI interactive documentation at `http://localhost:8000/docs`.

### 1. Health Probe
```http
GET /health
```
**Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "champion_model": "Random Forest",
  "api_version": "1.0.0"
}
```

### 2. Single Customer Prediction
```http
POST /predict
Content-Type: application/json
```
**Request Body:**
```json
{
  "customerID": "7590-VHVEG",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
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
  "TotalCharges": 29.85
}
```
**Response (200 OK):**
```json
{
  "customer_id": "7590-VHVEG",
  "churn_prediction": 1,
  "churn_probability": 0.6974,
  "risk_tier": "High",
  "decision_threshold": 0.25,
  "actionable_recommendation": "CRITICAL: Initiate high-priority retention outreach. Propose a 1-year contract with a 15% promotional discount, free TechSupport upgrade, and switch to automatic billing.",
  "key_risk_drivers": [
    "Month-to-month contract (flexible cancellation)",
    "Short tenure / Early customer lifecycle (< 1 year)",
    "Manual Electronic check payment method",
    "No Online Security add-on protection"
  ]
}
```

### 3. Batch Scoring
```http
POST /predict/batch
```
Accepts an array `{"customers": [...]}` and returns summarized cohort metrics with individual predictions.

---

## 🛠️ Installation & Execution Guide

### Local Environment Setup
```bash
# Clone and enter project
cd customer_churn_pipeline

# Install dependencies
pip install -r requirements.txt

# Run full pipeline (training, tuning, evaluation, figures generation)
python run_pipeline.py

# Run automated tests
pytest tests/ -v

# Start FastAPI server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Docker Container Deployment
```bash
# Build Docker image
docker build -t customer-churn-api:latest .

# Run container
docker run -d -p 8000:8000 --name churn-service customer-churn-api:latest

# Probe health
curl http://localhost:8000/health
```
