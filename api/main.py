from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import Literal
import sys
sys.path.append("src")
from predict import load_model, predict_churn

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a telecom customer will churn based on their profile.",
    version="2.0.0"
)

model = load_model()

# ── Input Schema ───────────────────────────────────────────────────────────────
# FIX from v1: Previously accepted raw integers (e.g. InternetService: 1)
# which required callers to know the LabelEncoder mapping internally.
# Now accepts human-readable strings — usable by any frontend or analyst.

class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: float
    TotalCharges: float

    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 845.50
            }
        }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is running!",
        "version": "2.0.0",
        "docs": "/docs",
        "changes": "Now accepts human-readable string inputs instead of encoded integers"
    }

@app.get("/health")
def health():
    """Health check endpoint — useful for monitoring and load balancers."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(customer: CustomerData):
    """
    Predicts churn probability for a single customer.

    Returns:
    - churn_prediction: 0 (will not churn) or 1 (will churn)
    - churn_probability: probability score between 0 and 1
    - interpretation: human-readable label
    """
    try:
        result = predict_churn(model, customer.dict())
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/model-info")
def model_info():
    """Returns information about the currently loaded model."""
    return {
        "model_type": type(model).__name__,
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "class_weight": str(model.class_weight),
        "n_features": model.n_features_in_
    }