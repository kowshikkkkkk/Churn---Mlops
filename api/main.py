from fastapi import FastAPI
from pydantic import BaseModel
import sys
sys.path.append("src")
from predict import load_model, predict_churn

app = FastAPI()
model = load_model()

class CustomerData(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: int
    Partner: int
    Dependents: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running!"}

@app.post("/predict")
def predict(customer: CustomerData):
    result = predict_churn(model, customer.dict())
    return result