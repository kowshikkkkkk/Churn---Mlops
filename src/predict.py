import joblib
import pandas as pd

# Exact column order from training
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

def load_model():
    model = joblib.load("models/model.pkl")
    return model

def predict_churn(model, data: dict):
    df = pd.DataFrame([data])
    df = df[FEATURE_COLUMNS]  # reorder columns to match training
    prediction = model.predict(df)
    probability = model.predict_proba(df)[0][1]
    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": round(float(probability), 4)
    }