import joblib
import pandas as pd

def load_model():
    model = joblib.load("models/model.pkl")
    return model

def load_feature_columns():
    """
    Loads the exact column list saved during training.

    FIX from v1: Previously had a hardcoded list of 19 integer-encoded columns.
    After switching to One-Hot Encoding, we have more columns with different names.
    Loading from the saved schema ensures inference always matches training exactly.
    """
    return joblib.load("models/feature_columns.pkl")

def predict_churn(model, data: dict):
    """
    Runs inference on a single customer record.

    data: dict with raw string values e.g. {"InternetService": "Fiber optic", ...}
    The API layer is responsible for passing already-encoded data here,
    OR we build a DataFrame and use pd.get_dummies to align columns.
    """
    # Build single-row DataFrame
    df = pd.DataFrame([data])

    # One-Hot Encode the input the same way training did
    df = pd.get_dummies(df, drop_first=True)

    # Load expected columns from training
    feature_columns = load_feature_columns()

    # Align: add missing columns as 0, remove extra columns, fix order
    # This is critical — model will break if column order doesn't match training
    df = df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(df)
    probability = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": round(float(probability), 4),
        "interpretation": "Will churn" if prediction[0] == 1 else "Will not churn"
    }