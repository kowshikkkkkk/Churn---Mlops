import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath: str):
    # Load
    df = pd.read_csv(filepath)

    # Fix TotalCharges (it's a string in the raw data)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    # Drop customer ID (not useful for ML)
    df.drop(columns=["customerID"], inplace=True)

    # Encode all categorical columns to numbers
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    # Split features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y

if __name__ == "__main__":
    X, y = load_and_preprocess("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(f"✅ Features shape: {X.shape}")
    print(f"✅ Target shape: {y.shape}")
    print(f"\nSample data:\n{X.head()}")