import pandas as pd
import joblib
import os

def load_and_preprocess(filepath: str, save_artifacts: bool = False):
    """
    Loads, cleans, and encodes the Telco churn dataset.

    Changes from v1:
    - Replaced LabelEncoder with pd.get_dummies() (One-Hot Encoding)
      Reason: LabelEncoder injected fake ordinal relationships into
      nominal categorical columns like InternetService, Contract, etc.
    - Saves column schema to models/feature_columns.pkl when save_artifacts=True
      Reason: Inference must use exact same columns in exact same order as training.
    - Encodes target column (Churn) explicitly as binary int
      Reason: get_dummies would otherwise also encode the target.
    """

    # ── Load ───────────────────────────────────────────────────────────────
    df = pd.read_csv(filepath)

    # ── Fix TotalCharges (stored as string in raw CSV) ─────────────────────
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    # ── Drop customerID (not a predictive feature) ─────────────────────────
    df.drop(columns=["customerID"], inplace=True)

    # ── Encode target variable explicitly ──────────────────────────────────
    # Must do this BEFORE get_dummies so Churn isn't one-hot encoded
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # ── One-Hot Encode all categorical columns ─────────────────────────────
    # drop_first=True removes one redundant column per feature
    # e.g. if is_DSL=0 and is_FiberOptic=0, we know it's "No" — no need for a 3rd column
    # This avoids multicollinearity (dummy variable trap)
    df = pd.get_dummies(df, drop_first=True)

    # ── Split features and target ──────────────────────────────────────────
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # ── Save column schema for inference ──────────────────────────────────
    # Critical: inference must produce columns in the exact same order
    if save_artifacts:
        os.makedirs("models", exist_ok=True)
        joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
        print(f"✅ Feature columns saved to models/feature_columns.pkl")
        print(f"   Total features after One-Hot Encoding: {len(X.columns)}")

    return X, y


if __name__ == "__main__":
    X, y = load_and_preprocess(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        save_artifacts=True
    )
    print(f"\n✅ Features shape : {X.shape}")
    print(f"✅ Target shape   : {y.shape}")
    print(f"\nFeature columns:\n{X.columns.tolist()}")
    print(f"\nClass distribution:\n{y.value_counts()}")
    print(f"Churn rate: {y.mean():.2%}")