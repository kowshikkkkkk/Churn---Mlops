import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from preprocess import load_and_preprocess
import os
mlflow.set_tracking_uri("sqlite:///" + os.path.join(os.getcwd(), "mlflow.db"))

def train_model(data_path: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """
    Trains RandomForest on churn data and logs everything to MLflow.

    Changes from v1:
    - Added class_weight='balanced' to handle class imbalance
      Reason: ~74% non-churn vs ~26% churn — model was biased toward majority class
              giving misleading 79% accuracy. Balanced weights penalize missing churners more.
    - Fixed roc_auc_score: now uses predict_proba instead of predict
      Reason: roc_auc_score on binary predictions (0/1) is less meaningful than
              on probabilities — it loses the threshold information.
    - Added feature importance logging to MLflow
      Reason: Shows which features drive churn — valuable for business stakeholders
              and makes MLflow usage actually meaningful.
    - Wrapped in a function so retrain.py can call it cleanly
      Reason: Previously, code ran at module level — importing train.py would
              trigger training immediately (side effect bug).
    - save_artifacts=True so feature_columns.pkl is saved during training
    """

    # ── Load & split ───────────────────────────────────────────────────────
    X, y = load_and_preprocess(data_path, save_artifacts=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # stratify preserves class ratio
    )

    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Churn rate train : {y_train.mean():.2%}")
    print(f"Churn rate test  : {y_test.mean():.2%}")

    # ── MLflow Experiment ──────────────────────────────────────────────────
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():

        # Parameters
        n_estimators = 100
        max_depth    = 5
        class_weight = "balanced"

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth",    max_depth)
        mlflow.log_param("class_weight", class_weight)
        mlflow.log_param("stratify",     True)

        # ── Train ──────────────────────────────────────────────────────────
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,   # FIX: handles imbalanced classes
            random_state=42
        )
        model.fit(X_train, y_train)

        # ── Evaluate ───────────────────────────────────────────────────────
        preds      = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)[:, 1]   # FIX: use probabilities for AUC

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, pred_proba)          # FIX: probabilities not binary preds

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc",  auc)

        # ── Feature Importance ─────────────────────────────────────────────
        # FIX: log feature importances — was missing in v1
        feature_importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        # Log top 10 features as individual MLflow metrics
        for feature, importance in feature_importance.head(10).items():
            mlflow.log_metric(f"importance_{feature}", round(importance, 4))

        # Save full feature importance as CSV artifact
        fi_path = os.path.join(os.getcwd(), "models", "feature_importance.csv")
        feature_importance.to_csv(fi_path, header=["importance"])
        mlflow.log_artifact(fi_path)

        print(f"\n📊 Top 5 features driving churn:")
        for feat, imp in feature_importance.head(5).items():
            print(f"   {feat:<35} {imp:.4f}")

        # ── Save model ─────────────────────────────────────────────────────
        mlflow.sklearn.log_model(model, "random_forest_model")
        joblib.dump(model, "models/model.pkl")

        print(f"\n✅ Accuracy : {acc:.4f}")
        print(f"✅ F1 Score : {f1:.4f}")
        print(f"✅ ROC-AUC  : {auc:.4f}")
        print("✅ Model saved to models/model.pkl")
        print("🎯 Model logged to MLflow!")

    return model


def retrain_model(data_path: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """
    Triggered by retrain.py when drift is detected.
    Logs to a separate MLflow experiment to distinguish from baseline runs.

    NOTE: In production, data_path should point to NEW incoming data, not the
    original CSV. Retraining on the same data that caused drift is ineffective.
    """
    print("🔄 Retraining model with latest data...")

    X, y = load_and_preprocess(data_path, save_artifacts=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("churn-prediction-retrain")

    with mlflow.start_run():
        n_estimators = 100
        max_depth    = 5

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth",    max_depth)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("trigger",      "drift_detected")

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42
        )
        model.fit(X_train, y_train)

        preds      = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, pred_proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc",  auc)

        # Feature importance
        feature_importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        for feature, importance in feature_importance.head(10).items():
            mlflow.log_metric(f"importance_{feature}", round(importance, 4))

        mlflow.sklearn.log_model(model, "random_forest_model")
        joblib.dump(model, "models/model.pkl")

        print(f"✅ Retrained — Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        print("✅ New model saved to models/model.pkl")

    return model


if __name__ == "__main__":
    train_model()