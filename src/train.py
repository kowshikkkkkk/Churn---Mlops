import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from preprocess import load_and_preprocess

# ── Load & split data ──────────────────────────────────────
X, y = load_and_preprocess("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── MLflow Experiment ──────────────────────────────────────
mlflow.set_experiment("churn-prediction")

with mlflow.start_run():

    # Parameters
    n_estimators = 100
    max_depth = 5

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc",  auc)

    # Save model
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"✅ Accuracy : {acc:.4f}")
    print(f"✅ F1 Score : {f1:.4f}")
    print(f"✅ ROC-AUC  : {auc:.4f}")
    print("🎯 Model logged to MLflow!")