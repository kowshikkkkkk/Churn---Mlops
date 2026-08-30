"""
Evaluation Script — Model Performance Gates for CD
===================================================

This script:
1. Loads the trained model
2. Evaluates against test set
3. Compares vs. baseline (previous version)
4. BLOCKS deployment if metrics regress
5. Outputs scores for GitHub Actions

Run: python scripts/evaluate_model.py
"""

import os
import sys
import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

sys.path.append("src")
from preprocess import load_and_preprocess


# ── Constants ──────────────────────────────────────────────────────────────────
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "models/metrics.json"
BASELINE_METRICS_PATH = "models/baseline_metrics.json"


def load_model_and_data():
    """Load trained model and test data."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    
    model = joblib.load(MODEL_PATH)
    X, y = load_and_preprocess(DATA_PATH)
    
    # Split into train/test (same split as training)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
    }
    
    return metrics, y_pred, y_pred_proba


def load_baseline():
    """Load baseline metrics from previous run."""
    if os.path.exists(BASELINE_METRICS_PATH):
        with open(BASELINE_METRICS_PATH, "r") as f:
            return json.load(f)
    return None


def save_metrics(metrics, is_baseline=False):
    """Save metrics to disk."""
    path = BASELINE_METRICS_PATH if is_baseline else METRICS_PATH
    
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Metrics saved to {path}")


def check_performance_gates(current, baseline, tolerance=0.05):
    """
    Check if current metrics pass gates vs. baseline.
    
    Rules:
    1. F1 score must be >= 0.60 (absolute gate)
    2. F1 score must not regress more than 5% vs. baseline
    3. ROC-AUC must not regress more than 3% vs. baseline
    
    Returns:
        (passed: bool, message: str)
    """
    passed = True
    messages = []
    
    # Gate 1: Absolute F1 threshold
    if current["f1_score"] < 0.60:
        passed = False
        messages.append(
            f"❌ GATE FAILED: F1 score {current['f1_score']:.4f} < 0.60 (minimum)"
        )
    else:
        messages.append(
            f"✅ Gate 1 passed: F1 score {current['f1_score']:.4f} >= 0.60"
        )
    
    # Gate 2: F1 regression check
    if baseline:
        f1_regression = baseline["f1_score"] - current["f1_score"]
        if f1_regression > tolerance:
            passed = False
            messages.append(
                f"❌ GATE FAILED: F1 regressed by {f1_regression:.4f} "
                f"({baseline['f1_score']:.4f} → {current['f1_score']:.4f})"
            )
        else:
            messages.append(
                f"✅ Gate 2 passed: F1 regression {f1_regression:.4f} within {tolerance*100:.0f}% tolerance"
            )
        
        # Gate 3: ROC-AUC regression check
        auc_regression = baseline["roc_auc"] - current["roc_auc"]
        if auc_regression > 0.03:
            passed = False
            messages.append(
                f"❌ GATE FAILED: ROC-AUC regressed by {auc_regression:.4f} "
                f"({baseline['roc_auc']:.4f} → {current['roc_auc']:.4f})"
            )
        else:
            messages.append(
                f"✅ Gate 3 passed: ROC-AUC regression {auc_regression:.4f} within 3% tolerance"
            )
    else:
        messages.append("ℹ️  First run detected — no baseline to compare against")
    
    return passed, "\n".join(messages)


def print_detailed_report(metrics, y_test, y_pred):
    """Print detailed evaluation report."""
    print("\n" + "="*70)
    print("MODEL EVALUATION REPORT")
    print("="*70)
    
    print("\n📊 METRICS")
    print("-" * 70)
    for metric, value in metrics.items():
        print(f"  {metric.upper():.<20} {value:.4f}")
    
    print("\n📈 CONFUSION MATRIX")
    print("-" * 70)
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives:  {cm[0, 0]:>6}")
    print(f"  False Positives: {cm[0, 1]:>6}")
    print(f"  False Negatives: {cm[1, 0]:>6}")
    print(f"  True Positives:  {cm[1, 1]:>6}")
    
    print("\n📋 CLASSIFICATION REPORT")
    print("-" * 70)
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    
    print("\n" + "="*70)


def main():
    """Main evaluation pipeline."""
    print("🔍 Starting Model Evaluation...\n")
    
    # Step 1: Load model and data
    print("Step 1: Loading model and test data...")
    model, X_test, y_test = load_model_and_data()
    print(f"  ✅ Loaded model with {model.n_features_in_} features")
    print(f"  ✅ Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Step 2: Evaluate
    print("\nStep 2: Evaluating model performance...")
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Step 3: Load baseline
    print("\nStep 3: Checking against baseline metrics...")
    baseline = load_baseline()
    
    # Step 4: Check gates
    print("\nStep 4: Performance gates check...")
    gates_passed, gate_messages = check_performance_gates(metrics, baseline)
    print(gate_messages)
    
    # Step 5: Save metrics
    print("\nStep 5: Saving metrics...")
    save_metrics(metrics, is_baseline=False)
    if not baseline:
        save_metrics(metrics, is_baseline=True)
        print("  ℹ️  First baseline established")
    
    # Step 6: Print report
    print_detailed_report(metrics, y_test, y_pred)
    
    # Step 7: Exit with appropriate code
    if gates_passed:
        print("\n✅ All performance gates PASSED! Ready for deployment.")
        return 0
    else:
        print("\n❌ Performance gates FAILED! Blocking deployment.")
        print("   Revert changes or improve model before deploying.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)