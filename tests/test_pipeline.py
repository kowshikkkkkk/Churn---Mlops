"""
Unit Tests for Churn MLOps Pipeline
=====================================
Run with: pytest tests/test_pipeline.py -v

What we're testing:
- Preprocessing produces correct shape and no nulls
- Saved feature columns match actual training columns
- Model loads and produces valid predictions
- Prediction output is properly bounded
- Drift detection returns a boolean without crashing
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
import joblib

sys.path.append("src")

from preprocess import load_and_preprocess

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_PATH    = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH   = "models/model.pkl"
COLUMNS_PATH = "models/feature_columns.pkl"


# ── Test 1: Preprocessing Shape ────────────────────────────────────────────────
def test_preprocessing_shape():
    """
    After preprocessing, we expect:
    - 30 feature columns (One-Hot Encoded)
    - More than 7000 rows (dataset has ~7043, a few dropped for null TotalCharges)
    - Target variable has exactly 2 unique values (0 and 1)
    """
    X, y = load_and_preprocess(DATA_PATH)

    assert X.shape[1] == 30, (
        f"Expected 30 features after One-Hot Encoding, got {X.shape[1]}"
    )
    assert X.shape[0] > 7000, (
        f"Expected more than 7000 rows, got {X.shape[0]}"
    )
    assert len(y.unique()) == 2, (
        f"Target should have 2 classes (0 and 1), got {y.unique()}"
    )
    print(f"\n✅ Shape test passed: {X.shape[0]} rows, {X.shape[1]} features")


# ── Test 2: No Nulls After Preprocessing ──────────────────────────────────────
def test_no_nulls_after_preprocessing():
    """
    After preprocessing, there should be zero null values.
    TotalCharges has spaces in raw data that become NaN — we drop those rows.
    """
    X, y = load_and_preprocess(DATA_PATH)

    null_count_X = X.isnull().sum().sum()
    null_count_y = y.isnull().sum()

    assert null_count_X == 0, (
        f"Found {null_count_X} null values in features"
    )
    assert null_count_y == 0, (
        f"Found {null_count_y} null values in target"
    )
    print(f"\n✅ No nulls test passed")


# ── Test 3: Feature Columns Match Saved Schema ────────────────────────────────
def test_feature_columns_match_schema():
    """
    The saved feature_columns.pkl must match the actual columns produced
    by preprocessing. If these don't match, inference will silently break.

    This test catches the case where someone changes preprocessing
    without regenerating the saved schema.
    """
    assert os.path.exists(COLUMNS_PATH), (
        f"feature_columns.pkl not found at {COLUMNS_PATH}. Run train.py first."
    )

    X, _ = load_and_preprocess(DATA_PATH)
    saved_columns = joblib.load(COLUMNS_PATH)

    assert list(X.columns) == saved_columns, (
        f"Mismatch between actual columns and saved schema.\n"
        f"Actual: {list(X.columns)}\n"
        f"Saved : {saved_columns}"
    )
    print(f"\n✅ Column schema test passed: {len(saved_columns)} columns match")


# ── Test 4: Model Prediction Output ───────────────────────────────────────────
def test_model_prediction_output():
    """
    Model must:
    - Load without errors
    - Return prediction of 0 or 1
    - Return probability between 0.0 and 1.0
    - Not crash on valid input
    """
    assert os.path.exists(MODEL_PATH), (
        f"model.pkl not found at {MODEL_PATH}. Run train.py first."
    )
    assert os.path.exists(COLUMNS_PATH), (
        f"feature_columns.pkl not found. Run train.py first."
    )

    model          = joblib.load(MODEL_PATH)
    feature_cols   = joblib.load(COLUMNS_PATH)

    # Create a dummy input row with all zeros (valid after reindex)
    dummy_input = pd.DataFrame(
        [np.zeros(len(feature_cols))],
        columns=feature_cols
    )

    prediction  = model.predict(dummy_input)[0]
    probability = model.predict_proba(dummy_input)[0][1]

    assert prediction in [0, 1], (
        f"Prediction should be 0 or 1, got {prediction}"
    )
    assert 0.0 <= probability <= 1.0, (
        f"Probability should be between 0 and 1, got {probability}"
    )
    print(f"\n✅ Prediction test passed: pred={prediction}, prob={probability:.4f}")


# ── Test 5: Class Imbalance Is Handled ────────────────────────────────────────
def test_class_imbalance_ratio():
    """
    Verifies the dataset is imbalanced (churn rate between 20-35%).
    This documents the known issue and confirms class_weight='balanced' is needed.
    If churn rate were 50%, we wouldn't need class weights.
    """
    _, y = load_and_preprocess(DATA_PATH)

    churn_rate = y.mean()

    assert 0.20 <= churn_rate <= 0.35, (
        f"Expected churn rate between 20-35%, got {churn_rate:.2%}"
    )
    print(f"\n✅ Class imbalance test passed: churn rate = {churn_rate:.2%}")


# ── Test 6: Drift Detection Returns Boolean ───────────────────────────────────
def test_drift_detection_returns_bool():
    """
    run_drift_report() must:
    - Complete without crashing
    - Return a boolean (True = drift detected, False = no drift)
    - Save the HTML report

    Note: We're testing the function CONTRACT, not the drift result itself
    (since drift is simulated and will always be True in current implementation)
    """
    from monitor import run_drift_report

    result = run_drift_report()

    assert isinstance(result, bool), (
        f"run_drift_report() should return a boolean, got {type(result)}"
    )
    assert os.path.exists("reports/drift_report.html"), (
        "Drift report HTML was not saved"
    )
    print(f"\n✅ Drift detection test passed: drift_detected={result}")