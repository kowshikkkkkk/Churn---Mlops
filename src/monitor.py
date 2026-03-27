import pandas as pd
import sys
sys.path.append("src")
from evidently import Report
from evidently.presets import DataDriftPreset
from preprocess import load_and_preprocess

def run_drift_report():
    # Load original training data as reference
    X_ref, _ = load_and_preprocess("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Simulate new incoming data with drift
    X_current = X_ref.copy()
    X_current["MonthlyCharges"] = X_current["MonthlyCharges"] * 1.3
    X_current["tenure"] = X_current["tenure"] * 0.7
    X_current["TotalCharges"] = X_current["TotalCharges"] * 1.2

    # Build and run report — result is returned, not stored
    report = Report([DataDriftPreset()])
    my_eval = report.run(X_current, X_ref)

    # Save as HTML
    my_eval.save_html("reports/drift_report.html")
    print("✅ Drift report saved to reports/drift_report.html")

    # Check drift using correct method for v0.7.21
    try:
        result = my_eval.load_dict()
        drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    except Exception:
        print("⚠️  Could not parse drift result — assuming drift detected")
        drift_detected = True

    if drift_detected:
        print("⚠️  DRIFT DETECTED!")
    else:
        print("✅ No drift detected!")

    return drift_detected