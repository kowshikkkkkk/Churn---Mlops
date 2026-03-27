import sys
sys.path.append("src")
from monitor import run_drift_report
from train import retrain_model

def retrain_pipeline():
    print("🔍 Checking for data drift...")
    drift_detected = run_drift_report()

    if drift_detected:
        print("⚠️  Drift detected — triggering retraining...")
        retrain_model()
        print("✅ Retraining complete — new model is live!")
    else:
        print("✅ No drift detected — model is healthy, no retraining needed!")

if __name__ == "__main__":
    retrain_pipeline()