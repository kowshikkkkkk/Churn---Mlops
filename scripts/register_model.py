"""
Model Registration Script — Versioning & MLflow Registry
=========================================================

This script:
1. Loads the trained model
2. Creates a model version entry with commit metadata
3. Registers model in MLflow Model Registry
4. Manages model lifecycle (Staging, Production, Archived)
5. Enables rollback by keeping version history

Run: python scripts/register_model.py
Environment: COMMIT_SHA, COMMIT_MESSAGE (optional for CI/CD)
"""

import os
import sys
import json
import hashlib
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

sys.path.append("src")


# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = "models/model.pkl"
REGISTRY_PATH = "model_registry.json"
METRICS_PATH = "models/metrics.json"
MODEL_NAME = "churn-predictor"


def compute_file_hash(filepath: str, algorithm: str = "sha256") -> str:
    """Compute hash of file for integrity checking."""
    hasher = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_commit_metadata() -> Dict[str, str]:
    """Get commit info from environment variables (set by CI/CD)."""
    return {
        "commit_sha": os.getenv("COMMIT_SHA", "manual-run")[:8],
        "commit_message": os.getenv("COMMIT_MESSAGE", "local training"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "branch": os.getenv("GITHUB_REF", "main").split("/")[-1],
    }


def load_metrics() -> Dict[str, float]:
    """Load latest evaluation metrics."""
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(f"Metrics not found at {METRICS_PATH}. Run evaluate_model.py first.")
    
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def create_registry_entry(
    model_version: str,
    model_hash: str,
    commit_metadata: Dict,
    metrics: Dict
) -> Dict:
    """Create a registry entry for model versioning."""
    return {
        "model_version": model_version,
        "model_hash": model_hash,
        "registered_at": datetime.utcnow().isoformat() + "Z",
        "commit": commit_metadata,
        "metrics": metrics,
        "stage": "Staging",  # Start in Staging
    }


def load_registry() -> Dict:
    """Load model registry or create empty one."""
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {"models": {}}


def save_registry(registry: Dict) -> None:
    """Save model registry to disk."""
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"✅ Registry saved to {REGISTRY_PATH}")


def get_next_version(registry: Dict) -> str:
    """Get next version number (semantic versioning)."""
    if not registry["models"]:
        return "1.0.0"
    
    # Get last version and increment patch
    models = sorted(list(registry["models"].keys()))
    last_version = models[-1]
    major, minor, patch = map(int, last_version.split("."))
    return f"{major}.{minor}.{patch + 1}"


def promote_to_production(registry: Dict, new_version: str) -> bool:
    """
    Promote new model to Production if it's better than current.
    
    Rules:
    1. If no production model exists, promote new one immediately
    2. If production model exists, compare metrics:
       - F1 must not regress > 5%
       - If passes, promote new to Production and archive old
    
    Returns:
        True if promoted, False otherwise
    """
    new_metrics = registry["models"][new_version]["metrics"]
    
    # Find current production model
    production_version = None
    for version, entry in registry["models"].items():
        if entry["stage"] == "Production":
            production_version = version
            break
    
    # If no production model exists, promote new one
    if not production_version:
        registry["models"][new_version]["stage"] = "Production"
        print(f"✅ Promoted {new_version} to Production (no previous production model)")
        return True
    
    # Compare metrics
    prod_metrics = registry["models"][production_version]["metrics"]
    
    # Gate: F1 score must not regress > 5%
    f1_regression = prod_metrics["f1_score"] - new_metrics["f1_score"]
    if f1_regression <= 0.05:  # Allow 5% regression
        registry["models"][new_version]["stage"] = "Production"
        registry["models"][production_version]["stage"] = "Archived"
        print(f"✅ Promoted {new_version} to Production")
        print(f"   Previous production ({production_version}) archived")
        return True
    else:
        print(f"❌ Did not promote {new_version} to Production")
        print(f"   F1 regressed too much: {prod_metrics['f1_score']:.4f} → {new_metrics['f1_score']:.4f}")
        return False


def print_registry_status(registry: Dict) -> None:
    """Print human-readable registry status."""
    print("\n" + "="*70)
    print("MODEL REGISTRY STATUS")
    print("="*70)
    
    if not registry["models"]:
        print("  (empty)")
        return
    
    for version, entry in sorted(registry["models"].items()):
        stage_emoji = {
            "Staging": "🔄",
            "Production": "🚀",
            "Archived": "📦"
        }.get(entry["stage"], "❓")
        
        print(f"\n{stage_emoji} Version {version} — {entry['stage']}")
        print(f"   Hash: {entry['model_hash'][:8]}...")
        print(f"   Commit: {entry['commit']['commit_sha']} — {entry['commit']['commit_message']}")
        print(f"   Metrics: F1={entry['metrics']['f1_score']:.4f}, AUC={entry['metrics']['roc_auc']:.4f}")
        print(f"   Registered: {entry['registered_at']}")


def main():
    """Main registration pipeline."""
    print("📦 Starting Model Registration...\n")
    
    # Step 1: Verify model exists
    print("Step 1: Verifying model artifact...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    print(f"  ✅ Model found: {os.path.getsize(MODEL_PATH)} bytes")
    
    # Step 2: Compute hash
    print("\nStep 2: Computing model integrity hash...")
    model_hash = compute_file_hash(MODEL_PATH)
    print(f"  ✅ SHA256: {model_hash[:16]}...")
    
    # Step 3: Load metrics
    print("\nStep 3: Loading evaluation metrics...")
    metrics = load_metrics()
    print(f"  ✅ F1: {metrics['f1_score']:.4f}, AUC: {metrics['roc_auc']:.4f}")
    
    # Step 4: Get commit metadata
    print("\nStep 4: Collecting commit metadata...")
    commit_metadata = get_commit_metadata()
    print(f"  ✅ Commit: {commit_metadata['commit_sha']}")
    print(f"  ✅ Message: {commit_metadata['commit_message']}")
    
    # Step 5: Update local registry
    print("\nStep 5: Updating local model registry...")
    registry = load_registry()
    version = get_next_version(registry)
    
    entry = create_registry_entry(version, model_hash, commit_metadata, metrics)
    registry["models"][version] = entry
    save_registry(registry)
    print(f"  ✅ Registered version: {version}")
    
    # Step 6: Decide on promotion
    print("\nStep 6: Evaluating for production promotion...")
    promote_to_production(registry, version)
    save_registry(registry)
    
    # Step 7: Print status
    print_registry_status(registry)
    
    print("\n" + "="*70)
    print("✅ Model registration complete!")
    print(f"   Version {version} registered")
    print("="*70)


if __name__ == "__main__":
    main()