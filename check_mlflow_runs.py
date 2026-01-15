"""
Quick script to check MLflow runs and models.
Run this to see what models and runs are tracked.
"""
import sys
from pathlib import Path

try:
    import mlflow
except ImportError:
    print("❌ MLflow not installed. Install with: pip install mlflow")
    sys.exit(1)


def check_mlflow_runs():
    """Check MLflow runs and models."""
    print("=" * 60)
    print("MLflow Models & Runs Status")
    print("=" * 60)
    
    # Check tracking URI
    tracking_uri = mlflow.get_tracking_uri()
    print(f"📊 Tracking URI: {tracking_uri}")
    print()
    
    # List experiments
    try:
        experiments = mlflow.search_experiments()
        print(f"📁 Experiments: {len(experiments)}")
        for exp in experiments:
            print(f"  • {exp.name} (ID: {exp.experiment_id})")
            
            # Get runs for this experiment
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=5)
            if not runs.empty:
                print(f"    Recent runs: {len(runs)}")
                for idx, run in runs.head(3).iterrows():
                    run_name = run.get("tags.mlflow.runName", run.get("run_id", "Unknown"))
                    print(f"      - {run_name}")
                    # Check for models
                    try:
                        artifacts = mlflow.list_artifacts(run_id=run["run_id"])
                        models = [a.path for a in artifacts if "facenet" in a.path.lower() or a.path.endswith(".keras")]
                        if models:
                            print(f"        ✓ Models: {', '.join(models)}")
                    except Exception:
                        pass
            print()
    except Exception as e:
        print(f"⚠️  Error accessing MLflow: {e}")
        print("   Make sure MLflow is running or tracking URI is correct")
        return
    
    # Check for registered models
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        models = client.search_registered_models()
        if models:
            print("🤖 Registered Models:")
            for model in models:
                print(f"  • {model.name}")
                print(f"    Versions: {len(model.latest_versions)}")
                for version in model.latest_versions[:3]:
                    print(f"      - v{version.version} ({version.current_stage})")
        else:
            print("🤖 No registered models yet")
    except Exception as e:
        print(f"⚠️  Could not check registered models: {e}")
    
    print()
    print("=" * 60)
    print("Quick Commands:")
    print("  Start UI:        mlflow ui --port 5000")
    print("  View runs:       mlflow runs list")
    print("  Load model:      mlflow.tensorflow.load_model('runs:/<run-id>/facenet')")
    print("=" * 60)


if __name__ == "__main__":
    check_mlflow_runs()
