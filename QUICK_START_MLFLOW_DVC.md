# Quick Start: MLflow & DVC

## MLflow Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start MLflow UI (Optional)
```bash
mlflow ui --port 5000
# Open http://localhost:5000 in browser
```

### 3. Use in Your Code
```python
from mlflow_utils import start_verification_run, log_verification_metrics

# Track a verification
with start_verification_run(run_name="test"):
    log_verification_metrics(
        distance=0.65,
        threshold=0.7,
        is_match=True,
        identity="John"
    )
```

### 4. View Results
- Open MLflow UI: http://localhost:5000
- Or check `./mlruns/` directory

---

## DVC Quick Start (5 minutes)

### 1. Initialize DVC
```bash
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

### 2. Track Your Dataset
```bash
# Track images directory
dvc add images/

# Track datasets
dvc add datasets/

# Commit to Git
git add images.dvc datasets.dvc .gitignore
git commit -m "Add datasets to DVC"
```

### 3. Push to Storage (Optional)
```bash
# Configure local storage
dvc remote add -d local ./dvc_storage

# Push data
dvc push
```

### 4. Pull Data
```bash
# Pull latest version
dvc pull images
```

---

## Integration with attend_app.py

### Add MLflow Tracking to Verification

```python
# At the top of attend_app.py
from mlflow_utils import (
    start_verification_run,
    log_mtcnn_params,
    log_facenet_params,
    log_verification_metrics,
)

# In _attendance_ui() function, wrap verification:
with start_verification_run(run_name="attendance_verification"):
    # Log parameters
    log_mtcnn_params(
        min_confidence=min_confidence,
        margin_ratio=margin_ratio,
        use_alignment=use_alignment,
    )
    log_facenet_params(
        use_prewhitening=use_prewhitening,
        use_tta_flip=use_tta_flip,
    )
    
    # Your existing verification code...
    # After getting best_dist and best_name:
    
    log_verification_metrics(
        distance=best_dist,
        threshold=threshold,
        is_match=is_match,
        identity=best_name,
    )
```

---

## Common Commands

### MLflow
```bash
# Start UI
mlflow ui --port 5000

# List experiments
mlflow experiments list

# View specific run
mlflow ui --experiment-id 0
```

### DVC
```bash
# Track file/directory
dvc add <path>

# Push to remote
dvc push

# Pull from remote
dvc pull

# Show status
dvc status

# Show data pipeline
dvc dag
```

---

## Next Steps

1. Read `MLFLOW_DVC_GUIDE.md` for detailed documentation
2. Integrate MLflow into your verification functions
3. Set up DVC for your datasets
4. Configure remote storage (S3, GCS, etc.)
