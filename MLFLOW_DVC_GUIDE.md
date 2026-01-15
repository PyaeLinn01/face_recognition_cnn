# MLflow & DVC Integration Guide

This guide explains how to use MLflow for model tracking and DVC for dataset versioning in the face recognition system.

## Table of Contents

1. [MLflow Setup](#mlflow-setup)
2. [DVC Setup](#dvc-setup)
3. [Integration with Face Recognition](#integration-with-face-recognition)
4. [Usage Examples](#usage-examples)
5. [Docker Integration](#docker-integration)

---

## MLflow Setup

### Installation

MLflow is already added to `requirements.txt`. Install it:

```bash
pip install -r requirements.txt
```

### Quick Start

**1. Start MLflow Tracking Server (Optional - for remote tracking)**

```bash
# Local file-based tracking (default)
# No server needed - uses ./mlruns directory

# Or start MLflow UI to view experiments
mlflow ui --port 5000
# Then open http://localhost:5000
```

**2. Set Tracking URI (Optional)**

```bash
# For local file storage (default)
export MLFLOW_TRACKING_URI=file:./mlruns

# For remote server
export MLFLOW_TRACKING_URI=http://localhost:5000

# For SQLite backend
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

### Basic Usage in Code

```python
from mlflow_utils import (
    start_verification_run,
    log_mtcnn_params,
    log_facenet_params,
    log_verification_metrics,
    log_facenet_model,
)

# Start a run
with start_verification_run(run_name="test_verification"):
    # Log parameters
    log_mtcnn_params(min_confidence=0.90, margin_ratio=0.2, use_alignment=True)
    log_facenet_params(use_prewhitening=False, use_tta_flip=True)
    
    # Log metrics
    log_verification_metrics(distance=0.65, threshold=0.7, is_match=True, identity="John")
    
    # Log model
    model = get_facenet_model()
    log_facenet_model(model, model_name="facenet_v1")
```

---

## DVC Setup

### Installation

DVC is already added to `requirements.txt`. Install it:

```bash
pip install -r requirements.txt
```

### Initialization

**1. Initialize DVC in your repository**

```bash
# Initialize DVC (creates .dvc directory)
dvc init

# If using Git, commit DVC files
git add .dvc .dvcignore dvc.yaml
git commit -m "Initialize DVC"
```

**2. Configure Remote Storage (Optional)**

**Local Storage:**
```bash
dvc remote add -d local ./dvc_storage
```

**S3 Storage:**
```bash
dvc remote add -d s3 s3://your-bucket-name/dvc-storage
dvc remote modify s3 endpointurl https://s3.amazonaws.com
# Configure credentials via AWS CLI or environment variables
```

**Google Cloud Storage:**
```bash
dvc remote add -d gs gs://your-bucket-name/dvc-storage
```

### Basic Usage

**1. Track Dataset**

```bash
# Track images directory
dvc add images/

# Track datasets directory
dvc add datasets/

# Track weights directory
dvc add weights/

# This creates .dvc files and adds them to .gitignore
```

**2. Commit to Git**

```bash
git add images.dvc datasets.dvc weights.dvc .gitignore
git commit -m "Add datasets to DVC"
```

**3. Push to Remote Storage**

```bash
# Push data to remote storage
dvc push

# Pull data from remote storage
dvc pull
```

**4. Update Dataset**

```bash
# After modifying data
dvc add images/
dvc commit images.dvc
dvc push
```

---

## Integration with Face Recognition

### MLflow Integration in attend_app.py

The `mlflow_utils.py` module provides functions to track:

- **MTCNN Parameters**: Detection confidence, margin ratio, alignment settings
- **FaceNet Parameters**: Prewhitening, TTA flip, model architecture
- **Verification Metrics**: Distance, threshold, match results
- **Model Artifacts**: Saved models, weights, sample images
- **Attendance Metrics**: Total verifications, accuracy, average distance

### Example: Track Verification Run

```python
from mlflow_utils import (
    start_verification_run,
    log_mtcnn_params,
    log_facenet_params,
    log_verification_metrics,
)

# In your verification function
with start_verification_run(run_name="attendance_verification"):
    # Log model parameters
    log_mtcnn_params(
        min_confidence=min_confidence,
        margin_ratio=margin_ratio,
        use_alignment=use_alignment,
    )
    log_facenet_params(
        use_prewhitening=use_prewhitening,
        use_tta_flip=use_tta_flip,
    )
    
    # After verification
    log_verification_metrics(
        distance=best_dist,
        threshold=threshold,
        is_match=is_match,
        identity=best_name,
    )
```

### Example: Track Model Training

```python
from mlflow_utils import (
    start_training_run,
    log_facenet_model,
    log_dataset_info,
)

with start_training_run(run_name="facenet_training"):
    # Log dataset info
    log_dataset_info(
        num_images=len(image_paths),
        num_identities=len(identities),
        dataset_path=Path("images/"),
    )
    
    # Train or load model
    model = get_facenet_model()
    
    # Log model
    log_facenet_model(model, model_name="facenet_attendance")
```

---

## Usage Examples

### Example 1: Track Face Registration

```python
from mlflow_utils import start_verification_run, log_mtcnn_params, log_dataset_info

def register_face_with_tracking(name: str, images: List[Path]):
    with start_verification_run(run_name=f"register_{name}"):
        log_mtcnn_params(min_confidence=0.90, margin_ratio=0.2, use_alignment=True)
        log_dataset_info(num_images=len(images), num_identities=1)
        
        # Your registration logic here
        # ...
```

### Example 2: Track Attendance Session

```python
from mlflow_utils import (
    start_verification_run,
    log_attendance_metrics,
    log_verification_metrics,
)

def track_attendance_session():
    total = 0
    successful = 0
    distances = []
    
    with start_verification_run(run_name="attendance_session"):
        # Your attendance logic
        for verification in verifications:
            total += 1
            if verification.is_match:
                successful += 1
            distances.append(verification.distance)
            log_verification_metrics(
                distance=verification.distance,
                threshold=verification.threshold,
                is_match=verification.is_match,
            )
        
        # Log session summary
        log_attendance_metrics(
            total_verifications=total,
            successful_matches=successful,
            avg_distance=np.mean(distances) if distances else 0.0,
        )
```

### Example 3: Version Dataset with DVC

```bash
# Initial setup
dvc init
dvc remote add -d local ./dvc_storage

# Track datasets
dvc add images/
dvc add datasets/
dvc add weights/

# Commit to Git
git add images.dvc datasets.dvc weights.dvc .gitignore
git commit -m "Add datasets to DVC"

# Push to storage
dvc push

# Later, pull latest version
dvc pull images
```

### Example 4: Reproduce Experiment

```bash
# Run DVC pipeline
dvc repro

# This will:
# 1. Check dataset versions
# 2. Prepare dataset if needed
# 3. Train model
# 4. Evaluate model
# All tracked in MLflow
```

---

## Docker Integration

### Update Dockerfile

Add MLflow and DVC environment variables:

```dockerfile
# MLflow tracking URI
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

# DVC remote (if using)
ENV DVC_REMOTE=local
```

### Run with MLflow UI

```bash
# Start MLflow UI in Docker
docker run -p 5000:5000 \
  -v $(pwd)/mlruns:/app/mlruns \
  face-recognition-cnn \
  mlflow ui --host 0.0.0.0 --port 5000
```

### Run with DVC

```bash
# Mount DVC storage
docker run -p 8501:8501 \
  -v $(pwd)/dvc_storage:/app/dvc_storage \
  -v $(pwd)/.dvc:/app/.dvc \
  face-recognition-cnn
```

---

## Best Practices

### MLflow

1. **Use meaningful run names**: Include date, experiment type, parameters
2. **Log all parameters**: Even if they seem minor, they affect results
3. **Log metrics regularly**: Track performance over time
4. **Tag important runs**: Use `mlflow.set_tag()` for important experiments
5. **Version models**: Register models with `registered_model_name`

### DVC

1. **Track large files**: Use DVC for datasets, models, weights
2. **Use .dvcignore**: Exclude unnecessary files
3. **Regular commits**: Commit DVC files to Git regularly
4. **Backup remotes**: Use cloud storage for important data
5. **Document changes**: Add notes when updating datasets

### Combined Workflow

1. **Version dataset with DVC** before training
2. **Track training with MLflow** including dataset version
3. **Register best models** in MLflow Model Registry
4. **Reproduce experiments** using DVC + MLflow

---

## Troubleshooting

### MLflow Issues

**Problem**: MLflow UI not showing runs
- **Solution**: Check `MLFLOW_TRACKING_URI` matches where runs are saved
- **Solution**: Ensure `mlruns/` directory exists and is writable

**Problem**: Model logging fails
- **Solution**: Ensure model is saved before logging
- **Solution**: Check disk space for artifacts

### DVC Issues

**Problem**: `dvc pull` fails
- **Solution**: Check remote storage configuration
- **Solution**: Verify credentials for cloud storage

**Problem**: Large files not tracked
- **Solution**: Ensure files are in tracked directories
- **Solution**: Check `.dvcignore` doesn't exclude needed files

---

## Next Steps

1. **Set up MLflow tracking** in your verification functions
2. **Initialize DVC** and track your datasets
3. **Create experiments** to compare different parameter settings
4. **Register models** in MLflow Model Registry
5. **Set up CI/CD** to automatically track experiments

For more information:
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
