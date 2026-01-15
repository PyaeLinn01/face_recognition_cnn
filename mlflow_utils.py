"""
MLflow utilities for tracking MTCNN and FaceNet models.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf


# MLflow tracking URI (can be set via environment variable)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def init_mlflow_experiment(experiment_name: str = "face_recognition") -> str:
    """Initialize or get MLflow experiment."""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        return experiment_id
    except Exception as e:
        print(f"Warning: Could not set up MLflow experiment: {e}")
        return "0"


def log_mtcnn_params(
    min_confidence: float,
    margin_ratio: float,
    use_alignment: bool,
    image_size: int = 96,
) -> None:
    """Log MTCNN detection parameters to MLflow."""
    mlflow.log_params({
        "mtcnn_min_confidence": min_confidence,
        "mtcnn_margin_ratio": margin_ratio,
        "mtcnn_use_alignment": use_alignment,
        "mtcnn_image_size": image_size,
        "detector_type": "MTCNN",
    })


def log_facenet_params(
    use_prewhitening: bool,
    use_tta_flip: bool,
    input_shape: tuple = (96, 96, 3),
    embedding_size: int = 128,
) -> None:
    """Log FaceNet model parameters to MLflow."""
    mlflow.log_params({
        "facenet_use_prewhitening": use_prewhitening,
        "facenet_use_tta_flip": use_tta_flip,
        "facenet_input_shape": str(input_shape),
        "facenet_embedding_size": embedding_size,
        "model_type": "FaceNet",
    })


def log_verification_metrics(
    distance: float,
    threshold: float,
    is_match: bool,
    identity: Optional[str] = None,
) -> None:
    """Log face verification metrics."""
    mlflow.log_metrics({
        "verification_distance": distance,
        "verification_threshold": threshold,
        "verification_match": 1.0 if is_match else 0.0,
    })
    if identity:
        mlflow.log_param("verified_identity", identity)


def log_model_artifacts(
    model_path: Optional[Path] = None,
    weights_dir: Optional[Path] = None,
    images_dir: Optional[Path] = None,
) -> None:
    """Log model artifacts to MLflow."""
    if model_path and model_path.exists():
        mlflow.log_artifact(str(model_path), "models")
    
    if weights_dir and weights_dir.exists():
        mlflow.log_artifacts(str(weights_dir), "weights")
    
    if images_dir and images_dir.exists():
        # Log sample images (first 5 per identity)
        sample_count = 0
        for img_path in sorted(images_dir.rglob("*.jpg"))[:20]:  # Limit to 20 samples
            mlflow.log_artifact(str(img_path), "sample_images")
            sample_count += 1
        mlflow.log_param("sample_images_logged", sample_count)


def log_facenet_model(model: tf.keras.Model, model_name: str = "facenet") -> None:
    """Log FaceNet TensorFlow model to MLflow."""
    try:
        mlflow.tensorflow.log_model(
            model,
            artifact_path=model_name,
            registered_model_name=model_name,
        )
        print(f"✓ Logged {model_name} model to MLflow")
    except Exception as e:
        print(f"Warning: Could not log model to MLflow: {e}")


def start_verification_run(
    run_name: Optional[str] = None,
    experiment_name: str = "face_recognition",
) -> mlflow.ActiveRun:
    """Start a new MLflow run for face verification."""
    experiment_id = init_mlflow_experiment(experiment_name)
    
    if run_name is None:
        run_name = f"verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return mlflow.start_run(run_name=run_name, experiment_id=experiment_id)


def start_training_run(
    run_name: Optional[str] = None,
    experiment_name: str = "face_recognition",
) -> mlflow.ActiveRun:
    """Start a new MLflow run for model training."""
    experiment_id = init_mlflow_experiment(experiment_name)
    
    if run_name is None:
        run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return mlflow.start_run(run_name=run_name, experiment_id=experiment_id)


def log_dataset_info(
    num_images: int,
    num_identities: int,
    dataset_path: Optional[Path] = None,
) -> None:
    """Log dataset information."""
    mlflow.log_params({
        "dataset_num_images": num_images,
        "dataset_num_identities": num_identities,
    })
    if dataset_path:
        mlflow.log_param("dataset_path", str(dataset_path))


def log_attendance_metrics(
    total_verifications: int,
    successful_matches: int,
    avg_distance: float,
) -> None:
    """Log attendance system metrics."""
    accuracy = successful_matches / total_verifications if total_verifications > 0 else 0.0
    mlflow.log_metrics({
        "attendance_total_verifications": total_verifications,
        "attendance_successful_matches": successful_matches,
        "attendance_accuracy": accuracy,
        "attendance_avg_distance": avg_distance,
    })
