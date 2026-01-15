"""
Example: How to integrate MLflow tracking into attend_app.py

This file shows how to add MLflow tracking to your face recognition system.
Copy relevant parts into attend_app.py or app.py.
"""
from mlflow_utils import (
    start_verification_run,
    log_mtcnn_params,
    log_facenet_params,
    log_verification_metrics,
    log_attendance_metrics,
    log_dataset_info,
)

# Example 1: Track face registration
def register_face_with_mlflow(name: str, num_images: int):
    """Register face with MLflow tracking."""
    with start_verification_run(run_name=f"register_{name}"):
        log_mtcnn_params(
            min_confidence=0.90,
            margin_ratio=0.20,
            use_alignment=True,
        )
        log_facenet_params(
            use_prewhitening=False,
            use_tta_flip=True,
        )
        log_dataset_info(
            num_images=num_images,
            num_identities=1,
        )
        # Your registration logic here
        print(f"Registered {name} with {num_images} images")


# Example 2: Track verification with MLflow
def verify_face_with_mlflow(
    query_emb,
    database,
    threshold: float,
    min_confidence: float,
    margin_ratio: float,
    use_alignment: bool,
    use_prewhitening: bool,
    use_tta_flip: bool,
):
    """Verify face and track with MLflow."""
    with start_verification_run(run_name="face_verification"):
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
        
        # Find best match
        best_name = None
        best_dist = float("inf")
        for identity, ref_emb in database.items():
            dist = float(np.linalg.norm(query_emb - ref_emb))
            if dist < best_dist:
                best_dist = dist
                best_name = identity
        
        # Log metrics
        is_match = best_dist < threshold
        log_verification_metrics(
            distance=best_dist,
            threshold=threshold,
            is_match=is_match,
            identity=best_name,
        )
        
        return best_name, best_dist, is_match


# Example 3: Track attendance session
def track_attendance_session(verifications: list):
    """Track entire attendance session with MLflow."""
    with start_verification_run(run_name="attendance_session"):
        total = len(verifications)
        successful = sum(1 for v in verifications if v.get("is_match", False))
        distances = [v.get("distance", 0.0) for v in verifications]
        
        # Log session metrics
        log_attendance_metrics(
            total_verifications=total,
            successful_matches=successful,
            avg_distance=np.mean(distances) if distances else 0.0,
        )
        
        # Log individual verifications
        for v in verifications:
            log_verification_metrics(
                distance=v.get("distance", 0.0),
                threshold=v.get("threshold", 0.7),
                is_match=v.get("is_match", False),
                identity=v.get("identity"),
            )


# Example 4: Add to attend_app.py verification function
"""
# In _attendance_ui() function, add MLflow tracking:

from mlflow_utils import start_verification_run, log_verification_metrics

# Before verification
with start_verification_run(run_name="attendance_verification"):
    # Your existing verification code...
    
    # After getting best_name and best_dist
    log_verification_metrics(
        distance=best_dist,
        threshold=threshold,
        is_match=is_match,
        identity=best_name,
    )
"""
