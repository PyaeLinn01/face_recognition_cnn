"""
MLflow Tracking for Face Recognition Models (FaceNet & MTCNN)

This module provides experiment tracking for:
- FaceNet model: Face embedding generation and verification
- MTCNN: Face detection

Run this script to log model experiments to MLflow:
    python mlflowexp.py

View results:
    mlflow ui --port 5000

Then open http://localhost:5000 in your browser.
"""

import mlflow
import mlflow.keras
import mlflow.tensorflow
import numpy as np
import cv2
import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import backend as K

# MongoDB imports
try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("Warning: pymongo not available. Install with: pip install pymongo")

# Local imports
from fr_utils import load_weights_from_FaceNet, load_dataset
from inception_blocks_v2 import faceRecoModel

# Set MLflow tracking URI (local directory)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# MongoDB Configuration
MONGODB_CONNECTION_STRING = os.getenv(
    "MONGODB_CONNECTION_STRING", 
    "mongodb://localhost:27017/"
)
MONGODB_DATABASE_NAME = os.getenv(
    "MONGODB_DATABASE_NAME",
    "face_attendance"
)

# Constants
TARGET_SIZE = (96, 96)
EMBEDDING_SIZE = 128


# ==================== MongoDB Helper Functions ====================

def get_mongodb_client():
    """Get MongoDB client."""
    if not MONGODB_AVAILABLE:
        raise RuntimeError("MongoDB not available - install pymongo")
    return MongoClient(MONGODB_CONNECTION_STRING)


def get_mongodb_collection(collection_name: str):
    """Get MongoDB collection."""
    client = get_mongodb_client()
    db = client[MONGODB_DATABASE_NAME]
    return db[collection_name]


def load_faces_from_mongodb() -> Dict[str, List[np.ndarray]]:
    """
    Load face images from MongoDB.
    
    Returns:
        Dictionary mapping user names to list of face images (as numpy arrays)
    """
    if not MONGODB_AVAILABLE:
        print("MongoDB not available")
        return {}
    
    try:
        collection = get_mongodb_collection("face_images")
        images_by_name = {}
        
        for doc in collection.find().sort("name", 1).sort("image_index", 1):
            name = doc.get("name")
            img_bytes = doc.get("image_bytes", b"")
            
            if not name or not img_bytes:
                continue
                
            if name not in images_by_name:
                images_by_name[name] = []
            
            # Decode image bytes to numpy array
            try:
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    images_by_name[name].append(img_bgr)
            except Exception as e:
                print(f"Error decoding image for {name}: {e}")
                continue
        
        print(f"Loaded {sum(len(v) for v in images_by_name.values())} images for {len(images_by_name)} identities from MongoDB")
        return images_by_name
        
    except Exception as e:
        print(f"Error loading faces from MongoDB: {e}")
        return {}


def load_users_from_mongodb() -> List[Dict]:
    """Load user data from MongoDB."""
    if not MONGODB_AVAILABLE:
        return []
    
    try:
        collection = get_mongodb_collection("users")
        users = list(collection.find({}, {"password": 0}))  # Exclude password
        return users
    except Exception as e:
        print(f"Error loading users from MongoDB: {e}")
        return []


def load_attendance_from_mongodb() -> List[Dict]:
    """Load attendance records from MongoDB."""
    if not MONGODB_AVAILABLE:
        return []
    
    try:
        collection = get_mongodb_collection("attendance")
        records = list(collection.find().sort("timestamp", -1).limit(1000))
        return records
    except Exception as e:
        print(f"Error loading attendance from MongoDB: {e}")
        return []


def get_mongodb_stats() -> Dict:
    """Get MongoDB collection statistics."""
    stats = {}
    try:
        client = get_mongodb_client()
        db = client[MONGODB_DATABASE_NAME]
        
        stats["face_images_count"] = db["face_images"].count_documents({})
        stats["users_count"] = db["users"].count_documents({})
        stats["attendance_count"] = db["attendance"].count_documents({})
        
        # Count unique identities in face_images
        pipeline = [{"$group": {"_id": "$name"}}, {"$count": "unique_names"}]
        result = list(db["face_images"].aggregate(pipeline))
        stats["unique_identities"] = result[0]["unique_names"] if result else 0
        
    except Exception as e:
        print(f"Error getting MongoDB stats: {e}")
    
    return stats


def get_model_summary_stats(model) -> Dict:
    """Extract model architecture statistics."""
    total_params = model.count_params()
    trainable_params = sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    # Count layers by type
    layer_counts = {}
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "total_layers": len(model.layers),
        "layer_types": layer_counts
    }


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Triplet loss function for face verification.
    
    Arguments:
        y_true: Not used, but required by Keras
        y_pred: List containing anchor, positive, and negative embeddings
        alpha: Margin for triplet loss (default 0.2)
    
    Returns:
        Loss value
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Compute distance between anchor and positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    
    # Compute distance between anchor and negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    
    # Compute triplet loss
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss


def compute_embedding_quality_metrics_from_images(model, images_by_name: Dict[str, List[np.ndarray]]) -> Dict:
    """
    Compute quality metrics for embeddings generated by the model.
    Uses images loaded from MongoDB.
    
    Metrics include:
    - Mean embedding norm
    - Embedding variance
    - Inference time
    """
    embeddings = []
    inference_times = []
    
    for name, images in images_by_name.items():
        for img_bgr in images[:5]:  # Limit to 5 images per person
            start_time = time.time()
            try:
                # Convert BGR to RGB and normalize
                img_rgb = img_bgr[..., ::-1].astype(np.float32) / 255.0
                
                # Ensure correct size
                if img_rgb.shape[:2] != (96, 96):
                    img_rgb = cv2.resize(img_rgb, (96, 96), interpolation=cv2.INTER_AREA)
                
                # Get embedding
                x = np.array([img_rgb])
                emb = model.predict_on_batch(x)[0]
                emb = emb / max(np.linalg.norm(emb), 1e-12)
                
                inference_times.append(time.time() - start_time)
                embeddings.append(emb)
            except Exception as e:
                print(f"Error processing image for {name}: {e}")
                continue
    
    if not embeddings:
        return {}
    
    embeddings = np.array(embeddings)
    
    metrics = {
        "mean_embedding_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
        "std_embedding_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
        "mean_inference_time_ms": float(np.mean(inference_times) * 1000),
        "min_inference_time_ms": float(np.min(inference_times) * 1000),
        "max_inference_time_ms": float(np.max(inference_times) * 1000),
        "embedding_dimension": EMBEDDING_SIZE,
        "num_test_images": len(embeddings)
    }
    
    return metrics


def compute_verification_metrics_from_mongodb(
    model, 
    database: Dict[str, np.ndarray], 
    images_by_name: Dict[str, List[np.ndarray]],
    threshold: float = 0.5
) -> Dict:
    """
    Compute verification metrics (accuracy, precision, recall, F1) using MongoDB data.
    
    Args:
        model: FaceNet model
        database: Dictionary of name -> embedding (reference embeddings)
        images_by_name: Dictionary of name -> list of face images
        threshold: Distance threshold for verification
    
    Returns:
        Dictionary of metrics
    """
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    distances = []
    
    db_names = list(database.keys())
    if len(db_names) < 2:
        return {}
    
    # Test each image against its own identity (positive) and a different identity (negative)
    for name, images in images_by_name.items():
        if name not in database:
            continue
            
        for img_bgr in images[:3]:  # Test up to 3 images per person
            try:
                # Convert and get embedding
                img_rgb = img_bgr[..., ::-1].astype(np.float32) / 255.0
                if img_rgb.shape[:2] != (96, 96):
                    img_rgb = cv2.resize(img_rgb, (96, 96), interpolation=cv2.INTER_AREA)
                
                x = np.array([img_rgb])
                emb = model.predict_on_batch(x)[0]
                emb = emb / max(np.linalg.norm(emb), 1e-12)
                query_emb = np.expand_dims(emb, axis=0)
                
                # Test positive pair (same identity)
                ref_emb = database[name]
                dist = float(np.linalg.norm(query_emb - ref_emb))
                distances.append(dist)
                predicted_same = dist < threshold
                
                if predicted_same:
                    true_positives += 1
                else:
                    false_negatives += 1
                
                # Test negative pair (different identity)
                other_names = [n for n in db_names if n != name]
                if other_names:
                    other_name = other_names[0]
                    other_ref_emb = database[other_name]
                    dist_other = float(np.linalg.norm(query_emb - other_ref_emb))
                    predicted_same_other = dist_other < threshold
                    
                    if not predicted_same_other:
                        true_negatives += 1
                    else:
                        false_positives += 1
                        
            except Exception as e:
                print(f"Error in verification for {name}: {e}")
                continue
    
    total = true_positives + true_negatives + false_positives + false_negatives
    
    if total == 0:
        return {}
    
    accuracy = (true_positives + true_negatives) / total
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    
    return {
        "verification_accuracy": float(accuracy),
        "verification_precision": float(precision),
        "verification_recall": float(recall),
        "verification_f1": float(f1),
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "mean_distance": float(np.mean(distances)) if distances else 0.0,
        "std_distance": float(np.std(distances)) if distances else 0.0,
        "threshold_used": threshold
    }


def compute_intra_inter_class_distances(database: Dict[str, np.ndarray]) -> Dict:
    """
    Compute intra-class and inter-class distance statistics.
    
    This helps evaluate embedding quality:
    - Low intra-class distance: Same person's embeddings are close
    - High inter-class distance: Different people's embeddings are far apart
    """
    names = list(database.keys())
    embeddings = [database[name].flatten() for name in names]
    
    if len(embeddings) < 2:
        return {}
    
    # Compute pairwise distances
    inter_distances = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            inter_distances.append(dist)
    
    return {
        "mean_inter_class_distance": float(np.mean(inter_distances)),
        "std_inter_class_distance": float(np.std(inter_distances)),
        "min_inter_class_distance": float(np.min(inter_distances)),
        "max_inter_class_distance": float(np.max(inter_distances)),
        "num_identities": len(names)
    }


def log_facenet_experiment(run_name: str = None):
    """
    Log FaceNet model experiment to MLflow.
    
    This logs:
    - Model parameters (architecture, input shape, embedding size)
    - Training configuration
    - Performance metrics from MongoDB data
    - Model artifacts
    """
    print("=" * 60)
    print("FaceNet MLflow Experiment")
    print("=" * 60)
    
    # Set experiment
    experiment_name = "FaceNet_Face_Recognition"
    mlflow.set_experiment(experiment_name)
    
    if run_name is None:
        run_name = f"facenet_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        print(f"Starting MLflow run: {run_name}")
        
        # ============ Load Model ============
        print("\n[1/6] Loading FaceNet model...")
        K.set_image_data_format("channels_last")
        model = faceRecoModel(input_shape=(96, 96, 3))
        load_weights_from_FaceNet(model)
        model.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
        print(f"Total Parameters: {model.count_params():,}")
        
        # ============ Log Parameters ============
        print("\n[2/6] Logging model parameters...")
        
        # Model architecture parameters
        mlflow.log_param("model_name", "FaceNet")
        mlflow.log_param("model_type", "Inception-based CNN")
        mlflow.log_param("input_shape", "(96, 96, 3)")
        mlflow.log_param("embedding_size", EMBEDDING_SIZE)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("loss_function", "triplet_loss")
        mlflow.log_param("triplet_margin_alpha", 0.2)
        mlflow.log_param("verification_threshold", 0.5)
        mlflow.log_param("channels_format", "channels_last")
        mlflow.log_param("data_source", "MongoDB")
        mlflow.log_param("mongodb_database", MONGODB_DATABASE_NAME)
        
        # Get model summary stats
        model_stats = get_model_summary_stats(model)
        mlflow.log_param("total_params", model_stats["total_params"])
        mlflow.log_param("trainable_params", model_stats["trainable_params"])
        mlflow.log_param("non_trainable_params", model_stats["non_trainable_params"])
        mlflow.log_param("total_layers", model_stats["total_layers"])
        
        # Log layer type counts
        for layer_type, count in model_stats["layer_types"].items():
            mlflow.log_param(f"layer_count_{layer_type}", count)
        
        # ============ Log MongoDB Stats ============
        print("\n[3/6] Fetching MongoDB statistics...")
        mongo_stats = get_mongodb_stats()
        for key, value in mongo_stats.items():
            mlflow.log_metric(f"mongodb_{key}", value)
            print(f"  {key}: {value}")
        
        # ============ Load Data from MongoDB ============
        print("\n[4/6] Loading face images from MongoDB...")
        images_by_name = load_faces_from_mongodb()
        
        if not images_by_name:
            print("  No face images found in MongoDB!")
            mlflow.log_param("registered_identities", 0)
            mlflow.log_param("total_images", 0)
        else:
            total_images = sum(len(imgs) for imgs in images_by_name.values())
            mlflow.log_param("registered_identities", len(images_by_name))
            mlflow.log_param("total_images", total_images)
            print(f"  Loaded {total_images} images for {len(images_by_name)} identities")
        
        # ============ Build Embedding Database ============
        print("\n[5/6] Building embedding database...")
        database = {}
        
        for name, images in images_by_name.items():
            per_identity_embs = []
            for img_bgr in images:
                try:
                    # Convert BGR to RGB and normalize
                    img_rgb = img_bgr[..., ::-1].astype(np.float32) / 255.0
                    
                    # Ensure correct size
                    if img_rgb.shape[:2] != (96, 96):
                        img_rgb = cv2.resize(img_rgb, (96, 96), interpolation=cv2.INTER_AREA)
                    
                    # Get embedding
                    x = np.array([img_rgb])
                    emb = model.predict_on_batch(x)[0]
                    emb = emb / max(np.linalg.norm(emb), 1e-12)
                    per_identity_embs.append(np.expand_dims(emb, axis=0))
                except Exception as e:
                    print(f"  Error processing image for {name}: {e}")
                    continue
            
            if per_identity_embs:
                # Average embeddings for this identity
                mean_emb = np.mean(np.concatenate(per_identity_embs, axis=0), axis=0)
                mean_emb = mean_emb / max(np.linalg.norm(mean_emb), 1e-12)
                database[name] = np.expand_dims(mean_emb, axis=0)
                print(f"  Built embedding for: {name} ({len(per_identity_embs)} images)")
        
        # ============ Compute Metrics ============
        print("\n[6/6] Computing performance metrics...")
        
        # Embedding quality metrics
        if images_by_name:
            embedding_metrics = compute_embedding_quality_metrics_from_images(model, images_by_name)
            for key, value in embedding_metrics.items():
                mlflow.log_metric(key, value)
                print(f"  {key}: {value:.4f}")
        
        # Inter-class distance metrics
        if len(database) >= 2:
            distance_metrics = compute_intra_inter_class_distances(database)
            for key, value in distance_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
                    print(f"  {key}: {value:.4f}")
        
        # Verification metrics
        if len(database) >= 2 and images_by_name:
            verification_metrics = compute_verification_metrics_from_mongodb(
                model, database, images_by_name, threshold=0.5
            )
            for key, value in verification_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
                    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # ============ Log Artifacts ============
        print("\n[7/7] Logging model artifacts...")
        
        # Save model summary to file
        summary_file = "model_summary.txt"
        with open(summary_file, "w") as f:
            f.write("FaceNet Model Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Data Source: MongoDB ({MONGODB_DATABASE_NAME})\n\n")
            model.summary(print_fn=lambda x: f.write(x + "\n"))
            f.write(f"\n\nDatabase Identities: {list(database.keys())}\n")
        mlflow.log_artifact(summary_file)
        os.remove(summary_file)
        
        # Save database info
        db_info_file = "database_info.json"
        db_info = {
            "data_source": "MongoDB",
            "mongodb_database": MONGODB_DATABASE_NAME,
            "identities": list(database.keys()),
            "count": len(database),
            "embedding_size": EMBEDDING_SIZE,
            "images_per_identity": {name: len(imgs) for name, imgs in images_by_name.items()},
            "timestamp": datetime.now().isoformat()
        }
        with open(db_info_file, "w") as f:
            json.dump(db_info, f, indent=2)
        mlflow.log_artifact(db_info_file)
        os.remove(db_info_file)
        
        # Save MongoDB stats
        mongo_stats_file = "mongodb_stats.json"
        with open(mongo_stats_file, "w") as f:
            json.dump(mongo_stats, f, indent=2)
        mlflow.log_artifact(mongo_stats_file)
        os.remove(mongo_stats_file)
        
        print("\n" + "=" * 60)
        print("FaceNet experiment logged successfully!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print("=" * 60)
        
        return mlflow.active_run().info.run_id


def log_mtcnn_experiment(run_name: str = None):
    """
    Log MTCNN face detection experiment to MLflow.
    
    This logs:
    - Detection parameters
    - Performance metrics (speed, accuracy)
    - Detection statistics
    """
    print("\n" + "=" * 60)
    print("MTCNN MLflow Experiment")
    print("=" * 60)
    
    # Import MTCNN
    try:
        from mtcnn.mtcnn import MTCNN
    except ImportError:
        print("MTCNN not available. Install with: pip install mtcnn")
        return None
    
    # Set experiment
    experiment_name = "MTCNN_Face_Detection"
    mlflow.set_experiment(experiment_name)
    
    if run_name is None:
        run_name = f"mtcnn_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        print(f"Starting MLflow run: {run_name}")
        
        # ============ Initialize Detector ============
        print("\n[1/4] Initializing MTCNN detector...")
        
        # MTCNN parameters (use defaults, log them for reference)
        min_face_size = 20
        scale_factor = 0.709
        confidence_thresholds = [0.6, 0.7, 0.7]  # P-Net, R-Net, O-Net
        
        # Initialize MTCNN with default settings (API varies by version)
        detector = MTCNN()
        print("MTCNN detector initialized")
        
        # ============ Log Parameters ============
        print("\n[2/4] Logging MTCNN parameters...")
        
        mlflow.log_param("model_name", "MTCNN")
        mlflow.log_param("model_type", "Multi-task Cascaded CNN")
        mlflow.log_param("stages", "P-Net, R-Net, O-Net")
        mlflow.log_param("min_face_size", min_face_size)
        mlflow.log_param("scale_factor", scale_factor)
        mlflow.log_param("pnet_threshold", confidence_thresholds[0])
        mlflow.log_param("rnet_threshold", confidence_thresholds[1])
        mlflow.log_param("onet_threshold", confidence_thresholds[2])
        mlflow.log_param("output_type", "bounding_box + keypoints")
        mlflow.log_param("keypoints", "left_eye, right_eye, nose, mouth_left, mouth_right")
        mlflow.log_param("data_source", "MongoDB")
        
        # ============ Run Detection Tests on MongoDB Images ============
        print("\n[3/4] Running face detection tests on MongoDB images...")
        
        # Load images from MongoDB
        images_by_name = load_faces_from_mongodb()
        detection_results = []
        inference_times = []
        total_faces_detected = 0
        
        if images_by_name:
            for name, images in images_by_name.items():
                for idx, img_bgr in enumerate(images[:5]):  # Test up to 5 images per person
                    try:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        
                        start_time = time.time()
                        faces = detector.detect_faces(img_rgb)
                        inference_time = time.time() - start_time
                        
                        inference_times.append(inference_time)
                        total_faces_detected += len(faces)
                        
                        # Get confidence scores
                        confidences = [f.get("confidence", 0) for f in faces]
                        
                        detection_results.append({
                            "identity": name,
                            "image_index": idx,
                            "faces_detected": len(faces),
                            "inference_time_ms": inference_time * 1000,
                            "confidences": confidences,
                            "image_size": img_bgr.shape[:2]
                        })
                        
                        print(f"  {name}[{idx}]: {len(faces)} faces, {inference_time*1000:.1f}ms")
                        
                    except Exception as e:
                        print(f"  Error processing {name}[{idx}]: {e}")
                        continue
        else:
            print("  No images found in MongoDB for MTCNN testing")
        
        # ============ Log Metrics ============
        print("\n[4/4] Logging detection metrics...")
        
        if inference_times:
            mlflow.log_metric("mean_inference_time_ms", float(np.mean(inference_times) * 1000))
            mlflow.log_metric("min_inference_time_ms", float(np.min(inference_times) * 1000))
            mlflow.log_metric("max_inference_time_ms", float(np.max(inference_times) * 1000))
            mlflow.log_metric("std_inference_time_ms", float(np.std(inference_times) * 1000))
            
            print(f"  Mean inference time: {np.mean(inference_times)*1000:.2f}ms")
            print(f"  Min inference time: {np.min(inference_times)*1000:.2f}ms")
            print(f"  Max inference time: {np.max(inference_times)*1000:.2f}ms")
        
        if detection_results:
            faces_per_image = [r["faces_detected"] for r in detection_results]
            all_confidences = [c for r in detection_results for c in r["confidences"]]
            
            mlflow.log_metric("total_images_processed", len(detection_results))
            mlflow.log_metric("total_faces_detected", total_faces_detected)
            mlflow.log_metric("mean_faces_per_image", float(np.mean(faces_per_image)))
            mlflow.log_metric("detection_rate", float(sum(1 for f in faces_per_image if f > 0) / len(faces_per_image)))
            
            if all_confidences:
                mlflow.log_metric("mean_confidence", float(np.mean(all_confidences)))
                mlflow.log_metric("min_confidence", float(np.min(all_confidences)))
                mlflow.log_metric("max_confidence", float(np.max(all_confidences)))
            
            print(f"  Total images: {len(detection_results)}")
            print(f"  Total faces: {total_faces_detected}")
            print(f"  Detection rate: {sum(1 for f in faces_per_image if f > 0) / len(faces_per_image):.2%}")
        
        # Save detection results
        results_file = "detection_results.json"
        with open(results_file, "w") as f:
            json.dump(detection_results, f, indent=2, default=str)
        mlflow.log_artifact(results_file)
        os.remove(results_file)
        
        print("\n" + "=" * 60)
        print("MTCNN experiment logged successfully!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print("=" * 60)
        
        return mlflow.active_run().info.run_id


def log_combined_pipeline_experiment(run_name: str = None):
    """
    Log combined MTCNN + FaceNet pipeline experiment.
    
    This tests the full face recognition pipeline:
    1. Face detection (MTCNN)
    2. Face alignment and preprocessing
    3. Embedding generation (FaceNet)
    4. Face verification
    """
    print("\n" + "=" * 60)
    print("Combined Pipeline MLflow Experiment")
    print("=" * 60)
    
    try:
        from mtcnn.mtcnn import MTCNN
    except ImportError:
        print("MTCNN not available")
        return None
    
    experiment_name = "Face_Recognition_Pipeline"
    mlflow.set_experiment(experiment_name)
    
    if run_name is None:
        run_name = f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        print(f"Starting MLflow run: {run_name}")
        
        # Initialize models
        print("\n[1/3] Loading models...")
        K.set_image_data_format("channels_last")
        facenet_model = faceRecoModel(input_shape=(96, 96, 3))
        load_weights_from_FaceNet(facenet_model)
        mtcnn_detector = MTCNN()
        print("Models loaded successfully")
        
        # Log pipeline parameters
        print("\n[2/3] Logging pipeline parameters...")
        mlflow.log_param("pipeline_name", "MTCNN + FaceNet")
        mlflow.log_param("detection_model", "MTCNN")
        mlflow.log_param("embedding_model", "FaceNet")
        mlflow.log_param("input_size_facenet", "(96, 96, 3)")
        mlflow.log_param("embedding_dim", 128)
        mlflow.log_param("verification_threshold", 0.5)
        mlflow.log_param("min_detection_confidence", 0.9)
        mlflow.log_param("data_source", "MongoDB")
        mlflow.log_param("mongodb_database", MONGODB_DATABASE_NAME)
        
        # Test pipeline on MongoDB images
        print("\n[3/3] Testing full pipeline on MongoDB images...")
        
        # Load images from MongoDB
        images_by_name = load_faces_from_mongodb()
        pipeline_times = []
        successful_recognitions = 0
        total_tests = 0
        
        if images_by_name:
            for name, images in images_by_name.items():
                for idx, img_bgr in enumerate(images[:3]):  # Test up to 3 images per person
                    try:
                        start_time = time.time()
                        
                        # Step 1: Convert to RGB and detect
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        faces = mtcnn_detector.detect_faces(img_rgb)
                        
                        if not faces:
                            # Image is already a face crop, process directly
                            face_resized = cv2.resize(img_rgb, TARGET_SIZE)
                            face_normalized = face_resized.astype(np.float32) / 255.0
                        else:
                            # Step 2: Get best face
                            best_face = max(faces, key=lambda x: x.get("confidence", 0))
                            x, y, w, h = best_face["box"]
                            
                            # Step 3: Crop and resize
                            face_crop = img_rgb[max(0,y):y+h, max(0,x):x+w]
                            if face_crop.size == 0:
                                face_crop = img_rgb
                            
                            face_resized = cv2.resize(face_crop, TARGET_SIZE)
                            face_normalized = face_resized.astype(np.float32) / 255.0
                        
                        # Step 4: Get embedding
                        embedding = facenet_model.predict(np.expand_dims(face_normalized, axis=0))
                        
                        pipeline_time = time.time() - start_time
                        pipeline_times.append(pipeline_time)
                        total_tests += 1
                        
                        if embedding is not None and embedding.shape[-1] == 128:
                            successful_recognitions += 1
                        
                        print(f"  {name}[{idx}]: Pipeline time {pipeline_time*1000:.1f}ms")
                        
                    except Exception as e:
                        print(f"  Error processing {name}[{idx}]: {e}")
                        continue
        else:
            print("  No images found in MongoDB for pipeline testing")
        
        # Log pipeline metrics
        if pipeline_times:
            mlflow.log_metric("mean_pipeline_time_ms", float(np.mean(pipeline_times) * 1000))
            mlflow.log_metric("min_pipeline_time_ms", float(np.min(pipeline_times) * 1000))
            mlflow.log_metric("max_pipeline_time_ms", float(np.max(pipeline_times) * 1000))
            mlflow.log_metric("pipeline_success_rate", float(successful_recognitions / max(1, total_tests)))
            mlflow.log_metric("total_tests", total_tests)
            mlflow.log_metric("successful_recognitions", successful_recognitions)
            
            print(f"\n  Mean pipeline time: {np.mean(pipeline_times)*1000:.2f}ms")
            print(f"  Success rate: {successful_recognitions}/{total_tests}")
        
        print("\n" + "=" * 60)
        print("Pipeline experiment logged successfully!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print("=" * 60)
        
        return mlflow.active_run().info.run_id


def run_all_experiments():
    """Run all MLflow experiments."""
    print("\n" + "=" * 70)
    print("        RUNNING ALL MLFLOW EXPERIMENTS")
    print("=" * 70)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Run FaceNet experiment
    facenet_run_id = log_facenet_experiment(f"facenet_{timestamp}")
    
    # Run MTCNN experiment
    mtcnn_run_id = log_mtcnn_experiment(f"mtcnn_{timestamp}")
    
    # Run combined pipeline experiment
    pipeline_run_id = log_combined_pipeline_experiment(f"pipeline_{timestamp}")
    
    print("\n" + "=" * 70)
    print("        ALL EXPERIMENTS COMPLETED")
    print("=" * 70)
    print("\nRun IDs:")
    print(f"  FaceNet:  {facenet_run_id}")
    print(f"  MTCNN:    {mtcnn_run_id}")
    print(f"  Pipeline: {pipeline_run_id}")
    print("\nTo view results, run:")
    print("  mlflow ui --port 5000")
    print("\nThen open http://localhost:5000 in your browser")
    print("=" * 70)
    
    return {
        "facenet": facenet_run_id,
        "mtcnn": mtcnn_run_id,
        "pipeline": pipeline_run_id
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MLflow tracking for Face Recognition models")
    parser.add_argument(
        "--experiment", 
        choices=["facenet", "mtcnn", "pipeline", "all"],
        default="all",
        help="Which experiment to run (default: all)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name"
    )
    
    args = parser.parse_args()
    
    if args.experiment == "facenet":
        log_facenet_experiment(args.run_name)
    elif args.experiment == "mtcnn":
        log_mtcnn_experiment(args.run_name)
    elif args.experiment == "pipeline":
        log_combined_pipeline_experiment(args.run_name)
    else:
        run_all_experiments()