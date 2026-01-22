"""
REST API Backend for Face Attendance System
Serves ML models (FaceNet, MTCNN) and handles MongoDB operations

This API provides the same face detection and verification logic as attend_app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import os
import base64
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, Tuple, Optional, List

# Import MongoDB utilities
try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

# Import MTCNN for face detection
try:
    from mtcnn.mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False

# Import TensorFlow and FaceNet model
try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from fr_utils import load_weights_from_FaceNet
    from inception_blocks_v2 import faceRecoModel
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Configuration
MONGODB_CONNECTION_STRING = os.getenv(
    "MONGODB_CONNECTION_STRING", 
    "mongodb://localhost:27017/"
)
MONGODB_DATABASE_NAME = os.getenv(
    "MONGODB_DATABASE_NAME",
    "face_attendance"
)

TARGET_SIZE = (96, 96)

# Cache models
_model_cache = {}
_detector_cache = {}
_database_cache = {}


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


def get_facenet_model():
    """Get or create FaceNet model (cached)."""
    if "facenet" not in _model_cache:
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow not available")
        K.set_image_data_format("channels_last")
        model = faceRecoModel(input_shape=(96, 96, 3))
        load_weights_from_FaceNet(model)
        _model_cache["facenet"] = model
        print("FaceNet model loaded successfully")
    return _model_cache["facenet"]


def get_mtcnn_detector():
    """Get or create MTCNN detector (cached)."""
    if "mtcnn" not in _detector_cache:
        if not MTCNN_AVAILABLE:
            raise RuntimeError("MTCNN not available - install mtcnn")
        _detector_cache["mtcnn"] = MTCNN()
        print("MTCNN detector loaded successfully")
    return _detector_cache["mtcnn"]


def _best_face_box(faces: List[dict], min_confidence: float = 0.90) -> Optional[Tuple[int, int, int, int]]:
    """Find the best face box from MTCNN detection results."""
    best = None
    best_score = -1.0
    for f in faces:
        conf = float(f.get("confidence", 0.0))
        if conf < min_confidence:
            continue
        x, y, w, h = f.get("box", (0, 0, 0, 0))
        area = float(max(0, int(w)) * max(0, int(h)))
        score = conf * area
        if score > best_score:
            best_score = score
            best = (int(x), int(y), int(w), int(h))
    return best


def _crop_with_margin(img: np.ndarray, box: Tuple[int, int, int, int], margin_ratio: float = 0.20) -> np.ndarray:
    """Crop face with margin."""
    h_img, w_img = img.shape[:2]
    x, y, w, h = box
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w_img, x + w + mx)
    y2 = min(h_img, y + h + my)
    if x2 <= x1 or y2 <= y1:
        return img
    return img[y1:y2, x1:x2]


def _center_square_crop(img: np.ndarray) -> np.ndarray:
    """Center square crop."""
    h, w = img.shape[:2]
    side = min(h, w)
    y_off = (h - side) // 2
    x_off = (w - side) // 2
    return img[y_off:y_off + side, x_off:x_off + side]


def _prewhiten(img: np.ndarray) -> np.ndarray:
    """Apply prewhitening normalization."""
    mean = np.mean(img)
    std = np.std(img)
    std_adj = max(std, 1.0 / np.sqrt(img.size))
    return (img - mean) / std_adj


def _align_face_by_keypoints(
    img_rgb: np.ndarray,
    keypoints: dict,
    output_size: int = 96
) -> Optional[np.ndarray]:
    """Align face using eye keypoints."""
    left_eye = keypoints.get("left_eye")
    right_eye = keypoints.get("right_eye")
    if left_eye is None or right_eye is None:
        return None
    
    try:
        lx, ly = int(left_eye[0]), int(left_eye[1])
        rx, ry = int(right_eye[0]), int(right_eye[1])
    except (TypeError, ValueError):
        return None
    
    dx, dy = rx - lx, ry - ly
    angle = np.degrees(np.arctan2(dy, dx))
    dist = np.hypot(dx, dy)
    if dist < 10:
        return None
    
    desired_dist = output_size * 0.3
    scale = desired_dist / dist
    eye_center = ((lx + rx) / 2, (ly + ry) / 2)
    M = cv2.getRotationMatrix2D(eye_center, angle, scale)
    M[0, 2] += output_size / 2 - eye_center[0]
    M[1, 2] += output_size * 0.35 - eye_center[1]
    
    aligned = cv2.warpAffine(
        img_rgb, M, (output_size, output_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned


def preprocess_and_detect_face(
    image_bytes: bytes,
    use_detection: bool = True,
    min_confidence: float = 0.90,
    margin_ratio: float = 0.20,
    use_alignment: bool = True,
) -> Tuple[np.ndarray, bool, Optional[dict]]:
    """
    Preprocess image and detect face using MTCNN.
    
    Returns:
        - processed_rgb: 96x96 RGB image ready for FaceNet
        - face_detected: whether a face was detected
        - face_info: detection info (box, keypoints, confidence)
    """
    # Decode image
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode image")
    
    img_rgb = img_bgr[..., ::-1]  # BGR to RGB
    face_detected = False
    face_info = None
    processed = img_rgb
    
    if use_detection and MTCNN_AVAILABLE:
        detector = get_mtcnn_detector()
        faces = detector.detect_faces(img_rgb)
        
        if faces:
            # Find best face
            best = None
            best_score = -1.0
            for f in faces:
                conf = float(f.get("confidence", 0.0))
                if conf < min_confidence:
                    continue
                x, y, w, h = f.get("box", (0, 0, 0, 0))
                area = float(max(0, int(w)) * max(0, int(h)))
                score = conf * area
                if score > best_score:
                    best_score = score
                    best = f
            
            if best is not None:
                face_detected = True
                face_info = best
                box = best.get("box", (0, 0, 0, 0))
                keypoints = best.get("keypoints", {})
                
                # Try alignment first
                if use_alignment and keypoints:
                    aligned = _align_face_by_keypoints(img_rgb, keypoints, output_size=96)
                    if aligned is not None:
                        processed = aligned
                    else:
                        # Fall back to cropping
                        processed = _crop_with_margin(img_rgb, tuple(box), margin_ratio)
                else:
                    processed = _crop_with_margin(img_rgb, tuple(box), margin_ratio)
    
    # Resize to 96x96
    if processed.shape[:2] != TARGET_SIZE:
        processed = _center_square_crop(processed)
        processed = cv2.resize(processed, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    
    return processed, face_detected, face_info


def get_embedding(
    processed_rgb: np.ndarray,
    use_prewhitening: bool = False,
    use_tta_flip: bool = True,
) -> np.ndarray:
    """Get FaceNet embedding from preprocessed 96x96 RGB image.
    
    Returns embedding with shape (1, 128) to match attend_app.py format.
    """
    model = get_facenet_model()
    
    # Normalize to [0, 1] range
    img = processed_rgb.astype(np.float32) / 255.0
    if use_prewhitening:
        img = _prewhiten(img)
    
    # Round to match app.py behavior
    img = np.around(img, decimals=12)
    
    # Get embedding
    x = np.array([img])
    emb = model.predict_on_batch(x)[0]
    
    # Test-time augmentation with horizontal flip
    if use_tta_flip:
        img_flip = np.ascontiguousarray(img[:, ::-1, :])
        x2 = np.array([img_flip])
        emb2 = model.predict_on_batch(x2)[0]
        emb = (emb + emb2) / 2.0
        # Only normalize after TTA (matching app.py)
        emb = emb / max(np.linalg.norm(emb), 1e-12)
    
    # Return with shape (1, 128) to match attend_app.py
    return np.expand_dims(emb, axis=0)


def load_reference_database() -> Dict[str, np.ndarray]:
    """Load reference database from MongoDB and build embeddings.
    
    Images in MongoDB are already preprocessed 96x96 faces (cropped, aligned by MTCNN).
    We just decode, normalize, and compute embeddings.
    """
    if "database" in _database_cache:
        return _database_cache["database"]
    
    try:
        collection = get_mongodb_collection("face_images")
        images_by_name = {}
        
        for doc in collection.find().sort("name", 1).sort("image_index", 1):
            name = doc.get("name")
            if name not in images_by_name:
                images_by_name[name] = []
            images_by_name[name].append(doc.get("image_bytes", b""))
        
        if not images_by_name:
            print("No faces registered in MongoDB")
            return {}
        
        print(f"Loading {len(images_by_name)} identities from MongoDB...")
        database = {}
        model = get_facenet_model()
        
        for name, image_bytes_list in images_by_name.items():
            per_identity_embs = []
            for img_bytes in image_bytes_list:
                if not img_bytes:
                    continue
                try:
                    # Decode pre-processed 96x96 face image from MongoDB
                    arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        continue
                    
                    # Image is already 96x96 face crop, just convert to RGB and normalize
                    # Match attend_app.py exactly
                    img_rgb = img_bgr[..., ::-1].astype(np.float32) / 255.0
                    
                    # Ensure correct size
                    if img_rgb.shape[:2] != (96, 96):
                        img_rgb = cv2.resize(img_rgb, (96, 96), interpolation=cv2.INTER_AREA)
                    
                    # Get embedding (same as attend_app.py)
                    x = np.array([img_rgb])
                    emb = model.predict_on_batch(x)[0]
                    emb = emb / max(np.linalg.norm(emb), 1e-12)
                    
                    # Store with shape (1, 128) like attend_app.py
                    per_identity_embs.append(np.expand_dims(emb, axis=0))
                except Exception as e:
                    print(f"Error processing image for {name}: {e}")
                    continue
            
            if per_identity_embs:
                # Average embeddings for this identity (same as attend_app.py)
                mean_emb = np.mean(np.concatenate(per_identity_embs, axis=0), axis=0)
                mean_emb = mean_emb / max(np.linalg.norm(mean_emb), 1e-12)
                # Store with shape (1, 128)
                database[name] = np.expand_dims(mean_emb, axis=0)
                print(f"  Loaded {name} with {len(per_identity_embs)} embeddings")
        
        _database_cache["database"] = database
        print(f"Reference database loaded: {len(database)} identities")
        return database
        
    except Exception as e:
        print(f"Error loading reference database: {e}")
        return {}


def clear_database_cache():
    """Clear the cached database to force reload."""
    _database_cache.clear()


# ==================== API ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "Face Attendance API",
        "models": {
            "facenet": "loaded" if "facenet" in _model_cache else "not_loaded",
            "mtcnn": "available" if MTCNN_AVAILABLE else "unavailable"
        },
        "mongodb": "connected" if MONGODB_AVAILABLE else "unavailable"
    }), 200


@app.route('/api/v1/detect-face', methods=['POST'])
def detect_face():
    """
    Detect faces in an image, return bounding boxes, and identify the person.
    Used for real-time face detection overlay on camera feed.
    
    Request:
    {
        "image_base64": "...",  # Base64 encoded image
        "min_confidence": 0.90,
        "identify": true  # Whether to also identify the person
    }
    
    Response:
    {
        "faces": [
            {
                "box": [x, y, width, height],
                "confidence": 0.99,
                "keypoints": {...},
                "identity": "John",  # If identified
                "match_confidence": 0.85  # Match confidence (1 - distance)
            }
        ],
        "count": 1
    }
    """
    try:
        data = request.json
        image_base64 = data.get("image_base64")
        min_confidence = data.get("min_confidence", 0.90)
        identify = data.get("identify", True)  # Default to identify
        threshold = data.get("threshold", 0.7)  # Distance threshold for matching
        
        if not image_base64:
            return jsonify({"error": "Missing image"}), 400
        
        # Decode image
        image_bytes = base64.b64decode(image_base64)
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return jsonify({"error": "Failed to decode image"}), 400
        
        img_rgb = img_bgr[..., ::-1]  # BGR to RGB
        
        # Detect faces
        detector = get_mtcnn_detector()
        faces = detector.detect_faces(img_rgb)
        
        # Load database for identification
        database = None
        if identify:
            database = load_reference_database()
        
        # Filter by confidence and format response
        detected_faces = []
        for face in faces:
            conf = float(face.get("confidence", 0.0))
            if conf >= min_confidence:
                box = face.get("box", [0, 0, 0, 0])
                keypoints = face.get("keypoints", {})
                
                face_data = {
                    "box": [int(b) for b in box],  # [x, y, width, height]
                    "confidence": round(conf, 4),
                    "keypoints": {k: [int(v[0]), int(v[1])] for k, v in keypoints.items()} if keypoints else {},
                    "identity": None,
                    "match_confidence": 0.0
                }
                
                # Try to identify the face if database exists
                if database and identify:
                    try:
                        # Use proper face alignment like app.py
                        aligned_face = None
                        
                        # Try alignment first (preferred method)
                        if keypoints and "left_eye" in keypoints and "right_eye" in keypoints:
                            aligned_face = _align_face_by_keypoints(img_rgb, keypoints, output_size=96)
                        
                        # Fall back to crop with margin if alignment fails
                        if aligned_face is None:
                            x, y, w, h = box
                            face_crop = _crop_with_margin(img_rgb, (x, y, w, h), margin_ratio=0.20)
                            face_crop = _center_square_crop(face_crop)
                            aligned_face = cv2.resize(face_crop, (96, 96), interpolation=cv2.INTER_AREA)
                        
                        # Get embedding (no TTA for real-time speed)
                        # Normalize same way as reference database
                        img_normalized = aligned_face.astype(np.float32) / 255.0
                        x_batch = np.array([img_normalized])
                        emb = get_facenet_model().predict_on_batch(x_batch)[0]
                        emb = emb / max(np.linalg.norm(emb), 1e-12)
                        query_emb = np.expand_dims(emb, axis=0)
                        
                        # Find best match
                        best_name = None
                        best_dist = float("inf")
                        
                        for identity, ref_emb in database.items():
                            dist = float(np.linalg.norm(query_emb - ref_emb))
                            if dist < best_dist:
                                best_dist = dist
                                best_name = identity
                        
                        if best_dist < threshold:
                            face_data["identity"] = best_name
                            face_data["match_confidence"] = round(max(0, 1.0 - best_dist), 3)
                    except Exception as e:
                        # Identification failed, continue without identity
                        print(f"Identification error: {e}")
                
                detected_faces.append(face_data)
        
        return jsonify({
            "faces": detected_faces,
            "count": len(detected_faces),
            "image_width": img_bgr.shape[1],
            "image_height": img_bgr.shape[0]
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/register-face', methods=['POST'])
def register_face():
    """
    Register a new face for a person with face detection and preprocessing.
    """
    try:
        data = request.json
        name = data.get("name")
        image_base64 = data.get("image_base64")
        image_index = data.get("image_index", 1)
        
        if not name or not image_base64:
            return jsonify({"error": "Missing name or image"}), 400
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_base64)
        
        # Process with face detection using MTCNN (like attend_app.py)
        # This detects face, crops with margin, aligns using keypoints, and resizes to 96x96
        processed_rgb, face_detected, face_info = preprocess_and_detect_face(
            image_bytes,
            use_detection=True,
            min_confidence=0.90,
            margin_ratio=0.20,
            use_alignment=True,
        )
        
        if not face_detected:
            return jsonify({
                "error": "No face detected in image. Please ensure your face is clearly visible.",
                "face_detected": False
            }), 400
        
        # Encode processed 96x96 face image to bytes for storage
        processed_bgr = processed_rgb[..., ::-1]  # RGB to BGR for cv2
        _, processed_bytes = cv2.imencode(".jpg", processed_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        processed_bytes = processed_bytes.tobytes()
        
        # Save to MongoDB with same schema as attend_app.py
        collection = get_mongodb_collection("face_images")
        
        doc = {
            "name": name,
            "image_index": image_index,
            "image_bytes": processed_bytes,  # Store processed 96x96 face
            "size": (96, 96),  # Match attend_app.py schema
            "face_detected": face_detected,
            "face_confidence": face_info.get("confidence") if face_info else None,
            "created_at": datetime.now(),
        }
        
        result = collection.insert_one(doc)
        
        # Clear cache to force reload on next verify
        clear_database_cache()
        
        return jsonify({
            "success": True,
            "id": str(result.inserted_id),
            "name": name,
            "index": image_index,
            "face_detected": face_detected,
            "face_confidence": face_info.get("confidence") if face_info else None,
        }), 201
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/verify-face', methods=['POST'])
def verify_face():
    """
    Verify a face against registered identities using FaceNet + MTCNN.
    """
    try:
        data = request.json
        image_base64 = data.get("image_base64")
        threshold = data.get("threshold", 0.7)
        use_detection = data.get("use_detection", True)
        
        if not image_base64:
            return jsonify({"error": "Missing image"}), 400
        
        # Decode image
        image_bytes = base64.b64decode(image_base64)
        
        # Process with face detection
        processed_rgb, face_detected, face_info = preprocess_and_detect_face(
            image_bytes,
            use_detection=use_detection,
            min_confidence=0.90,
            margin_ratio=0.20,
            use_alignment=True,
        )
        
        # Get embedding for input image
        query_emb = get_embedding(processed_rgb, use_prewhitening=False, use_tta_flip=True)
        
        # Load reference database
        database = load_reference_database()
        
        if not database:
            return jsonify({
                "matched": False,
                "error": "No registered faces in database",
                "face_detected": face_detected,
            }), 200
        
        # Find best match
        best_name = None
        best_dist = float("inf")
        
        for identity, ref_emb in database.items():
            dist = float(np.linalg.norm(query_emb - ref_emb))
            if dist < best_dist:
                best_dist = dist
                best_name = identity
        
        is_match = best_dist < threshold
        
        return jsonify({
            "matched": is_match,
            "identity": best_name if is_match else None,
            "distance": float(best_dist),
            "threshold": threshold,
            "confidence": max(0, 1.0 - float(best_dist)) if is_match else 0.0,
            "face_detected": face_detected,
            "face_confidence": face_info.get("confidence") if face_info else None,
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/attendance/record', methods=['POST'])
def record_attendance():
    """Record attendance for a verified person."""
    try:
        data = request.json
        name = data.get("name")
        identity = data.get("identity")
        distance = data.get("distance", 0.0)
        
        if not identity:
            return jsonify({"error": "Missing identity"}), 400
        
        collection = get_mongodb_collection("attendance")
        
        doc = {
            "timestamp": datetime.now(),
            "entered_name": name or identity,
            "matched_identity": identity,
            "distance": float(distance),
        }
        
        result = collection.insert_one(doc)
        
        return jsonify({
            "success": True,
            "id": str(result.inserted_id),
            "recorded_at": doc["timestamp"].isoformat()
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/attendance/recent', methods=['GET'])
def get_recent_attendance():
    """Get recent attendance records."""
    try:
        limit = request.args.get("limit", 50, type=int)
        
        collection = get_mongodb_collection("attendance")
        records = list(
            collection.find()
            .sort("timestamp", -1)
            .limit(limit)
        )
        
        # Convert ObjectId to string and datetime to ISO format
        for record in records:
            record["_id"] = str(record["_id"])
            record["timestamp"] = record["timestamp"].isoformat()
        
        return jsonify({
            "count": len(records),
            "records": records
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/faces/list', methods=['GET'])
def list_registered_faces():
    """List all registered faces."""
    try:
        collection = get_mongodb_collection("face_images")
        
        # Get unique names with count
        pipeline = [
            {"$group": {"_id": "$name", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        results = list(collection.aggregate(pipeline))
        
        faces = [{"name": r["_id"], "count": r["count"]} for r in results]
        
        return jsonify({
            "count": len(faces),
            "faces": faces
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/database/reload', methods=['POST'])
def reload_database():
    """Force reload of the reference database."""
    try:
        clear_database_cache()
        database = load_reference_database()
        return jsonify({
            "success": True,
            "identities": len(database),
            "names": list(database.keys())
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== AUTH ENDPOINTS ====================

@app.route('/api/v1/auth/signup', methods=['POST'])
def signup():
    """Register a new user."""
    try:
        data = request.json
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")
        name = data.get("name", "").strip()
        
        if not email or not password or not name:
            return jsonify({"error": "Email, password, and name are required"}), 400
        
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400
        
        collection = get_mongodb_collection("users")
        
        # Check if user already exists
        existing = collection.find_one({"email": email})
        if existing:
            return jsonify({"error": "User with this email already exists"}), 400
        
        # Simple password hashing (in production, use bcrypt)
        import hashlib
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        doc = {
            "email": email,
            "password_hash": password_hash,
            "name": name,
            "created_at": datetime.now(),
        }
        
        result = collection.insert_one(doc)
        
        return jsonify({
            "success": True,
            "user": {
                "id": str(result.inserted_id),
                "email": email,
                "name": name,
            }
        }), 201
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/auth/login', methods=['POST'])
def login():
    """Login user."""
    try:
        data = request.json
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")
        
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
        
        collection = get_mongodb_collection("users")
        
        # Find user
        user = collection.find_one({"email": email})
        if not user:
            return jsonify({"error": "Invalid email or password"}), 401
        
        # Check password
        import hashlib
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        if user.get("password_hash") != password_hash:
            return jsonify({"error": "Invalid email or password"}), 401
        
        return jsonify({
            "success": True,
            "user": {
                "id": str(user["_id"]),
                "email": user["email"],
                "name": user["name"],
            }
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/auth/user/<user_id>', methods=['GET'])
def get_user(user_id):
    """Get user by ID."""
    try:
        from bson.objectid import ObjectId
        collection = get_mongodb_collection("users")
        
        user = collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        return jsonify({
            "user": {
                "id": str(user["_id"]),
                "email": user["email"],
                "name": user["name"],
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Pre-load models at startup
    print("Loading models...")
    try:
        get_facenet_model()
        get_mtcnn_detector()
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not pre-load models: {e}")
    
    # Development server
    # Using port 5001 to avoid conflict with macOS AirPlay Receiver on port 5000
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('API_PORT', 5001)),
        debug=True
    )
