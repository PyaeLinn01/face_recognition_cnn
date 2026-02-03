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
from functools import wraps
import hmac

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

# Import PyTorch for anti-spoofing liveness detection
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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

# Liveness detection paths
LIVENESS_MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "Face_liveness_detection_static/resources/anti_spoof_models"
)
LIVENESS_DETECTION_MODEL = os.path.join(
    os.path.dirname(__file__),
    "Face_liveness_detection_static/resources/detection_model"
)

# API Key protection
API_KEY_ENV = "FACE_API_KEY"
API_KEY_HEADER = "x-api-key"

# Cache models
_model_cache = {}
_detector_cache = {}
_database_cache = {}
_liveness_cache = {}


# ==================== LIVENESS DETECTION ====================

def parse_liveness_model_name(model_name):
    """Parse model name to get input dimensions and model type."""
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]
    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    return int(h_input), int(w_input), model_type, scale


def get_liveness_kernel(height, width):
    """Get kernel size for liveness model."""
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


class LivenessDetector:
    """Anti-spoofing liveness detection using MiniFAS models."""
    
    def __init__(self, model_dir, device_id=0):
        self.model_dir = model_dir
        self.device = torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        )
        self.models = {}
        self.face_detector = None
        self._init_face_detector()
    
    def _init_face_detector(self):
        """Initialize RetinaFace detector."""
        caffemodel = os.path.join(LIVENESS_DETECTION_MODEL, "Widerface-RetinaFace.caffemodel")
        deploy = os.path.join(LIVENESS_DETECTION_MODEL, "deploy.prototxt")
        if os.path.exists(caffemodel) and os.path.exists(deploy):
            self.face_detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
            print("Liveness face detector loaded")
        else:
            print(f"Warning: Liveness detector models not found at {LIVENESS_DETECTION_MODEL}")
    
    def get_bbox(self, img):
        """Get face bounding box using RetinaFace."""
        if self.face_detector is None:
            return None
        
        import math
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img_resized = cv2.resize(
                img,
                (int(192 * math.sqrt(aspect_ratio)), int(192 / math.sqrt(aspect_ratio))),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            img_resized = img
        
        blob = cv2.dnn.blobFromImage(img_resized, 1, mean=(104, 117, 123))
        self.face_detector.setInput(blob, 'data')
        out = self.face_detector.forward('detection_out').squeeze()
        
        if len(out.shape) == 1:
            return None
        
        max_conf_index = np.argmax(out[:, 2])
        if out[max_conf_index, 2] < 0.6:
            return None
        
        left = int(out[max_conf_index, 3] * width)
        top = int(out[max_conf_index, 4] * height)
        right = int(out[max_conf_index, 5] * width)
        bottom = int(out[max_conf_index, 6] * height)
        
        return [left, top, right - left + 1, bottom - top + 1]
    
    def _get_new_box(self, src_w, src_h, bbox, scale):
        """Get scaled bounding box for cropping."""
        x, y, box_w, box_h = bbox
        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
        
        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w / 2 + x, box_h / 2 + y
        
        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2
        
        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1
        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1
        
        return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)
    
    def crop_face(self, org_img, bbox, scale, out_w, out_h, crop=True):
        """Crop face region for liveness model input."""
        if not crop:
            return cv2.resize(org_img, (out_w, out_h))
        
        src_h, src_w = org_img.shape[:2]
        left_top_x, left_top_y, right_bottom_x, right_bottom_y = self._get_new_box(
            src_w, src_h, bbox, scale
        )
        
        img = org_img[left_top_y:right_bottom_y + 1, left_top_x:right_bottom_x + 1]
        return cv2.resize(img, (out_w, out_h))
    
    def _load_model(self, model_path):
        """Load anti-spoof model."""
        if model_path in self.models:
            return self.models[model_path]
        
        # Import model classes - add Face_liveness_detection_static to path for 'from src.xxx' imports
        import sys
        liveness_root = os.path.join(os.path.dirname(__file__), "Face_liveness_detection_static")
        if liveness_root not in sys.path:
            sys.path.insert(0, liveness_root)
        
        from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
        
        MODEL_MAPPING = {
            'MiniFASNetV1': MiniFASNetV1,
            'MiniFASNetV2': MiniFASNetV2,
            'MiniFASNetV1SE': MiniFASNetV1SE,
            'MiniFASNetV2SE': MiniFASNetV2SE
        }
        
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_liveness_model_name(model_name)
        kernel_size = get_liveness_kernel(h_input, w_input)
        
        model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)
        
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = next(keys)
        
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                new_state_dict[key[7:]] = value
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        
        model.eval()
        self.models[model_path] = model
        return model
    
    def predict(self, img, model_path):
        """Run prediction on cropped face image."""
        import sys
        liveness_root = os.path.join(os.path.dirname(__file__), "Face_liveness_detection_static")
        if liveness_root not in sys.path:
            sys.path.insert(0, liveness_root)
        
        from src.data_io import transform as trans
        
        test_transform = trans.Compose([trans.ToTensor()])
        img_tensor = test_transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        model = self._load_model(model_path)
        
        with torch.no_grad():
            result = model.forward(img_tensor)
            result = F.softmax(result, dim=1).cpu().numpy()
        
        return result
    
    def check_liveness(self, image):
        """
        Check if a face is real or fake.
        
        Args:
            image: RGB numpy array of the face image
            
        Returns:
            dict: {
                'is_real': bool,
                'score': float (0-1, higher = more likely real),
                'label': str ('Real' or 'Fake'),
                'face_detected': bool
            }
        """
        # Get face bounding box
        bbox = self.get_bbox(image)
        
        if bbox is None:
            return {
                'is_real': False,
                'score': 0.0,
                'label': 'No Face',
                'face_detected': False,
                'bbox': None
            }
        
        prediction = np.zeros((1, 3))
        
        # Run prediction with all available models
        for model_name in os.listdir(self.model_dir):
            if not model_name.endswith('.pth'):
                continue
            
            h_input, w_input, model_type, scale = parse_liveness_model_name(model_name)
            
            crop = scale is not None
            if crop:
                img = self.crop_face(image, bbox, scale, w_input, h_input, crop=True)
            else:
                img = self.crop_face(image, bbox, 1.0, w_input, h_input, crop=False)
            
            if img is None:
                continue
            
            model_path = os.path.join(self.model_dir, model_name)
            prediction += self.predict(img, model_path)
        
        # Get final prediction
        label_idx = np.argmax(prediction)
        score = float(prediction[0][label_idx] / 2)  # Normalize by number of models
        
        is_real = bool(label_idx == 1)  # Convert numpy bool to Python bool
        
        # Convert bbox to Python list (it may contain numpy values)
        bbox_list = [int(x) for x in bbox] if bbox is not None else None
        
        return {
            'is_real': is_real,
            'score': score,
            'label': 'Real' if is_real else 'Fake',
            'face_detected': True,
            'bbox': bbox_list
        }


def get_liveness_detector():
    """Get or create liveness detector (cached)."""
    if "liveness" not in _liveness_cache:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available - install torch")
        if not os.path.exists(LIVENESS_MODEL_DIR):
            raise RuntimeError(f"Liveness models not found at {LIVENESS_MODEL_DIR}")
        
        _liveness_cache["liveness"] = LivenessDetector(LIVENESS_MODEL_DIR, device_id=0)
        print("Liveness detector loaded successfully")
    
    return _liveness_cache["liveness"]


def require_api_key(f):
    """Decorator to enforce API key validation on protected endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        expected_key = os.getenv(API_KEY_ENV)
        if not expected_key:
            # No key configured - allow in dev mode
            return f(*args, **kwargs)
        
        provided_key = request.headers.get(API_KEY_HEADER)
        if not provided_key:
            return jsonify({"error": "Missing API key. Include x-api-key header."}), 401
        
        if not hmac.compare_digest(provided_key.strip(), expected_key.strip()):
            return jsonify({"error": "Invalid API key"}), 401
        
        return f(*args, **kwargs)
    return decorated_function


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
    liveness_available = TORCH_AVAILABLE and os.path.exists(LIVENESS_MODEL_DIR)
    return jsonify({
        "status": "ok",
        "service": "Face Attendance API",
        "models": {
            "facenet": "loaded" if "facenet" in _model_cache else "not_loaded",
            "mtcnn": "available" if MTCNN_AVAILABLE else "unavailable",
            "liveness": "available" if liveness_available else "unavailable"
        },
        "mongodb": "connected" if MONGODB_AVAILABLE else "unavailable"
    }), 200


@app.route('/api/v1/liveness-check', methods=['POST'])
@require_api_key
def liveness_check():
    """
    Check if a face is real (live) or fake (spoofed).
    Uses MiniFAS anti-spoofing models.
    
    Request:
    {
        "image_base64": "...",  # Base64 encoded image
    }
    
    Response:
    {
        "is_real": true/false,
        "score": 0.95,  # Confidence score (0-1)
        "label": "Real" or "Fake",
        "face_detected": true/false,
        "bbox": [x, y, width, height] or null
    }
    """
    try:
        data = request.json
        image_base64 = data.get("image_base64")
        
        if not image_base64:
            return jsonify({"error": "Missing image"}), 400
        
        # Decode image
        image_bytes = base64.b64decode(image_base64)
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return jsonify({"error": "Failed to decode image"}), 400
        
        img_rgb = img_bgr[..., ::-1]  # BGR to RGB
        
        # Run liveness detection
        detector = get_liveness_detector()
        result = detector.check_liveness(img_rgb)
        
        return jsonify(result), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/detect-face', methods=['POST'])
@require_api_key
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
        threshold = data.get("threshold", 0.6)  # Distance threshold for matching
        
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
@require_api_key
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
@require_api_key
def verify_face():
    """
    Verify a face against registered identities using FaceNet + MTCNN.
    """
    try:
        data = request.json
        image_base64 = data.get("image_base64")
        threshold = data.get("threshold", 0.6)
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

def init_default_users():
    """Create default admin and teacher accounts if they don't exist."""
    import hashlib
    collection = get_mongodb_collection("users")
    
    default_users = [
        {
            "email": "admin@gmail.com",
            "password": "123456",
            "name": "Admin User",
            "role": "admin"
        },
        {
            "email": "teacher@gmail.com",
            "password": "123456",
            "name": "Teacher User",
            "role": "teacher"
        }
    ]
    
    for user_data in default_users:
        existing = collection.find_one({"email": user_data["email"]})
        if not existing:
            password_hash = hashlib.sha256(user_data["password"].encode()).hexdigest()
            doc = {
                "email": user_data["email"],
                "password_hash": password_hash,
                "name": user_data["name"],
                "role": user_data["role"],
                "created_at": datetime.now(),
            }
            collection.insert_one(doc)
            print(f"Created default {user_data['role']} account: {user_data['email']}")


@app.route('/api/v1/auth/signup', methods=['POST'])
def signup():
    """Register a new user (students only via signup)."""
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
            "role": "student",  # Default role for signup
            "created_at": datetime.now(),
        }
        
        result = collection.insert_one(doc)
        
        return jsonify({
            "success": True,
            "user": {
                "id": str(result.inserted_id),
                "email": email,
                "name": name,
                "role": "student",
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
                "role": user.get("role", "student"),
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
                "role": user.get("role", "student"),
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== ADMIN ENDPOINTS ====================

@app.route('/api/v1/admin/majors', methods=['GET'])
def get_majors():
    """Get all majors."""
    try:
        collection = get_mongodb_collection("majors")
        majors = list(collection.find())
        return jsonify({
            "majors": [{"id": str(m["_id"]), "name": m["name"], "description": m.get("description", "")} for m in majors]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/admin/majors', methods=['POST'])
def create_major():
    """Create a new major."""
    try:
        data = request.json
        name = data.get("name", "").strip()
        description = data.get("description", "").strip()
        
        if not name:
            return jsonify({"error": "Name is required"}), 400
        
        collection = get_mongodb_collection("majors")
        doc = {"name": name, "description": description, "created_at": datetime.now()}
        result = collection.insert_one(doc)
        
        return jsonify({
            "success": True,
            "major": {"id": str(result.inserted_id), "name": name, "description": description}
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/admin/majors/<major_id>', methods=['DELETE'])
def delete_major(major_id):
    """Delete a major."""
    try:
        from bson.objectid import ObjectId
        collection = get_mongodb_collection("majors")
        result = collection.delete_one({"_id": ObjectId(major_id)})
        if result.deleted_count == 0:
            return jsonify({"error": "Major not found"}), 404
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/admin/subjects', methods=['GET'])
def get_subjects():
    """Get all subjects."""
    try:
        collection = get_mongodb_collection("subjects")
        subjects = list(collection.find())
        return jsonify({
            "subjects": [{"id": str(s["_id"]), "name": s["name"], "code": s.get("code", "")} for s in subjects]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/admin/subjects', methods=['POST'])
def create_subject():
    """Create a new subject."""
    try:
        data = request.json
        name = data.get("name", "").strip()
        code = data.get("code", "").strip()
        
        if not name:
            return jsonify({"error": "Name is required"}), 400
        
        collection = get_mongodb_collection("subjects")
        doc = {"name": name, "code": code, "created_at": datetime.now()}
        result = collection.insert_one(doc)
        
        return jsonify({
            "success": True,
            "subject": {"id": str(result.inserted_id), "name": name, "code": code}
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/admin/subjects/<subject_id>', methods=['DELETE'])
def delete_subject(subject_id):
    """Delete a subject."""
    try:
        from bson.objectid import ObjectId
        collection = get_mongodb_collection("subjects")
        result = collection.delete_one({"_id": ObjectId(subject_id)})
        if result.deleted_count == 0:
            return jsonify({"error": "Subject not found"}), 404
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/admin/teachers', methods=['GET'])
def get_teachers():
    """Get all teachers."""
    try:
        collection = get_mongodb_collection("users")
        teachers = list(collection.find({"role": "teacher"}))
        return jsonify({
            "teachers": [{"id": str(t["_id"]), "name": t["name"], "email": t["email"]} for t in teachers]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/admin/teachers', methods=['POST'])
def create_teacher():
    """Create a new teacher account."""
    try:
        import hashlib
        data = request.json
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")
        name = data.get("name", "").strip()
        
        if not email or not password or not name:
            return jsonify({"error": "Email, password, and name are required"}), 400
        
        collection = get_mongodb_collection("users")
        existing = collection.find_one({"email": email})
        if existing:
            return jsonify({"error": "User with this email already exists"}), 400
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        doc = {
            "email": email,
            "password_hash": password_hash,
            "name": name,
            "role": "teacher",
            "created_at": datetime.now(),
        }
        result = collection.insert_one(doc)
        
        return jsonify({
            "success": True,
            "teacher": {"id": str(result.inserted_id), "name": name, "email": email}
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/admin/teachers/<teacher_id>', methods=['DELETE'])
def delete_teacher(teacher_id):
    """Delete a teacher."""
    try:
        from bson.objectid import ObjectId
        collection = get_mongodb_collection("users")
        result = collection.delete_one({"_id": ObjectId(teacher_id), "role": "teacher"})
        if result.deleted_count == 0:
            return jsonify({"error": "Teacher not found"}), 404
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/admin/students', methods=['GET'])
def get_students():
    """Get all students."""
    try:
        collection = get_mongodb_collection("users")
        students = list(collection.find({"role": "student"}))
        
        # Get face registration status
        faces_collection = get_mongodb_collection("faces")
        
        result = []
        for s in students:
            face_count = faces_collection.count_documents({"name": s["name"]})
            result.append({
                "id": str(s["_id"]),
                "name": s["name"],
                "email": s["email"],
                "face_registered": face_count > 0,
                "face_count": face_count
            })
        
        return jsonify({"students": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/admin/students/<student_id>', methods=['DELETE'])
def delete_student(student_id):
    """Delete a student."""
    try:
        from bson.objectid import ObjectId
        collection = get_mongodb_collection("users")
        result = collection.delete_one({"_id": ObjectId(student_id), "role": "student"})
        if result.deleted_count == 0:
            return jsonify({"error": "Student not found"}), 404
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== TEACHER ENDPOINTS ====================

@app.route('/api/v1/teacher/attendance', methods=['GET'])
def get_all_attendance():
    """Get all attendance records (for teachers)."""
    try:
        collection = get_mongodb_collection("attendance")
        records = list(collection.find().sort("timestamp", -1).limit(500))
        
        return jsonify({
            "records": [{
                "id": str(r["_id"]),
                "timestamp": r["timestamp"].isoformat() if isinstance(r["timestamp"], datetime) else r["timestamp"],
                "entered_name": r.get("entered_name", ""),
                "matched_identity": r.get("matched_identity", ""),
                "distance": r.get("distance", 0),
            } for r in records]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/teacher/attendance/stats', methods=['GET'])
def get_attendance_stats():
    """Get attendance statistics."""
    try:
        collection = get_mongodb_collection("attendance")
        
        # Get total records
        total = collection.count_documents({})
        
        # Get unique students
        unique_students = len(collection.distinct("matched_identity"))
        
        # Get today's count
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_count = collection.count_documents({"timestamp": {"$gte": today}})
        
        return jsonify({
            "total_records": total,
            "unique_students": unique_students,
            "today_count": today_count
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
    
    # Initialize default admin/teacher accounts
    try:
        init_default_users()
    except Exception as e:
        print(f"Warning: Could not init default users: {e}")
    
    # Development server
    # Using port 5001 to avoid conflict with macOS AirPlay Receiver on port 5000
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('API_PORT', 5001)),
        debug=True
    )
