import csv
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

from app import (
    CURRENT_DIR,
    embedding_from_bytes,
    get_facenet_model,
    load_reference_database,
)

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    from bson.binary import Binary
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

try:
    import mlflow
    from mlflow_utils import (
        start_verification_run,
        log_mtcnn_params,
        log_facenet_params,
        log_verification_metrics,
        log_dataset_info,
        log_attendance_metrics,
    )
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# Custom CSS for modern UI styling
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 0.9rem;
    }
    
    /* Status cards */
    .status-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 0.5rem 0;
    }
    
    .status-card.success {
        border-left-color: #28a745;
        background: linear-gradient(90deg, rgba(40, 167, 69, 0.1), transparent);
    }
    
    .status-card.warning {
        border-left-color: #ffc107;
        background: linear-gradient(90deg, rgba(255, 193, 7, 0.1), transparent);
    }
    
    .status-card.error {
        border-left-color: #dc3545;
        background: linear-gradient(90deg, rgba(220, 53, 69, 0.1), transparent);
    }
    
    .status-card.info {
        border-left-color: #17a2b8;
        background: linear-gradient(90deg, rgba(23, 162, 184, 0.1), transparent);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #eee;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 5px;
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        color: #2c3e50;
        background-color: white;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e8eaf0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Data table styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Camera input styling */
    .stCameraInput > div {
        border-radius: 16px;
        overflow: hidden;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #2c3e50;
    }
    
    .info-box strong {
        color: #1a252f;
    }
    
    /* MLflow/DVC result cards */
    .result-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .result-card h4 {
        margin: 0 0 0.5rem 0;
        color: #333;
    }
    
    .result-card .stat {
        display: inline-block;
        background: #f0f2f6;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-top: 0.3rem;
    }
    
    /* Progress indicator */
    .progress-ring {
        display: inline-block;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: conic-gradient(#667eea var(--progress), #e0e0e0 0);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Animated gradient text */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .animated-title {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
"""


IMAGES_ROOT = CURRENT_DIR / "images"
ATTENDANCE_CSV = CURRENT_DIR / "attendance.csv"
TARGET_SIZE = (96, 96)
MONGODB_COLLECTION_NAME = "face_images"
MONGODB_ATTENDANCE_COLLECTION_NAME = "attendance"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def _get_mongodb_client(connection_string: str):
    """Get MongoDB client (cached)."""
    if not MONGODB_AVAILABLE:
        raise ImportError("pymongo not installed. Run: pip install pymongo")
    try:
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command("ping")
        return client
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        raise ConnectionError(f"Failed to connect to MongoDB: {e}")


def _get_mongodb_collection(connection_string: str, database_name: str, collection_name: str = MONGODB_COLLECTION_NAME):
    """Get MongoDB collection."""
    client = _get_mongodb_client(connection_string)
    db = client[database_name]
    return db[collection_name]


def _save_image_to_mongodb(
    image_bytes: bytes,
    name: str,
    image_index: int,
    connection_string: str,
    database_name: str,
) -> bool:
    """Save processed 96x96 image to MongoDB."""
    if not MONGODB_AVAILABLE:
        return False
    
    try:
        collection = _get_mongodb_collection(connection_string, database_name)
        doc = {
            "name": name,
            "image_index": image_index,
            "image_bytes": Binary(image_bytes),
            "created_at": datetime.now(),
            "size": TARGET_SIZE,
        }
        collection.insert_one(doc)
        return True
    except Exception as e:
        st.error(f"MongoDB save error: {e}")
        return False


def _load_images_from_mongodb(
    connection_string: str,
    database_name: str,
) -> Dict[str, List[bytes]]:
    """Load all images from MongoDB, grouped by name."""
    if not MONGODB_AVAILABLE:
        return {}
    
    try:
        collection = _get_mongodb_collection(connection_string, database_name)
        images_by_name: Dict[str, List[bytes]] = {}
        
        for doc in collection.find().sort("name", 1).sort("image_index", 1):
            name = doc.get("name", "")
            img_bytes = doc.get("image_bytes")
            if name and img_bytes:
                if name not in images_by_name:
                    images_by_name[name] = []
                images_by_name[name].append(bytes(img_bytes))
        
        return images_by_name
    except Exception as e:
        st.error(f"MongoDB load error: {e}")
        return {}


def _build_reference_db_from_mongodb(
    connection_string: str,
    database_name: str,
    use_prewhitening: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """Build reference database from MongoDB images."""
    model = get_facenet_model()
    images_by_name = _load_images_from_mongodb(connection_string, database_name)
    
    if not images_by_name:
        raise RuntimeError("No images found in MongoDB")
    
    db: Dict[str, np.ndarray] = {}
    ref_paths: Dict[str, str] = {}
    per_identity: Dict[str, List[np.ndarray]] = {}
    
    for name, image_bytes_list in images_by_name.items():
        for img_bytes in image_bytes_list:
            # Decode image bytes to BGR
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            
            # Image is already 96x96, just convert to RGB and normalize
            img_rgb = img_bgr[..., ::-1].astype(np.float32) / 255.0
            if use_prewhitening:
                img_rgb = _prewhiten(img_rgb)
            
            x = np.array([img_rgb])
            emb = model.predict_on_batch(x)[0]
            emb = emb / max(np.linalg.norm(emb), 1e-12)
            per_identity.setdefault(name, []).append(np.expand_dims(emb, axis=0))
    
    # Average embeddings per identity
    for identity, embs in per_identity.items():
        if embs:
            mean_emb = np.mean(np.concatenate(embs, axis=0), axis=0)
            mean_emb = mean_emb / max(np.linalg.norm(mean_emb), 1e-12)
            db[identity] = np.expand_dims(mean_emb, axis=0)
            ref_paths[identity] = f"MongoDB:{identity}"
    
    return db, ref_paths


@st.cache_resource
def _get_mtcnn() -> "MTCNN":
    from mtcnn.mtcnn import MTCNN

    return MTCNN()


def _draw_rounded_rect(
    img_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    radius: int = 12,
) -> None:
    """Draw a rounded rectangle (in-place) on BGR image."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    r = max(0, int(radius))

    # Clamp radius to box size
    r = min(r, abs(x2 - x1) // 2, abs(y2 - y1) // 2)
    if r <= 0:
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
        return

    # Straight segments
    cv2.line(img_bgr, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img_bgr, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img_bgr, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img_bgr, (x2, y1 + r), (x2, y2 - r), color, thickness)

    # Corner arcs
    cv2.ellipse(img_bgr, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img_bgr, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img_bgr, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img_bgr, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def _best_face_box(faces: List[dict], min_confidence: float) -> Optional[Tuple[int, int, int, int]]:
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


def _crop_with_margin_bgr(img_bgr: np.ndarray, box: Tuple[int, int, int, int], margin_ratio: float) -> np.ndarray:
    h_img, w_img = img_bgr.shape[:2]
    x, y, w, h = box
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w_img, x + w + mx)
    y2 = min(h_img, y + h + my)
    if x2 <= x1 or y2 <= y1:
        return img_bgr
    return img_bgr[y1:y2, x1:x2]


def _center_square_crop(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    return img_bgr[y1 : y1 + side, x1 : x1 + side]


def _prewhiten(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    mean = np.mean(x)
    std = np.std(x)
    std_adj = max(std, 1.0 / np.sqrt(x.size))
    return (x - mean) / std_adj


def _embedding_from_bgr_face(
    face_bgr: np.ndarray,
    model,
    use_prewhitening: bool,
) -> np.ndarray:
    face = cv2.resize(face_bgr, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    img = face[..., ::-1].astype(np.float32) / 255.0  # RGB, 0..1
    if use_prewhitening:
        img = _prewhiten(img)
    x = np.array([img])
    emb = model.predict_on_batch(x)[0]
    emb = emb / max(np.linalg.norm(emb), 1e-12)
    return np.expand_dims(emb, axis=0)


def _save_camera_image(
    image_bytes: bytes,
    save_path: Optional[Path] = None,
    mongodb_connection_string: Optional[str] = None,
    mongodb_database_name: Optional[str] = None,
    name: Optional[str] = None,
    image_index: Optional[int] = None,
) -> bytes:
    """Decode, detect face with MTCNN, crop, resize to 96x96, then save.
    
    Returns: processed image bytes (96x96 JPEG)
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode camera image")

    # Face detection (similar to resize_images.py)
    detector = _get_mtcnn()
    img_rgb = img_bgr[..., ::-1]
    faces = detector.detect_faces(img_rgb)

    crop = img_bgr
    margin_ratio = 0.20
    best = _best_face_box(faces, min_confidence=0.90)
    if best is not None:
        crop = _crop_with_margin_bgr(img_bgr, best, margin_ratio=margin_ratio)
    crop = _center_square_crop(crop)
    crop = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # Encode to JPEG bytes
    _, processed_bytes = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    processed_bytes = processed_bytes.tobytes()

    # Save to filesystem if path provided
    if save_path is not None:
        _ensure_dir(save_path.parent)
        cv2.imwrite(str(save_path), crop)

    # Save to MongoDB if connection provided
    if mongodb_connection_string and mongodb_database_name and name is not None and image_index is not None:
        _save_image_to_mongodb(
            processed_bytes,
            name,
            image_index,
            mongodb_connection_string,
            mongodb_database_name,
        )

    return processed_bytes


def _save_attendance_to_mongodb(
    name: str,
    identity: str,
    distance: float,
    connection_string: str,
    database_name: str,
) -> bool:
    """Save attendance record to MongoDB."""
    if not MONGODB_AVAILABLE:
        return False
    
    try:
        collection = _get_mongodb_collection(
            connection_string,
            database_name,
            MONGODB_ATTENDANCE_COLLECTION_NAME,
        )
        doc = {
            "timestamp": datetime.now(),
            "entered_name": name,
            "matched_identity": identity,
            "distance": float(distance),
        }
        collection.insert_one(doc)
        return True
    except Exception as e:
        st.error(f"MongoDB attendance save error: {e}")
        return False


def _load_attendance_from_mongodb(
    connection_string: str,
    database_name: str,
    max_rows: int = 200,
) -> List[List[str]]:
    """Load attendance records from MongoDB."""
    if not MONGODB_AVAILABLE:
        return []
    
    try:
        collection = _get_mongodb_collection(
            connection_string,
            database_name,
            MONGODB_ATTENDANCE_COLLECTION_NAME,
        )
        
        rows: List[List[str]] = []
        rows.append(["timestamp", "entered_name", "matched_identity", "distance"])
        
        # Get most recent records
        for doc in collection.find().sort("timestamp", -1).limit(max_rows):
            timestamp = doc.get("timestamp", datetime.now())
            if isinstance(timestamp, datetime):
                timestamp_str = timestamp.isoformat(timespec="seconds")
            else:
                timestamp_str = str(timestamp)
            
            rows.append([
                timestamp_str,
                doc.get("entered_name", ""),
                doc.get("matched_identity", ""),
                f"{doc.get('distance', 0.0):.4f}",
            ])
        
        return rows
    except Exception as e:
        st.error(f"MongoDB attendance load error: {e}")
        return []


def _append_attendance_row(
    name: str,
    identity: str,
    distance: float,
    use_mongodb: bool = False,
    mongodb_connection_string: Optional[str] = None,
    mongodb_database_name: Optional[str] = None,
    also_save_csv: bool = True,
) -> None:
    """Save attendance record to CSV and/or MongoDB."""
    # Save to MongoDB if enabled
    if use_mongodb and mongodb_connection_string and mongodb_database_name:
        _save_attendance_to_mongodb(
            name,
            identity,
            distance,
            mongodb_connection_string,
            mongodb_database_name,
        )
    
    # Save to CSV if enabled
    if also_save_csv:
        _ensure_dir(ATTENDANCE_CSV.parent)
        file_exists = ATTENDANCE_CSV.exists()
        with ATTENDANCE_CSV.open("a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "entered_name", "matched_identity", "distance"])
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    name,
                    identity,
                    f"{distance:.4f}",
                ]
            )


def _load_attendance_rows(
    max_rows: int = 200,
    use_mongodb: bool = False,
    mongodb_connection_string: Optional[str] = None,
    mongodb_database_name: Optional[str] = None,
) -> List[List[str]]:
    """Load attendance records from MongoDB or CSV."""
    # Try MongoDB first if enabled
    if use_mongodb and mongodb_connection_string and mongodb_database_name:
        rows = _load_attendance_from_mongodb(
            mongodb_connection_string,
            mongodb_database_name,
            max_rows=max_rows,
        )
        if rows:
            return rows
    
    # Fallback to CSV
    if not ATTENDANCE_CSV.exists():
        return []
    rows: List[List[str]] = []
    with ATTENDANCE_CSV.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    body = body[-max_rows:]
    return [header] + body


def _run_command(cmd: list) -> Tuple[str, int]:
    """Run a command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=str(CURRENT_DIR))
        return result.stdout.strip(), 0
    except subprocess.CalledProcessError as e:
        return e.stderr.strip(), e.returncode
    except FileNotFoundError:
        return "Command not found", 1


def _get_dvc_status() -> Dict:
    """Get DVC tracking status."""
    status = {
        "initialized": False,
        "status": "unknown",
        "tracked_datasets": [],
        "remote": None,
        "git_tracked": False,
    }
    
    dvc_dir = CURRENT_DIR / ".dvc"
    if not dvc_dir.exists():
        return status
    
    status["initialized"] = True
    
    # Check DVC status
    output, code = _run_command(["dvc", "status"])
    if code == 0:
        if "nothing to commit" in output.lower() or not output:
            status["status"] = "up_to_date"
        else:
            status["status"] = output
    else:
        status["status"] = f"error: {output}"
    
    # Find all .dvc files
    dvc_files = list(CURRENT_DIR.rglob("*.dvc"))
    for dvc_file in sorted(dvc_files):
        if ".dvc" not in str(dvc_file.parent):  # Skip .dvc folder itself
            rel_path = dvc_file.relative_to(CURRENT_DIR)
            data_path = rel_path.with_suffix("")
            full_data_path = CURRENT_DIR / data_path
            
            dataset_info = {
                "path": str(data_path),
                "dvc_file": str(rel_path),
                "exists": full_data_path.exists(),
                "size_mb": 0,
            }
            
            if full_data_path.exists():
                try:
                    size = sum(f.stat().st_size for f in full_data_path.rglob("*") if f.is_file())
                    dataset_info["size_mb"] = round(size / (1024 * 1024), 2)
                except Exception:
                    pass
            
            status["tracked_datasets"].append(dataset_info)
    
    # Check remote storage
    output, code = _run_command(["dvc", "remote", "list"])
    if code == 0 and output:
        status["remote"] = output.strip()
    
    # Check .dvc files in git
    output, code = _run_command(["git", "status", "--porcelain"])
    if code == 0:
        status["git_tracked"] = True
    
    return status


def _get_mlflow_runs() -> Dict:
    """Get MLflow runs and experiments info."""
    status = {
        "available": MLFLOW_AVAILABLE,
        "tracking_uri": None,
        "experiments": [],
        "total_runs": 0,
        "recent_runs": [],
        "registered_models": [],
    }
    
    if not MLFLOW_AVAILABLE:
        return status
    
    try:
        status["tracking_uri"] = mlflow.get_tracking_uri()
        
        # List experiments
        experiments = mlflow.search_experiments()
        for exp in experiments:
            exp_info = {
                "name": exp.name,
                "id": exp.experiment_id,
                "runs": [],
            }
            
            # Get runs for this experiment
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=10)
            if not runs.empty:
                status["total_runs"] += len(runs)
                for idx, run in runs.head(5).iterrows():
                    run_info = {
                        "name": run.get("tags.mlflow.runName", run.get("run_id", "Unknown")[:8]),
                        "run_id": run.get("run_id", ""),
                        "status": run.get("status", "UNKNOWN"),
                        "start_time": run.get("start_time", None),
                        "metrics": {},
                        "has_model": False,
                    }
                    
                    # Extract key metrics
                    for col in runs.columns:
                        if col.startswith("metrics."):
                            metric_name = col.replace("metrics.", "")
                            val = run.get(col)
                            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                                run_info["metrics"][metric_name] = val
                    
                    # Check for models
                    try:
                        artifacts = mlflow.list_artifacts(run_id=run["run_id"])
                        for a in artifacts:
                            if "facenet" in a.path.lower() or a.path.endswith(".keras"):
                                run_info["has_model"] = True
                                break
                    except Exception:
                        pass
                    
                    exp_info["runs"].append(run_info)
                    if len(status["recent_runs"]) < 5:
                        status["recent_runs"].append(run_info)
            
            status["experiments"].append(exp_info)
        
        # Check for registered models
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            models = client.search_registered_models()
            for model in models:
                model_info = {
                    "name": model.name,
                    "versions": len(model.latest_versions),
                    "latest_version": None,
                }
                if model.latest_versions:
                    latest = model.latest_versions[0]
                    model_info["latest_version"] = {
                        "version": latest.version,
                        "stage": latest.current_stage,
                    }
                status["registered_models"].append(model_info)
        except Exception:
            pass
        
    except Exception as e:
        status["error"] = str(e)
    
    return status


def _render_metric_card(value: str, label: str, color: str = "#667eea") -> str:
    """Render a styled metric card."""
    return f"""
    <div style="
        background: linear-gradient(135deg, {color} 0%, {color}aa 100%);
        padding: 1.2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px {color}44;
        margin: 0.5rem 0;
    ">
        <h3 style="margin: 0; font-size: 1.8rem; font-weight: 700;">{value}</h3>
        <p style="margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.85rem;">{label}</p>
    </div>
    """


def _render_status_badge(status: str, text: str) -> str:
    """Render a status badge."""
    colors = {
        "success": "#28a745",
        "warning": "#ffc107", 
        "error": "#dc3545",
        "info": "#17a2b8",
    }
    color = colors.get(status, "#6c757d")
    return f"""
    <span style="
        background: {color};
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
    ">{text}</span>
    """


def _build_reference_db(
    use_detection: bool,
    require_detection: bool,
    min_confidence: float,
    margin_ratio: float,
    use_alignment: bool,
    use_prewhitening: bool,
    use_tta_flip: bool,
    use_mongodb: bool = False,
    mongodb_connection_string: Optional[str] = None,
    mongodb_database_name: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """Build reference database from filesystem or MongoDB."""
    if use_mongodb and mongodb_connection_string and mongodb_database_name:
        try:
            return _build_reference_db_from_mongodb(
                mongodb_connection_string,
                mongodb_database_name,
                use_prewhitening=use_prewhitening,
            )
        except Exception as e:
            st.warning(f"MongoDB load failed, falling back to filesystem: {e}")
            # Fall through to filesystem
    
    # Filesystem fallback
    if not IMAGES_ROOT.exists():
        raise FileNotFoundError(f"Missing images folder: {IMAGES_ROOT}")

    image_paths = sorted(
        [
            p
            for p in IMAGES_ROOT.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    if not image_paths:
        raise RuntimeError(f"No reference images found in {IMAGES_ROOT}")

    signature = tuple((str(p), p.stat().st_mtime) for p in image_paths)
    return load_reference_database(
        str(IMAGES_ROOT),
        signature,
        use_detection=use_detection,
        require_detection=require_detection,
        min_confidence=min_confidence,
        margin_ratio=margin_ratio,
        use_alignment=use_alignment,
        use_prewhitening=use_prewhitening,
        use_tta_flip=use_tta_flip,
    )


def _register_face_ui() -> None:
    st.markdown('<p class="section-header">👤 Register New Face</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if "captured_count" not in st.session_state:
        st.session_state.captured_count = 0
    if "registered_name" not in st.session_state:
        st.session_state.registered_name = ""
    if "captured_images" not in st.session_state:
        st.session_state.captured_images = []
    
    num_images = 4
    
    # Sidebar settings (collapsed by default)
    with st.sidebar.expander("⚙️ Storage Settings", expanded=False):
        use_mongodb = st.checkbox("Use MongoDB", value=True, key="register_use_mongodb")
        save_to_filesystem = st.checkbox("Also save to filesystem", value=False, key="save_to_filesystem_cb")
        use_mlflow = st.checkbox("Enable MLflow tracking", value=False, disabled=not MLFLOW_AVAILABLE, key="register_mlflow")
    
    mongodb_connection_string = None
    mongodb_database_name = None
    
    if use_mongodb and MONGODB_AVAILABLE:
        default_conn = os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017/")
        default_db = os.getenv("MONGODB_DATABASE_NAME", "face_attendance")
        try:
            secrets = st.secrets.get("mongodb", {})
            default_conn = secrets.get("connection_string", default_conn)
            default_db = secrets.get("database_name", default_db)
        except Exception:
            pass
        mongodb_connection_string = default_conn
        mongodb_database_name = default_db
    
    # ============ STEP 1: Enter Name ============
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 0.8rem 1.2rem; border-radius: 10px; margin-bottom: 1rem;">
        <strong>Step 1:</strong> Enter the person's name
    </div>
    """, unsafe_allow_html=True)
    
    col_name, col_reset = st.columns([3, 1])
    with col_name:
        name = st.text_input(
            "Person Name",
            value=st.session_state.registered_name,
            placeholder="e.g., John Smith",
            key="person_name_input",
            label_visibility="collapsed"
        )
    with col_reset:
        if st.button("🔄 Start Over", use_container_width=True, key="reset_btn"):
            st.session_state.captured_count = 0
            st.session_state.registered_name = ""
            st.session_state.captured_images = []
            st.rerun()
    
    # Update registered name in session
    if name and name != st.session_state.registered_name:
        st.session_state.registered_name = name
        st.session_state.captured_count = 0
        st.session_state.captured_images = []
    
    if not name:
        st.warning("👆 Please enter a name to begin registration.")
        return
    
    # ============ PROGRESS INDICATOR ============
    st.markdown("")
    
    # Visual progress with circles
    progress_html = '<div style="display: flex; justify-content: center; gap: 15px; margin: 1.5rem 0;">'
    for i in range(1, num_images + 1):
        if i <= st.session_state.captured_count:
            # Completed
            progress_html += f'''
            <div style="width: 60px; height: 60px; border-radius: 50%; 
                        background: linear-gradient(135deg, #28a745, #20c997); 
                        display: flex; align-items: center; justify-content: center;
                        color: white; font-weight: bold; font-size: 1.2rem;
                        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4);">
                ✓
            </div>'''
        elif i == st.session_state.captured_count + 1:
            # Current
            progress_html += f'''
            <div style="width: 60px; height: 60px; border-radius: 50%; 
                        background: linear-gradient(135deg, #667eea, #764ba2); 
                        display: flex; align-items: center; justify-content: center;
                        color: white; font-weight: bold; font-size: 1.2rem;
                        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                        animation: pulse 2s infinite;">
                {i}
            </div>'''
        else:
            # Pending
            progress_html += f'''
            <div style="width: 60px; height: 60px; border-radius: 50%; 
                        background: #e0e0e0; 
                        display: flex; align-items: center; justify-content: center;
                        color: #999; font-weight: bold; font-size: 1.2rem;">
                {i}
            </div>'''
    progress_html += '</div>'
    
    # Add pulse animation
    progress_html += '''
    <style>
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    </style>
    '''
    
    st.markdown(progress_html, unsafe_allow_html=True)
    
    # Progress text
    if st.session_state.captured_count >= num_images:
        st.markdown(f"""
        <div style="text-align: center; color: #28a745; font-size: 1.2rem; font-weight: 600; margin: 1rem 0;">
            ✅ All {num_images} photos captured for <strong>{name}</strong>!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align: center; color: #666; margin: 0.5rem 0;">
            Photo <strong>{st.session_state.captured_count + 1}</strong> of <strong>{num_images}</strong> 
            &nbsp;•&nbsp; Registering: <strong>{name}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # ============ STEP 2: Capture Photos ============
    if st.session_state.captured_count < num_images:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 0.8rem 1.2rem; border-radius: 10px; margin: 1.5rem 0 1rem 0;">
            <strong>Step 2:</strong> Capture photo {st.session_state.captured_count + 1} of {num_images}
        </div>
        """, unsafe_allow_html=True)
        
        # Tips for current photo
        tips = [
            "📷 Look straight at the camera",
            "📷 Turn slightly to your left",
            "📷 Turn slightly to your right", 
            "📷 Tilt your head slightly"
        ]
        current_tip = tips[st.session_state.captured_count % len(tips)]
        
        st.info(f"💡 **Tip:** {current_tip}")
        
        # Camera
        camera_image = st.camera_input(
            f"Capture Photo {st.session_state.captured_count + 1}",
            key=f"camera_{st.session_state.captured_count}",
            label_visibility="collapsed"
        )
        
        # Save button - big and prominent
        if camera_image:
            st.markdown("")
            if st.button(
                f"📸 Save Photo {st.session_state.captured_count + 1}",
                use_container_width=True,
                type="primary",
                key="save_photo_btn"
            ):
                save_path = None
                if save_to_filesystem:
                    identity_dir = IMAGES_ROOT / name
                    _ensure_dir(identity_dir)
                    filename = f"{name}_{st.session_state.captured_count + 1}.jpg"
                    save_path = identity_dir / filename

                try:
                    _save_camera_image(
                        camera_image.getvalue(),
                        save_path=save_path,
                        mongodb_connection_string=mongodb_connection_string if use_mongodb else None,
                        mongodb_database_name=mongodb_database_name if use_mongodb else None,
                        name=name if use_mongodb else None,
                        image_index=st.session_state.captured_count + 1 if use_mongodb else None,
                    )
                    
                    st.session_state.captured_count += 1
                    st.session_state.captured_images.append(st.session_state.captured_count)
                    
                    # MLflow logging on first image
                    if use_mlflow and MLFLOW_AVAILABLE and st.session_state.captured_count == 1:
                        try:
                            with start_verification_run(run_name=f"register_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                                log_mtcnn_params(min_confidence=0.90, margin_ratio=0.20, use_alignment=True)
                                log_facenet_params(use_prewhitening=False, use_tta_flip=True)
                                log_dataset_info(num_images=num_images, num_identities=1)
                        except Exception:
                            pass
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Failed to save: {e}")
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #888;">
                📷 Take a photo using the camera above, then click <strong>Save</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # ============ COMPLETION ============
    else:
        st.balloons()
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                    color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
            <h2 style="margin: 0 0 0.5rem 0;">🎉 Registration Complete!</h2>
            <p style="margin: 0; font-size: 1.1rem;">
                Successfully registered <strong>{name}</strong> with {num_images} photos.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👤 Register Another Person", use_container_width=True, key="register_another"):
                st.session_state.captured_count = 0
                st.session_state.registered_name = ""
                st.session_state.captured_images = []
                st.rerun()
        with col2:
            st.markdown("""
            <div style="background: #f0f2f6; padding: 0.8rem; border-radius: 8px; text-align: center;">
                👉 Switch to <strong>📸 Attendance</strong> tab to verify faces
            </div>
            """, unsafe_allow_html=True)


def _attendance_ui() -> None:
    st.markdown('<p class="section-header">📸 Real‑time Attendance</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>🔍 How it works:</strong> The camera verifies faces against all registered identities 
        from MongoDB (or filesystem) and automatically records attendance when a match is found.
    </div>
    """, unsafe_allow_html=True)

    # MongoDB settings
    st.sidebar.markdown("### 🗄️ Data Source")
    use_mongodb = st.sidebar.checkbox("Use MongoDB", value=True, key="attendance_use_mongodb")
    
    mongodb_connection_string = None
    mongodb_database_name = None
    
    if use_mongodb:
        if not MONGODB_AVAILABLE:
            st.sidebar.error("❌ pymongo not installed. Run: pip install pymongo")
        else:
            # Try to get from environment variables first (for Docker), then secrets, then defaults
            default_conn = os.getenv("MONGODB_CONNECTION_STRING", None)
            default_db = os.getenv("MONGODB_DATABASE_NAME", None)
            
            if default_conn is None or default_db is None:
                try:
                    secrets = st.secrets.get("mongodb", {})
                    default_conn = secrets.get("connection_string", "mongodb://localhost:27017/")
                    default_db = secrets.get("database_name", "face_attendance")
                except Exception:
                    default_conn = default_conn or "mongodb://localhost:27017/"
                    default_db = default_db or "face_attendance"
            
            mongodb_connection_string = st.sidebar.text_input(
                "MongoDB Connection String",
                value=default_conn,
                type="default",
                key="attendance_mongo_conn",
            )
            mongodb_database_name = st.sidebar.text_input(
                "Database Name",
                value=default_db,
                key="attendance_mongo_db",
            )
    
    # Attendance storage settings
    st.sidebar.markdown("### 💾 Attendance Storage")
    save_attendance_to_mongodb = st.sidebar.checkbox(
        "Save attendance to MongoDB",
        value=True,
        disabled=not (use_mongodb and mongodb_connection_string and mongodb_database_name),
        key="save_attendance_mongodb_cb",
    )
    also_save_csv = st.sidebar.checkbox("Also save to CSV", value=False, key="also_save_csv_cb")

    st.sidebar.markdown("### 🎛️ Verification Settings")
    threshold = st.sidebar.slider("Match threshold", 0.0, 2.0, 0.7, 0.01, help="Lower = stricter matching", key="threshold_slider")
    
    with st.sidebar.expander("🔧 Advanced Settings", expanded=False):
        use_detection = st.checkbox("Use MTCNN face detection", value=True, key="use_detection_cb")
        require_detection = st.checkbox("Require face detection", value=False, key="require_detection_cb")
        min_confidence = st.slider("Min face confidence", 0.0, 1.0, 0.90, 0.01, key="min_conf_slider")
        margin_ratio = st.slider("Face crop margin", 0.0, 0.6, 0.2, 0.05, key="margin_slider")
        use_alignment = st.checkbox("Align face (eyes/nose)", value=True, key="use_alignment_cb")
        use_prewhitening = st.checkbox("Prewhiten (normalize)", value=False, key="use_prewhitening_cb")
        use_tta_flip = st.checkbox("Flip TTA (average)", value=True, key="use_tta_flip_cb")
    
    # Set MLflow to False since UI is removed
    use_mlflow = False

    try:
        database, ref_paths = _build_reference_db(
            use_detection=use_detection,
            require_detection=require_detection,
            min_confidence=min_confidence,
            margin_ratio=margin_ratio,
            use_alignment=use_alignment,
            use_prewhitening=use_prewhitening,
            use_tta_flip=use_tta_flip,
            use_mongodb=use_mongodb and mongodb_connection_string and mongodb_database_name,
            mongodb_connection_string=mongodb_connection_string,
            mongodb_database_name=mongodb_database_name,
        )
    except Exception as e:
        st.error(f"Failed to build reference database: {e}")
        return

    if not database:
        st.error("No reference identities available. Please register at least one person first.")
        return

    # Log dataset info to MLflow if enabled
    if use_mlflow and MLFLOW_AVAILABLE:
        try:
            num_fs_images = len(list(IMAGES_ROOT.rglob("*.jpg"))) if IMAGES_ROOT.exists() else 0
            num_mongo_images = len(database) * 4 if use_mongodb else 0
            log_dataset_info(
                num_images=num_fs_images + num_mongo_images,
                num_identities=len(database),
                dataset_path=IMAGES_ROOT if not use_mongodb else None,
            )
        except Exception as e:
            st.sidebar.warning(f"MLflow logging error: {e}")

    model = get_facenet_model()
    
    # Initialize MLflow run if enabled (start run and log params once per session)
    if use_mlflow and MLFLOW_AVAILABLE:
        if "mlflow_run_id" not in st.session_state:
            try:
                import mlflow
                from mlflow_utils import init_mlflow_experiment
                experiment_id = init_mlflow_experiment("face_recognition")
                run = mlflow.start_run(
                    run_name=f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    experiment_id=experiment_id,
                    nested=False,
                )
                st.session_state.mlflow_run_id = run.info.run_id
                # Log parameters once at session start
                log_mtcnn_params(
                    min_confidence=min_confidence,
                    margin_ratio=margin_ratio,
                    use_alignment=use_alignment,
                )
                log_facenet_params(
                    use_prewhitening=use_prewhitening,
                    use_tta_flip=use_tta_flip,
                )
                
                # Log models as artifacts (FaceNet model and MTCNN config)
                try:
                    from mlflow_utils import log_facenet_model, log_model_artifacts
                    # Log FaceNet model
                    log_facenet_model(model, model_name="facenet")
                    # Log weights directory and sample images
                    log_model_artifacts(
                        weights_dir=CURRENT_DIR / "weights",
                        images_dir=IMAGES_ROOT if IMAGES_ROOT.exists() else None,
                    )
                    # Log MTCNN configuration
                    import mlflow
                    mlflow.log_param("mtcnn_version", "0.1.1")
                    mlflow.log_param("mtcnn_model_type", "MTCNN Face Detector")
                    mlflow.log_text("MTCNN is a pre-trained face detection library", "mtcnn_info.txt")
                except Exception as e:
                    st.sidebar.warning(f"MLflow model logging error: {e}")
                
                # Log models as artifacts
                try:
                    from mlflow_utils import log_facenet_model, log_model_artifacts
                    # Log FaceNet model
                    log_facenet_model(model, model_name="facenet")
                    # Log MTCNN info and weights directory
                    log_model_artifacts(
                        weights_dir=CURRENT_DIR / "weights",
                        images_dir=IMAGES_ROOT if IMAGES_ROOT.exists() else None,
                    )
                    # Log MTCNN configuration
                    import mlflow
                    mlflow.log_param("mtcnn_version", "0.1.1")
                    mlflow.log_param("mtcnn_model_type", "MTCNN Face Detector")
                except Exception as e:
                    st.sidebar.warning(f"MLflow model logging error: {e}")
            except Exception as e:
                st.sidebar.warning(f"MLflow initialization error: {e}")
                st.session_state.mlflow_run_id = None
        # Ensure we're using the active run for logging
        elif st.session_state.mlflow_run_id:
            try:
                import mlflow
                # Set active run if not already active
                if mlflow.active_run() is None:
                    mlflow.start_run(run_id=st.session_state.mlflow_run_id)
            except Exception:
                pass  # Run might be closed, will create new one if needed

    st.markdown("### 🎥 Live Camera Verification")
    st.caption("📹 If live camera doesn’t load, install dependencies and restart Streamlit.")

    try:
        from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
        import av
    except Exception:
        st.warning(
            "Live camera requires `streamlit-webrtc`. "
            "Run `pip install -r requirements.txt` and restart, or use the snapshot verifier below."
        )
        camera_image = st.camera_input("Camera – capture frame for verification")

        if st.button("Verify & Record Attendance (snapshot)", disabled=(camera_image is None)):
            if camera_image is None:
                st.error("No image captured from camera.")
                return

            try:
                query_emb, preview_rgb = embedding_from_bytes(
                    camera_image.getvalue(),
                    model,
                    use_detection=use_detection,
                    require_detection=require_detection,
                    min_confidence=min_confidence,
                    margin_ratio=margin_ratio,
                    use_alignment=use_alignment,
                    use_prewhitening=use_prewhitening,
                    use_tta_flip=use_tta_flip,
                )
            except Exception as e:
                st.error(f"Failed to process camera frame: {e}")
                return

            if use_detection:
                st.image(preview_rgb, caption="Detected / aligned face", use_container_width=True)

            best_name = None
            best_dist = float("inf")
            for identity, ref_emb in database.items():
                dist = float(np.linalg.norm(query_emb - ref_emb))
                if dist < best_dist:
                    best_dist = dist
                    best_name = identity

            if best_name is None:
                st.error("Could not find any identity in database.")
                return

            is_match = best_dist < threshold
            st.write({"best_identity": best_name, "distance": best_dist, "threshold": threshold, "match": is_match})

            # Log to MLflow if enabled
            if use_mlflow and MLFLOW_AVAILABLE:
                try:
                    log_verification_metrics(
                        distance=best_dist,
                        threshold=threshold,
                        is_match=is_match,
                        identity=best_name,
                    )
                except Exception as e:
                    st.sidebar.warning(f"MLflow logging error: {e}")

            if is_match:
                _append_attendance_row(
                    best_name,
                    best_name,
                    best_dist,
                    use_mongodb=save_attendance_to_mongodb and mongodb_connection_string and mongodb_database_name,
                    mongodb_connection_string=mongodb_connection_string if save_attendance_to_mongodb else None,
                    mongodb_database_name=mongodb_database_name if save_attendance_to_mongodb else None,
                    also_save_csv=also_save_csv,
                )
                success_msg = f"Attendance recorded for **{best_name}**"
                if save_attendance_to_mongodb:
                    success_msg += " (saved to MongoDB)"
                if also_save_csv:
                    success_msg += " (saved to CSV)"
                st.success(success_msg)
            else:
                st.warning("Face did not match any registered identity with the given threshold.")
    else:
        # Live processor with rounded bounding boxes + continuous identity estimate.
        class FaceAttendanceProcessor(VideoProcessorBase):
            def __init__(self) -> None:
                self.detector = _get_mtcnn()
                self.frame_idx = 0
                self.last_best: Optional[Tuple[str, float]] = None

            def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
                self.frame_idx += 1
                img_bgr = frame.to_ndarray(format="bgr24")

                img_rgb = img_bgr[..., ::-1]
                faces = self.detector.detect_faces(img_rgb)

                # Draw all faces (rounded)
                for f in faces:
                    conf = float(f.get("confidence", 0.0))
                    if conf < min_confidence:
                        continue
                    x, y, w, h = f.get("box", (0, 0, 0, 0))
                    x1, y1 = max(0, int(x)), max(0, int(y))
                    x2, y2 = max(0, int(x + w)), max(0, int(y + h))
                    _draw_rounded_rect(img_bgr, x1, y1, x2, y2, color=(0, 255, 0), thickness=2, radius=14)

                # For speed, run recognition only every few frames
                if self.frame_idx % 6 == 0:
                    best_box = _best_face_box(faces, min_confidence=min_confidence)
                    if best_box is not None:
                        face = _crop_with_margin_bgr(img_bgr, best_box, margin_ratio=margin_ratio)
                        face = _center_square_crop(face)
                        q = _embedding_from_bgr_face(face, model=model, use_prewhitening=use_prewhitening)

                        best_name = None
                        best_dist = float("inf")
                        for identity, ref_emb in database.items():
                            dist = float(np.linalg.norm(q - ref_emb))
                            if dist < best_dist:
                                best_dist = dist
                                best_name = identity
                        if best_name is not None:
                            self.last_best = (best_name, float(best_dist))

                # Overlay label
                if self.last_best is not None:
                    bn, bd = self.last_best
                    is_match = bd < threshold
                    label = f"{bn}  dist={bd:.3f}  {'MATCH' if is_match else 'NO MATCH'}"
                    color = (0, 200, 0) if is_match else (0, 0, 255)
                    cv2.putText(
                        img_bgr,
                        label,
                        (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

        ctx = webrtc_streamer(
            key="attendance-live",
            video_processor_factory=FaceAttendanceProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("When it shows MATCH, you can record attendance.")
        with col_b:
            if st.button("Record attendance (from live match)"):
                if ctx.video_processor is None or getattr(ctx.video_processor, "last_best", None) is None:
                    st.error("No match data yet. Wait for the camera to detect a face.")
                else:
                    best_name, best_dist = ctx.video_processor.last_best
                    if float(best_dist) < threshold:
                        # Log to MLflow if enabled
                        if use_mlflow and MLFLOW_AVAILABLE:
                            try:
                                log_verification_metrics(
                                    distance=float(best_dist),
                                    threshold=threshold,
                                    is_match=True,
                                    identity=best_name,
                                )
                            except Exception as e:
                                st.sidebar.warning(f"MLflow logging error: {e}")
                        
                        _append_attendance_row(
                            best_name,
                            best_name,
                            float(best_dist),
                            use_mongodb=save_attendance_to_mongodb and mongodb_connection_string and mongodb_database_name,
                            mongodb_connection_string=mongodb_connection_string if save_attendance_to_mongodb else None,
                            mongodb_database_name=mongodb_database_name if save_attendance_to_mongodb else None,
                            also_save_csv=also_save_csv,
                        )
                        success_msg = f"Attendance recorded for **{best_name}**"
                        if save_attendance_to_mongodb:
                            success_msg += " (saved to MongoDB)"
                        if also_save_csv:
                            success_msg += " (saved to CSV)"
                        st.success(success_msg)
                    else:
                        st.warning("⚠️ Current live frame is not a MATCH. Move closer / adjust lighting.")

    # Attendance Records Section
    rows = _load_attendance_rows(
        max_rows=200,
        use_mongodb=use_mongodb and mongodb_connection_string and mongodb_database_name,
        mongodb_connection_string=mongodb_connection_string,
        mongodb_database_name=mongodb_database_name,
    )
    if rows:
        st.markdown('<p class="section-header">📋 Recent Attendance</p>', unsafe_allow_html=True)
        import pandas as pd  # Local import to keep top imports minimal

        df = pd.DataFrame(rows[1:], columns=rows[0])
        
        # Add some styling to the dataframe
        if not df.empty:
            # Show stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(_render_metric_card(str(len(df)), "Total Records", "#667eea"), unsafe_allow_html=True)
            with col2:
                unique_names = df["matched_identity"].nunique() if "matched_identity" in df.columns else 0
                st.markdown(_render_metric_card(str(unique_names), "Unique People", "#28a745"), unsafe_allow_html=True)
            with col3:
                if "timestamp" in df.columns and len(df) > 0:
                    last_time = df["timestamp"].iloc[0] if len(df) > 0 else "N/A"
                    st.markdown(_render_metric_card(str(last_time)[:10], "Last Check-in", "#764ba2"), unsafe_allow_html=True)
        
        st.dataframe(df, use_container_width=True, height=300)
    else:
        st.info("📭 No attendance records yet. Start verifying faces to record attendance!")


def _mlflow_results_ui() -> None:
    """MLflow Results Dashboard UI."""
    st.markdown('<p class="section-header">📊 MLflow Experiment Tracking</p>', unsafe_allow_html=True)
    
    if not MLFLOW_AVAILABLE:
        st.error("❌ MLflow is not installed. Install with: `pip install mlflow`")
        st.code("pip install mlflow", language="bash")
        return
    
    # Get MLflow status
    with st.spinner("Loading MLflow data..."):
        mlflow_status = _get_mlflow_runs()
    
    if "error" in mlflow_status:
        st.error(f"⚠️ Error accessing MLflow: {mlflow_status['error']}")
        return
    
    # Overview metrics
    st.markdown("### 📈 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(_render_metric_card(
            str(len(mlflow_status["experiments"])), 
            "Experiments", 
            "#667eea"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(_render_metric_card(
            str(mlflow_status["total_runs"]), 
            "Total Runs", 
            "#28a745"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(_render_metric_card(
            str(len(mlflow_status["registered_models"])), 
            "Registered Models", 
            "#764ba2"
        ), unsafe_allow_html=True)
    
    with col4:
        models_with_artifacts = sum(1 for r in mlflow_status["recent_runs"] if r.get("has_model"))
        st.markdown(_render_metric_card(
            str(models_with_artifacts), 
            "Runs with Models", 
            "#f5576c"
        ), unsafe_allow_html=True)
    
    st.markdown("")
    
    # Tracking URI info
    st.markdown("### 🔗 Connection Info")
    st.info(f"**Tracking URI:** `{mlflow_status['tracking_uri']}`")
    
    col1, col2 = st.columns(2)
    with col1:
        st.code("mlflow ui --port 5000", language="bash")
        st.caption("Run this command to start the MLflow web UI")
    with col2:
        if st.button("🔄 Refresh MLflow Data", key="refresh_mlflow"):
            st.rerun()
    
    # Recent Runs
    st.markdown("### 🏃 Recent Runs")
    
    if not mlflow_status["recent_runs"]:
        st.info("No runs recorded yet. Start using the Attendance tab to create runs!")
    else:
        for run in mlflow_status["recent_runs"]:
            with st.expander(f"🔹 {run['name']} {'✅' if run['has_model'] else ''}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Run ID:** `{run['run_id'][:12]}...`")
                    st.markdown(f"**Status:** {_render_status_badge('success' if run['status'] == 'FINISHED' else 'info', run['status'])}", unsafe_allow_html=True)
                    
                    if run.get("start_time"):
                        try:
                            start_time = datetime.fromtimestamp(run["start_time"] / 1000)
                            st.markdown(f"**Started:** {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        except Exception:
                            pass
                
                with col2:
                    if run["has_model"]:
                        st.success("🤖 Model logged")
                    else:
                        st.info("📝 Metrics only")
                
                if run["metrics"]:
                    st.markdown("**📊 Metrics:**")
                    metrics_cols = st.columns(min(len(run["metrics"]), 4))
                    for i, (k, v) in enumerate(run["metrics"].items()):
                        with metrics_cols[i % 4]:
                            if isinstance(v, float):
                                st.metric(k, f"{v:.4f}")
                            else:
                                st.metric(k, str(v))
    
    # Registered Models
    st.markdown("### 🤖 Registered Models")
    
    if not mlflow_status["registered_models"]:
        st.info("No registered models yet. Models are automatically registered during verification runs.")
    else:
        for model in mlflow_status["registered_models"]:
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**{model['name']}**")
                with col2:
                    st.markdown(f"Versions: **{model['versions']}**")
                with col3:
                    if model.get("latest_version"):
                        lv = model["latest_version"]
                        st.markdown(f"v{lv['version']} ({lv['stage']})")
    
    # Experiments detail
    st.markdown("### 🧪 Experiments")
    
    for exp in mlflow_status["experiments"]:
        with st.expander(f"📁 {exp['name']} (ID: {exp['id']})", expanded=False):
            if exp["runs"]:
                st.markdown(f"**Total runs:** {len(exp['runs'])}")
                
                # Create a simple table for runs
                import pandas as pd
                run_data = []
                for r in exp["runs"]:
                    run_data.append({
                        "Name": r["name"],
                        "Status": r["status"],
                        "Has Model": "✅" if r["has_model"] else "❌",
                        "Metrics": ", ".join([f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in list(r["metrics"].items())[:3]]),
                    })
                
                if run_data:
                    st.dataframe(pd.DataFrame(run_data), use_container_width=True)
            else:
                st.info("No runs in this experiment")
    
    # Quick commands section
    st.markdown("### 🛠️ Quick Commands")
    
    st.markdown("""
    | Command | Description |
    |---------|-------------|
    | `mlflow ui --port 5000` | Start MLflow web interface |
    | `mlflow runs list` | List all runs in terminal |
    | `python check_mlflow_runs.py` | Check runs status |
    """)


def _dvc_status_ui() -> None:
    """DVC Dataset Tracking Status UI."""
    st.markdown('<p class="section-header">📦 DVC Dataset Tracking</p>', unsafe_allow_html=True)
    
    # Get DVC status
    with st.spinner("Checking DVC status..."):
        dvc_status = _get_dvc_status()
    
    # Overview
    st.markdown("### 📈 Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        init_status = "✅" if dvc_status["initialized"] else "❌"
        st.markdown(_render_metric_card(
            init_status, 
            "DVC Initialized", 
            "#28a745" if dvc_status["initialized"] else "#dc3545"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(_render_metric_card(
            str(len(dvc_status["tracked_datasets"])), 
            "Tracked Datasets", 
            "#667eea"
        ), unsafe_allow_html=True)
    
    with col3:
        remote_status = "✅" if dvc_status["remote"] else "❌"
        st.markdown(_render_metric_card(
            remote_status, 
            "Remote Configured", 
            "#28a745" if dvc_status["remote"] else "#ffc107"
        ), unsafe_allow_html=True)
    
    st.markdown("")
    
    if not dvc_status["initialized"]:
        st.warning("⚠️ DVC is not initialized in this project.")
        st.code("dvc init", language="bash")
        st.caption("Run this command to initialize DVC")
        return
    
    # Status
    st.markdown("### 📊 Sync Status")
    
    if dvc_status["status"] == "up_to_date":
        st.success("✅ All datasets are up to date with remote!")
    elif "error" in str(dvc_status["status"]).lower():
        st.error(f"❌ {dvc_status['status']}")
    else:
        st.warning(f"⚠️ Changes detected:\n{dvc_status['status']}")
    
    # Tracked datasets
    st.markdown("### 📁 Tracked Datasets")
    
    if not dvc_status["tracked_datasets"]:
        st.info("No datasets tracked yet.")
        st.code("dvc add data/your_dataset", language="bash")
        st.caption("Use this command to start tracking a dataset")
    else:
        for dataset in dvc_status["tracked_datasets"]:
            with st.expander(f"📂 {dataset['path']}", expanded=True):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**DVC File:** `{dataset['dvc_file']}`")
                
                with col2:
                    if dataset["exists"]:
                        st.markdown(_render_status_badge("success", "✓ Local"), unsafe_allow_html=True)
                    else:
                        st.markdown(_render_status_badge("warning", "⚠ Not Local"), unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"**Size:** {dataset['size_mb']:.2f} MB")
                
                if not dataset["exists"]:
                    st.code(f"dvc pull {dataset['path']}", language="bash")
                    st.caption("Run this to download the dataset")
    
    # Remote storage
    st.markdown("### 💾 Remote Storage")
    
    if dvc_status["remote"]:
        st.success(f"✅ Remote configured: `{dvc_status['remote']}`")
    else:
        st.warning("⚠️ No remote storage configured")
        st.code("dvc remote add -d myremote /path/to/storage", language="bash")
        st.caption("Configure a remote to share datasets across machines")
    
    # Actions
    st.markdown("### 🛠️ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Status", key="refresh_dvc"):
            st.rerun()
    
    with col2:
        if st.button("📥 Pull All Data", key="dvc_pull"):
            with st.spinner("Pulling data from remote..."):
                output, code = _run_command(["dvc", "pull"])
                if code == 0:
                    st.success("✅ Data pulled successfully!")
                    st.code(output or "All data up to date")
                else:
                    st.error(f"❌ Pull failed: {output}")
    
    with col3:
        if st.button("📤 Push All Data", key="dvc_push"):
            with st.spinner("Pushing data to remote..."):
                output, code = _run_command(["dvc", "push"])
                if code == 0:
                    st.success("✅ Data pushed successfully!")
                    st.code(output or "All data synced")
                else:
                    st.error(f"❌ Push failed: {output}")
    
    # Quick commands section
    st.markdown("### 📖 DVC Commands Reference")
    
    st.markdown("""
    | Command | Description |
    |---------|-------------|
    | `dvc status` | Check what's changed |
    | `dvc add <path>` | Track a new dataset |
    | `dvc pull` | Download tracked data |
    | `dvc push` | Upload data to remote |
    | `dvc repro` | Reproduce pipeline |
    | `python check_dvc_status.py` | Full status check |
    """)


def main() -> None:
    st.set_page_config(
        page_title="Face Attendance System", 
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header with animated title
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 class="animated-title" style="font-size: 2.5rem; margin: 0;">🎯 Face Attendance System</h1>
        <p style="color: #666; margin-top: 0.5rem;">Powered by FaceNet • MTCNN</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar header
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #667eea; margin: 0;">⚙️ Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Status")
    
    # MongoDB status
    if MONGODB_AVAILABLE:
        st.sidebar.markdown("✅ MongoDB: Available")
    else:
        st.sidebar.markdown("⚠️ MongoDB: Not installed")
    
    st.sidebar.markdown("---")
    
    # Main content with tabs
    tab1, tab2 = st.tabs([
        "👤 Register Face",
        "📸 Attendance"
    ])
    
    with tab1:
        _register_face_ui()
    
    with tab2:
        _attendance_ui()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        <p style="margin: 0;">Built with ❤️ using Streamlit • TensorFlow • OpenCV</p>
        <p style="margin: 0.25rem 0 0 0; font-size: 0.85rem;">
            📖 Check <a href="QUICK_VIEW_GUIDE.md">Quick View Guide</a> | 
            <a href="WHERE_TO_FIND.md">Where to Find</a> for more info
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()




