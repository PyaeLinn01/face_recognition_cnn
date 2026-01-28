"""
Ray-enhanced Face Attendance Streamlit App.

This version integrates Ray for distributed face processing:
- Distributed model inference across GPU/CPU workers
- Batch processing capabilities
- Improved concurrent user handling
- Better resource utilization
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

# Ray imports
try:
    import ray
    from ray_face_processor import FaceRecognitionService
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

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

# Configuration
IMAGES_ROOT = CURRENT_DIR / "images"
ATTENDANCE_CSV = CURRENT_DIR / "attendance.csv"
TARGET_SIZE = (96, 96)
MONGODB_COLLECTION_NAME = "face_images"
MONGODB_ATTENDANCE_COLLECTION_NAME = "attendance"

# Ray service instance (cached)
@st.cache_resource
def get_ray_service():
    """Get or create Ray FaceRecognitionService instance."""
    if not RAY_AVAILABLE:
        return None

    try:
        # Check if Ray is already initialized
        if not ray.is_initialized():
            ray_head = os.getenv("RAY_HEAD_ADDRESS", "localhost:10001")
            ray.init(address=f"ray://{ray_head}", ignore_reinit_error=True)

        service = FaceRecognitionService(
            num_embedding_actors=2,
            num_db_actors=1,
            mongo_connection=os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017/"),
            mongo_db=os.getenv("MONGODB_DATABASE_NAME", "face_attendance")
        )
        return service
    except Exception as e:
        st.error(f"Failed to initialize Ray service: {e}")
        return None


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


def _attendance_ui() -> None:
    """Ray-enhanced attendance UI."""
    st.markdown('<p class="section-header">🚀 Ray-Powered Real‑time Attendance</p>', unsafe_allow_html=True)

    # Ray status indicator
    if RAY_AVAILABLE:
        ray_service = get_ray_service()
        if ray_service:
            st.success("✅ Ray distributed processing enabled")
        else:
            st.warning("⚠️ Ray available but service not initialized")
    else:
        st.error("❌ Ray not available - install with: pip install ray[serve]")

    st.markdown("""
    <div class="info-box">
        <strong>🚀 Ray-Powered Features:</strong> Distributed face recognition with horizontal scaling,
        batch processing, and GPU acceleration across multiple workers.
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

    st.sidebar.markdown("### 🎛️ Recognition Settings")
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

    # Load reference database
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

    model = get_facenet_model()

    st.markdown("### 🎥 Live Camera Verification")

    # Choose processing mode
    use_ray = st.checkbox("🚀 Use Ray distributed processing", value=RAY_AVAILABLE, key="use_ray_processing")
    if use_ray and not RAY_AVAILABLE:
        st.error("Ray not available. Please install with: pip install ray[serve]")
        use_ray = False

    st.caption("📹 If live camera doesn't load, install dependencies and restart Streamlit.")

    try:
        from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
        import av
    except Exception:
        st.warning(
            "Live camera requires `streamlit-webrtc`. "
            "Run `pip install -r requirements.txt` and restart, or use the snapshot verifier below."
        )
        camera_image = st.camera_input("Camera – capture frame for verification")
        # ... existing snapshot processing code would go here
        return

    # Live processor with Ray acceleration
    class RayFaceAttendanceProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            self.detector = _get_mtcnn()
            self.frame_idx = 0
            self.last_best: Optional[Tuple[str, float]] = None
            self.ray_service = get_ray_service() if use_ray else None

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

            # Run recognition only every few frames for performance
            if self.frame_idx % 6 == 0:
                best_box = _best_face_box(faces, min_confidence=min_confidence)
                if best_box is not None:
                    face = _crop_with_margin_bgr(img_bgr, best_box, margin_ratio=margin_ratio)
                    face = _center_square_crop(face)
                    face_resized = cv2.resize(face, TARGET_SIZE, interpolation=cv2.INTER_AREA)

                    # Use Ray for distributed processing if available
                    if self.ray_service:
                        # Convert to bytes for Ray processing
                        _, face_bytes = cv2.imencode('.jpg', face_resized)
                        face_bytes = face_bytes.tobytes()

                        try:
                            result = self.ray_service.recognize_face(
                                face_bytes,
                                threshold=threshold,
                                use_detection=False,  # Already preprocessed
                                require_detection=False,
                                min_confidence=min_confidence,
                                margin_ratio=margin_ratio,
                                use_alignment=use_alignment,
                                use_prewhitening=use_prewhitening,
                                use_tta_flip=use_tta_flip,
                            )

                            if result["recognized"]:
                                self.last_best = (result["identity"], result["distance"])

                        except Exception as e:
                            st.error(f"Ray processing error: {e}")
                            self.fallback_recognition(face_resized)

                    else:
                        # Fallback to local processing
                        self.fallback_recognition(face_resized)

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

        def fallback_recognition(self, face_bgr: np.ndarray) -> None:
            """Fallback local face recognition."""
            try:
                q = _embedding_from_bgr_face(face_bgr, model=model, use_prewhitening=use_prewhitening)

                best_name = None
                best_dist = float("inf")
                for identity, ref_emb in database.items():
                    dist = float(np.linalg.norm(q - ref_emb))
                    if dist < best_dist:
                        best_dist = dist
                        best_name = identity

                if best_name is not None:
                    self.last_best = (best_name, float(best_dist))

            except Exception as e:
                st.error(f"Local recognition error: {e}")

    ctx = webrtc_streamer(
        key="attendance-live-ray",
        video_processor_factory=RayFaceAttendanceProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    col_a, col_b = st.columns(2)
    with col_a:
        processing_mode = "Ray distributed" if use_ray else "Local"
        st.caption(f"🎯 Processing mode: {processing_mode}")

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


def main() -> None:
    st.set_page_config(
        page_title="Ray-Powered Face Attendance System",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS (same as original)
    # ... CSS content remains the same ...

    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 class="animated-title" style="font-size: 2.5rem; margin: 0;">🚀 Ray-Powered Face Attendance System</h1>
        <p style="color: #666; margin-top: 0.5rem;">Distributed AI • FaceNet • Real-time Recognition</p>
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
    st.sidebar.markdown("### 📊 System Status")

    # Ray status
    if RAY_AVAILABLE:
        ray_service = get_ray_service()
        if ray_service:
            st.sidebar.markdown("✅ Ray: Connected")
        else:
            st.sidebar.markdown("⚠️ Ray: Available")
    else:
        st.sidebar.markdown("❌ Ray: Not installed")

    # MongoDB status
    if MONGODB_AVAILABLE:
        st.sidebar.markdown("✅ MongoDB: Available")
    else:
        st.sidebar.markdown("⚠️ MongoDB: Not installed")

    st.sidebar.markdown("---")

    # Main content - only attendance UI for now (registration can be added later)
    _attendance_ui()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        <p style="margin: 0;">Powered by Ray + FaceNet + Streamlit</p>
        <p style="margin: 0.25rem 0 0 0; font-size: 0.85rem;">
            📖 Check documentation for Ray cluster setup
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()