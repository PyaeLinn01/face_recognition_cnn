import csv
import os
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
    st.subheader("Register / Enroll Face")
    st.markdown(
        "Capture **4 images** of the same person from slightly different angles.\n"
        "They will be saved to MongoDB (and optionally to filesystem) and used as reference anchors."
    )

    # MongoDB settings
    st.sidebar.markdown("### MongoDB Settings")
    use_mongodb = st.sidebar.checkbox("Use MongoDB", value=True)
    
    # MLflow tracking option
    st.sidebar.markdown("### MLflow Tracking")
    use_mlflow = st.sidebar.checkbox("Enable MLflow tracking", value=False, disabled=not MLFLOW_AVAILABLE, key="register_mlflow")
    if not MLFLOW_AVAILABLE:
        st.sidebar.warning("MLflow not available. Install: pip install mlflow")
    elif use_mlflow:
        st.sidebar.info("📊 Models logged to MLflow")
    
    mongodb_connection_string = None
    mongodb_database_name = None
    
    if use_mongodb:
        if not MONGODB_AVAILABLE:
            st.sidebar.error("pymongo not installed. Run: pip install pymongo")
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
                help="e.g., mongodb://localhost:27017/ or mongodb+srv://user:pass@cluster.mongodb.net/",
            )
            mongodb_database_name = st.sidebar.text_input(
                "Database Name",
                value=default_db,
                help="Database name to store face images",
            )
            
            if mongodb_connection_string and mongodb_database_name:
                try:
                    _get_mongodb_client(mongodb_connection_string)
                    st.sidebar.success("✓ MongoDB connected")
                except Exception as e:
                    st.sidebar.error(f"MongoDB connection failed: {e}")

    # Also save to filesystem option
    save_to_filesystem = st.sidebar.checkbox("Also save to filesystem", value=False)

    name = st.text_input("Person name (used as folder/collection key)", "")
    # Fixed to exactly 4 snapshots
    num_images = 4

    if not name:
        st.info("Enter a name to start capturing.")

    camera_image = st.camera_input("Capture face image")

    if "captured_count" not in st.session_state:
        st.session_state.captured_count = 0

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Captured for this session", st.session_state.captured_count)
    with col2:
        st.metric("Target snapshots", int(num_images))

    # Disable saving when we already have 4 snapshots
    save_disabled = camera_image is None or not name or st.session_state.captured_count >= num_images

    if st.button("Save snapshot", disabled=save_disabled):
        if camera_image is None:
            st.error("No image captured from camera.")
            return
        if not name:
            st.error("Please enter a name before saving.")
            return

        count = st.session_state.captured_count
        save_path = None
        if save_to_filesystem:
            identity_dir = IMAGES_ROOT / name
            _ensure_dir(identity_dir)
            filename = f"{name}_{count + 1}.jpg"
            save_path = identity_dir / filename

        try:
            _save_camera_image(
                camera_image.getvalue(),
                save_path=save_path,
                mongodb_connection_string=mongodb_connection_string if use_mongodb else None,
                mongodb_database_name=mongodb_database_name if use_mongodb else None,
                name=name if use_mongodb else None,
                image_index=count + 1 if use_mongodb else None,
            )
        except Exception as e:
            st.error(f"Failed to save snapshot: {e}")
            return

        st.session_state.captured_count = count + 1
        
        # Log to MLflow if enabled (only on first image)
        if use_mlflow and MLFLOW_AVAILABLE and count == 0:
            try:
                with start_verification_run(run_name=f"register_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
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
            except Exception as e:
                st.sidebar.warning(f"MLflow logging error: {e}")
        
        success_msg = f"Saved snapshot #{st.session_state.captured_count}"
        if save_to_filesystem and save_path:
            success_msg += f" to `{save_path}`"
        if use_mongodb and mongodb_connection_string:
            success_msg += f" and MongoDB ({mongodb_database_name})"
        st.success(success_msg)

        if st.session_state.captured_count >= num_images:
            st.balloons()
            st.info(
                f"Captured {st.session_state.captured_count} snapshots for **{name}**. "
                "You can switch to the *Attendance* tab to start verification."
            )


def _attendance_ui() -> None:
    st.subheader("Real‑time Attendance (Camera Verification)")

    st.markdown(
        "This mode uses the camera to verify a face against all registered identities "
        "from MongoDB (or filesystem) and records attendance."
    )

    # MongoDB settings
    st.sidebar.markdown("### Data Source")
    use_mongodb = st.sidebar.checkbox("Use MongoDB", value=True)
    
    mongodb_connection_string = None
    mongodb_database_name = None
    
    if use_mongodb:
        if not MONGODB_AVAILABLE:
            st.sidebar.error("pymongo not installed. Run: pip install pymongo")
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
    st.sidebar.markdown("### Attendance Storage")
    save_attendance_to_mongodb = st.sidebar.checkbox(
        "Save attendance to MongoDB",
        value=True,
        disabled=not (use_mongodb and mongodb_connection_string and mongodb_database_name),
    )
    also_save_csv = st.sidebar.checkbox("Also save to CSV", value=False)

    st.sidebar.markdown("### Verification Settings")
    threshold = st.sidebar.slider("Match threshold", 0.0, 2.0, 0.7, 0.01)
    use_detection = st.sidebar.checkbox("Use MTCNN face detection", value=True)
    require_detection = st.sidebar.checkbox("Require face detection", value=False)
    min_confidence = st.sidebar.slider("Min face confidence", 0.0, 1.0, 0.90, 0.01)
    margin_ratio = st.sidebar.slider("Face crop margin", 0.0, 0.6, 0.2, 0.05)
    use_alignment = st.sidebar.checkbox("Align face (eyes/nose)", value=True)
    use_prewhitening = st.sidebar.checkbox("Prewhiten (normalize)", value=False)
    use_tta_flip = st.sidebar.checkbox("Flip TTA (average)", value=True)
    
    # MLflow tracking option
    st.sidebar.markdown("### MLflow Tracking")
    use_mlflow = st.sidebar.checkbox("Enable MLflow tracking", value=False, disabled=not MLFLOW_AVAILABLE)
    if not MLFLOW_AVAILABLE:
        st.sidebar.warning("MLflow not available. Install: pip install mlflow")
    elif use_mlflow:
        st.sidebar.info("📊 View models: `mlflow ui --port 5000`")
        try:
            import mlflow
            tracking_uri = mlflow.get_tracking_uri()
            if tracking_uri.startswith("file:"):
                runs_dir = tracking_uri.replace("file:", "")
                if Path(runs_dir).exists():
                    num_runs = len(list(Path(runs_dir).rglob("meta.yaml")))
                    st.sidebar.caption(f"✓ {num_runs} runs tracked")
        except Exception:
            pass

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

    entered_name = st.text_input("Person name (optional, for logging only)", "")

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

    st.markdown("### Live camera (rounded detection box)")
    st.caption("If live camera doesn’t load, install dependencies and restart Streamlit.")

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
                log_name = entered_name.strip() or best_name
                _append_attendance_row(
                    log_name,
                    best_name,
                    best_dist,
                    use_mongodb=save_attendance_to_mongodb and mongodb_connection_string and mongodb_database_name,
                    mongodb_connection_string=mongodb_connection_string if save_attendance_to_mongodb else None,
                    mongodb_database_name=mongodb_database_name if save_attendance_to_mongodb else None,
                    also_save_csv=also_save_csv,
                )
                success_msg = f"Attendance recorded for **{log_name}**"
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
                        log_name = entered_name.strip() or best_name
                        
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
                            log_name,
                            best_name,
                            float(best_dist),
                            use_mongodb=save_attendance_to_mongodb and mongodb_connection_string and mongodb_database_name,
                            mongodb_connection_string=mongodb_connection_string if save_attendance_to_mongodb else None,
                            mongodb_database_name=mongodb_database_name if save_attendance_to_mongodb else None,
                            also_save_csv=also_save_csv,
                        )
                        success_msg = f"Attendance recorded for **{log_name}**"
                        if save_attendance_to_mongodb:
                            success_msg += " (saved to MongoDB)"
                        if also_save_csv:
                            success_msg += " (saved to CSV)"
                        st.success(success_msg)
                    else:
                        st.warning("Current live frame is not a MATCH. Move closer / adjust lighting.")

    # MLflow & DVC Status Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 View Results")
    
    st.sidebar.markdown("**MLflow Models:**")
    st.sidebar.code("mlflow ui --port 5000\n# Then open: http://localhost:5000")
    st.sidebar.markdown("📍 **FaceNet**: Run → Artifacts → `facenet/`")
    st.sidebar.markdown("📍 **MTCNN**: Run → Parameters → `mtcnn_*`")
    
    st.sidebar.markdown("**DVC Dataset:**")
    st.sidebar.code("python check_dvc_status.py\n# Or: dvc status")
    st.sidebar.markdown("📍 **Dataset**: `data/mongodb_export/`")
    
    st.sidebar.markdown("**📖 Guides:**")
    st.sidebar.markdown("- [Quick View Guide](QUICK_VIEW_GUIDE.md)")
    st.sidebar.markdown("- [Where to Find](WHERE_TO_FIND.md)")

    rows = _load_attendance_rows(
        max_rows=200,
        use_mongodb=use_mongodb and mongodb_connection_string and mongodb_database_name,
        mongodb_connection_string=mongodb_connection_string,
        mongodb_database_name=mongodb_database_name,
    )
    if rows:
        st.markdown("### Recent Attendance")
        import pandas as pd  # Local import to keep top imports minimal

        df = pd.DataFrame(rows[1:], columns=rows[0])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No attendance records yet.")


def main() -> None:
    st.set_page_config(page_title="Face Attendance (FaceNet)", layout="centered")
    st.title("Face Attendance System (FaceNet)")

    tab = st.sidebar.radio("Mode", ("Register face", "Attendance"), index=0)

    if tab == "Register face":
        _register_face_ui()
    else:
        _attendance_ui()


if __name__ == "__main__":
    main()

