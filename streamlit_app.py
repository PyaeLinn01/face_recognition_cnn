import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend as K
from mtcnn.mtcnn import MTCNN

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from inception_blocks_v2 import faceRecoModel
from fr_utils import load_weights_from_FaceNet


def _best_face(faces: List[Dict[str, Any]], min_confidence: float) -> Optional[Dict[str, Any]]:
    if not faces:
        return None

    best = None
    best_score = -1.0
    for f in faces:
        conf = float(f.get("confidence", 0.0))
        x, y, w, h = f.get("box", (0, 0, 0, 0))
        area = float(max(0, int(w)) * max(0, int(h)))
        score = conf * area
        if score > best_score:
            best_score = score
            best = f

    if best is None:
        return None
    if float(best.get("confidence", 0.0)) < min_confidence:
        return None
    return best


def _crop_with_margin(img_rgb: np.ndarray, box: Tuple[int, int, int, int], margin_ratio: float) -> np.ndarray:
    h_img, w_img = img_rgb.shape[:2]
    x, y, w, h = box

    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w_img, x + w + mx)
    y2 = min(h_img, y + h + my)

    if x2 <= x1 or y2 <= y1:
        return img_rgb

    return img_rgb[y1:y2, x1:x2]


def _prewhiten(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    mean = np.mean(x)
    std = np.std(x)
    std_adj = max(std, 1.0 / np.sqrt(x.size))
    y = (x - mean) / std_adj
    return y


def _align_face_by_keypoints(img_rgb: np.ndarray, keypoints: Dict[str, Any], output_size: int = 96) -> Optional[np.ndarray]:
    try:
        le = keypoints["left_eye"]
        re_ = keypoints["right_eye"]
        nose = keypoints["nose"]
    except Exception:
        return None

    src = np.array([le, re_, nose], dtype=np.float32)

    w = float(output_size)
    h = float(output_size)
    dst = np.array(
        [
            [0.35 * w, 0.35 * h],
            [0.65 * w, 0.35 * h],
            [0.50 * w, 0.55 * h],
        ],
        dtype=np.float32,
    )

    M, _inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return None

    aligned = cv2.warpAffine(
        img_rgb,
        M,
        (output_size, output_size),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned


@st.cache_resource
def get_mtcnn_detector() -> MTCNN:
    return MTCNN()


def _preprocess_bgr_image(
    img_bgr: np.ndarray,
    use_detection: bool,
    require_detection: bool,
    min_confidence: float,
    margin_ratio: float,
    use_alignment: bool,
    use_prewhitening: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if img_bgr is None:
        raise ValueError("Could not decode image")

    img_rgb = img_bgr[..., ::-1]
    preview_rgb = img_rgb

    if use_detection:
        detector = get_mtcnn_detector()
        faces = detector.detect_faces(img_rgb)
        best = _best_face(faces, min_confidence=min_confidence)
        if best is None:
            if require_detection:
                raise ValueError("No face detected (or confidence below threshold)")
        else:
            x, y, w, h = best.get("box", (0, 0, 0, 0))
            box = (int(x), int(y), int(w), int(h))
            preview_rgb = _crop_with_margin(img_rgb, box, margin_ratio)
            if use_alignment:
                aligned = _align_face_by_keypoints(img_rgb, best.get("keypoints", {}), output_size=96)
                if aligned is not None:
                    preview_rgb = aligned

    resized = cv2.resize(preview_rgb, (96, 96), interpolation=cv2.INTER_AREA)
    img = resized.astype(np.float32) / 255.0
    if use_prewhitening:
        img = _prewhiten(img)
    img = np.around(img, decimals=12)
    return img, preview_rgb


def _embedding_from_preprocessed(
    img_hwc: np.ndarray,
    model: tf.keras.Model,
    use_tta_flip: bool,
) -> np.ndarray:
    x = np.array([img_hwc])
    emb = model.predict_on_batch(x)[0]

    if use_tta_flip:
        img_flip = np.ascontiguousarray(img_hwc[:, ::-1, :])
        x2 = np.array([img_flip])
        emb2 = model.predict_on_batch(x2)[0]
        emb = (emb + emb2) / 2.0
        emb = emb / max(np.linalg.norm(emb), 1e-12)

    return np.expand_dims(emb, axis=0)


def embedding_from_bytes(
    image_bytes: bytes,
    model: tf.keras.Model,
    use_detection: bool,
    require_detection: bool,
    min_confidence: float,
    margin_ratio: float,
    use_alignment: bool,
    use_prewhitening: bool,
    use_tta_flip: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img, preview_rgb = _preprocess_bgr_image(
        img_bgr,
        use_detection=use_detection,
        require_detection=require_detection,
        min_confidence=min_confidence,
        margin_ratio=margin_ratio,
        use_alignment=use_alignment,
        use_prewhitening=use_prewhitening,
    )
    return _embedding_from_preprocessed(img, model, use_tta_flip=use_tta_flip), preview_rgb


def embedding_from_path(
    image_path: Path,
    model: tf.keras.Model,
    use_detection: bool,
    require_detection: bool,
    min_confidence: float,
    margin_ratio: float,
    use_alignment: bool,
    use_prewhitening: bool,
    use_tta_flip: bool,
) -> np.ndarray:
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    img, _preview_rgb = _preprocess_bgr_image(
        img_bgr,
        use_detection=use_detection,
        require_detection=require_detection,
        min_confidence=min_confidence,
        margin_ratio=margin_ratio,
        use_alignment=use_alignment,
        use_prewhitening=use_prewhitening,
    )
    return _embedding_from_preprocessed(img, model, use_tta_flip=use_tta_flip)


def _identity_from_path(images_root: Path, image_path: Path) -> str:
    try:
        rel = image_path.relative_to(images_root)
    except Exception:
        rel = image_path

    if len(rel.parts) >= 2:
        return rel.parts[0]

    stem = image_path.stem
    stem = re.sub(r"[_-]\d+$", "", stem)
    return stem


@st.cache_resource
def get_facenet_model() -> tf.keras.Model:
    K.set_image_data_format("channels_last")
    model = faceRecoModel(input_shape=(96, 96, 3))
    load_weights_from_FaceNet(model)
    return model


@st.cache_data
def load_reference_database(
    images_dir: str,
    signature: Tuple[Tuple[str, float], ...],
    use_detection: bool,
    require_detection: bool,
    min_confidence: float,
    margin_ratio: float,
    use_alignment: bool,
    use_prewhitening: bool,
    use_tta_flip: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    model = get_facenet_model()
    db: Dict[str, np.ndarray] = {}
    paths: Dict[str, str] = {}
    per_identity: Dict[str, List[np.ndarray]] = {}
    per_identity_path: Dict[str, str] = {}

    images_root = Path(images_dir)

    for p_str, _mtime in signature:
        p = Path(p_str)
        identity = _identity_from_path(images_root, p)
        emb = embedding_from_path(
            p,
            model,
            use_detection=use_detection,
            require_detection=require_detection,
            min_confidence=min_confidence,
            margin_ratio=margin_ratio,
            use_alignment=use_alignment,
            use_prewhitening=use_prewhitening,
            use_tta_flip=use_tta_flip,
        )
        per_identity.setdefault(identity, []).append(emb)
        per_identity_path.setdefault(identity, str(p))

    for identity, embs in per_identity.items():
        mean_emb = np.mean(np.concatenate(embs, axis=0), axis=0)
        mean_emb = mean_emb / max(np.linalg.norm(mean_emb), 1e-12)
        db[identity] = np.expand_dims(mean_emb, axis=0)
        paths[identity] = per_identity_path[identity]

    return db, paths


def main() -> None:
    st.set_page_config(page_title="Face Verification (FaceNet)", layout="centered")

    st.title("Face Verification (FaceNet)")

    images_dir = CURRENT_DIR / "images"
    if not images_dir.exists():
        st.error(f"Missing images folder: {images_dir}")
        return

    image_paths = sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    if not image_paths:
        st.warning(f"No reference images found in {images_dir}")
        return

    mode = st.sidebar.selectbox("Mode", ["Verify", "Identify"], index=0)
    threshold = st.sidebar.slider("Threshold", min_value=0.0, max_value=2.0, value=0.7, step=0.01)
    use_detection = st.sidebar.checkbox("Use MTCNN face detection", value=True)
    require_detection = st.sidebar.checkbox("Require face detection", value=False)
    min_confidence = st.sidebar.slider("Min face confidence", min_value=0.0, max_value=1.0, value=0.90, step=0.01)
    margin_ratio = st.sidebar.slider("Face crop margin", min_value=0.0, max_value=0.6, value=0.2, step=0.05)
    use_alignment = st.sidebar.checkbox("Align face (eyes/nose)", value=True)
    use_prewhitening = st.sidebar.checkbox("Prewhiten (normalize)", value=False)
    use_tta_flip = st.sidebar.checkbox("Flip TTA (average)", value=True)

    signature = tuple((str(p), p.stat().st_mtime) for p in image_paths)
    try:
        database, ref_paths = load_reference_database(
            str(images_dir),
            signature,
            use_detection=use_detection,
            require_detection=require_detection,
            min_confidence=min_confidence,
            margin_ratio=margin_ratio,
            use_alignment=use_alignment,
            use_prewhitening=use_prewhitening,
            use_tta_flip=use_tta_flip,
        )
    except Exception as e:
        st.error(f"Failed to build reference database: {e}")
        return

    identities = sorted(database.keys())

    selected_identity = None
    if mode == "Verify":
        selected_identity = st.sidebar.selectbox("Reference identity", identities)

    uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.error("Please upload an image first.")
        return

    model = get_facenet_model()

    try:
        query_emb, cropped_rgb = embedding_from_bytes(
            uploaded.getvalue(),
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
        st.error(f"Failed to process uploaded image: {e}")
        return

    if use_detection:
        st.image(cropped_rgb, caption="Detected/cropped face", use_container_width=True)

    if mode == "Verify":
        if selected_identity is None:
            st.error("Please select an identity.")
            return

        ref_emb = database[selected_identity]
        dist = float(np.linalg.norm(query_emb - ref_emb))
        is_match = dist < threshold

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Reference")
            st.image(ref_paths[selected_identity], caption=selected_identity, use_container_width=True)
        with col2:
            st.subheader("Result")
            st.write({"identity": selected_identity, "distance": dist, "threshold": threshold, "match": is_match})

    else:
        best_name = None
        best_dist = float("inf")

        for name, ref_emb in database.items():
            dist = float(np.linalg.norm(query_emb - ref_emb))
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_name is None:
            st.error("No reference images available.")
            return

        is_match = best_dist < threshold

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Best match")
            st.image(ref_paths[best_name], caption=best_name, use_container_width=True)
        with col2:
            st.subheader("Result")
            st.write({"best_identity": best_name, "distance": best_dist, "threshold": threshold, "match": is_match})


if __name__ == "__main__":
    main()
