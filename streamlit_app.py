import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend as K

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from inception_blocks_v2 import faceRecoModel
from fr_utils import load_weights_from_FaceNet


def _preprocess_bgr_image(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        raise ValueError("Could not decode image")

    img_rgb = img_bgr[..., ::-1]
    img_rgb = cv2.resize(img_rgb, (96, 96), interpolation=cv2.INTER_AREA)
    img = np.around(img_rgb / 255.0, decimals=12)
    return img


def _embedding_from_preprocessed(img_hwc: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    x = np.array([img_hwc])
    emb = model.predict_on_batch(x)
    return emb


def embedding_from_bytes(image_bytes: bytes, model: tf.keras.Model) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = _preprocess_bgr_image(img_bgr)
    return _embedding_from_preprocessed(img, model)


def embedding_from_path(image_path: Path, model: tf.keras.Model) -> np.ndarray:
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    img = _preprocess_bgr_image(img_bgr)
    return _embedding_from_preprocessed(img, model)


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
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    model = get_facenet_model()
    db: Dict[str, np.ndarray] = {}
    paths: Dict[str, str] = {}

    for p_str, _mtime in signature:
        p = Path(p_str)
        identity = p.stem
        db[identity] = embedding_from_path(p, model)
        paths[identity] = str(p)

    return db, paths


def main() -> None:
    st.set_page_config(page_title="Face Verification (FaceNet)", layout="centered")

    st.title("Face Verification (FaceNet)")

    images_dir = CURRENT_DIR / "images"
    if not images_dir.exists():
        st.error(f"Missing images folder: {images_dir}")
        return

    image_paths = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )

    if not image_paths:
        st.warning(f"No reference images found in {images_dir}")
        return

    signature = tuple((str(p), p.stat().st_mtime) for p in image_paths)
    database, ref_paths = load_reference_database(str(images_dir), signature)
    identities = sorted(database.keys())

    mode = st.sidebar.selectbox("Mode", ["Verify", "Identify"], index=0)
    threshold = st.sidebar.slider("Threshold", min_value=0.0, max_value=2.0, value=0.7, step=0.01)

    selected_identity = None
    if mode == "Verify":
        selected_identity = st.sidebar.selectbox("Reference identity", identities)

    uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        st.image(uploaded, caption="Uploaded image", use_container_width=True)

    run = st.button("Run")

    if not run:
        return

    if uploaded is None:
        st.error("Please upload an image first.")
        return

    model = get_facenet_model()

    try:
        query_emb = embedding_from_bytes(uploaded.getvalue(), model)
    except Exception as e:
        st.error(f"Failed to process uploaded image: {e}")
        return

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
