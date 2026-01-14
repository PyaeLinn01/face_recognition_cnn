import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import streamlit as st

from app import (
    CURRENT_DIR,
    embedding_from_bytes,
    get_facenet_model,
    load_reference_database,
)


IMAGES_ROOT = CURRENT_DIR / "images"
ATTENDANCE_CSV = CURRENT_DIR / "attendance.csv"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_camera_image(image_bytes: bytes, save_path: Path) -> None:
    """Decode, detect face with MTCNN, crop, resize to 96x96, then save."""
    from mtcnn.mtcnn import MTCNN  # local import to keep startup light

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode camera image")

    # Face detection (similar to resize_images.py)
    detector = MTCNN()
    img_rgb = img_bgr[..., ::-1]
    faces = detector.detect_faces(img_rgb)

    # Pick best face by confidence * area
    best = None
    best_score = -1.0
    for f in faces:
        conf = float(f.get("confidence", 0.0))
        x, y, w, h = f.get("box", (0, 0, 0, 0))
        area = float(max(0, int(w)) * max(0, int(h)))
        score = conf * area
        if score > best_score:
            best_score = score
            best = (int(x), int(y), int(w), int(h))

    crop = img_bgr
    margin_ratio = 0.20
    if best is not None:
        h_img, w_img = img_bgr.shape[:2]
        x, y, w, h = best
        mx = int(w * margin_ratio)
        my = int(h * margin_ratio)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w_img, x + w + mx)
        y2 = min(h_img, y + h + my)
        if x2 > x1 and y2 > y1:
            crop = img_bgr[y1:y2, x1:x2]

    # Center square crop
    h, w = crop.shape[:2]
    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    crop = crop[y1 : y1 + side, x1 : x1 + side]

    # Resize to 96x96
    crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)

    _ensure_dir(save_path.parent)
    cv2.imwrite(str(save_path), crop)


def _append_attendance_row(name: str, identity: str, distance: float) -> None:
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


def _load_attendance_rows(max_rows: int = 200) -> List[List[str]]:
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
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
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
        "They will be saved under `images/&lt;name&gt;/` and used as reference anchors."
    )

    name = st.text_input("Person name (used as folder name)", "")
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

        identity_dir = IMAGES_ROOT / name
        _ensure_dir(identity_dir)

        count = st.session_state.captured_count
        filename = f"{name}_{count + 1}.jpg"
        save_path = identity_dir / filename
        try:
            _save_camera_image(camera_image.getvalue(), save_path)
        except Exception as e:
            st.error(f"Failed to save snapshot: {e}")
            return

        st.session_state.captured_count = count + 1
        st.success(f"Saved snapshot #{st.session_state.captured_count} to `{save_path}`")

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
        "in the `images/` folder and records attendance."
    )

    threshold = st.sidebar.slider("Match threshold", 0.0, 2.0, 0.7, 0.01)
    use_detection = st.sidebar.checkbox("Use MTCNN face detection", value=True)
    require_detection = st.sidebar.checkbox("Require face detection", value=False)
    min_confidence = st.sidebar.slider("Min face confidence", 0.0, 1.0, 0.90, 0.01)
    margin_ratio = st.sidebar.slider("Face crop margin", 0.0, 0.6, 0.2, 0.05)
    use_alignment = st.sidebar.checkbox("Align face (eyes/nose)", value=True)
    use_prewhitening = st.sidebar.checkbox("Prewhiten (normalize)", value=False)
    use_tta_flip = st.sidebar.checkbox("Flip TTA (average)", value=True)

    try:
        database, ref_paths = _build_reference_db(
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

    if not database:
        st.error("No reference identities available. Please register at least one person first.")
        return

    entered_name = st.text_input("Person name (optional, for logging only)", "")

    camera_image = st.camera_input("Camera – capture frame for verification")

    model = get_facenet_model()

    if st.button("Verify & Record Attendance", disabled=(camera_image is None)):
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

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Best match")
            st.image(ref_paths[best_name], caption=best_name, use_container_width=True)
        with col2:
            st.subheader("Result")
            st.write(
                {
                    "best_identity": best_name,
                    "distance": best_dist,
                    "threshold": threshold,
                    "match": is_match,
                }
            )

        if is_match:
            log_name = entered_name.strip() or best_name
            _append_attendance_row(log_name, best_name, best_dist)
            st.success(f"Attendance recorded for **{log_name}**.")
        else:
            st.warning("Face did not match any registered identity with the given threshold.")

    rows = _load_attendance_rows()
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

