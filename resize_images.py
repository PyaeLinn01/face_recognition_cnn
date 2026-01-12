#!/usr/bin/env python3
"""
Resize all images in the 'images/' folder to 96x96 (square) and overwrite them.
This script ensures the reference database uses consistent 96x96 images.
"""

import argparse
import os
from typing import List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

TARGET_SIZE = (96, 96)
IMAGES_DIR = Path(__file__).resolve().parent / "images"


def _get_detector() -> MTCNN:
    return MTCNN()


def _best_face_box(faces: List[dict]) -> Optional[Tuple[int, int, int, int]]:
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
            best = (int(x), int(y), int(w), int(h))

    return best


def _crop_with_margin(img: np.ndarray, box: Tuple[int, int, int, int], margin_ratio: float) -> np.ndarray:
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
    h, w = img.shape[:2]
    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    return img[y1 : y1 + side, x1 : x1 + side]


def _resize_to_target(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    th, tw = target_size[1], target_size[0]
    h, w = img.shape[:2]
    if (w, h) == (tw, th):
        return img

    interpolation = cv2.INTER_AREA if (w > tw or h > th) else cv2.INTER_CUBIC
    return cv2.resize(img, (tw, th), interpolation=interpolation)


def resize_image(
    in_path: Path,
    out_path: Path,
    detector: Optional[MTCNN],
    use_detection: bool,
    margin_ratio: float,
) -> bool:
    """Resize a single image to 96x96 and overwrite it."""
    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to read {in_path}")
        return False

    crop = img
    if use_detection and detector is not None:
        img_rgb = img[..., ::-1]
        faces = detector.detect_faces(img_rgb)
        box = _best_face_box(faces)
        if box is not None:
            crop = _crop_with_margin(img, box, margin_ratio)

    crop = _center_square_crop(crop)
    resized = _resize_to_target(crop, TARGET_SIZE)

    cv2.imwrite(str(out_path), resized)
    print(f"Resized {in_path.name} -> {out_path.name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", nargs="?", default=str(IMAGES_DIR), help="Folder containing images to resize")
    parser.add_argument("--no-detect", action="store_true", help="Disable MTCNN face detection before resizing")
    parser.add_argument("--margin", type=float, default=0.20, help="Margin ratio around detected face")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"Images folder not found: {input_dir}")
        return

    exts = {".jpg", ".jpeg", ".png", ".avif"}
    image_files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not image_files:
        print("No images found to resize.")
        return

    detector: Optional[MTCNN] = None
    use_detection = not bool(args.no_detect)
    if use_detection:
        detector = _get_detector()

    for p in image_files:
        resize_image(p, p, detector=detector, use_detection=use_detection, margin_ratio=float(args.margin))

    print(f"\nResized {len(image_files)} images to {TARGET_SIZE} in {input_dir}")


if __name__ == "__main__":
    main()
