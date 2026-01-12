#!/usr/bin/env python3
"""
Resize all images in the 'images/' folder to 96x96 (square) and overwrite them.
This script ensures the reference database uses consistent 96x96 images.
"""

import os
from pathlib import Path

import cv2

TARGET_SIZE = (96, 96)
IMAGES_DIR = Path(__file__).resolve().parent / "images"


def resize_image(in_path: Path, out_path: Path) -> bool:
    """Resize a single image to 96x96, preserving aspect ratio with letterboxing if needed."""
    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to read {in_path}")
        return False

    h, w = img.shape[:2]
    # Compute scaling to fit within 96x96 while preserving aspect ratio
    scale = min(TARGET_SIZE[0] / w, TARGET_SIZE[1] / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Letterbox to exactly 96x96 (center the image)
    top = (TARGET_SIZE[1] - new_h) // 2
    bottom = TARGET_SIZE[1] - new_h - top
    left = (TARGET_SIZE[0] - new_w) // 2
    right = TARGET_SIZE[0] - new_w - left

    letterboxed = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # black padding
    )

    cv2.imwrite(str(out_path), letterboxed)
    print(f"Resized {in_path.name} -> {out_path.name}")
    return True


def main() -> None:
    if not IMAGES_DIR.is_dir():
        print(f"Images folder not found: {IMAGES_DIR}")
        return

    image_files = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png",".avif"}]
    if not image_files:
        print("No images found to resize.")
        return

    for p in image_files:
        resize_image(p, p)

    print(f"\nResized {len(image_files)} images to {TARGET_SIZE} in {IMAGES_DIR}")


if __name__ == "__main__":
    main()
