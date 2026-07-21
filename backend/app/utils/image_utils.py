"""Image loading, resizing and caching helpers."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ImageLoadError(Exception):
    """Raised when an image cannot be read or is corrupted."""


def read_image_bgr(path: Path) -> np.ndarray:
    """Read an image from disk as BGR numpy array, raising on failure."""
    if not path.exists():
        raise ImageLoadError(f"Image not found: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageLoadError(f"Corrupted or unsupported image file: {path}")
    return img


def read_image_rgb(path: Path) -> np.ndarray:
    bgr = read_image_bgr(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def get_image_dimensions(path: Path) -> tuple[int, int]:
    """Return (width, height) without loading full pixel data when possible."""
    img = read_image_bgr(path)
    h, w = img.shape[:2]
    return w, h


def downscale_if_needed(img: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float]:
    """Downscale image so max(h, w) <= max_dimension. Returns (image, scale_factor)."""
    h, w = img.shape[:2]
    largest = max(h, w)
    if largest <= max_dimension:
        return img, 1.0
    scale = max_dimension / largest
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def compute_file_hash(path: Path) -> str:
    """Compute a short hash of a file for cache invalidation."""
    stat = path.stat()
    key = f"{path.name}-{stat.st_size}-{stat.st_mtime_ns}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def encode_jpeg(img_bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ImageLoadError("Failed to encode image as JPEG")
    return buf.tobytes()


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name,
    )
