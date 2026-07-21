"""Binary mask <-> polygon conversion using OpenCV contour extraction."""
from __future__ import annotations

import logging

import cv2
import numpy as np

from app.models.schemas import Point

logger = logging.getLogger(__name__)


def mask_to_polygon(
    mask: np.ndarray,
    epsilon_ratio: float = 0.002,
    min_points: int = 3,
) -> list[Point]:
    """Convert a binary mask to a single simplified polygon (normalized 0-1 coords).

    Picks the largest external contour (handles noisy SAM output), simplifies it
    with Douglas-Peucker, and normalizes to image-relative coordinates.
    """
    h, w = mask.shape[:2]
    mask_u8 = (mask.astype(np.uint8)) * 255

    # Morphological close to remove small holes/speckles before contouring.
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 4:
        return []

    perimeter = cv2.arcLength(largest, True)
    epsilon = max(epsilon_ratio * perimeter, 0.5)
    simplified = cv2.approxPolyDP(largest, epsilon, True)

    points = simplified.reshape(-1, 2)
    if len(points) < min_points:
        # Fall back to convex hull of the raw contour if simplification collapsed it.
        hull = cv2.convexHull(largest).reshape(-1, 2)
        points = hull

    if len(points) < min_points:
        return []

    return [Point(x=float(px) / w, y=float(py) / h) for px, py in points]


def polygon_to_mask(polygon: list[Point], width: int, height: int) -> np.ndarray:
    """Rasterize a normalized polygon into a binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(polygon) < 3:
        return mask.astype(bool)
    pts = np.array([[int(p.x * width), int(p.y * height)] for p in polygon], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def simplify_polygon(polygon: list[Point], width: int, height: int, epsilon_ratio: float = 0.002) -> list[Point]:
    """Re-simplify an existing (possibly hand-edited) polygon."""
    if len(polygon) < 4:
        return polygon
    pts = np.array([[p.x * width, p.y * height] for p in polygon], dtype=np.float32).reshape(-1, 1, 2)
    perimeter = cv2.arcLength(pts, True)
    epsilon = max(epsilon_ratio * perimeter, 0.5)
    simplified = cv2.approxPolyDP(pts, epsilon, True).reshape(-1, 2)
    if len(simplified) < 3:
        return polygon
    return [Point(x=float(px) / width, y=float(py) / height) for px, py in simplified]


def clip_polygon_to_image(polygon: list[Point]) -> list[Point]:
    """Clamp all polygon points into the [0, 1] normalized image bounds."""
    return [Point(x=min(1.0, max(0.0, p.x)), y=min(1.0, max(0.0, p.y))) for p in polygon]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0
