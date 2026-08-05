"""Binary mask <-> polygon conversion using OpenCV contour extraction."""
from __future__ import annotations

import logging

import cv2
import numpy as np

from app.models.schemas import Point

logger = logging.getLogger(__name__)


MIN_CONTOUR_AREA = 4.0

# Below this, simplification is treated as "off" rather than "very light" -
# cv2.approxPolyDP with a tiny-but-nonzero epsilon still moves points.
NO_SIMPLIFY_EPSILON_RATIO = 1e-9


def mask_to_polygon(
    mask: np.ndarray,
    epsilon_ratio: float = 0.002,
    min_points: int = 3,
) -> list[Point]:
    """Convert a binary mask to a single simplified polygon (normalized 0-1 coords).

    Picks the largest external contour (handles noisy SAM output), simplifies it
    with Douglas-Peucker, and normalizes to image-relative coordinates.

    **Lossy by design, and only appropriate for blob-shaped components.** For
    thin/branching defects use `mask_to_polygons()` instead - see its docstring
    for why the two behaviours are separate functions rather than one with a
    flag defaulting to the wrong thing.
    """
    polygons = mask_to_polygons(mask, epsilon_ratio=epsilon_ratio, min_points=min_points, all_contours=False)
    return polygons[0] if polygons else []


def mask_to_polygons(
    mask: np.ndarray,
    epsilon_ratio: float = 0.002,
    min_points: int = 3,
    all_contours: bool = True,
) -> list[list[Point]]:
    """Convert a binary mask to one polygon per external contour.

    `all_contours=True` with `epsilon_ratio=0` is the **fine-structure** path,
    used for classes flagged `fine_structure` on `dataset_classes` (crack,
    corrosion, wheel shelling). Both parts matter, and both were previously
    impossible:

    - **Keeping every contour.** The single-polygon path takes
      `max(contours, key=cv2.contourArea)`, so a crack that branches, or breaks
      into several collinear segments - which is what real cracks do - lost
      every piece but the biggest. The pipeline scores crack/corrosion on
      Dice/IoU **plus length-recall** (docs/pipeline.md §5.4), and dropping
      segments hits length-recall directly and silently.
    - **Not simplifying.** Douglas-Peucker's epsilon is a fraction of the
      contour *perimeter*. A hairline crack has a long perimeter relative to
      its width, so the epsilon that flatters a coupler outline erases the
      structure being measured.

    The morphological close is also skipped on the fine-structure path: a 3x3
    close can bridge two nearby crack segments into one, or fill the gap that
    made them two, which changes measured length.
    """
    h, w = mask.shape[:2]
    mask_u8 = (mask.astype(np.uint8)) * 255

    simplify = epsilon_ratio > NO_SIMPLIFY_EPSILON_RATIO
    if simplify:
        # Morphological close to remove small holes/speckles before contouring.
        # Deliberately not applied when preserving fine structure - see above.
        kernel = np.ones((3, 3), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    selected = contours if all_contours else [max(contours, key=cv2.contourArea)]

    scored: list[tuple[float, list[Point]]] = []
    for contour in selected:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        if simplify:
            perimeter = cv2.arcLength(contour, True)
            epsilon = max(epsilon_ratio * perimeter, 0.5)
            points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(points) < min_points:
                # Fall back to convex hull of the raw contour if simplification collapsed it.
                points = cv2.convexHull(contour).reshape(-1, 2)
        else:
            # Raw contour points, as found. No hull fallback either: a convex
            # hull of a crack is a blob, which is worse than dropping a
            # degenerate contour.
            points = contour.reshape(-1, 2)

        if len(points) < min_points:
            continue

        scored.append((area, [Point(x=float(px) / w, y=float(py) / h) for px, py in points]))

    # Largest-area first, so a consumer taking only the first gets the dominant
    # piece - point count would rank a long thin contour above a bigger blob.
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [polygon for _area, polygon in scored]


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
