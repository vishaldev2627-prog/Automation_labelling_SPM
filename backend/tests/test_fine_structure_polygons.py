"""Tests for the fine-structure mask path (W-3 / N-3).

What was wrong: `mask_to_polygon` kept only `max(contours, key=cv2.contourArea)`
and simplified with Douglas-Peucker at `epsilon = 0.002 * perimeter`. For a
crack that branches or breaks into segments, every piece but the biggest was
discarded, and the surviving one was smoothed. The pipeline scores crack and
corrosion on Dice/IoU **plus length-recall** (docs/pipeline.md §5.4) and wheel
shelling on Dice plus shelling length (§5.5) - so the loss landed precisely on
the measured quantity, silently.

`underbelly` being area-cam footage (confirmed 2026-08-05) makes this
load-bearing for two families, not one: `p2_under_crackseg` as well as
`p3_wheel_shelling`.

Pure geometry, no database and no SAM2.

    cd backend && python -m pytest tests/test_fine_structure_polygons.py
    cd backend && python -m tests.test_fine_structure_polygons   # no pytest
"""
from __future__ import annotations

import numpy as np

from app.services.polygon_service import mask_to_polygon, mask_to_polygons, polygon_to_mask

H, W = 200, 200


def _two_segment_crack() -> np.ndarray:
    """Two disjoint thin strips - a crack broken into segments, which is what
    real cracks look like. The second is deliberately smaller so the old
    largest-contour-only rule would drop it."""
    mask = np.zeros((H, W), dtype=bool)
    mask[40:44, 20:120] = True  # longer segment
    mask[40:44, 140:180] = True  # shorter segment, separated by a gap
    return mask


def _single_blob() -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[50:150, 50:150] = True
    return mask


# ------------------------------------- the regression: dropped crack segments

def test_single_polygon_path_still_drops_the_smaller_segment() -> None:
    """Documents the old behaviour, which is still correct for blob-shaped
    components and is why the two paths are separate functions."""
    polygon = mask_to_polygon(_two_segment_crack())
    assert polygon  # got something
    mask = polygon_to_mask(polygon, W, H)
    # The far segment (x >= 140) is absent from the reconstruction.
    assert not mask[40:44, 145:175].any()


def test_all_contours_path_keeps_both_segments() -> None:
    polygons = mask_to_polygons(_two_segment_crack(), epsilon_ratio=0.0, all_contours=True)
    assert len(polygons) == 2

    union = np.zeros((H, W), dtype=bool)
    for piece in polygons:
        union |= polygon_to_mask(piece, W, H)
    # Both segments present.
    assert union[41, 60]
    assert union[41, 160]


def test_fine_structure_path_recovers_far_more_of_the_mask() -> None:
    """The end-to-end claim: for a segmented thin defect, all-contours plus
    no-simplification reconstructs substantially more of the original mask than
    the default path."""
    original = _two_segment_crack()

    lossy = polygon_to_mask(mask_to_polygon(original), W, H)

    faithful = np.zeros((H, W), dtype=bool)
    for piece in mask_to_polygons(original, epsilon_ratio=0.0, all_contours=True):
        faithful |= polygon_to_mask(piece, W, H)

    def recall(reconstructed: np.ndarray) -> float:
        return float(np.logical_and(original, reconstructed).sum() / original.sum())

    assert recall(faithful) > recall(lossy)
    assert recall(faithful) > 0.9


# ------------------------------------------------- simplification is really off

def _comb() -> np.ndarray:
    """A bar with fine teeth - stands in for the small-scale detail a thin
    defect's outline carries. Teeth are 1px wide, well under the epsilon
    Douglas-Peucker derives from this shape's long perimeter, so simplification
    genuinely erases them."""
    mask = np.zeros((H, W), dtype=bool)
    mask[100:110, 20:180] = True
    for x in range(20, 180, 3):
        mask[90:100, x] = True
    return mask


def test_epsilon_zero_preserves_more_vertices_than_default_simplification() -> None:
    mask = _comb()
    unsimplified = mask_to_polygons(mask, epsilon_ratio=0.0, all_contours=True)[0]
    simplified = mask_to_polygons(mask, epsilon_ratio=0.002, all_contours=False)[0]
    assert len(unsimplified) > len(simplified)


def test_epsilon_zero_round_trips_the_shape_far_more_faithfully() -> None:
    """The vertex count is a proxy; this is the property that matters - how much
    of the original shape survives being turned into a polygon and back."""
    mask = _comb()

    def iou(polygon) -> float:
        reconstructed = polygon_to_mask(polygon, W, H)
        return float(
            np.logical_and(mask, reconstructed).sum() / np.logical_or(mask, reconstructed).sum()
        )

    unsimplified = iou(mask_to_polygons(mask, epsilon_ratio=0.0, all_contours=True)[0])
    simplified = iou(mask_to_polygons(mask, epsilon_ratio=0.002, all_contours=False)[0])
    assert unsimplified > simplified


def test_blob_classes_are_unaffected_by_the_change() -> None:
    """Non-fine-structure classes must keep their existing behaviour - a
    component outline is a blob and simplifying it is a feature."""
    polygon = mask_to_polygon(_single_blob())
    # A rectangle simplifies to roughly its corners, as it always did.
    assert 4 <= len(polygon) <= 8
    mask = polygon_to_mask(polygon, W, H)
    original = _single_blob()
    iou = np.logical_and(mask, original).sum() / np.logical_or(mask, original).sum()
    assert iou > 0.97


# --------------------------------------------------------------- housekeeping

def test_polygons_are_ordered_largest_area_first() -> None:
    """A consumer taking only the first piece must get the dominant one -
    ranking by point count would put a long thin contour above a bigger blob."""
    mask = np.zeros((H, W), dtype=bool)
    mask[10:20, 10:20] = True  # small
    mask[100:180, 100:180] = True  # large
    polygons = mask_to_polygons(mask, epsilon_ratio=0.0, all_contours=True)
    assert len(polygons) == 2
    areas = [polygon_to_mask(p, W, H).sum() for p in polygons]
    assert areas[0] > areas[1]


def test_empty_mask_yields_nothing_on_both_paths() -> None:
    empty = np.zeros((H, W), dtype=bool)
    assert mask_to_polygons(empty, epsilon_ratio=0.0, all_contours=True) == []
    assert mask_to_polygon(empty) == []


def test_speck_below_min_area_is_dropped() -> None:
    mask = np.zeros((H, W), dtype=bool)
    mask[5, 5] = True
    assert mask_to_polygons(mask, epsilon_ratio=0.0, all_contours=True) == []


def test_no_morphological_close_on_the_fine_structure_path() -> None:
    """A 3x3 close can bridge two nearby crack segments into one, changing
    measured length. Two strips separated by a 2px gap must stay two."""
    mask = np.zeros((H, W), dtype=bool)
    mask[40:44, 20:100] = True
    mask[40:44, 102:180] = True  # 2px gap
    assert len(mask_to_polygons(mask, epsilon_ratio=0.0, all_contours=True)) == 2


def test_extra_polygons_survive_a_payload_round_trip() -> None:
    """`extra_polygons` lives in the JSONB payload and in GenerateMaskResponse.
    If either drops it, a branching crack silently loses its pieces on the next
    autosave - the frontend holds objects in memory and posts them back."""
    from app.models.schemas import AnnotationObject, BoundingBox, GenerateMaskResponse, Point

    pieces = [[Point(x=0.5, y=0.5), Point(x=0.6, y=0.5), Point(x=0.6, y=0.6)]]
    obj = AnnotationObject(
        id="a",
        class_id=0,
        bbox=BoundingBox(x_center=0.5, y_center=0.5, width=0.2, height=0.2),
        polygon=[Point(x=0.1, y=0.1), Point(x=0.2, y=0.1), Point(x=0.2, y=0.2)],
        extra_polygons=pieces,
    )
    reloaded = AnnotationObject.model_validate(obj.model_dump(mode="json"))
    assert len(reloaded.extra_polygons) == 1
    assert len(reloaded.extra_polygons[0]) == 3

    response = GenerateMaskResponse(
        object_id="a", polygon=obj.polygon, extra_polygons=pieces, confidence=0.9,
        all_scores=[0.9], selected_mask_index=0,
    )
    assert len(response.extra_polygons) == 1


def test_legacy_object_payload_defaults_extra_polygons_empty() -> None:
    from app.models.schemas import AnnotationObject

    legacy = {
        "id": "a",
        "class_id": 0,
        "class_name": "coupler",
        "bbox": {"x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.2},
        "polygon": [],
        "confidence": 0.8,
        "all_mask_scores": [0.8],
        "selected_mask_index": 0,
        "status": "auto_generated",
        "visible": True,
        "source": "detection_box",
        "propagated_from_image_id": None,
    }
    assert AnnotationObject.model_validate(legacy).extra_polygons == []


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - hand-rolled runner wants everything
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    raise SystemExit(1 if failures else 0)
