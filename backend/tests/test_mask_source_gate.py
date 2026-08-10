"""Tests for MaskSource (YOLO11-seg Option B): which model produced an
object's *current* polygon, and why that has to gate auto-accept
independently of the confidence numbers.

The risk this locks down: the 0.95 mask-confidence bar in auto_accept_service
was calibrated against SAM2's own score distribution. A YOLO11-seg
detector-predicted polygon reports a *joint* box/mask confidence that isn't
the same reliability claim at the same number - so a detector-sourced mask
must never be auto-accept eligible until SAM2 has actually produced or
confirmed it, regardless of how high its own confidence reads.

No database, no GPU, no SAM2 - same style as test_confidence_split.py.

    cd backend && python -m pytest tests/test_mask_source_gate.py
    cd backend && python -m tests.test_mask_source_gate   # no pytest
"""
from __future__ import annotations

from app.models.schemas import AnnotationObject, BoundingBox, MaskSource, ObjectStatus
from app.services.auto_accept_service import _object_is_acceptable

BBOX = {"x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.2}
POLYGON = [{"x": 0.4, "y": 0.4}, {"x": 0.6, "y": 0.4}, {"x": 0.6, "y": 0.6}]


def _obj(**overrides) -> dict:
    base = {
        "id": "abc123",
        "class_id": 1,
        "class_name": "coupler",
        "bbox": BBOX,
        "polygon": POLYGON,
        "detector_confidence": 0.99,
        "mask_confidence": 0.99,
        "mask_source": "sam2",
        "all_mask_scores": [0.99],
        "selected_mask_index": 0,
        "status": "auto_generated",
        "visible": True,
        "source": "detection_box",
        "propagated_from_image_id": None,
    }
    base.update(overrides)
    return base


# ----------------------------------------------------- the auto-accept gate

def test_sam2_mask_source_passes() -> None:
    assert _object_is_acceptable(_obj(mask_source="sam2"), {1}) is True


def test_detector_mask_source_never_passes_regardless_of_confidence() -> None:
    """The core regression this file locks down. A detector-predicted
    polygon at the same numeric confidence as a SAM2 one must still fail -
    it's not the same reliability claim."""
    obj = _obj(mask_source="detector", detector_confidence=0.999, mask_confidence=0.999)
    assert _object_is_acceptable(obj, {1}) is False


def test_explicit_null_mask_source_fails() -> None:
    """A real object with no mask at all (mask_source explicitly None, not
    absent) must fail - distinct from the 'key absent = legacy' case below."""
    assert _object_is_acceptable(_obj(mask_source=None), {1}) is False


def test_absent_mask_source_key_treated_as_legacy_sam2() -> None:
    """Rows saved before mask_source existed have the key *absent* entirely,
    never a `None` a validator already resolved to. Before this field
    existed, SAM2 was the only thing that ever produced a mask - so an
    absent key must not silently break every pre-existing eligible object
    (test_confidence_split.py's test_both_high_passes is exactly this shape:
    a legacy fixture with no mask_source key at all)."""
    obj = _obj()
    del obj["mask_source"]
    assert _object_is_acceptable(obj, {1}) is True


def test_absent_mask_source_key_still_needs_the_confidence_bars() -> None:
    """Legacy-key-absent is not a bypass of the other two gates."""
    obj = _obj(detector_confidence=0.30)
    del obj["mask_source"]
    assert _object_is_acceptable(obj, {1}) is False


# ------------------------------------------------- AnnotationObject.mask_source

def test_detector_seeded_object_keeps_its_explicit_mask_source() -> None:
    obj = AnnotationObject(
        id="1", class_id=0, bbox=BoundingBox(**BBOX),
        polygon=[{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.1}, {"x": 0.2, "y": 0.2}],
        detector_confidence=0.9, mask_confidence=0.9,
        mask_source=MaskSource.DETECTOR, status=ObjectStatus.AUTO_GENERATED,
    )
    assert obj.mask_source == MaskSource.DETECTOR


def test_legacy_polygon_with_no_mask_source_key_backfills_to_sam2() -> None:
    """Every polygon persisted before this field existed came from SAM2,
    unambiguously - unlike the confidence-field migration, there is nothing
    to guess here."""
    legacy_payload = {
        "id": "2", "class_id": 0, "bbox": BBOX, "polygon": POLYGON, "mask_confidence": 0.8,
    }
    obj = AnnotationObject.model_validate(legacy_payload)
    assert obj.mask_source == MaskSource.SAM2


def test_no_polygon_no_mask_source_key_stays_none() -> None:
    """A box-only object (nothing has generated a mask yet) must not be
    backfilled to sam2 just because the key is missing - there's no mask to
    attribute to anyone."""
    obj = AnnotationObject(id="3", class_id=0, bbox=BoundingBox(**BBOX))
    assert obj.mask_source is None


def test_detector_value_survives_a_save_and_reload_round_trip() -> None:
    """The backfill validator must never clobber a genuine `detector` value
    on reload - it only fires when the field is truly absent/None, and a
    payload saved after this field existed always carries its real value
    explicitly."""
    obj = AnnotationObject(
        id="1", class_id=0, bbox=BoundingBox(**BBOX), polygon=POLYGON,
        detector_confidence=0.9, mask_confidence=0.9, mask_source=MaskSource.DETECTOR,
    )
    reloaded = AnnotationObject.model_validate(obj.model_dump(mode="json"))
    assert reloaded.mask_source == MaskSource.DETECTOR


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
