"""Regression tests for the detector/mask confidence split (M0.2 / P-3).

The bug this locks down: `AnnotationObject.confidence` held three different
things - 0.0 for a box read from a YOLO label file (no signal), ultralytics'
`box.conf` for a detector-produced box, and SAM2's *mask* score once
mask_generation_service ran and overwrote whichever of those was there.
`auto_accept_service`'s 0.95 gate read that field as a detector confidence,
so after mask generation it was deciding whether to skip human review based
on how cleanly SAM2 segmented rather than how sure anything was of the class.

No database, no GPU, no SAM2: the auto-accept gate is exercised through
`_object_is_acceptable`, which operates on the saved-payload dicts, and the
legacy-payload migration through the pydantic model directly.

    cd backend && python -m pytest tests/test_confidence_split.py
    cd backend && python -m tests.test_confidence_split   # no pytest
"""
from __future__ import annotations

from app.models.schemas import AnnotationObject, BoundingBox, ObjectSource, ObjectStatus
from app.services.auto_accept_service import (
    DETECTOR_CONFIDENCE_THRESHOLD,
    MASK_CONFIDENCE_THRESHOLD,
    _object_is_acceptable,
)
from app.services.triage_service import _detector_confidences, _low_confidence_tier, _no_signal_tier

BBOX = {"x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.2}


def _obj(**overrides) -> dict:
    """A saved-payload object dict, the shape these consumers actually read."""
    base = {
        "id": "abc123",
        "class_id": 1,
        "class_name": "coupler",
        "bbox": BBOX,
        "polygon": [],
        "detector_confidence": 0.99,
        "mask_confidence": 0.99,
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

def test_high_mask_score_alone_does_not_pass_the_gate() -> None:
    """The core regression. A clean SAM2 mask on a box the detector was
    unsure about must not skip human review."""
    obj = _obj(detector_confidence=0.30, mask_confidence=0.99)
    assert _object_is_acceptable(obj, {1}) is False


def test_high_detector_confidence_alone_does_not_pass_the_gate() -> None:
    obj = _obj(detector_confidence=0.99, mask_confidence=0.40)
    assert _object_is_acceptable(obj, {1}) is False


def test_both_high_passes() -> None:
    assert _object_is_acceptable(_obj(), {1}) is True


def test_missing_detector_confidence_fails_closed() -> None:
    """No signal is not evidence - a label-file box, or any pre-split
    historical object, must never be auto-accept eligible."""
    obj = _obj(detector_confidence=None, mask_confidence=1.0)
    assert _object_is_acceptable(obj, {1}) is False
    # Absent key, not just None - older payloads won't have it at all.
    obj_without_key = _obj()
    del obj_without_key["detector_confidence"]
    assert _object_is_acceptable(obj_without_key, {1}) is False


def test_ineligible_class_fails_regardless_of_confidence() -> None:
    assert _object_is_acceptable(_obj(detector_confidence=1.0, mask_confidence=1.0), set()) is False


def test_thresholds_are_boundary_inclusive() -> None:
    obj = _obj(
        detector_confidence=DETECTOR_CONFIDENCE_THRESHOLD, mask_confidence=MASK_CONFIDENCE_THRESHOLD
    )
    assert _object_is_acceptable(obj, {1}) is True


# ------------------------------------------ legacy payload read-time mapping

def test_legacy_payload_confidence_maps_to_mask_confidence_only() -> None:
    """Rows written before the split hold a single `confidence`. It becomes
    mask_confidence; detector_confidence stays None, so historical data is
    not auto-accept eligible until reprocessed (the conservative direction)."""
    legacy = {
        "id": "x",
        "class_id": 0,
        "class_name": "coupler",
        "bbox": BBOX,
        "polygon": [],
        "confidence": 0.97,
        "all_mask_scores": [0.97],
        "selected_mask_index": 0,
        "status": "auto_generated",
        "visible": True,
        "source": "detection_box",
        "propagated_from_image_id": None,
    }
    obj = AnnotationObject.model_validate(legacy)
    assert obj.mask_confidence == 0.97
    assert obj.detector_confidence is None
    assert _object_is_acceptable(obj.model_dump(mode="json"), {0}) is False


def test_new_payload_is_not_rewritten_by_the_legacy_mapping() -> None:
    """A payload that already carries both fields must pass through untouched,
    even if a stale `confidence` key is also present."""
    obj = AnnotationObject.model_validate(
        {**_obj(class_id=0), "confidence": 0.10, "detector_confidence": 0.96, "mask_confidence": 0.98}
    )
    assert obj.detector_confidence == 0.96
    assert obj.mask_confidence == 0.98


def test_defaults_when_neither_confidence_is_supplied() -> None:
    obj = AnnotationObject(
        id="x",
        class_id=0,
        bbox=BoundingBox(**BBOX),
        status=ObjectStatus.PENDING,
        source=ObjectSource.MANUAL,
    )
    assert obj.detector_confidence is None
    assert obj.mask_confidence == 0.0


def test_round_trip_survives_save_and_reload() -> None:
    """model_dump -> (JSONB) -> model_validate must not lose or re-map either
    field; this is exactly what save_state/get_state do."""
    original = AnnotationObject.model_validate(_obj(detector_confidence=0.42, mask_confidence=0.84))
    reloaded = AnnotationObject.model_validate(original.model_dump(mode="json"))
    assert reloaded.detector_confidence == 0.42
    assert reloaded.mask_confidence == 0.84


# ------------------------------------------------------------- triage tiers

def test_low_confidence_tier_ranks_on_detector_confidence_not_mask_score() -> None:
    states = {
        # Confident class, ugly mask - must NOT be flagged as low-confidence.
        "clean_mask_unsure_class": {"objects": [_obj(detector_confidence=0.90, mask_confidence=0.10)]},
        # Unsure class, beautiful mask - must be flagged.
        "ugly_mask_sure_class": {"objects": [_obj(detector_confidence=0.20, mask_confidence=0.99)]},
    }
    names = {k: f"{k}.jpg" for k in states}
    tier = _low_confidence_tier(states, names)
    assert [t.image_id for t in tier] == ["ugly_mask_sure_class"]


def test_low_confidence_tier_ignores_no_signal_objects() -> None:
    states = {"from_label_file": {"objects": [_obj(detector_confidence=None, mask_confidence=0.05)]}}
    assert _low_confidence_tier(states, {"from_label_file": "a.jpg"}) == []


def test_no_signal_tier_surfaces_them_instead() -> None:
    states = {
        "from_label_file": {"objects": [_obj(detector_confidence=None)]},
        "has_signal": {"objects": [_obj(detector_confidence=0.7)]},
        "no_objects_at_all": {"objects": []},
    }
    names = {k: f"{k}.jpg" for k in states}
    tier = _no_signal_tier(states, names, excluded=set())
    assert [t.image_id for t in tier] == ["from_label_file"]


def test_no_signal_tier_respects_exclusions() -> None:
    states = {"from_label_file": {"objects": [_obj(detector_confidence=None)]}}
    names = {"from_label_file": "a.jpg"}
    assert _no_signal_tier(states, names, excluded={"from_label_file"}) == []


def test_detector_confidences_drops_none_without_treating_it_as_zero() -> None:
    state = {
        "objects": [
            _obj(detector_confidence=None),
            _obj(detector_confidence=0.8),
            _obj(detector_confidence=0.0),  # a real zero IS a signal, and is kept
        ]
    }
    assert _detector_confidences(state) == [0.8, 0.0]


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
