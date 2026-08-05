"""Tests for the explicit-negative state (W-2 / C-3).

`VB-BufferBoundary`'s `buffer_visible` classifier trains on both classes, so
frames where the buffer is *absent* are half its signal
(`FINAL_AIML_ARCHITECTURE` §2.2, R-BUFFER). Before this change there was no way
to say "a human looked and there is nothing here", and `export_service` dropped
every empty frame outright (`if not objects: skipped += 1; continue`) - so
negatives could not be produced at all.

These tests pin the three-way distinction that makes it work: confirmed-empty
(a negative), not-yet-annotated (skip), and confirmed-empty-because-everything
-was-rejected (also a negative).

No database and no SAM2 - the label-writing convention and the model's own
invariants are what matter here, and both are exercised directly.

    cd backend && python -m pytest tests/test_confirmed_negatives.py
    cd backend && python -m tests.test_confirmed_negatives   # no pytest
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from app.models.schemas import (
    AnnotationObject,
    BoundingBox,
    ImageAnnotations,
    ObjectStatus,
    SaveAnnotationRequest,
)
from app.utils.yolo_utils import write_segmentation_label_file

BBOX = BoundingBox(x_center=0.5, y_center=0.5, width=0.2, height=0.2)


def _obj(status: ObjectStatus = ObjectStatus.CONFIRMED) -> AnnotationObject:
    return AnnotationObject(id="a", class_id=0, class_name="coupler", bbox=BBOX, status=status)


# --------------------------------------------------- the model's own defaults

def test_flag_defaults_off_so_existing_data_is_unaffected() -> None:
    """Every already-stored payload lacks this key; it must not read as a
    confirmed negative, or the next export would emit thousands of them."""
    annotations = ImageAnnotations(image_id="x", file_name="x.jpg", width=100, height=100)
    assert annotations.no_objects_confirmed is False


def test_legacy_payload_without_the_key_is_not_a_negative() -> None:
    legacy = {
        "image_id": "x",
        "file_name": "x.jpg",
        "width": 100,
        "height": 100,
        "objects": [],
        "completed": True,
        "last_modified": 1.0,
    }
    assert ImageAnnotations.model_validate(legacy).no_objects_confirmed is False


def test_save_request_flag_is_tri_state() -> None:
    """None means "leave stored value alone" so an autosave can't clear a
    confirmation; False is an explicit human retraction. A plain bool would
    silently un-confirm on every autosave."""
    assert SaveAnnotationRequest(image_id="x", objects=[]).no_objects_confirmed is None
    assert (
        SaveAnnotationRequest(image_id="x", objects=[], no_objects_confirmed=False).no_objects_confirmed
        is False
    )
    assert (
        SaveAnnotationRequest(image_id="x", objects=[], no_objects_confirmed=True).no_objects_confirmed
        is True
    )


# ------------------------------------- the router's confirmation-clearing rule

def _apply_router_rules(stored: ImageAnnotations, request: SaveAnnotationRequest) -> ImageAnnotations:
    """Mirror of routers/images.py's save path, so the rule is tested without
    standing up FastAPI + Postgres. Kept deliberately small; if the router
    changes, this must change with it."""
    stored.objects = request.objects
    stored.completed = request.mark_completed or stored.completed
    if request.no_objects_confirmed is not None:
        stored.no_objects_confirmed = request.no_objects_confirmed
    if any(o.status != ObjectStatus.REJECTED for o in stored.objects):
        stored.no_objects_confirmed = False
    return stored


def test_autosave_does_not_clear_an_existing_confirmation() -> None:
    stored = ImageAnnotations(
        image_id="x", file_name="x.jpg", width=100, height=100, no_objects_confirmed=True, completed=True
    )
    result = _apply_router_rules(stored, SaveAnnotationRequest(image_id="x", objects=[]))
    assert result.no_objects_confirmed is True


def test_adding_a_live_object_clears_the_confirmation() -> None:
    """The assertion described an empty frame. Once something is in it the
    assertion is false, and keeping it would export a populated frame as a
    negative."""
    stored = ImageAnnotations(
        image_id="x", file_name="x.jpg", width=100, height=100, no_objects_confirmed=True
    )
    result = _apply_router_rules(stored, SaveAnnotationRequest(image_id="x", objects=[_obj()]))
    assert result.no_objects_confirmed is False


def test_rejected_objects_do_not_count_as_present() -> None:
    """"The detector proposed these, I rejected all of them, there is genuinely
    nothing here" is one of the real ways a negative gets made."""
    stored = ImageAnnotations(
        image_id="x", file_name="x.jpg", width=100, height=100, no_objects_confirmed=True
    )
    result = _apply_router_rules(
        stored,
        SaveAnnotationRequest(image_id="x", objects=[_obj(ObjectStatus.REJECTED)]),
    )
    assert result.no_objects_confirmed is True


def test_explicit_retraction_is_honoured() -> None:
    stored = ImageAnnotations(
        image_id="x", file_name="x.jpg", width=100, height=100, no_objects_confirmed=True, completed=True
    )
    result = _apply_router_rules(
        stored, SaveAnnotationRequest(image_id="x", objects=[], no_objects_confirmed=False)
    )
    assert result.no_objects_confirmed is False


def test_retraction_does_not_un_complete_the_image() -> None:
    """Matches the backend's existing rule that `completed` never goes back to
    false on a save."""
    stored = ImageAnnotations(
        image_id="x", file_name="x.jpg", width=100, height=100, no_objects_confirmed=True, completed=True
    )
    result = _apply_router_rules(
        stored, SaveAnnotationRequest(image_id="x", objects=[], no_objects_confirmed=False)
    )
    assert result.completed is True


# ------------------------------------------------ the YOLO negative convention

def test_empty_object_list_writes_a_truly_empty_label_file() -> None:
    """A negative sample is an image plus a zero-byte label file. A stray "\\n"
    is a malformed label line to some YOLO loaders, so emptiness must be
    exact, not "one blank line"."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "frame.txt"
        write_segmentation_label_file(path, [])
        assert path.exists()
        assert path.read_bytes() == b""


def test_non_empty_object_list_still_writes_labels() -> None:
    from app.models.schemas import Point

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "frame.txt"
        polygon = [Point(x=0.1, y=0.1), Point(x=0.2, y=0.1), Point(x=0.2, y=0.2)]
        write_segmentation_label_file(path, [(3, polygon)])
        content = path.read_text(encoding="utf-8")
        assert content.startswith("3 ")
        assert content.endswith("\n")


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
