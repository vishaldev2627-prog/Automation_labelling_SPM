"""Tests for the component-condition label axis (W-1 / C-1).

Why it exists: the pipeline team confirmed `side_view` feeds `p1_side_damage`,
which `docs/pipeline.md` §5.2 specifies as a **crop classifier** outputting
condition ∈ {ok, broken, missing, ...}. Our objects carried component identity
(`class_id`) only, so `side_view` supplied that model's crops but none of its
labels.

The property these tests defend hardest: **absent means unassessed, never
"ok"**. Roughly 2000 images were annotated before this field existed; promoting
"nobody looked" to "in good condition" on a defect-detection system is the wrong
direction of error, and it is the kind of default that is invisible once it is
in the data.

    cd backend && python -m pytest tests/test_condition_labels.py
    cd backend && python -m tests.test_condition_labels   # no pytest
"""
from __future__ import annotations

from app.models.schemas import (
    AnnotationObject,
    BoundingBox,
    Condition,
    ExportRequest,
    ObjectStatus,
)

BBOX = BoundingBox(x_center=0.5, y_center=0.5, width=0.2, height=0.2)


def _obj(**overrides) -> AnnotationObject:
    base = {"id": "a", "class_id": 0, "class_name": "coupler", "bbox": BBOX}
    base.update(overrides)
    return AnnotationObject(**base)


# ---------------------------------------------- the vocabulary is theirs, exactly

def test_vocabulary_matches_pipeline_md_section_5_2() -> None:
    """Verbatim from the doc. Pinned so a paraphrase or a "tidy-up" of these
    names shows up as a failure - the pipeline trains against this exact set."""
    assert {c.value for c in Condition} == {
        "ok",
        "broken",
        "missing",
        "hanging",
        "displaced",
        "dislocated",
        "leaking",
        "damaged",
        "securing_broken",
        "securing_hanging",
        "fiba_red",
        "sparking",
        "binding",
    }


def test_leaking_condition_exists_and_is_not_the_excluded_leakage_class() -> None:
    """`exclude_classes: [leakage]` excludes a synthetic *component class* from
    serving. `leaking` is a *condition* and is legitimate. Similar names,
    different things - filtering this out would silently drop real labels."""
    assert Condition.LEAKING.value == "leaking"
    assert "leakage" not in {c.value for c in Condition}


# ------------------------------------- unassessed is not "ok" (the core property)

def test_default_is_unassessed_not_ok() -> None:
    assert _obj().condition is None


def test_legacy_object_payload_stays_unassessed() -> None:
    """Every object stored before this field existed. It must not read as `ok`,
    or ~2000 images silently become "these components are fine" training data
    nobody ever asserted."""
    legacy = {
        "id": "a",
        "class_id": 0,
        "class_name": "coupler",
        "bbox": {"x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.2},
        "polygon": [],
        "confidence": 0.8,
        "all_mask_scores": [0.8],
        "selected_mask_index": 0,
        "status": "confirmed",
        "visible": True,
        "source": "detection_box",
        "propagated_from_image_id": None,
    }
    assert AnnotationObject.model_validate(legacy).condition is None


def test_ok_is_distinguishable_from_unassessed_after_a_round_trip() -> None:
    """The whole point: a human asserting "this component is fine" must survive
    persistence as something different from nobody having looked."""
    assessed = AnnotationObject.model_validate(
        _obj(condition=Condition.OK).model_dump(mode="json")
    )
    unassessed = AnnotationObject.model_validate(_obj().model_dump(mode="json"))
    assert assessed.condition is Condition.OK
    assert unassessed.condition is None
    assert assessed.condition != unassessed.condition


def test_condition_survives_a_payload_round_trip() -> None:
    reloaded = AnnotationObject.model_validate(
        _obj(condition=Condition.SECURING_BROKEN).model_dump(mode="json")
    )
    assert reloaded.condition is Condition.SECURING_BROKEN


def test_unrecognised_condition_does_not_validate() -> None:
    import pydantic

    try:
        _obj(condition="slightly_wonky")
    except pydantic.ValidationError:
        pass
    else:
        raise AssertionError("an unrecognised condition should not validate")


# ------------------------------------------------------- crop-export selection

def _selected_for_export(objects: list[AnnotationObject]) -> list[AnnotationObject]:
    """Mirror of export_service._write_condition_crops' selection rule."""
    return [o for o in objects if o.status != ObjectStatus.REJECTED and o.condition is not None]


def test_only_assessed_objects_are_selected_for_condition_crops() -> None:
    objects = [
        _obj(id="assessed", condition=Condition.BROKEN),
        _obj(id="unassessed"),
    ]
    assert [o.id for o in _selected_for_export(objects)] == ["assessed"]


def test_rejected_objects_are_excluded_even_when_assessed() -> None:
    objects = [_obj(id="r", condition=Condition.BROKEN, status=ObjectStatus.REJECTED)]
    assert _selected_for_export(objects) == []


def test_condition_names_are_safe_as_directory_names() -> None:
    """Crops are foldered by condition (crops/<split>/<condition>/...), which is
    the layout YOLO11-cls and the usual EfficientNet pipelines expect."""
    for condition in Condition:
        assert condition.value
        assert "/" not in condition.value and "\\" not in condition.value
        assert " " not in condition.value
        assert condition.value == condition.value.strip()


def test_crop_export_is_on_by_default_but_a_no_op_without_conditions() -> None:
    """Defaults on so nobody has to remember a flag, and costs nothing on
    datasets annotated before the field existed - no object has a condition, so
    nothing is written."""
    assert ExportRequest().include_condition_crops is True
    assert _selected_for_export([_obj(), _obj(id="b")]) == []


# -------------------------------------------------------------- crop geometry

def test_crop_margin_expands_and_clamps_at_the_frame_edge() -> None:
    """A component at the edge should get a smaller margin, never a shifted or
    wrapped crop. Mirrors the arithmetic in _write_condition_crops."""
    img_w, img_h = 1000, 800
    margin_pct = 7.5

    # Bbox flush against the left edge.
    bbox = BoundingBox(x_center=0.05, y_center=0.5, width=0.1, height=0.2)
    x1, y1, x2, y2 = bbox.to_xyxy(img_w, img_h)
    margin_x = int(round((x2 - x1) * margin_pct / 100.0))
    margin_y = int(round((y2 - y1) * margin_pct / 100.0))
    cx1 = max(0, x1 - margin_x)
    cy1 = max(0, y1 - margin_y)
    cx2 = min(img_w, x2 + margin_x)
    cy2 = min(img_h, y2 + margin_y)

    assert cx1 == 0  # clamped, not negative
    assert cx2 <= img_w and cy2 <= img_h
    assert cx2 > cx1 and cy2 > cy1
    # Still grew on the side that had room.
    assert cx2 > x2


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
