"""Tests for the coach_type field (W-4 / C-5).

Why the field exists: the pipeline's component manifest is
coach-type-conditional (`pipeline.md` §8, which refuses to guess a manifest for
an unknown type), and their R10 "label scarcity - needs label counts" cannot be
answered by an aggregate per-class count, which hides a class well covered on
LHB and absent on ICF.

Which values: **LHB and ICF only** (confirmed 2026-08-05), matching
`pipeline.md` §5.7's 2-class coach-type classifier and
`FINAL_AIML_ARCHITECTURE` §9's detection record. An earlier reply had listed
four (adding Vande Bharat and hybrid); those are deliberately not labelable, so
the tool cannot emit a coach type the pipeline's classifier and per-type
manifests have no slot for. These tests pin the enum so widening it again is a
deliberate act, not a drift.

    cd backend && python -m pytest tests/test_coach_type.py
    cd backend && python -m tests.test_coach_type   # no pytest
"""
from __future__ import annotations

from app.models.schemas import (
    CoachType,
    ImageAnnotations,
    SaveAnnotationRequest,
    SetCoachTypeRequest,
)


def _annotations(**overrides) -> ImageAnnotations:
    base = {"image_id": "x", "file_name": "x.jpg", "width": 100, "height": 100}
    base.update(overrides)
    return ImageAnnotations(**base)


def test_only_lhb_icf_and_unknown_are_labelable() -> None:
    """Pinned deliberately: the pipeline's coach-type classifier is 2-class and
    its manifests are per-type, so emitting any other value would produce a
    label with no slot downstream."""
    assert {c.value for c in CoachType} == {"LHB", "ICF", "unknown"}


def test_default_is_unknown_not_a_guessed_type() -> None:
    """`pipeline.md` §8 refuses to select a manifest for an unknown coach type
    rather than assuming one; defaulting to LHB because it's commonest would be
    inventing data."""
    assert _annotations().coach_type is CoachType.UNKNOWN


def test_legacy_payload_without_the_key_reads_as_unknown() -> None:
    legacy = {
        "image_id": "x",
        "file_name": "x.jpg",
        "width": 100,
        "height": 100,
        "objects": [],
        "completed": True,
        "last_modified": 1.0,
    }
    assert ImageAnnotations.model_validate(legacy).coach_type is CoachType.UNKNOWN


def test_save_request_coach_type_is_tri_state() -> None:
    """None means "leave the stored value alone", so an autosave cannot reset a
    coach type someone set."""
    assert SaveAnnotationRequest(image_id="x", objects=[]).coach_type is None
    assert (
        SaveAnnotationRequest(image_id="x", objects=[], coach_type=CoachType.ICF).coach_type
        is CoachType.ICF
    )


def test_round_trip_through_json_preserves_the_value() -> None:
    """This field lives in the JSONB payload, so dump -> store -> validate is
    the real persistence path."""
    original = _annotations(coach_type=CoachType.ICF)
    reloaded = ImageAnnotations.model_validate(original.model_dump(mode="json"))
    assert reloaded.coach_type is CoachType.ICF


def test_values_are_safe_as_manifest_keys_and_path_segments() -> None:
    """They end up as per-coach-type keys in the snapshot manifest's label
    counts, so no spaces or separators needing escaping."""
    for coach_type in CoachType:
        assert " " not in coach_type.value
        assert "/" not in coach_type.value and "\\" not in coach_type.value


def test_bulk_request_defaults_to_empty_image_ids() -> None:
    """Empty means "every already-saved image still marked unknown" - never
    "every image", so a bulk apply cannot overwrite a per-image correction."""
    request = SetCoachTypeRequest(coach_type=CoachType.LHB)
    assert request.image_ids == []


def test_bulk_request_rejects_an_unknown_type_string() -> None:
    import pydantic

    try:
        SetCoachTypeRequest(coach_type="Shatabdi")
    except pydantic.ValidationError:
        pass
    else:
        raise AssertionError("an unrecognised coach type should not validate")


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
