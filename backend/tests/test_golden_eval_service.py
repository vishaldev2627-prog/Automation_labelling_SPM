"""Tests for evaluating a trained detector against the golden set (M6).

`evaluate_on_golden_set` itself needs a real YOLO checkpoint and torch, so
it's verified live against the running dev stack rather than mocked here
(same reasoning as test_wheel_unwrap.py staying away from mocking OpenCV).
What's tested directly: the "no golden set yet" guard, and the dataset-
assembly step, which is pure file I/O against a duck-typed dataset service.

    cd backend && python -m pytest tests/test_golden_eval_service.py
    cd backend && python -m tests.test_golden_eval_service   # no pytest
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from app.models.schemas import AnnotationObject, BoundingBox, ImageAnnotations, ObjectStatus
from app.services.golden_eval_service import NoGoldenSetError, _assemble_golden_dataset, evaluate_on_golden_set

BBOX = BoundingBox(x_center=0.5, y_center=0.5, width=0.2, height=0.2)


class _FakeDatasetService:
    """Duck-typed stand-in exposing only what _assemble_golden_dataset uses -
    get_annotations and get_image_path - so this is testable without a real
    DatasetService/Postgres."""

    def __init__(self, root: Path, annotations: dict[str, ImageAnnotations]) -> None:
        self._root = root
        self._annotations = annotations

    def get_annotations(self, image_id: str) -> ImageAnnotations:
        return self._annotations[image_id]

    def get_image_path(self, image_id: str) -> Path:
        path = self._root / f"{image_id}.jpg"
        path.write_bytes(b"fake-image-bytes")
        return path


def _annotations(image_id: str, objects: list[AnnotationObject], no_objects_confirmed: bool = False) -> ImageAnnotations:
    return ImageAnnotations(
        image_id=image_id, file_name=f"{image_id}.jpg", width=100, height=100,
        objects=objects, no_objects_confirmed=no_objects_confirmed,
    )


def _obj(class_id: int = 0, status: ObjectStatus = ObjectStatus.CONFIRMED) -> AnnotationObject:
    return AnnotationObject(id="o1", class_id=class_id, class_name="coupler", bbox=BBOX, status=status)


def test_empty_golden_set_raises_no_golden_set_error() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            evaluate_on_golden_set(
                ds=None, golden_image_ids=set(), model_path=Path("unused.pt"),
                classes=["coupler"], staging_root=Path(tmp),
            )
        except NoGoldenSetError:
            pass
        else:
            raise AssertionError("expected NoGoldenSetError for an empty golden set")


def test_labeled_golden_image_is_written_with_its_label_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = _FakeDatasetService(root, {"img1": _annotations("img1", [_obj()])})
        data_yaml, written = _assemble_golden_dataset(ds, {"img1"}, root / "staging", ["coupler"])
        assert written == 1
        assert (root / "staging" / "val" / "images" / "img1.jpg").exists()
        label_text = (root / "staging" / "val" / "labels" / "img1.txt").read_text()
        assert label_text.startswith("0 ")


def test_confirmed_negative_golden_image_gets_an_empty_label_file() -> None:
    """A golden image with no objects but no_objects_confirmed=True is a
    real negative example, same convention export_service uses - it must
    still be included, with a truly empty label file."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = _FakeDatasetService(root, {"img1": _annotations("img1", [], no_objects_confirmed=True)})
        data_yaml, written = _assemble_golden_dataset(ds, {"img1"}, root / "staging", ["coupler"])
        assert written == 1
        assert (root / "staging" / "val" / "labels" / "img1.txt").read_text() == ""


def test_unconfirmed_empty_golden_image_is_skipped() -> None:
    """No objects and no explicit confirmation - not evaluable, not a
    negative example either, just unannotated."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = _FakeDatasetService(root, {"img1": _annotations("img1", [], no_objects_confirmed=False)})
        _data_yaml, written = _assemble_golden_dataset(ds, {"img1"}, root / "staging", ["coupler"])
        assert written == 0


def test_rejected_objects_are_excluded_from_the_label_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = _FakeDatasetService(
            root, {"img1": _annotations("img1", [_obj(status=ObjectStatus.REJECTED)])}
        )
        _data_yaml, written = _assemble_golden_dataset(ds, {"img1"}, root / "staging", ["coupler"])
        # A rejected-only image has no live objects and wasn't explicitly
        # confirmed empty - same as the unconfirmed-empty case, skipped.
        assert written == 0


def test_data_yaml_points_val_at_the_golden_images() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = _FakeDatasetService(root, {"img1": _annotations("img1", [_obj()])})
        data_yaml, _written = _assemble_golden_dataset(ds, {"img1"}, root / "staging", ["coupler", "wheel"])
        content = data_yaml.read_text()
        assert "val/images" in content
        assert "nc: 2" in content


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
