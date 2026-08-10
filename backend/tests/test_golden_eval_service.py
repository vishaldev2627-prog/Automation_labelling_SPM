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

from app.models.schemas import AnnotationObject, BoundingBox, ImageAnnotations, ObjectStatus, Point
from app.services.golden_eval_service import (
    InsufficientGoldenCoverageError,
    NoGoldenSetError,
    _assemble_golden_dataset,
    _compute_fp_fn,
    evaluate_on_golden_set,
)

BBOX = BoundingBox(x_center=0.5, y_center=0.5, width=0.2, height=0.2)
# _assemble_golden_dataset now writes YOLO-seg polygon lines, not derived
# bboxes (YOLO11-seg move) - a bbox alone no longer produces a label line,
# so every fixture object needs a real (>=3 point) polygon too.
POLYGON = [Point(x=0.4, y=0.4), Point(x=0.6, y=0.4), Point(x=0.6, y=0.6), Point(x=0.4, y=0.6)]


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
    return AnnotationObject(
        id="o1", class_id=class_id, class_name="coupler", bbox=BBOX, polygon=POLYGON, status=status
    )


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
        data_yaml, written, instance_counts = _assemble_golden_dataset(ds, {"img1"}, root / "staging", ["coupler"])
        assert written == 1
        assert (root / "staging" / "val" / "images" / "img1.jpg").exists()
        label_text = (root / "staging" / "val" / "labels" / "img1.txt").read_text()
        assert label_text.startswith("0 ")
        assert instance_counts == {0: 1}


def test_confirmed_negative_golden_image_gets_an_empty_label_file() -> None:
    """A golden image with no objects but no_objects_confirmed=True is a
    real negative example, same convention export_service uses - it must
    still be included, with a truly empty label file."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = _FakeDatasetService(root, {"img1": _annotations("img1", [], no_objects_confirmed=True)})
        data_yaml, written, _instance_counts = _assemble_golden_dataset(ds, {"img1"}, root / "staging", ["coupler"])
        assert written == 1
        assert (root / "staging" / "val" / "labels" / "img1.txt").read_text() == ""


def test_unconfirmed_empty_golden_image_is_skipped() -> None:
    """No objects and no explicit confirmation - not evaluable, not a
    negative example either, just unannotated."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = _FakeDatasetService(root, {"img1": _annotations("img1", [], no_objects_confirmed=False)})
        _data_yaml, written, _instance_counts = _assemble_golden_dataset(ds, {"img1"}, root / "staging", ["coupler"])
        assert written == 0


def test_rejected_objects_are_excluded_from_the_label_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = _FakeDatasetService(
            root, {"img1": _annotations("img1", [_obj(status=ObjectStatus.REJECTED)])}
        )
        _data_yaml, written, _instance_counts = _assemble_golden_dataset(ds, {"img1"}, root / "staging", ["coupler"])
        # A rejected-only image has no live objects and wasn't explicitly
        # confirmed empty - same as the unconfirmed-empty case, skipped.
        assert written == 0


def test_data_yaml_points_val_at_the_golden_images() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = _FakeDatasetService(root, {"img1": _annotations("img1", [_obj()])})
        data_yaml, _written, _instance_counts = _assemble_golden_dataset(ds, {"img1"}, root / "staging", ["coupler", "wheel"])
        content = data_yaml.read_text()
        assert "val/images" in content
        assert "nc: 2" in content


# ------------------------------------------------- InsufficientGoldenCoverageError (Module 7)

def test_insufficient_golden_coverage_raises_with_offending_class_ids() -> None:
    """Raised BEFORE any model.val() call - a bogus model_path proves
    ultralytics was never reached (a real attempt would fail with a
    file-not-found style error, not this one)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = _FakeDatasetService(root, {"img1": _annotations("img1", [_obj(class_id=0)])})
        try:
            evaluate_on_golden_set(
                ds, {"img1"}, Path("/nonexistent/model.pt"), ["coupler", "wheel"], root / "staging",
                candidate_class_ids={0, 1},  # class 1 has zero golden coverage
            )
            raise AssertionError("expected InsufficientGoldenCoverageError")
        except InsufficientGoldenCoverageError as exc:
            assert exc.class_ids == [1]


def test_classes_with_coverage_still_flagged_alongside_a_missing_one() -> None:
    """Lists every uncovered class at once, not just the first - mirrors
    should_promote()'s own 'collect every failing check' philosophy."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = _FakeDatasetService(root, {"img1": _annotations("img1", [_obj(class_id=0)])})
        try:
            evaluate_on_golden_set(
                ds, {"img1"}, Path("/nonexistent/model.pt"), ["coupler", "wheel", "axle"], root / "staging",
                candidate_class_ids={0, 1, 2},  # both 1 and 2 have zero coverage
            )
            raise AssertionError("expected InsufficientGoldenCoverageError")
        except InsufficientGoldenCoverageError as exc:
            assert exc.class_ids == [1, 2]


def test_covered_candidate_classes_do_not_raise() -> None:
    """candidate_class_ids only checks coverage for the classes actually
    passed in - a class present in the golden set is never a problem."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = _FakeDatasetService(root, {"img1": _annotations("img1", [_obj(class_id=0)])})
        # Coverage check passes (class 0 has 1 instance); the function then
        # proceeds to real ultralytics loading, which fails on the bogus
        # path - a DIFFERENT exception than InsufficientGoldenCoverageError,
        # which is exactly what proves the coverage check let it through.
        try:
            evaluate_on_golden_set(
                ds, {"img1"}, Path("/nonexistent/model.pt"), ["coupler"], root / "staging",
                candidate_class_ids={0},
            )
            raise AssertionError("expected some ultralytics loading error, not silence")
        except InsufficientGoldenCoverageError:
            raise AssertionError("class 0 has coverage - must not raise this")
        except Exception:
            pass  # any other exception (bogus model path) is expected here


# -------------------------------------------------------------- _compute_fp_fn (Module 7)

class _FakeTensor:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _FakeFpFnBox:
    def __init__(self, class_id: int, conf: float):
        self.cls = _FakeTensor(class_id)
        self.conf = _FakeTensor(conf)


class _FakeFpFnMasks:
    def __init__(self, xyn):
        self.xyn = xyn


class _FakeFpFnResult:
    def __init__(self, boxes, masks):
        self.boxes = boxes
        self.masks = masks


class _FakeFpFnModel:
    """Maps image_id (derived from the fake image path) -> canned
    predict() result, so each test controls exactly what the model
    'sees' for each golden image."""

    def __init__(self, results_by_image_id: dict[str, list]) -> None:
        self._results_by_image_id = results_by_image_id

    def predict(self, path, conf, verbose=False):
        image_id = Path(path).stem
        return self._results_by_image_id.get(image_id, [])


# A polygon that exactly overlaps POLYGON's box - perfect IoU=1.0 match.
MATCHING_XYN = [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]
# Far away from POLYGON - zero overlap.
FAR_AWAY_XYN = [[0.05, 0.05], [0.1, 0.05], [0.1, 0.1], [0.05, 0.1]]


def test_perfect_prediction_match_is_a_true_positive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = _FakeDatasetService(root, {"img1": _annotations("img1", [_obj(class_id=0)])})
        model = _FakeFpFnModel({
            "img1": [_FakeFpFnResult(
                boxes=[_FakeFpFnBox(class_id=0, conf=0.9)],
                masks=_FakeFpFnMasks([MATCHING_XYN]),
            )]
        })
        rates = _compute_fp_fn(ds, {"img1"}, model, confidence_threshold=0.25, iou_threshold=0.5)
        assert rates[0]["false_positive_rate"] == 0.0
        assert rates[0]["false_negative_rate"] == 0.0


def test_no_predictions_at_all_makes_every_ground_truth_a_false_negative() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = _FakeDatasetService(root, {"img1": _annotations("img1", [_obj(class_id=0)])})
        model = _FakeFpFnModel({"img1": []})
        rates = _compute_fp_fn(ds, {"img1"}, model, confidence_threshold=0.25, iou_threshold=0.5)
        assert rates[0]["false_negative_rate"] == 1.0


def test_prediction_with_no_matching_ground_truth_is_a_false_positive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # No ground truth at all for class 0 in this image.
        ds = _FakeDatasetService(root, {"img1": _annotations("img1", [])})
        model = _FakeFpFnModel({
            "img1": [_FakeFpFnResult(
                boxes=[_FakeFpFnBox(class_id=0, conf=0.9)],
                masks=_FakeFpFnMasks([FAR_AWAY_XYN]),
            )]
        })
        rates = _compute_fp_fn(ds, {"img1"}, model, confidence_threshold=0.25, iou_threshold=0.5)
        assert rates[0]["false_positive_rate"] == 1.0


def test_prediction_below_iou_threshold_counts_as_both_fp_and_fn() -> None:
    """A prediction that doesn't clear the IoU bar for the one real
    instance present is a false positive (wrong outline) AND that real
    instance is still unmatched, so it's also a false negative - not a
    trade-off, genuinely two separate failures."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = _FakeDatasetService(root, {"img1": _annotations("img1", [_obj(class_id=0)])})
        model = _FakeFpFnModel({
            "img1": [_FakeFpFnResult(
                boxes=[_FakeFpFnBox(class_id=0, conf=0.9)],
                masks=_FakeFpFnMasks([FAR_AWAY_XYN]),  # wrong location, same class
            )]
        })
        rates = _compute_fp_fn(ds, {"img1"}, model, confidence_threshold=0.25, iou_threshold=0.5)
        assert rates[0]["false_positive_rate"] == 1.0
        assert rates[0]["false_negative_rate"] == 1.0


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
