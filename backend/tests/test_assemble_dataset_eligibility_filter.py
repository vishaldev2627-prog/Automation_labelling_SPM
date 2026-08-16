"""Tests for Module 3 of the class-incremental promotion plan -
docs/mlflow_class_incremental_architecture.md §D/§E: `_assemble_dataset`
only writes label lines for classes whose state is eligible/active.

No database, no GPU: `DetectorService._assemble_dataset` is exercised
directly against a duck-typed fake DatasetService (same style as
tests/test_golden_eval_service.py's _FakeDatasetService) writing to a real
temp directory (the label/image files themselves are the thing under
test).

    cd backend && python -m pytest tests/test_assemble_dataset_eligibility_filter.py
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import tempfile
from pathlib import Path

from app.models.schemas import AnnotationObject, BoundingBox, ImageAnnotations, ObjectStatus, Point
from app.services.detector_service import DetectorService

BBOX = BoundingBox(x_center=0.5, y_center=0.5, width=0.2, height=0.2)
POLY = [Point(x=0.4, y=0.4), Point(x=0.6, y=0.4), Point(x=0.6, y=0.6)]


class _FakeDatasetService:
    """Duck-typed stand-in exposing only what _assemble_dataset uses -
    image_ids, get_annotations, get_image_path, dataset_key."""

    def __init__(self, root: Path, annotations: Dict[str, ImageAnnotations]) -> None:
        self._root = root
        self._annotations = annotations

    def image_ids(self) -> List[str]:
        return list(self._annotations.keys())

    def get_annotations(self, image_id: str) -> ImageAnnotations:
        return self._annotations[image_id]

    def get_image_path(self, image_id: str) -> Path:
        path = self._root / f"{image_id}.jpg"
        path.write_bytes(b"fake-image-bytes")
        return path

    @property
    def dataset_key(self) -> str:
        return "fake_view"


def _obj(obj_id: str, class_id: int) -> AnnotationObject:
    return AnnotationObject(id=obj_id, class_id=class_id, bbox=BBOX, polygon=POLY, status=ObjectStatus.CONFIRMED)


def _ann(image_id: str, objects: List[AnnotationObject]) -> ImageAnnotations:
    return ImageAnnotations(image_id=image_id, file_name=f"{image_id}.jpg", width=100, height=100,
                             objects=objects, completed=True)


def _detector(annotations: Dict[str, ImageAnnotations], tmp: Path) -> Tuple[DetectorService, Path]:
    root = tmp / "images_root"
    root.mkdir()
    ds = _FakeDatasetService(root, annotations)
    det = DetectorService(dataset_service=ds, models_dir=tmp / "models")
    return det, root


def test_only_eligible_and_active_classes_get_label_lines() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # 2 images (not 1) - a single-image dataset gets duplicated into
        # both train and val by _assemble_dataset's own split logic
        # (image_ids[split_at:] or image_ids[:1]), unrelated to this
        # module's own change; keeping >=2 images avoids that quirk here.
        annotations = {
            "img1": _ann("img1", [_obj("o0", 0), _obj("o1", 1), _obj("o2", 2)]),
            "img2": _ann("img2", [_obj("o3", 0)]),
        }
        det, _root = _detector(annotations, tmp_path)
        # class 0 = active, class 1 = eligible, class 2 = discovered
        staging = tmp_path / "staging"
        data_yaml, num_images = det._assemble_dataset(staging, ["a", "b", "c"], trainable_class_ids={0, 1})

        label_files = list((staging / "train" / "labels").glob("*.txt")) + list(
            (staging / "val" / "labels").glob("*.txt")
        )
        assert num_images == 2
        img1_label = next(f for f in label_files if f.name == "img1.txt")
        lines = [l for l in img1_label.read_text().splitlines() if l.strip()]
        written_class_ids = {int(line.split()[0]) for line in lines}
        assert written_class_ids == {0, 1}


def test_image_with_mixed_eligible_and_ineligible_objects_still_writes_the_eligible_ones() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        annotations = {
            "img1": _ann("img1", [_obj("o0", 0), _obj("o1", 5)]),
            "img2": _ann("img2", [_obj("o2", 0)]),
        }
        det, _root = _detector(annotations, tmp_path)
        staging = tmp_path / "staging"
        data_yaml, num_images = det._assemble_dataset(staging, ["a"] * 6, trainable_class_ids={0})

        assert num_images == 2  # both images still written - not skipped
        label_files = list((staging / "train" / "labels").glob("*.txt")) + list(
            (staging / "val" / "labels").glob("*.txt")
        )
        img1_label = next(f for f in label_files if f.name == "img1.txt")
        content = img1_label.read_text()
        assert content.startswith("0 ")
        assert "5 " not in content


def test_data_yaml_names_and_nc_stay_the_full_unfiltered_class_map() -> None:
    """Ultralytics requires len(names) == nc exactly (verified live against
    the installed version - a true sparse dict is rejected) - the fix here
    is at the label-line level, not the class-map level, so data.yaml must
    stay untouched regardless of which classes are trainable."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        annotations = {"img1": _ann("img1", [_obj("o0", 0)])}
        det, _root = _detector(annotations, tmp_path)
        staging = tmp_path / "staging"
        det._assemble_dataset(staging, ["a", "b", "c"], trainable_class_ids={0})

        import yaml

        content = yaml.safe_load((staging / "data.yaml").read_text())
        assert content["nc"] == 3
        assert content["names"] == ["a", "b", "c"]


def test_all_classes_excluded_produces_zero_usable_images_not_a_crash() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        annotations = {"img1": _ann("img1", [_obj("o0", 0)])}
        det, _root = _detector(annotations, tmp_path)
        staging = tmp_path / "staging"
        data_yaml, num_images = det._assemble_dataset(staging, ["a"], trainable_class_ids=set())
        assert num_images == 0
        assert data_yaml.exists()  # still writes a valid (if untrainable-on) data.yaml, doesn't crash


def test_confirmed_empty_frame_still_counts_regardless_of_eligibility() -> None:
    """A no_objects_confirmed negative sample has no objects to filter at
    all - eligibility gating must not accidentally suppress it."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ann1 = ImageAnnotations(image_id="img1", file_name="img1.jpg", width=100, height=100,
                                 objects=[], completed=True, no_objects_confirmed=True)
        ann2 = ImageAnnotations(image_id="img2", file_name="img2.jpg", width=100, height=100,
                                 objects=[], completed=True, no_objects_confirmed=True)
        det, _root = _detector({"img1": ann1, "img2": ann2}, tmp_path)
        staging = tmp_path / "staging"
        _data_yaml, num_images = det._assemble_dataset(staging, ["a"], trainable_class_ids=set())
        assert num_images == 2


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
