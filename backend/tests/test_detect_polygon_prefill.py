"""Tests for DetectorService.detect()'s polygon-gating logic (YOLO11-seg
Option B): a detection's predicted mask polygon is only trusted and
returned when its confidence clears mask_confidence_threshold - the same
"never show a bad mask" bar SAM2's own path uses - and gracefully degrades
to box-only output for a checkpoint with no seg head at all (result.masks
is None), so a leftover pre-migration yolov8s.pt registry entry keeps
working.

Exercised against duck-typed fakes standing in for ultralytics' Results/
Boxes/Masks objects - no real model weights, no GPU, no download. Real
YOLO11-seg inference end-to-end is verified live against the running dev
stack (same reasoning test_golden_eval_service.py gives for its own
model.val() call).

    cd backend && python -m pytest tests/test_detect_polygon_prefill.py
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from app.services.detector_service import DetectorService


class _Tensor:
    """Just enough of a torch tensor's API for detect() to call - .item()
    and .tolist() are all it touches."""

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value

    def tolist(self):
        return self._value

    def __getitem__(self, idx):
        return _Tensor(self._value[idx])


class _FakeBox:
    def __init__(self, class_id: int, xyxy, conf: float):
        self.cls = _Tensor(class_id)
        self.xyxy = _Tensor([xyxy])
        self.conf = _Tensor(conf)


class _FakeMasks:
    def __init__(self, xyn):
        self.xyn = xyn


class _FakeResult:
    def __init__(self, boxes, masks, orig_shape=(100, 100)):
        self.boxes = boxes
        self.masks = masks
        self.orig_shape = orig_shape


def _detector(predict_return) -> DetectorService:
    """A DetectorService with is_active()/model loading stubbed out, so
    detect()'s own polygon-gating logic runs against a controlled fake
    model - no registry, no dataset_service, no real weights needed."""
    det = DetectorService(dataset_service=MagicMock(), models_dir=Path("/tmp/unused_models_dir"))
    det.is_active = lambda: True
    fake_model = MagicMock()
    fake_model.predict.return_value = predict_return
    det._ensure_model_loaded = lambda: fake_model
    return det


def test_confident_detection_gets_its_predicted_polygon() -> None:
    box = _FakeBox(class_id=0, xyxy=[10, 10, 30, 30], conf=0.9)
    xyn = [[[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]]
    det = _detector([_FakeResult(boxes=[box], masks=_FakeMasks(xyn))])
    out = det.detect(Path("img.jpg"), classes=["a"])
    assert len(out) == 1
    class_id, bbox, confidence, polygon = out[0]
    assert class_id == 0
    assert confidence == 0.9
    assert len(polygon) == 4


def test_low_confidence_detection_gets_box_only_no_polygon() -> None:
    """Below the mask-confidence bar: box-prompt-worthy, but the polygon
    isn't trusted - same 'never show a bad mask' rule SAM2 itself uses."""
    box = _FakeBox(class_id=0, xyxy=[10, 10, 30, 30], conf=0.3)
    xyn = [[[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]]
    det = _detector([_FakeResult(boxes=[box], masks=_FakeMasks(xyn))])
    out = det.detect(Path("img.jpg"), classes=["a"])
    assert len(out) == 1
    _, _, confidence, polygon = out[0]
    assert confidence == 0.3
    assert polygon == []


def test_confidence_exactly_at_threshold_is_not_trusted() -> None:
    """Boundary case: the gate is strictly `>`, matching
    mask_generation_service's own SAM2 threshold check - equal-to fails."""
    box = _FakeBox(class_id=0, xyxy=[10, 10, 30, 30], conf=0.5)  # default threshold is 0.5
    xyn = [[[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]]
    det = _detector([_FakeResult(boxes=[box], masks=_FakeMasks(xyn))])
    out = det.detect(Path("img.jpg"), classes=["a"])
    _, _, _, polygon = out[0]
    assert polygon == []


def test_legacy_detection_only_checkpoint_has_no_masks_attribute_set() -> None:
    """A pre-migration yolov8s.pt (detection-only) checkpoint's result has
    masks=None - must degrade to box-only for every detection, never crash."""
    box = _FakeBox(class_id=0, xyxy=[10, 10, 30, 30], conf=0.95)
    det = _detector([_FakeResult(boxes=[box], masks=None)])
    out = det.detect(Path("img.jpg"), classes=["a"])
    assert len(out) == 1
    _, _, confidence, polygon = out[0]
    assert confidence == 0.95
    assert polygon == []


def test_degenerate_two_point_contour_is_treated_as_no_polygon() -> None:
    """A contour ultralytics considers too small/degenerate to be a real
    polygon (<3 points) must not be handed to the annotation pre-fill."""
    box = _FakeBox(class_id=0, xyxy=[10, 10, 30, 30], conf=0.99)
    xyn = [[[0.1, 0.1], [0.3, 0.3]]]  # only 2 points
    det = _detector([_FakeResult(boxes=[box], masks=_FakeMasks(xyn))])
    out = det.detect(Path("img.jpg"), classes=["a"])
    _, _, _, polygon = out[0]
    assert polygon == []


def test_class_id_beyond_known_classes_is_filtered_out() -> None:
    """The active model's class list can be ahead of what this dataset view
    currently knows (e.g. a stale registry) - such detections are dropped
    entirely, not passed through with a garbage class_id."""
    boxes = [
        _FakeBox(class_id=0, xyxy=[10, 10, 30, 30], conf=0.9),
        _FakeBox(class_id=5, xyxy=[40, 40, 60, 60], conf=0.9),
    ]
    xyn = [
        [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3]],
        [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6]],
    ]
    det = _detector([_FakeResult(boxes=boxes, masks=_FakeMasks(xyn))])
    out = det.detect(Path("img.jpg"), classes=["only_one_class"])
    assert len(out) == 1
    assert out[0][0] == 0


def test_inactive_detector_returns_empty_without_touching_the_model() -> None:
    det = _detector([_FakeResult(boxes=[], masks=None)])
    det.is_active = lambda: False
    det._ensure_model_loaded = lambda: (_ for _ in ()).throw(AssertionError("must not load a model when inactive"))
    assert det.detect(Path("img.jpg"), classes=["a"]) == []


def test_no_results_at_all_returns_empty() -> None:
    det = _detector([])
    assert det.detect(Path("img.jpg"), classes=["a"]) == []


def test_no_detections_in_frame_returns_empty() -> None:
    det = _detector([_FakeResult(boxes=[], masks=_FakeMasks([]))])
    assert det.detect(Path("img.jpg"), classes=["a"]) == []


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
