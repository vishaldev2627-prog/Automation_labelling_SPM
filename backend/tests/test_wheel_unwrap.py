"""Tests for the wheel log-polar unwrap, generated at export time (W-5, D-Q5).

The problem this defends against: `pipeline.md` §5.5 scores wheel-shelling on
a log-polar unwrapped crop, a coordinate transform parameterised by circle
center, radius, and output size. A mask drawn *directly* in unwrapped space
is meaningless the moment those parameters change - and they will, since this
tool's wheel geometry constants are provisional (D-Q5: "wheel specs are to be
considered in your own accord for now"). So annotation stays in raw frame
space and the unwrap is regenerated at export - see log_polar_service's
module docstring and export_service._write_wheel_unwraps.

Two things are pinned here:

1. `find_wheel_circle` - the circle is seeded from the annotated `wheel`
   object's own bbox in *this* frame, not a fixed pixel constant (a fixed
   radius would be wrong the moment camera distance/zoom varies between
   frames, and there's no camera calibration yet to convert a real diameter
   into a per-frame pixel radius - see config.py's wheel_unwrap_* comment).
2. `unwrap_log_polar` - `cv2.warpPolar`'s own axis convention (`dsize=(w,h)`
   maps `w` to radius, `h` to angle - confirmed empirically, not documented)
   is the opposite of the wide-strip orientation this function must return,
   so getting the swap+transpose wrong would silently produce a tall narrow
   image instead of pipeline.md's "flat strip". A ring at a known radius is
   used to pin the correct final orientation.

Pure numpy/OpenCV, no DatasetService/Postgres - same reasoning as
test_split_integrity.py and test_snapshot_service.py.

    cd backend && python -m pytest tests/test_wheel_unwrap.py
    cd backend && python -m tests.test_wheel_unwrap   # no pytest
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from app.models.schemas import AnnotationObject, BoundingBox, ObjectStatus
from app.services.log_polar_service import find_wheel_circle, unwrap_log_polar


def _wheel_object(
    x_center: float = 0.5,
    y_center: float = 0.5,
    width: float = 0.4,
    height: float = 0.4,
    class_name: str = "wheel",
    status: ObjectStatus = ObjectStatus.CONFIRMED,
) -> AnnotationObject:
    return AnnotationObject(
        id="w1",
        class_id=0,
        class_name=class_name,
        bbox=BoundingBox(x_center=x_center, y_center=y_center, width=width, height=height),
        status=status,
    )


# --------------------------------------------------------- find_wheel_circle

def test_no_wheel_object_returns_none() -> None:
    """No annotated wheel in this frame - the caller must skip the unwrap
    rather than guess a circle from nothing."""
    circle = find_wheel_circle([], 1000, 1000, "wheel", radius_padding_pct=5.0)
    assert circle is None


def test_rejected_wheel_object_does_not_count() -> None:
    circle = find_wheel_circle(
        [_wheel_object(status=ObjectStatus.REJECTED)], 1000, 1000, "wheel", radius_padding_pct=5.0
    )
    assert circle is None


def test_class_name_matched_case_insensitively() -> None:
    """Class names are annotator-entered free text, same convention
    safety_critical/fine_structure keyword seeding already uses."""
    circle = find_wheel_circle(
        [_wheel_object(class_name="WHEEL")], 1000, 1000, "wheel", radius_padding_pct=0.0
    )
    assert circle is not None


def test_circle_center_and_radius_derived_from_bbox() -> None:
    """A centered, square 400x400px bbox on a 1000x1000 image: center should
    land at (500, 500), radius at half the box extent before padding."""
    circle = find_wheel_circle(
        [_wheel_object(x_center=0.5, y_center=0.5, width=0.4, height=0.4)],
        1000,
        1000,
        "wheel",
        radius_padding_pct=0.0,
    )
    assert circle is not None
    assert circle.center_x == 500.0
    assert circle.center_y == 500.0
    assert circle.radius == 200.0  # half of 400px


def test_radius_padding_expands_beyond_the_bbox() -> None:
    circle = find_wheel_circle(
        [_wheel_object(width=0.4, height=0.4)], 1000, 1000, "wheel", radius_padding_pct=10.0
    )
    assert circle is not None
    assert math.isclose(circle.radius, 220.0)  # 200px * 1.10


def test_non_square_bbox_uses_the_larger_half_extent() -> None:
    """Erring toward the larger dimension means the unwrap radius never clips
    inside the actual tire edge - only ever includes a sliver of background."""
    circle = find_wheel_circle(
        [_wheel_object(width=0.2, height=0.4)], 1000, 1000, "wheel", radius_padding_pct=0.0
    )
    assert circle is not None
    assert circle.radius == 200.0  # from height (0.4 * 1000 / 2), not width


# ------------------------------------------------------------ unwrap_log_polar

def test_unwrap_output_shape_matches_configured_dimensions() -> None:
    image = np.zeros((300, 300), dtype=np.uint8)
    circle = find_wheel_circle([_wheel_object()], 300, 300, "wheel", radius_padding_pct=0.0)
    assert circle is not None
    out = unwrap_log_polar(image, circle, out_width=256, out_height=64, log_scale=False)
    assert out.shape == (64, 256)


def test_unwrap_preserves_channel_count() -> None:
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    circle = find_wheel_circle([_wheel_object()], 300, 300, "wheel", radius_padding_pct=0.0)
    assert circle is not None
    out = unwrap_log_polar(image, circle, out_width=256, out_height=64, log_scale=False)
    assert out.shape == (64, 256, 3)


def test_a_ring_becomes_a_horizontal_band_spanning_the_width() -> None:
    """The orientation pin: cv2.warpPolar's own dsize=(w,h) maps w to radius
    and h to angle (opposite of what this function must return - a wide
    strip with radius on the short/height axis, circumference on the
    long/width axis). Getting the internal swap+transpose wrong would put a
    constant-radius ring in a narrow *vertical* band instead of a horizontal
    one - this is exactly the mistake made and caught while building this."""
    image = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(image, (150, 150), 100, 255, thickness=10)

    from app.services.log_polar_service import WheelCircle

    circle = WheelCircle(center_x=150.0, center_y=150.0, radius=140.0)
    out = unwrap_log_polar(image, circle, out_width=512, out_height=128, log_scale=False)

    ys, xs = np.where(out > 150)
    assert len(ys) > 0
    # Narrow band on the height (radius) axis...
    assert (ys.max() - ys.min()) < out.shape[0] * 0.3
    # ...spanning nearly the entire width (angle/circumference) axis.
    assert (xs.max() - xs.min()) > out.shape[1] * 0.8


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
