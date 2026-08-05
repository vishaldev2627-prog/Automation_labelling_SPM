"""Wheel log-polar unwrap, generated at export time (W-5, D-Q5).

**Why this exists at export, not at annotation time.** `docs/pipeline.md`
§5.5 runs wheel-shelling segmentation on a log-polar unwrapped crop - a
coordinate transform parameterised by circle center, radius, and output size.
A mask drawn directly in unwrapped space is meaningless under different
parameters: change the assumed radius and every previously-drawn mask
silently misaligns. Since this tool's wheel geometry constants are
provisional (D-Q5: "wheel specs are to be considered in your own accord for
now") and will change once real specs arrive, annotation stays in raw frame
space - where the data is ground truth - and the unwrap is regenerated at
export from current parameters. A spec change then becomes a re-export, not
a re-annotation of every wheel.

Deliberately pure / filesystem-and-numpy only, no DatasetService or Postgres
- mirrors export_service._enforce_split_integrity's split from its DB-facing
caller, so the transform itself is testable in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.models.schemas import AnnotationObject, ObjectStatus


@dataclass
class WheelCircle:
    center_x: float  # px
    center_y: float  # px
    radius: float  # px, already padded


def find_wheel_circle(
    objects: list[AnnotationObject],
    img_width: int,
    img_height: int,
    wheel_class_name: str,
    radius_padding_pct: float,
) -> Optional[WheelCircle]:
    """Seed the unwrap circle from the annotated wheel object's own bbox in
    this frame, rather than a fixed pixel constant - see module docstring for
    why. Returns None if no live (non-rejected) object of `wheel_class_name`
    exists on this image; the caller should skip the unwrap for that image
    rather than guess.

    Matched case-insensitively on `class_name`, the same convention the
    safety_critical/fine_structure keyword seeding already uses, since class
    names are annotator-entered free text.

    The bbox's larger half-extent becomes the radius - a wheel drawn as a
    tight box around the tire is very close to circular, and erring toward
    the larger dimension means the unwrap radius never clips inside the
    actual tire edge, only ever includes a sliver of background.
    """
    wheel = next(
        (
            obj
            for obj in objects
            if obj.status != ObjectStatus.REJECTED and obj.class_name.strip().lower() == wheel_class_name.lower()
        ),
        None,
    )
    if wheel is None:
        return None

    cx = wheel.bbox.x_center * img_width
    cy = wheel.bbox.y_center * img_height
    half_w = wheel.bbox.width * img_width / 2
    half_h = wheel.bbox.height * img_height / 2
    radius = max(half_w, half_h) * (1 + radius_padding_pct / 100.0)
    return WheelCircle(center_x=cx, center_y=cy, radius=radius)


def unwrap_log_polar(
    image: np.ndarray,
    circle: WheelCircle,
    out_width: int,
    out_height: int,
    log_scale: bool,
) -> np.ndarray:
    """Re-map the disc at `circle` into a flat `out_height` x `out_width`
    strip - circumference along the width (long) axis, radius along the
    height (short) axis - so a ring defect (shelling, tread crack) becomes an
    ordinary axis-aligned segmentation target (`docs/pipeline.md` §5.5's own
    definition of the term). `out_height` is intentionally small relative to
    `out_width`: only the outer ring (the tire/tread) carries any defect
    signal, not the wheel's interior.

    `log_scale=True` matches the literal "log-polar" naming this tool's own
    docs and manifest field use; flip to linear radial spacing with a config
    change if the pipeline team's actual model expects that instead - the
    flag exists because nothing here has been validated against their model.

    `cv2.warpPolar`'s own `dsize=(w, h)` maps `w` to the *radius* axis and `h`
    to the *angle* axis of its output (undocumented, confirmed empirically
    with a single known point at radius=50/angle=0 landing exactly where the
    radius formula predicts) - the opposite of the wide-strip orientation
    this function returns. So `dsize` is passed as `(out_height, out_width)`
    - radius bins first, angle bins second - and the raw result is
    transposed to put radius on the short axis and angle/circumference on
    the long one, matching every other image array here (rows=height,
    cols=width).
    """
    flags = cv2.WARP_POLAR_LOG if log_scale else cv2.WARP_POLAR_LINEAR
    raw = cv2.warpPolar(
        image,
        (out_height, out_width),
        (circle.center_x, circle.center_y),
        circle.radius,
        flags | cv2.INTER_LINEAR,
    )
    return np.transpose(raw, (1, 0, 2)) if raw.ndim == 3 else raw.T
