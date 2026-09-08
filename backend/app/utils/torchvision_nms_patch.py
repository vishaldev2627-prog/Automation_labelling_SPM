"""Pure-PyTorch NMS fallback for this Jetson deployment.

torchvision==0.15.2 here is a generic PyPI wheel, not one built against the
dustynv torch 2.1.0a0+41361538.nv23.06 nightly (see backend/Dockerfile.jetson's
own comment on the pairing). Its compiled C++ extension (`torchvision._C`)
has an ABI mismatch with that torch build - `torchvision.ops.nms` raises
"Couldn't load custom C++ ops" the moment it's actually called.

The Dockerfile comment assumed Ultralytics falls back to a pure-torch NMS
automatically when this happens - it doesn't (verified against the installed
version: `ultralytics/utils/nms.py::non_max_suppression` calls
`torchvision.ops.nms` unconditionally, no try/except). That's what breaks
every detector training run at the very first warmup inference
(`ultralytics.utils.checks.check_amp`), and therefore the "no recommendations
after 100+ annotated images" symptom - the detector this app is supposed to
auto-fill new images from has never been able to finish training here.

SAM2 and the mobilenet similarity encoder are unaffected (Dockerfile's own
comment: they only call pure-Python torchvision APIs), so this patches nms
only, not the whole torchvision.ops surface.
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_applied = False


def _pure_torch_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Greedy NMS via plain tensor ops - O(n^2), fine at YOLO per-image box
    counts (tens to low hundreds), unlike torchvision's compiled version this
    needs no C++/CUDA extension to work."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    x1, y1, x2, y2 = boxes.unbind(-1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)

    keep: list[int] = []
    while order.numel() > 0:
        i = order[0]
        keep.append(int(i))
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-9)
        order = rest[iou <= iou_threshold]

    return torch.as_tensor(keep, dtype=torch.int64, device=boxes.device)


def apply() -> None:
    """Idempotent - safe to call from multiple entrypoints."""
    global _applied
    if _applied:
        return

    import torchvision.ops as tv_ops
    import torchvision.ops.boxes as tv_boxes

    try:
        tv_boxes.nms(
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            0.5,
        )
        logger.info("torchvision.ops.nms works natively; skipping pure-torch NMS patch.")
        _applied = True
        return
    except Exception as exc:
        logger.warning("torchvision.ops.nms is broken (%s) - patching in a pure-torch fallback.", exc)

    tv_boxes.nms = _pure_torch_nms
    tv_ops.nms = _pure_torch_nms
    _applied = True
