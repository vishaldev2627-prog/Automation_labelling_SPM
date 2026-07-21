"""SAM2 inference service.

Wraps Meta's Segment Anything Model 2 for box-prompted (and point-prompted)
mask generation. Falls back gracefully with clear errors if SAM2 / torch /
a GPU are unavailable, and supports an ONNX runtime backend as an alternative
to the PyTorch checkpoint for lighter-weight / CPU deployments.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SAMNotAvailableError(RuntimeError):
    """Raised when SAM2 cannot be initialized (missing deps/checkpoint/GPU)."""


class MaskResult:
    __slots__ = ("masks", "scores")

    def __init__(self, masks: np.ndarray, scores: np.ndarray):
        self.masks = masks  # (N, H, W) bool
        self.scores = scores  # (N,)

    @property
    def best_index(self) -> int:
        return int(np.argmax(self.scores))


class SAMService:
    """Thread-safe singleton wrapper around a SAM2 image predictor."""

    def __init__(
        self,
        checkpoint_path: str,
        model_cfg: str,
        device: str = "auto",
        use_onnx: bool = False,
        onnx_encoder_path: str = "",
        onnx_decoder_path: str = "",
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._model_cfg = model_cfg
        self._device_pref = device
        self._use_onnx = use_onnx
        self._onnx_encoder_path = onnx_encoder_path
        self._onnx_decoder_path = onnx_decoder_path

        self._predictor = None
        self._device = None
        self._lock = threading.Lock()
        self._current_image_key: Optional[str] = None
        self._backend = "onnx" if use_onnx else "pytorch"
        self._init_error: Optional[str] = None
        self._initialize()

    # ------------------------------------------------------------------ init
    def _resolve_device(self) -> str:
        import torch

        if self._device_pref != "auto":
            return self._device_pref
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _initialize(self) -> None:
        try:
            if self._use_onnx:
                self._initialize_onnx()
            else:
                self._initialize_pytorch()
            logger.info("SAM service initialized (backend=%s, device=%s)", self._backend, self._device)
        except Exception as exc:  # noqa: BLE001 - we want to capture and surface any init failure
            self._init_error = str(exc)
            logger.exception("Failed to initialize SAM service; falling back to unavailable state")

    def _initialize_pytorch(self) -> None:
        import torch

        self._device = self._resolve_device()
        ckpt = Path(self._checkpoint_path)
        if not ckpt.exists():
            raise SAMNotAvailableError(
                f"SAM2 checkpoint not found at {ckpt}. Download it and set SAM_CHECKPOINT in .env "
                "(see README for download instructions)."
            )

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise SAMNotAvailableError(
                "The 'sam2' package is not installed. Run `pip install -e ./sam2` "
                "or `pip install sam-2` as documented in the README."
            ) from exc

        model = build_sam2(self._model_cfg, str(ckpt), device=self._device)
        self._predictor = SAM2ImagePredictor(model)

    def _initialize_onnx(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise SAMNotAvailableError("onnxruntime is not installed but SAM_USE_ONNX=true") from exc

        enc_path = Path(self._onnx_encoder_path)
        dec_path = Path(self._onnx_decoder_path)
        if not enc_path.exists() or not dec_path.exists():
            raise SAMNotAvailableError(
                f"ONNX encoder/decoder not found at {enc_path} / {dec_path}. "
                "Export SAM2 to ONNX or disable SAM_USE_ONNX."
            )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self._device_pref != "cpu" else [
            "CPUExecutionProvider"
        ]
        self._onnx_encoder = ort.InferenceSession(str(enc_path), providers=providers)
        self._onnx_decoder = ort.InferenceSession(str(dec_path), providers=providers)
        self._device = "onnxruntime"

    # --------------------------------------------------------------- status
    @property
    def is_available(self) -> bool:
        return self._init_error is None and (self._predictor is not None or self._backend == "onnx")

    @property
    def status(self) -> dict:
        return {
            "available": self.is_available,
            "backend": self._backend,
            "device": self._device,
            "error": self._init_error,
        }

    # ------------------------------------------------------------ inference
    def set_image(self, image_rgb: np.ndarray, cache_key: str) -> None:
        """Compute (or reuse cached) image embeddings for the given image."""
        if not self.is_available:
            raise SAMNotAvailableError(self._init_error or "SAM model unavailable")

        with self._lock:
            if self._current_image_key == cache_key:
                return
            self._predictor.set_image(image_rgb)
            self._current_image_key = cache_key

    def predict_box(
        self,
        image_rgb: np.ndarray,
        cache_key: str,
        box_xyxy: tuple[int, int, int, int],
        positive_points: Optional[list[tuple[float, float]]] = None,
        negative_points: Optional[list[tuple[float, float]]] = None,
    ) -> MaskResult:
        """Run SAM2 with a bounding-box prompt (plus optional point refinements)."""
        if not self.is_available:
            raise SAMNotAvailableError(self._init_error or "SAM model unavailable")

        self.set_image(image_rgb, cache_key)

        point_coords = None
        point_labels = None
        pts = [(p[0], p[1], 1) for p in (positive_points or [])] + [
            (p[0], p[1], 0) for p in (negative_points or [])
        ]
        if pts:
            point_coords = np.array([[p[0], p[1]] for p in pts], dtype=np.float32)
            point_labels = np.array([p[2] for p in pts], dtype=np.int32)

        with self._lock:
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=np.array(box_xyxy, dtype=np.float32),
                multimask_output=True,
            )
        return MaskResult(masks=masks.astype(bool), scores=scores)

    def predict_points(
        self,
        image_rgb: np.ndarray,
        cache_key: str,
        positive_points: list[tuple[float, float]],
        negative_points: Optional[list[tuple[float, float]]] = None,
    ) -> MaskResult:
        """Run SAM2 with only click prompts (magic wand / refine mode)."""
        if not self.is_available:
            raise SAMNotAvailableError(self._init_error or "SAM model unavailable")
        self.set_image(image_rgb, cache_key)

        pts = [(p[0], p[1], 1) for p in positive_points] + [(p[0], p[1], 0) for p in (negative_points or [])]
        point_coords = np.array([[p[0], p[1]] for p in pts], dtype=np.float32)
        point_labels = np.array([p[2] for p in pts], dtype=np.int32)

        with self._lock:
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        return MaskResult(masks=masks.astype(bool), scores=scores)


_service_singleton: Optional[SAMService] = None
_service_lock = threading.Lock()


def get_sam_service() -> SAMService:
    global _service_singleton
    if _service_singleton is None:
        with _service_lock:
            if _service_singleton is None:
                from app.config import get_settings

                s = get_settings()
                _service_singleton = SAMService(
                    checkpoint_path=s.sam_checkpoint,
                    model_cfg=s.sam_model_cfg,
                    device=s.sam_device,
                    use_onnx=s.sam_use_onnx,
                    onnx_encoder_path=s.sam_onnx_encoder_path,
                    onnx_decoder_path=s.sam_onnx_decoder_path,
                )
    return _service_singleton
