"""Visual similarity index over the dataset.

Computes a compact embedding per image with a small pretrained CNN
(MobileNetV3-Small, ImageNet features) and keeps an in-memory + on-disk
cosine-similarity index so near-duplicate images (e.g. adjacent video
frames, or repeated shots of the same component) can be found cheaply.

Deliberately not a "real" vector database — at dataset sizes in the low
thousands, a brute-force cosine similarity over a single in-memory matrix
is simpler, has no extra service to run, and is fast enough (milliseconds).
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from app.models.schemas import SimilarNeighbor, SimilarityIndexStatus
from app.services.dataset_service import DatasetService
from app.utils.file_utils import new_id
from app.utils.image_utils import compute_file_hash, read_image_rgb

logger = logging.getLogger(__name__)

INDEX_FILENAME = "_similarity_index.npz"
EMBED_INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SimilarityService:
    """Embeds images and answers nearest-neighbor queries against the index."""

    def __init__(self, dataset_service: DatasetService) -> None:
        self._ds = dataset_service
        self._model = None
        self._model_lock = threading.Lock()

        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._hashes: dict[str, str] = {}
        self._embeddings: Optional[np.ndarray] = None  # (N, D) float32, L2-normalized
        self._id_to_row: dict[str, int] = {}
        self._loaded_for_dir: Optional[Path] = None

        self._jobs: dict[str, SimilarityIndexStatus] = {}
        self._jobs_lock = threading.Lock()
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(max_workers=1)

    # ------------------------------------------------------------- model
    def _get_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    import torch
                    import torchvision

                    weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                    model = torchvision.models.mobilenet_v3_small(weights=weights)
                    model.classifier = torch.nn.Identity()  # use pooled features, not classifier logits
                    model.eval()
                    self._model = model
        return self._model

    def _embed_array(self, img_rgb: np.ndarray) -> np.ndarray:
        import cv2
        import torch

        resized = cv2.resize(img_rgb, (EMBED_INPUT_SIZE, EMBED_INPUT_SIZE), interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()
        with torch.no_grad():
            feats = self._get_model()(tensor)
        vec = feats.squeeze(0).numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_image(self, path: Path) -> np.ndarray:
        return self._embed_array(read_image_rgb(path))

    # --------------------------------------------------------------- index
    @property
    def _index_path(self) -> Path:
        return self._ds.state_dir / INDEX_FILENAME

    def _ensure_loaded(self) -> None:
        root = self._ds.state_dir
        with self._lock:
            if self._loaded_for_dir == root:
                return
            self._ids = []
            self._hashes = {}
            self._embeddings = None
            self._id_to_row = {}
            path = self._index_path
            if path.exists():
                try:
                    data = np.load(path, allow_pickle=True)
                    ids = list(data["ids"])
                    hashes = list(data["hashes"])
                    embeddings = data["embeddings"].astype(np.float32)
                    self._ids = ids
                    self._hashes = dict(zip(ids, hashes))
                    self._embeddings = embeddings
                    self._id_to_row = {iid: i for i, iid in enumerate(ids)}
                except Exception:
                    logger.exception("Failed to load similarity index at %s; starting fresh", path)
            self._loaded_for_dir = root

    def _persist(self) -> None:
        with self._lock:
            if not self._ids:
                return
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                self._index_path,
                ids=np.array(self._ids, dtype=object),
                hashes=np.array([self._hashes[i] for i in self._ids], dtype=object),
                embeddings=self._embeddings,
            )

    def _upsert(self, image_id: str, file_hash: str, vector: np.ndarray) -> None:
        with self._lock:
            row = self._id_to_row.get(image_id)
            if row is not None:
                self._embeddings[row] = vector
                self._hashes[image_id] = file_hash
                return
            self._ids.append(image_id)
            self._hashes[image_id] = file_hash
            if self._embeddings is None:
                self._embeddings = vector.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, vector.reshape(1, -1)])
            self._id_to_row[image_id] = len(self._ids) - 1

    def ensure_indexed(self, image_id: str, persist: bool = True) -> bool:
        """Embed a single image on-demand if it's missing or stale in the index.
        Returns True if it was (re)computed. `persist=False` lets bulk callers
        (see `_run_reindex`) batch disk writes instead of rewriting the whole
        index file after every single image."""
        self._ensure_loaded()
        path = self._ds.get_image_path(image_id)
        file_hash = compute_file_hash(path)
        with self._lock:
            if self._hashes.get(image_id) == file_hash:
                return False
        vector = self.embed_image(path)
        with self._lock:
            self._upsert(image_id, file_hash, vector)
            if persist:
                self._persist()
        return True

    def nearest_neighbors(self, image_id: str, k: int = 5, min_similarity: float = 0.0) -> list[SimilarNeighbor]:
        self._ensure_loaded()
        self.ensure_indexed(image_id)
        with self._lock:
            row = self._id_to_row.get(image_id)
            if row is None or self._embeddings is None or len(self._ids) < 2:
                return []
            query = self._embeddings[row]
            sims = self._embeddings @ query  # both L2-normalized -> cosine similarity
            order = np.argsort(-sims)
            results = []
            for idx in order:
                candidate_id = self._ids[idx]
                if candidate_id == image_id:
                    continue
                score = float(sims[idx])
                if score < min_similarity:
                    break
                try:
                    file_name = self._ds.get_image_path(candidate_id).name
                except Exception:
                    continue
                results.append(SimilarNeighbor(image_id=candidate_id, file_name=file_name, similarity=score))
                if len(results) >= k:
                    break
            return results

    # ---------------------------------------------------------------- jobs
    def start_reindex(self, image_ids: Optional[list[str]] = None) -> SimilarityIndexStatus:
        self._ensure_loaded()
        ids = image_ids or self._ds.image_ids()
        job_id = new_id()
        status = SimilarityIndexStatus(
            job_id=job_id,
            total=len(ids),
            processed=0,
            status="running",
            started_at=time.time(),
            updated_at=time.time(),
            indexed_images=len(self._ids),
        )
        with self._jobs_lock:
            self._jobs[job_id] = status
        self._executor.submit(self._run_reindex, job_id, ids)
        return status

    def _run_reindex(self, job_id: str, image_ids: list[str]) -> None:
        PERSIST_EVERY = 50
        since_persist = 0
        for image_id in image_ids:
            try:
                if self.ensure_indexed(image_id, persist=False):
                    since_persist += 1
            except Exception:
                logger.exception("Failed to index image %s for similarity search", image_id)
            if since_persist >= PERSIST_EVERY:
                with self._lock:
                    self._persist()
                since_persist = 0
            with self._jobs_lock:
                job = self._jobs[job_id]
                job.processed += 1
                job.indexed_images = len(self._ids)
                job.updated_at = time.time()
        if since_persist:
            with self._lock:
                self._persist()
        with self._jobs_lock:
            self._jobs[job_id].status = "completed"
            self._jobs[job_id].updated_at = time.time()
        logger.info("Similarity reindex job %s completed (%d images)", job_id, len(image_ids))

    def get_job(self, job_id: str) -> Optional[SimilarityIndexStatus]:
        return self._jobs.get(job_id)


_similarity_service: Optional[SimilarityService] = None
_lock = threading.Lock()


def get_similarity_service() -> SimilarityService:
    global _similarity_service
    if _similarity_service is None:
        with _lock:
            if _similarity_service is None:
                from app.services.dataset_service import get_dataset_service

                _similarity_service = SimilarityService(get_dataset_service())
    return _similarity_service
