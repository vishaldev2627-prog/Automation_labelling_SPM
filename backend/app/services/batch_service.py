"""Background batch mask generation across many images.

Uses a bounded thread pool so GPU inference isn't oversubscribed, and tracks
job progress in-memory so the frontend can poll for status.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.models.schemas import BatchJobStatus, ObjectStatus
from app.services.dataset_service import DatasetService
from app.services.mask_generation_service import MaskGenerationService
from app.utils.file_utils import new_id

logger = logging.getLogger(__name__)


class BatchService:
    def __init__(self, dataset_service: DatasetService, mask_service: MaskGenerationService, max_workers: int = 2) -> None:
        self._ds = dataset_service
        self._mask_service = mask_service
        self._max_workers = max_workers
        self._jobs: dict[str, BatchJobStatus] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)  # jobs run sequentially; inference itself is the bottleneck

    def start_job(self, image_ids: list[str], overwrite: bool) -> BatchJobStatus:
        job_id = new_id()
        status = BatchJobStatus(
            job_id=job_id,
            total=len(image_ids),
            processed=0,
            failed=0,
            status="running",
            current_image=None,
            started_at=time.time(),
            updated_at=time.time(),
        )
        with self._lock:
            self._jobs[job_id] = status

        self._executor.submit(self._run_job, job_id, image_ids, overwrite)
        return status

    def _run_job(self, job_id: str, image_ids: list[str], overwrite: bool) -> None:
        for image_id in image_ids:
            self._update(job_id, current_image=image_id)
            try:
                annotations = self._ds.get_annotations(image_id)
                needs_processing = overwrite or any(
                    obj.status == ObjectStatus.PENDING for obj in annotations.objects
                )
                if needs_processing:
                    annotations = self._mask_service.generate_all_masks(annotations, overwrite=overwrite)
                    self._ds.save_annotations(annotations)
                self._update(job_id, processed_delta=1)
            except Exception:
                logger.exception("Batch processing failed for image %s", image_id)
                self._update(job_id, failed_delta=1, processed_delta=1)

        self._update(job_id, status="completed", current_image=None)

    def _update(
        self,
        job_id: str,
        processed_delta: int = 0,
        failed_delta: int = 0,
        status: Optional[str] = None,
        current_image: Optional[str] = "__unset__",
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.processed += processed_delta
            job.failed += failed_delta
            if status:
                job.status = status
            if current_image != "__unset__":
                job.current_image = current_image
            job.updated_at = time.time()

    def get_job(self, job_id: str) -> Optional[BatchJobStatus]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[BatchJobStatus]:
        return list(self._jobs.values())


_batch_service: Optional[BatchService] = None
_lock = threading.Lock()


def get_batch_service() -> BatchService:
    global _batch_service
    if _batch_service is None:
        with _lock:
            if _batch_service is None:
                from app.config import get_settings
                from app.services.dataset_service import get_dataset_service
                from app.services.mask_generation_service import get_mask_generation_service

                settings = get_settings()
                _batch_service = BatchService(
                    get_dataset_service(), get_mask_generation_service(), max_workers=settings.batch_max_workers
                )
    return _batch_service
