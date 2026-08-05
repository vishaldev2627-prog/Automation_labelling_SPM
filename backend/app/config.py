"""Application configuration loaded from environment variables / .env file."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the annotation tool backend."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Dataset
    dataset_path: str = "../dataset"
    exports_path: str = "../exports"
    models_path: str = "../models"
    state_dir_name: str = ".annotation_state"

    # SAM2
    sam_checkpoint: str = "../models/sam2.1_hiera_large.pt"
    sam_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_device: Literal["cuda", "cpu", "mps", "auto"] = "auto"
    sam_use_onnx: bool = False
    sam_onnx_encoder_path: str = "../models/sam2_encoder.onnx"
    sam_onnx_decoder_path: str = "../models/sam2_decoder.onnx"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Processing
    max_image_dimension: int = 4096
    thumbnail_max_dimension: int = 1024
    polygon_epsilon_ratio: float = 0.002
    min_polygon_points: int = 3
    # Margin added around a component's bbox when cutting the crop the
    # p1_side_damage condition classifier trains on. 7.5% matches the
    # pipeline's own P1 crop margin (`homography.crop_margin_pct` in
    # FINAL_AIML_ARCHITECTURE §10) so training crops are cut the same way the
    # serving path cuts them - a different margin at train time than at
    # inference time is a silent domain shift.
    condition_crop_margin_pct: float = 7.5
    mask_confidence_threshold: float = 0.5
    batch_max_workers: int = 2

    # Similarity-based propagation (Phase 1): when an image is finalized,
    # carry its accepted objects onto near-duplicate images so they don't
    # need to be annotated from scratch.
    propagation_enabled: bool = True
    similarity_threshold: float = 0.85
    propagation_top_k: int = 5

    # Classes the pipeline never serves, so we must never ship labels for them.
    # `leakage` is all-synthetic (docs/pipeline.md §5.2, §12 and
    # FINAL_AIML_ARCHITECTURE §10: `exclude_classes: [leakage]`), and the build
    # plan requires excluding it at the tool level rather than trusting export
    # to filter it. Comma-separated; matched case-insensitively on class name.
    # Part of every class-map version's hashed content, so changing this mints a
    # new version rather than silently altering what past snapshots meant.
    exclude_classes: str = "leakage"

    # Object store for publishing dataset snapshots (M2). Points at the
    # `automation-minio-1` service in docker-compose by default. Publishing is
    # opt-in per export and never fails an export - the local snapshot is the
    # source of truth, the bucket is the staging copy the pipeline team pulls
    # from (build plan Q-C: we stage, they import into their own MLflow).
    # boto3 talks S3, so pointing this at real S3 later is a config change.
    s3_endpoint_url: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket: str = "vb-dataset-snapshots"
    s3_prefix: str = "snapshots"
    s3_region: str = "us-east-1"

    # Logging
    log_level: str = "INFO"
    log_file: str = "../logs/backend.log"

    # Postgres (annotation state / audit history - see app/db.py)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "annotator"
    postgres_password: str = "change-me"
    postgres_db: str = "annotator"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def dataset_dir(self) -> Path:
        return Path(self.dataset_path).resolve()

    @property
    def exports_dir(self) -> Path:
        return Path(self.exports_path).resolve()

    @property
    def models_dir(self) -> Path:
        return Path(self.models_path).resolve()

    @property
    def state_dir(self) -> Path:
        return self.dataset_dir / self.state_dir_name

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def exclude_class_list(self) -> list[str]:
        return [c.strip() for c in self.exclude_classes.split(",") if c.strip()]

    def is_excluded_class(self, name: str) -> bool:
        return name.strip().lower() in {c.lower() for c in self.exclude_class_list}


@lru_cache
def get_settings() -> Settings:
    return Settings()
