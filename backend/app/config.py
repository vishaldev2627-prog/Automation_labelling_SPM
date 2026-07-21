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
    state_dir_name: str = ".annotation_state"

    # SAM2
    sam_checkpoint: str = "../models/sam2.1_hiera_large.pt"
    sam_model_cfg: str = "sam2.1_hiera_l.yaml"
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
    batch_max_workers: int = 2

    # Logging
    log_level: str = "INFO"
    log_file: str = "../logs/backend.log"

    @property
    def dataset_dir(self) -> Path:
        return Path(self.dataset_path).resolve()

    @property
    def exports_dir(self) -> Path:
        return Path(self.exports_path).resolve()

    @property
    def state_dir(self) -> Path:
        return self.dataset_dir / self.state_dir_name

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
