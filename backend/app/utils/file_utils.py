"""Filesystem helpers: atomic JSON read/write, id generation."""
from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically to avoid corruption on crash mid-write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to read/parse JSON at %s; returning default", path)
        return default


def image_stem_to_label_path(labels_dir: Path, image_path: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"
