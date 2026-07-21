"""YOLO detection/segmentation label parsing and writing."""
from __future__ import annotations

import logging
from pathlib import Path

from app.models.schemas import BoundingBox, Point

logger = logging.getLogger(__name__)


class LabelParseError(Exception):
    """Raised when a YOLO label file/line is malformed."""


def parse_detection_label_file(path: Path) -> list[tuple[int, BoundingBox]]:
    """Parse a YOLO detection .txt file into (class_id, bbox) tuples.

    Malformed lines are skipped with a logged warning rather than aborting
    the whole file, since a single bad line shouldn't block the dataset.
    """
    results: list[tuple[int, BoundingBox]] = []
    if not path.exists():
        return results

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                logger.warning("Skipping malformed label line %s:%d -> %r", path, line_no, line)
                continue
            try:
                class_id = int(float(parts[0]))
                x, y, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                logger.warning("Skipping non-numeric label line %s:%d -> %r", path, line_no, line)
                continue

            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                logger.warning("Skipping out-of-range bbox %s:%d -> %r", path, line_no, line)
                continue

            results.append((class_id, BoundingBox(x_center=x, y_center=y, width=w, height=h)))

    return results


def format_segmentation_line(class_id: int, polygon: list[Point]) -> str:
    coords = " ".join(f"{p.x:.6f} {p.y:.6f}" for p in polygon)
    return f"{class_id} {coords}"


def write_segmentation_label_file(path: Path, objects: list[tuple[int, list[Point]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        format_segmentation_line(class_id, polygon)
        for class_id, polygon in objects
        if len(polygon) >= 3
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_class_names(dataset_dir: Path, num_fallback_classes: int = 0) -> list[str]:
    """Load class names from data.yaml or classes.txt, falling back to generic names."""
    for candidate in ("data.yaml", "data.yml", "dataset.yaml"):
        yaml_path = dataset_dir / candidate
        if yaml_path.exists():
            try:
                import yaml

                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                names = data.get("names")
                if isinstance(names, dict):
                    return [names[k] for k in sorted(names, key=lambda k: int(k))]
                if isinstance(names, list):
                    return list(names)
            except Exception:
                logger.exception("Failed to parse %s", yaml_path)

    classes_txt = dataset_dir / "classes.txt"
    if classes_txt.exists():
        return [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]

    return [f"class_{i}" for i in range(num_fallback_classes)]
