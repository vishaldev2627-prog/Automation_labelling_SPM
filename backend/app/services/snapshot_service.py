"""Immutable, content-addressed dataset snapshots (M2).

**Why.** `export_service` used to write into one mutable `exports/` directory, so
a re-export silently changed what a previously-handed-over dataset was. The
hand-rolled `*_before_27class_remap_*` and `*_before_synthetic_removal_*` backup
directories in this repo are what that costs in practice. Build plan Phase 5:
every export becomes an immutable, named snapshot, and the snapshot plus its
manifest is the handoff artifact the pipeline team imports (Q-C: we stage, they
import - we do not write their MLflow).

**Identity.** `snapshot_id` is a sha256 over the **data only**: every included
file's path and content hash, plus the class-map hash. Confirmed 2026-08-05.

    IN  the hash: images, labels, mask rasters, condition crops, class map
    NOT in hash: created_at, annotator/reviewer ids, review and audit counts,
                 per-class counts, split-integrity counts

Consequences, all intended:
  - Re-exporting unchanged data yields the **same** snapshot_id, so finalize is
    idempotent and a repeat export is free rather than a duplicate.
  - One more audit review, with the images and labels byte-identical, does not
    mint a new snapshot. That provenance is recorded in the manifest, where it
    describes the snapshot rather than defining it.
  - Renaming a class **does** change the id, even with identical label files -
    identical bytes mean different things under a different map, which is
    exactly the failure the class-map versioning work exists to prevent.

**Build order.** The directory is named after a hash of its own contents, so it
is built in a staging directory first, hashed, then moved into place. If the
target already exists, the content is by definition identical and the staging
copy is discarded.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SNAPSHOTS_DIRNAME = "snapshots"
STAGING_DIRNAME = ".staging"
MANIFEST_NAME = "manifest.json"
CHECKSUMS_NAME = "checksums.sha256"

# Files that describe the snapshot rather than being part of its data, and so
# must not be inside the hash of the data (the manifest cannot contain its own
# id, and checksums.sha256 is derived from the very list being hashed).
NON_DATA_FILES = {MANIFEST_NAME, CHECKSUMS_NAME}

MANIFEST_SCHEMA_VERSION = 1

# Which pipeline model families each dataset view feeds. Every entry traces to a
# confirmed answer from the pipeline team (see DECISIONS_LOG.md), not a guess:
#   side_view      -> D-Q1 = (b): feeds the crop classifier, NOT the P1 detection
#                     head, which trains on stitched panoramas we don't produce.
#   underbelly     -> D-Q5: area-cam footage, in scope for P2.
#   wheel_shelling -> masks must live in log-polar space (pipeline.md §5.5); the
#                     unwrap is not built yet, so this is declared but unfed.
#   buffer         -> D-Q2: buffer_visible, annotated as boxes, flag derived here.
# Names match FINAL_AIML_ARCHITECTURE §10's `models:` block.
VIEW_TARGET_FAMILIES: dict[str, list[str]] = {
    "side_view": ["p1_side_damage"],
    "underbelly": ["p2_under_anomaly", "p2_under_crackseg"],
    "wheel_shelling": ["p3_wheel_shelling"],
    "buffer": ["buffer_boundary"],
}


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Streamed, so a multi-GB snapshot doesn't need to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_index(root: Path) -> list[tuple[str, str, int]]:
    """Every data file under `root` as (posix relative path, sha256, size).

    Sorted by path so the ordering is reproducible across filesystems - `rglob`
    order is not guaranteed, and an unstable order would make the snapshot id
    depend on how the OS happened to walk the tree.

    Paths are stored posix-style for the same reason: a snapshot built on
    Windows and one built on Linux from identical data must hash identically.
    """
    entries: list[tuple[str, str, int]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in NON_DATA_FILES:
            continue
        entries.append((relative, file_sha256(path), path.stat().st_size))
    entries.sort(key=lambda entry: entry[0])
    return entries


def compute_snapshot_id(file_index: list[tuple[str, str, int]], class_map_hash: Optional[str]) -> str:
    """sha256 over the file set plus the class-map hash. See the module docstring
    for what is deliberately excluded.

    Size is not hashed - it is already implied by the content hash, and including
    it would only add a second way to say the same thing.

    A missing class-map hash is hashed as the literal string "unknown" rather
    than omitted, so a snapshot built without a resolvable class map cannot
    collide with one that has a real map.
    """
    payload = {
        "files": [[relative, digest] for relative, digest, _size in file_index],
        "class_map_hash": class_map_hash or "unknown",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_checksums(root: Path, file_index: list[tuple[str, str, int]]) -> None:
    """Standard `sha256sum`-compatible file, so the pipeline team can verify a
    transferred snapshot with `sha256sum -c` and no tooling from us."""
    lines = [f"{digest}  {relative}" for relative, digest, _size in file_index]
    (root / CHECKSUMS_NAME).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_manifest(
    *,
    snapshot_id: str,
    dataset_key: str,
    class_map_version: Optional[int],
    class_map_hash: Optional[str],
    class_names: list[str],
    exclude_classes: list[str],
    label_format: dict[int, str],
    file_index: list[tuple[str, str, int]],
    stats: dict,
) -> dict:
    """The handoff contract. Field names are ours to define (D-Q6).

    Deliberately includes the uncomfortable numbers.
    `provenance.grandfathered_unreviewed_images` counts images that are
    exportable without a second review because they were completed before that
    gate existed; if it is non-zero, a model trained on this snapshot cannot
    claim "all data second-reviewed". Better a field than a discovery.
    """
    view = Path(dataset_key).name
    total_bytes = sum(size for _relative, _digest, size in file_index)
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        # Not part of the hash - two byte-identical exports differ here and are
        # still the same snapshot.
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_view": view,
        "dataset_root": dataset_key,
        "target_families": VIEW_TARGET_FAMILIES.get(view, []),
        "class_map": {
            "version": class_map_version,
            "content_hash": class_map_hash,
            # String keys, not ints. JSON has no integer keys, so both the
            # on-disk manifest and the JSONB column stringify them - emitting
            # ints here would mean the in-memory manifest and the persisted one
            # were not equal, and any later comparison of the two would report a
            # spurious difference.
            "names": {str(i): name for i, name in enumerate(class_names)},
            # Excluded names are retained in `names` on purpose: class ids are
            # positional, so dropping one renumbers every class after it.
            "exclude_classes": exclude_classes,
        },
        "label_format": {str(class_id): fmt for class_id, fmt in sorted(label_format.items())},
        "splits": stats.get("splits", {}),
        "split_integrity": stats.get("split_integrity", {}),
        "provenance": stats.get("provenance", {}),
        "per_class_counts": stats.get("per_class_counts", {}),
        "spine_stamp_coverage": stats.get("spine_stamp_coverage", {}),
        # W-5: which provisional unwrap parameters produced this snapshot's
        # logpolar/ files, if any were written - see export_service's
        # `wheel_unwrap_params` for why `certified` is load-bearing, not
        # decoration. Present (with wheel_unwraps: 0) even for views that
        # never produce an unwrap, same "0 reads as legible, not broken" the
        # condition-crop counters already establish.
        "wheel_unwrap": stats.get("wheel_unwrap_params", {}),
        "counts": {
            "files": len(file_index),
            "total_bytes": total_bytes,
            "images": stats.get("exported", 0),
            "confirmed_negatives": stats.get("negatives", 0),
            "mask_rasters": stats.get("mask_rasters", 0),
            "wheel_unwraps": stats.get("wheel_unwraps", 0),
            "condition_crops": stats.get("condition_crops", 0),
            "excluded_labels_dropped": stats.get("excluded_labels_dropped", 0),
        },
        "files": {
            "images": "images/",
            "labels": "labels/",
            "masks": "masks/",
            "logpolar": "logpolar/",
            "crops": "crops/",
            "checksums": CHECKSUMS_NAME,
        },
    }


def new_staging_dir(exports_dir: Path) -> Path:
    """A staging directory to build into before the id is known.

    Under `exports/.staging/` rather than the system temp dir so the finalize
    step is a rename on the same filesystem, not a cross-device copy of a
    dataset that may be many gigabytes.
    """
    from app.utils.file_utils import new_id

    staging = exports_dir / STAGING_DIRNAME / new_id()
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def finalize(staging_dir: Path, exports_dir: Path, snapshot_id: str) -> tuple[Path, bool]:
    """Move a staged build to its content-addressed home.

    Returns `(path, created)`. `created=False` means this exact snapshot already
    existed, which is the normal outcome of re-exporting unchanged data - the
    staged copy is discarded and the existing directory is left untouched.
    Never overwrites: an existing snapshot is immutable by definition, and
    rewriting it would defeat the entire mechanism.
    """
    target = exports_dir / SNAPSHOTS_DIRNAME / snapshot_id.replace("sha256:", "sha256-")
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        logger.info("Snapshot %s already exists at %s; discarding staged copy", snapshot_id, target)
        shutil.rmtree(staging_dir, ignore_errors=True)
        return target, False

    shutil.move(str(staging_dir), str(target))
    logger.info("Finalized snapshot %s at %s", snapshot_id, target)
    return target, True


def write_manifest(root: Path, manifest: dict) -> None:
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def read_manifest(root: Path) -> Optional[dict]:
    path = root / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.exception("Could not read snapshot manifest at %s", path)
        return None


def cleanup_stale_staging(exports_dir: Path, max_age_seconds: int = 24 * 3600) -> int:
    """Remove staging directories left behind by a crashed or killed export.

    Called at the start of a build rather than on a timer: this process has no
    scheduler, and an abandoned staging dir is otherwise invisible until it
    fills the disk.
    """
    staging_root = exports_dir / STAGING_DIRNAME
    if not staging_root.exists():
        return 0

    import time

    now = time.time()
    removed = 0
    for candidate in staging_root.iterdir():
        if not candidate.is_dir():
            continue
        try:
            if now - candidate.stat().st_mtime > max_age_seconds:
                shutil.rmtree(candidate, ignore_errors=True)
                removed += 1
        except OSError:
            logger.exception("Could not inspect staging dir %s", candidate)
    if removed:
        logger.info("Removed %d stale staging directory/ies under %s", removed, staging_root)
    return removed
