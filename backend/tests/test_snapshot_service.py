"""Tests for content-addressed dataset snapshots (M2).

The problem being fixed: `export_service` wrote into one mutable `exports/`
directory, so a re-export silently changed what a previously-handed-over dataset
was. The `*_before_27class_remap_*` and `*_before_synthetic_removal_*` directories
in this repo are what that costs.

The identity rule, confirmed 2026-08-05, is **data only**:

    IN  the hash: file paths + file content hashes, and the class-map hash
    NOT in hash: created_at, annotator/reviewer ids, review counts, per-class
                 counts, split-integrity counts

Both halves are tested, because both are easy to break in a way nothing would
notice: hashing too much makes every export a new snapshot (content addressing
buys nothing), hashing too little makes two genuinely different datasets collide.

Real filesystem, no database - snapshot_service is deliberately pure
filesystem/hashing so it can be tested without Postgres.

    cd backend && python -m pytest tests/test_snapshot_service.py
    cd backend && python -m tests.test_snapshot_service   # no pytest
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.services import snapshot_service as snap


def _make_dataset(root: Path, label_text: str = "0 0.1 0.1 0.2 0.2\n") -> None:
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "train" / "a.png").write_bytes(b"fake-image-bytes")
    (root / "labels" / "train" / "a.txt").write_text(label_text, encoding="utf-8")
    (root / "classes.txt").write_text("coupler\n", encoding="utf-8")


def _index_and_id(root: Path, class_map_hash: str = "sha256:map"):
    index = snap.build_file_index(root)
    return index, snap.compute_snapshot_id(index, class_map_hash)


# ------------------------------------------------------------- identity: stable

def test_identical_content_yields_identical_snapshot_id() -> None:
    """The idempotence property. Without it, re-exporting unchanged data creates
    a duplicate snapshot every time and content-addressing is pointless."""
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a", Path(tmp) / "b"
        _make_dataset(a)
        _make_dataset(b)
        assert _index_and_id(a)[1] == _index_and_id(b)[1]


def test_snapshot_id_is_prefixed_and_full_length() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "d"
        _make_dataset(root)
        snapshot_id = _index_and_id(root)[1]
        assert snapshot_id.startswith("sha256:")
        assert len(snapshot_id) == len("sha256:") + 64


def test_manifest_and_checksums_are_excluded_from_the_hash() -> None:
    """The manifest cannot contain its own id, and checksums.sha256 is derived
    from the very list being hashed - including either would be circular."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "d"
        _make_dataset(root)
        index_before, id_before = _index_and_id(root)

        snap.write_checksums(root, index_before)
        snap.write_manifest(root, {"snapshot_id": id_before, "anything": "at all"})

        assert _index_and_id(root)[1] == id_before


# --------------------------------------------------------- identity: sensitive

def test_changed_label_content_changes_the_id() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a", Path(tmp) / "b"
        _make_dataset(a, label_text="0 0.1 0.1 0.2 0.2\n")
        _make_dataset(b, label_text="0 0.9 0.9 0.2 0.2\n")
        assert _index_and_id(a)[1] != _index_and_id(b)[1]


def test_an_extra_file_changes_the_id() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "d"
        _make_dataset(root)
        before = _index_and_id(root)[1]
        (root / "images" / "train" / "b.png").write_bytes(b"another-image")
        assert _index_and_id(root)[1] != before


def test_a_renamed_file_changes_the_id_even_with_identical_bytes() -> None:
    """Paths carry the split assignment (images/train vs images/val), so moving a
    file between splits must change identity even though no byte changed."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "d"
        _make_dataset(root)
        before = _index_and_id(root)[1]
        (root / "images" / "val").mkdir(parents=True, exist_ok=True)
        (root / "images" / "train" / "a.png").rename(root / "images" / "val" / "a.png")
        assert _index_and_id(root)[1] != before


def test_class_map_hash_participates_in_identity() -> None:
    """Identical label bytes mean different things under a different class map -
    exactly the failure the class-map versioning work exists to prevent."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "d"
        _make_dataset(root)
        index = snap.build_file_index(root)
        assert snap.compute_snapshot_id(index, "sha256:map-a") != snap.compute_snapshot_id(
            index, "sha256:map-b"
        )


def test_missing_class_map_cannot_collide_with_a_real_one() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "d"
        _make_dataset(root)
        index = snap.build_file_index(root)
        assert snap.compute_snapshot_id(index, None) != snap.compute_snapshot_id(index, "sha256:real")
        # Stable across calls despite being the "unknown" case.
        assert snap.compute_snapshot_id(index, None) == snap.compute_snapshot_id(index, None)


# ---------------------------------------------------------------- file index

def test_file_index_is_sorted_and_posix_pathed() -> None:
    """rglob order is not guaranteed, and a Windows-built snapshot must hash
    identically to a Linux-built one from the same data."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "d"
        _make_dataset(root)
        (root / "images" / "train" / "z.png").write_bytes(b"z")
        (root / "images" / "train" / "b.png").write_bytes(b"b")
        index = snap.build_file_index(root)
        paths = [relative for relative, _digest, _size in index]
        assert paths == sorted(paths)
        assert all("\\" not in p for p in paths)


def test_file_index_reports_sizes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "d"
        _make_dataset(root)
        index = snap.build_file_index(root)
        by_path = {relative: size for relative, _digest, size in index}
        assert by_path["images/train/a.png"] == len(b"fake-image-bytes")


def test_checksums_file_is_sha256sum_compatible() -> None:
    """So the pipeline team can verify a transfer with `sha256sum -c` and no
    tooling from us."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "d"
        _make_dataset(root)
        index = snap.build_file_index(root)
        snap.write_checksums(root, index)
        lines = (root / snap.CHECKSUMS_NAME).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == len(index)
        digest, _, path = lines[0].partition("  ")
        assert len(digest) == 64
        assert path == index[0][0]


# ------------------------------------------------------------------- finalize

def test_finalize_moves_staging_into_a_content_addressed_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        exports = Path(tmp) / "exports"
        staging = snap.new_staging_dir(exports)
        _make_dataset(staging)
        snapshot_id = _index_and_id(staging)[1]

        target, created = snap.finalize(staging, exports, snapshot_id)
        assert created is True
        assert not staging.exists()
        assert (target / "labels" / "train" / "a.txt").exists()
        # ':' is not a legal path character on Windows, so it is encoded.
        assert ":" not in target.name
        assert target.parent.name == snap.SNAPSHOTS_DIRNAME


def test_finalize_is_idempotent_and_never_overwrites() -> None:
    """Re-exporting unchanged data must leave the existing snapshot untouched -
    an immutable artifact that gets rewritten is not immutable."""
    with tempfile.TemporaryDirectory() as tmp:
        exports = Path(tmp) / "exports"

        first = snap.new_staging_dir(exports)
        _make_dataset(first)
        snapshot_id = _index_and_id(first)[1]
        target, created_first = snap.finalize(first, exports, snapshot_id)
        marker = target / "labels" / "train" / "a.txt"
        original_bytes = marker.read_bytes()

        second = snap.new_staging_dir(exports)
        _make_dataset(second)
        # Tamper with the staged copy so an overwrite would be detectable.
        (second / "labels" / "train" / "a.txt").write_text("tampered\n", encoding="utf-8")
        target_again, created_second = snap.finalize(second, exports, snapshot_id)

        assert created_first is True
        assert created_second is False
        assert target_again == target
        assert marker.read_bytes() == original_bytes
        assert not second.exists()


def test_staging_lives_under_exports_for_a_same_filesystem_move() -> None:
    """A cross-device move of a multi-GB dataset would be a copy. Staging must sit
    on the same filesystem as its destination."""
    with tempfile.TemporaryDirectory() as tmp:
        exports = Path(tmp) / "exports"
        staging = snap.new_staging_dir(exports)
        assert staging.parent == exports / snap.STAGING_DIRNAME


def test_cleanup_removes_only_stale_staging_dirs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        exports = Path(tmp) / "exports"
        fresh = snap.new_staging_dir(exports)
        stale = snap.new_staging_dir(exports)
        import os
        import time

        old = time.time() - 48 * 3600
        os.utime(stale, (old, old))

        removed = snap.cleanup_stale_staging(exports, max_age_seconds=24 * 3600)
        assert removed == 1
        assert fresh.exists()
        assert not stale.exists()


def test_cleanup_is_safe_when_nothing_has_been_staged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        assert snap.cleanup_stale_staging(Path(tmp) / "exports") == 0


# ------------------------------------------------------------------- manifest

def _manifest(**overrides) -> dict:
    kwargs = {
        "snapshot_id": "sha256:abc",
        "dataset_key": "/data/dataset/side_view",
        "class_map_version": 3,
        "class_map_hash": "sha256:map",
        "class_names": ["coupler", "leakage"],
        "exclude_classes": ["leakage"],
        "label_format": {0: "polygon", 1: "polygon"},
        "file_index": [("images/train/a.png", "deadbeef", 16)],
        "stats": {
            "exported": 1,
            "negatives": 0,
            "provenance": {"grandfathered_unreviewed_images": 49},
            "per_class_counts": {"LHB": {"0": 5}},
            "split_integrity": {"pseudo_synthetic_in_val_or_test": 0},
            "splits": {"train": {"images": 1, "labels": 1}},
            "spine_stamp_coverage": {"frames_missing_stamp": 1},
        },
    }
    kwargs.update(overrides)
    return snap.build_manifest(**kwargs)


def test_manifest_is_json_serializable() -> None:
    """It is written to disk and stored in a JSONB column; a stray set or Path
    would fail at the worst possible moment."""
    json.dumps(_manifest())


def test_manifest_derives_the_view_from_the_dataset_root() -> None:
    assert _manifest()["dataset_view"] == "side_view"


def test_manifest_declares_target_families_from_confirmed_decisions() -> None:
    """side_view feeds the crop classifier, not the P1 detection head - D-Q1=(b).
    Getting this wrong would tell the pipeline team a dataset is for a model it
    cannot train."""
    assert _manifest()["target_families"] == ["p1_side_damage"]
    assert _manifest(dataset_key="/d/underbelly")["target_families"] == [
        "p2_under_anomaly",
        "p2_under_crackseg",
    ]
    # An unrecognised view declares nothing rather than guessing.
    assert _manifest(dataset_key="/d/experimental")["target_families"] == []


def test_manifest_keeps_excluded_names_in_the_class_map() -> None:
    """Class ids are positional; dropping a name renumbers everything after it,
    which is the exact drift all of this exists to prevent."""
    class_map = _manifest()["class_map"]
    # Keys are strings: JSON has no integer keys, so the manifest must be written
    # the way it will be read back (see test_manifest_round_trips_through_disk).
    assert class_map["names"]["1"] == "leakage"
    assert class_map["exclude_classes"] == ["leakage"]


def test_manifest_surfaces_the_uncomfortable_provenance_number() -> None:
    """grandfathered_unreviewed_images non-zero means a model trained on this
    snapshot cannot claim all its data was second-reviewed. Better a field than a
    discovery."""
    assert _manifest()["provenance"]["grandfathered_unreviewed_images"] == 49


def test_manifest_records_per_class_counts_by_coach_type() -> None:
    assert _manifest()["per_class_counts"] == {"LHB": {"0": 5}}


def test_manifest_carries_the_class_map_hash_not_just_the_number() -> None:
    """The number is for humans; the hash is what actually proves two snapshots
    were built against the same map."""
    class_map = _manifest()["class_map"]
    assert class_map["version"] == 3
    assert class_map["content_hash"] == "sha256:map"


def test_manifest_totals_bytes_from_the_file_index() -> None:
    assert _manifest()["counts"]["total_bytes"] == 16
    assert _manifest()["counts"]["files"] == 1


def test_manifest_round_trips_through_disk() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = _manifest()
        snap.write_manifest(root, manifest)
        assert snap.read_manifest(root) == manifest


def test_read_manifest_returns_none_when_absent_or_corrupt() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert snap.read_manifest(root) is None
        (root / snap.MANIFEST_NAME).write_text("{not json", encoding="utf-8")
        assert snap.read_manifest(root) is None


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - hand-rolled runner wants everything
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    raise SystemExit(1 if failures else 0)
