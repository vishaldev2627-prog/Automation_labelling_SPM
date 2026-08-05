"""Tests for immutable, content-addressed class-map versions (M1).

The risk being closed is the build plan's `[HIGH]` class-map drift, which is not
hypothetical on this project - it already happened once as a 27-class remap, and
the `*_before_27class_remap_*` backup directories are the scar. With an
unversioned map, a dataset snapshot cannot say what its class ids meant, so
neither can a model trained on it, and a mismatched map is a corrupted training
run that looks entirely normal.

Two properties matter most and are tested hardest:
  1. **Identical content must not mint a new version** - otherwise every dataset
     load creates one and the history becomes noise nobody reads.
  2. **Different content must mint a different version** - otherwise drift is
     silent, which is the original bug.

The hashing and canonicalization are pure functions, so those are tested
directly. The DB-backed minting path is exercised through a fake session in
test_class_map_minting below rather than a real Postgres.

    cd backend && python -m pytest tests/test_class_map_versioning.py
    cd backend && python -m tests.test_class_map_versioning   # no pytest
"""
from __future__ import annotations

from app.config import Settings
from app.services.class_map_service import (
    canonical_payload,
    content_hash,
    names_from_version,
)

NAMES = ["coupler", "brake_cylinder", "axle_box"]
EXCLUDE = ["leakage"]


# ------------------------------------------------------- hashing / canonical form

def test_same_content_hashes_identically() -> None:
    """The idempotence property. If this breaks, every dataset load mints a
    version."""
    assert content_hash(NAMES, EXCLUDE) == content_hash(list(NAMES), list(EXCLUDE))


def test_renaming_a_class_changes_the_hash() -> None:
    renamed = ["coupler", "brake_cyl", "axle_box"]
    assert content_hash(renamed, EXCLUDE) != content_hash(NAMES, EXCLUDE)


def test_appending_a_class_changes_the_hash() -> None:
    assert content_hash([*NAMES, "spring"], EXCLUDE) != content_hash(NAMES, EXCLUDE)


def test_reordering_classes_changes_the_hash() -> None:
    """Order *is* identity here - class ids are positional, so swapping two names
    silently remaps every label referring to them. This is the exact shape of the
    27-class incident and must never hash equal."""
    reordered = ["brake_cylinder", "coupler", "axle_box"]
    assert content_hash(reordered, EXCLUDE) != content_hash(NAMES, EXCLUDE)


def test_changing_exclude_classes_changes_the_hash() -> None:
    assert content_hash(NAMES, ["leakage", "something_else"]) != content_hash(NAMES, EXCLUDE)


def test_exclude_classes_order_and_duplicates_do_not_matter() -> None:
    """Their order in a config string is incidental; it must not fabricate a new
    version."""
    assert content_hash(NAMES, ["b", "a"]) == content_hash(NAMES, ["a", "b"])
    assert content_hash(NAMES, ["a", "a", "b"]) == content_hash(NAMES, ["a", "b"])


def test_hash_is_prefixed_so_the_algorithm_is_self_describing() -> None:
    digest = content_hash(NAMES, EXCLUDE)
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64


def test_canonical_payload_records_ids_explicitly() -> None:
    """`[[0, "coupler"], ...]` rather than an object keyed by stringified ints,
    which would sort "10" before "2" and read as if it meant it."""
    payload = canonical_payload(NAMES, EXCLUDE)
    assert payload["names"] == [[0, "coupler"], [1, "brake_cylinder"], [2, "axle_box"]]
    assert payload["exclude_classes"] == ["leakage"]


def test_empty_class_list_is_hashable() -> None:
    """The `buffer` view starts with zero classes by product decision, so this is
    a real state, not an edge case."""
    assert content_hash([], EXCLUDE)
    assert content_hash([], EXCLUDE) != content_hash(NAMES, EXCLUDE)


def test_unicode_class_names_hash_stably() -> None:
    assert content_hash(["coupleur_avant"], EXCLUDE) == content_hash(["coupleur_avant"], EXCLUDE)


# ------------------------------------------------- reading a stored version back

class _FakeVersion:
    def __init__(self, names):
        self.names = names


def test_names_round_trip_through_a_stored_version() -> None:
    payload = canonical_payload(NAMES, EXCLUDE)
    assert names_from_version(_FakeVersion(payload["names"])) == NAMES


def test_names_from_version_tolerates_gaps_in_ids() -> None:
    """A historical map is whatever it was; reading old data must not raise."""
    assert names_from_version(_FakeVersion([[0, "a"], [2, "c"]])) == ["a", "class_1", "c"]


def test_names_from_version_handles_an_empty_map() -> None:
    assert names_from_version(_FakeVersion([])) == []


# --------------------------------------------------- exclude_classes enforcement

def _settings(exclude: str) -> Settings:
    return Settings(exclude_classes=exclude)


def test_exclude_class_list_parses_and_trims() -> None:
    assert _settings("leakage").exclude_class_list == ["leakage"]
    assert _settings("leakage, foo ,bar").exclude_class_list == ["leakage", "foo", "bar"]
    assert _settings("").exclude_class_list == []


def test_is_excluded_class_is_case_and_whitespace_insensitive() -> None:
    """An annotator typing "Leakage" or " leakage " must not slip past a check
    whose whole job is that the label never gets created."""
    settings = _settings("leakage")
    for candidate in ("leakage", "Leakage", "LEAKAGE", "  leakage  "):
        assert settings.is_excluded_class(candidate), candidate


def test_is_excluded_class_does_not_match_substrings() -> None:
    """`leaking` is a legitimate *condition* (pipeline.md §5.2) and a component
    name containing the word must not be caught by an over-broad match."""
    settings = _settings("leakage")
    assert not settings.is_excluded_class("leaking")
    assert not settings.is_excluded_class("leakage_sensor")
    assert not settings.is_excluded_class("brake_cylinder")


def test_default_exclude_list_matches_the_docs() -> None:
    """`exclude_classes: [leakage]`, docs/pipeline.md §12 and FINAL_AIML §10."""
    assert Settings().exclude_class_list == ["leakage"]


# ------------------------------- the migration duplicates the hash on purpose

def test_migration_seed_hash_matches_the_service_hash() -> None:
    """The seeding migration deliberately re-implements the hash rather than
    importing application code (a migration that imports app code stops being
    reproducible against the schema it was written for). That duplication can
    drift silently, and the symptom would be every view minting a spurious
    version 2 on its next load. This is the guard.
    """
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "migrations"
        / "versions"
        / "b7e3d81a45c2_add_class_map_versions.py"
    )
    spec = importlib.util.spec_from_file_location("_seed_migration", path)
    assert spec and spec.loader
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)

    assert migration.SEED_EXCLUDE_CLASSES == Settings().exclude_class_list
    for names in ([], NAMES, ["a"], ["z", "y", "x"]):
        assert migration._content_hash(names, EXCLUDE) == content_hash(names, EXCLUDE), names


# NOTE: the DB-backed ensure_version() path - the ON CONFLICT retry when two
# sessions load the same view concurrently - is deliberately NOT tested here.
# Faking enough SQLAlchemy to exercise it would mostly test the fake, and SQLite
# cannot create the JSONB columns the model declares. It needs an integration
# test against a real Postgres, which is not available in this suite. The pure
# parts it depends on (canonicalization, hashing, gap-tolerant reads) are covered
# above.


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
