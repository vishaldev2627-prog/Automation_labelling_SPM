"""Immutable, content-addressed class-map versions (M1).

**Why this exists.** The class map - which integer id means which component -
was previously an unversioned list living in `data.yaml`/`classes.txt`, rewritten
in place by `add_class()`. That is the single highest-severity risk named in
`annotation_module_build_plan.md` §5, and it is not hypothetical: this project
already went through a 27-class remap, evidenced by the `*_before_27class_remap_*`
backup directories. With an unversioned map, a dataset snapshot cannot say what
its class ids meant, so a model trained on it cannot either, and a mismatched map
is a corrupted training run that looks completely normal.

**The rule:** a class map is never edited. Any change to the id->name mapping or
to `exclude_classes` mints a **new version**. Old versions are never mutated or
deleted, so every past snapshot can always resolve the exact map it was built
against.

**Content-addressed, so minting is idempotent.** The version is keyed by a
sha256 of a canonical serialization. Loading a dataset ten times does not create
ten versions; it resolves to the same one. A genuine change produces a different
hash and therefore a new version, without anyone having to remember to bump
anything.

**What is deliberately NOT part of the hash:** per-class colour,
`safety_critical`, and `fine_structure`. Those are tool-side metadata - colour is
cosmetic, and the other two control export format and auto-accept gating rather
than class identity. A curator correcting a `safety_critical` flag should not
invalidate the class map every previous snapshot pinned. `fine_structure` does
affect the exported label format, but that belongs in the snapshot manifest's
per-class `label_format`, which describes one export, not the taxonomy.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.db_models import ClassMapVersion

logger = logging.getLogger(__name__)


def canonical_payload(names: list[str], exclude_classes: list[str]) -> dict:
    """The exact structure that gets hashed.

    `names` is serialized as an ordered `[id, name]` list rather than an object
    keyed by stringified ints: `{"10": ..., "2": ...}` sorts lexicographically,
    which is stable but reads as if 10 precedes 2. The list makes the id->name
    ordering explicit, which matters for a structure whose whole job is being
    unambiguous years later.

    `exclude_classes` is sorted and de-duplicated so its ordering in a config
    file can never produce a spurious new version.
    """
    return {
        "names": [[i, name] for i, name in enumerate(names)],
        "exclude_classes": sorted(set(exclude_classes)),
    }


def content_hash(names: list[str], exclude_classes: list[str]) -> str:
    payload = canonical_payload(names, exclude_classes)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def get_current(db: Session, dataset_key: str) -> Optional[ClassMapVersion]:
    """The highest-numbered version for this view, or None if none exists yet."""
    return db.execute(
        select(ClassMapVersion)
        .where(ClassMapVersion.dataset_view == dataset_key)
        .order_by(ClassMapVersion.version.desc())
        .limit(1)
    ).scalar_one_or_none()


def list_versions(db: Session, dataset_key: str) -> list[ClassMapVersion]:
    return list(
        db.execute(
            select(ClassMapVersion)
            .where(ClassMapVersion.dataset_view == dataset_key)
            .order_by(ClassMapVersion.version.asc())
        ).scalars()
    )


def get_by_hash(db: Session, dataset_key: str, hash_value: str) -> Optional[ClassMapVersion]:
    return db.execute(
        select(ClassMapVersion).where(
            ClassMapVersion.dataset_view == dataset_key,
            ClassMapVersion.content_hash == hash_value,
        )
    ).scalar_one_or_none()


def ensure_version(
    db: Session,
    dataset_key: str,
    names: list[str],
    exclude_classes: list[str],
    annotator_id: Optional[int] = None,
) -> ClassMapVersion:
    """Resolve this exact class map to a version, minting one if it is new.

    Idempotent by content hash, so this is safe to call on every dataset load -
    which is the point: drift gets recorded automatically rather than depending
    on someone remembering to version a change they made by editing `data.yaml`
    by hand.

    Returns the existing version when the content is unchanged. When it has
    changed, the new version's number is `max(existing) + 1`; numbers are
    per-view and never reused.
    """
    digest = content_hash(names, exclude_classes)

    existing = get_by_hash(db, dataset_key, digest)
    if existing is not None:
        return existing

    payload = canonical_payload(names, exclude_classes)
    next_version = (
        db.execute(
            select(func.coalesce(func.max(ClassMapVersion.version), 0)).where(
                ClassMapVersion.dataset_view == dataset_key
            )
        ).scalar_one()
        + 1
    )

    version = ClassMapVersion(
        dataset_view=dataset_key,
        version=next_version,
        content_hash=digest,
        names=payload["names"],
        exclude_classes=payload["exclude_classes"],
        created_by_id=annotator_id,
    )
    db.add(version)
    try:
        db.commit()
    except IntegrityError:
        # Two sessions loading the same view concurrently can both miss the
        # get_by_hash above and race to insert. Either unique constraint
        # (view+version or view+content_hash) rejects the loser, which then
        # re-reads: if it lost on content_hash the winner minted the identical
        # map and we want that row; if it lost on version number, retry once
        # with a fresh number.
        db.rollback()
        already = get_by_hash(db, dataset_key, digest)
        if already is not None:
            return already
        return ensure_version(db, dataset_key, names, exclude_classes, annotator_id)

    db.refresh(version)
    logger.info(
        "Minted class-map version %d for %s (%d classes, hash %s)",
        version.version,
        dataset_key,
        len(names),
        digest,
    )
    return version


def names_from_version(version: ClassMapVersion) -> list[str]:
    """Rebuild the ordered class-name list from a stored version.

    Tolerates gaps rather than assuming the ids were contiguous: a historical
    map is whatever it was, and this must not raise while reading old data.
    """
    if not version.names:
        return []
    by_id = {int(class_id): name for class_id, name in version.names}
    highest = max(by_id)
    return [by_id.get(i, f"class_{i}") for i in range(highest + 1)]
