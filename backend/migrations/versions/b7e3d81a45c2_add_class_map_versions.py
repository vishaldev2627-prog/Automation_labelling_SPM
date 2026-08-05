"""add class_map_versions, seed version 1 per existing dataset view

Immutable, content-addressed class-map versions (M1). See
app/services/class_map_service.py for the rationale; short version: the class
map was an unversioned list in data.yaml that add_class() rewrote in place, which
is the build plan's `[HIGH]` class-map-drift risk and already materialized once on
this project as a 27-class remap.

Seeding: every dataset view that already has `dataset_classes` rows gets a
version 1 minted from its current contents, so existing data has something real
to pin rather than starting at "unknown". The hash is computed here with the same
canonical serialization the service uses - if that ever changes, this migration's
seeded hashes stop matching and every view mints a version 2 on next load, which
is noisy but not wrong (nothing is lost, the old row stays).

Revision ID: b7e3d81a45c2
Revises: a4f1c9e207b3
Create Date: 2026-08-05

"""
from __future__ import annotations

import hashlib
import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b7e3d81a45c2'
down_revision: Union[str, None] = 'a4f1c9e207b3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Kept in sync with config.Settings.exclude_classes' default. Confirmed in
# docs/pipeline.md §12 and FINAL_AIML_ARCHITECTURE §10: `exclude_classes:
# [leakage]` - an all-synthetic class that is never shippable.
SEED_EXCLUDE_CLASSES = ["leakage"]


def _content_hash(names: list[str], exclude_classes: list[str]) -> str:
    """Duplicated from class_map_service on purpose: a migration must not import
    application code, or it stops being reproducible against the schema it was
    written for."""
    payload = {
        "names": [[i, name] for i, name in enumerate(names)],
        "exclude_classes": sorted(set(exclude_classes)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def upgrade() -> None:
    op.create_table(
        'class_map_versions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('dataset_view', sa.String(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('content_hash', sa.String(), nullable=False),
        sa.Column('names', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('exclude_classes', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('created_by_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['created_by_id'], ['annotators.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('dataset_view', 'version', name='uq_class_map_versions_view_version'),
        sa.UniqueConstraint('dataset_view', 'content_hash', name='uq_class_map_versions_view_hash'),
    )
    op.create_index(
        op.f('ix_class_map_versions_dataset_view'), 'class_map_versions', ['dataset_view'], unique=False
    )

    # Seed version 1 per view from whatever dataset_classes currently holds.
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            "SELECT dataset_view, class_id, name FROM dataset_classes ORDER BY dataset_view, class_id"
        )
    ).fetchall()

    by_view: dict[str, dict[int, str]] = {}
    for dataset_view, class_id, name in rows:
        by_view.setdefault(dataset_view, {})[int(class_id)] = name

    for dataset_view, id_to_name in by_view.items():
        if not id_to_name:
            continue
        # Tolerate gaps rather than assuming contiguous ids - a historical map is
        # whatever it was, and a migration is the wrong place to raise over it.
        highest = max(id_to_name)
        names = [id_to_name.get(i, f"class_{i}") for i in range(highest + 1)]
        bind.execute(
            sa.text(
                """
                INSERT INTO class_map_versions
                    (dataset_view, version, content_hash, names, exclude_classes)
                VALUES
                    (:dataset_view, 1, :content_hash, CAST(:names AS jsonb), CAST(:exclude AS jsonb))
                """
            ),
            {
                "dataset_view": dataset_view,
                "content_hash": _content_hash(names, SEED_EXCLUDE_CLASSES),
                "names": json.dumps([[i, name] for i, name in enumerate(names)]),
                "exclude": json.dumps(sorted(set(SEED_EXCLUDE_CLASSES))),
            },
        )


def downgrade() -> None:
    op.drop_index(op.f('ix_class_map_versions_dataset_view'), table_name='class_map_versions')
    op.drop_table('class_map_versions')
