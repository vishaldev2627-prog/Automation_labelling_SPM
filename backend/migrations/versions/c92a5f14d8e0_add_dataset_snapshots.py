"""add dataset_snapshots

Index over the immutable content-addressed snapshot directories (M2). The
snapshot data itself lives on disk under exports/snapshots/<snapshot_id>/; this
table makes "which snapshots exist, from which class map, published where" a
query rather than a directory listing.

Purely additive - no existing rows to migrate, since snapshots did not exist
before this. Nothing here backfills: previous exports overwrote a mutable
directory and cannot be reconstructed as snapshots after the fact, which is the
whole reason this table exists.

Revision ID: c92a5f14d8e0
Revises: b7e3d81a45c2
Create Date: 2026-08-05

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'c92a5f14d8e0'
down_revision: Union[str, None] = 'b7e3d81a45c2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'dataset_snapshots',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('snapshot_id', sa.String(), nullable=False),
        sa.Column('dataset_view', sa.String(), nullable=False),
        sa.Column('class_map_version', sa.Integer(), nullable=True),
        sa.Column('class_map_hash', sa.String(), nullable=True),
        sa.Column('manifest', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('local_path', sa.String(), nullable=False),
        sa.Column('file_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('total_bytes', sa.BigInteger(), server_default='0', nullable=False),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('published_uri', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column(
            'last_exported_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True
        ),
        sa.Column('created_by_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['created_by_id'], ['annotators.id'], ),
        sa.PrimaryKeyConstraint('id'),
        # Unique globally, not per view: snapshot_id is a content hash, so the
        # same file set and class map from two views genuinely is one dataset.
        sa.UniqueConstraint('snapshot_id', name='uq_dataset_snapshots_snapshot_id'),
    )
    op.create_index(
        op.f('ix_dataset_snapshots_snapshot_id'), 'dataset_snapshots', ['snapshot_id'], unique=False
    )
    op.create_index(
        op.f('ix_dataset_snapshots_dataset_view'), 'dataset_snapshots', ['dataset_view'], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_dataset_snapshots_dataset_view'), table_name='dataset_snapshots')
    op.drop_index(op.f('ix_dataset_snapshots_snapshot_id'), table_name='dataset_snapshots')
    op.drop_table('dataset_snapshots')
