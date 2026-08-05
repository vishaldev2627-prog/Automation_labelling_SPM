"""add golden_sets, golden_set_items

Structurally separate storage for the frozen golden eval set (M4, D-Q4:
"frozen per-class golden set, gating before the shadow canary"). Append-only,
mirroring class_map_versions: a curation round mints a new version rather
than editing an existing one's items.

Purely additive - nothing to backfill, since no golden set has existed before
this. Population (which images actually go in) is separately blocked on named
domain experts to curate (D-Q4); this migration only builds the storage and
permission gate so that write path exists from day one.

Revision ID: d5a7c0f3e8b1
Revises: c92a5f14d8e0
Create Date: 2026-08-05

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd5a7c0f3e8b1'
down_revision: Union[str, None] = 'c92a5f14d8e0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'golden_sets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('dataset_view', sa.String(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('created_by_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['created_by_id'], ['annotators.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('dataset_view', 'version', name='uq_golden_sets_view_version'),
    )
    op.create_index(op.f('ix_golden_sets_dataset_view'), 'golden_sets', ['dataset_view'], unique=False)

    op.create_table(
        'golden_set_items',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('golden_set_id', sa.Integer(), nullable=False),
        sa.Column('image_id', sa.String(), nullable=False),
        sa.Column('frozen_object_store_key', sa.String(), nullable=True),
        sa.Column('added_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['golden_set_id'], ['golden_sets.id'], ),
        sa.PrimaryKeyConstraint('id'),
        # Deliberately NOT unique on image_id alone - the same image can
        # legitimately appear in more than one version (see GoldenSetItem
        # docstring). Unique per (set, image) just prevents a double-add
        # within one curation round.
        sa.UniqueConstraint('golden_set_id', 'image_id', name='uq_golden_set_items_set_image'),
    )
    op.create_index(op.f('ix_golden_set_items_golden_set_id'), 'golden_set_items', ['golden_set_id'], unique=False)
    op.create_index(op.f('ix_golden_set_items_image_id'), 'golden_set_items', ['image_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_golden_set_items_image_id'), table_name='golden_set_items')
    op.drop_index(op.f('ix_golden_set_items_golden_set_id'), table_name='golden_set_items')
    op.drop_table('golden_set_items')
    op.drop_index(op.f('ix_golden_sets_dataset_view'), table_name='golden_sets')
    op.drop_table('golden_sets')
