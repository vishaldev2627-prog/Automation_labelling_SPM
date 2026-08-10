"""add class lifecycle state/tier to dataset_classes

Class lifecycle (docs/mlflow_class_incremental_architecture.md §D):
discovered -> collecting_data -> eligible -> active -> deprecated.

Every existing dataset_classes row predates this feature and has already
been trained on in real runs - the column default and this migration's
backfill both land on "active" unconditionally, never retroactively
demoting something already shipping. `tier` (cosmetic/structural/safety,
reusing pipeline.md's existing vocabulary) backfills from the existing
`safety_critical` flag: true -> safety, everything else -> structural
(never cosmetic by default - a conservative starting point for a human to
loosen later, not a claim this migration is making about any class).

Revision ID: e2f4a7b91c30
Revises: a1b2c3d4e5f6
Create Date: 2026-08-11

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e2f4a7b91c30'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('dataset_classes', sa.Column('state', sa.String(), server_default='active', nullable=False))
    op.add_column('dataset_classes', sa.Column('tier', sa.String(), server_default='structural', nullable=False))
    op.add_column('dataset_classes', sa.Column('deprecated_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('dataset_classes', sa.Column('deprecated_by_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_dataset_classes_deprecated_by_id_annotators',
        'dataset_classes', 'annotators', ['deprecated_by_id'], ['id'],
    )

    # server_default already backfills state="active" for every existing
    # row (Postgres applies it to the whole table on an ADD COLUMN NOT
    # NULL). tier needs a differentiated backfill against the pre-existing
    # safety_critical flag - the server_default alone would leave every
    # row at "structural" including the ones already marked safety_critical.
    op.execute(
        """
        UPDATE dataset_classes
        SET tier = 'safety'
        WHERE safety_critical = true
        """
    )


def downgrade() -> None:
    op.drop_constraint('fk_dataset_classes_deprecated_by_id_annotators', 'dataset_classes', type_='foreignkey')
    op.drop_column('dataset_classes', 'deprecated_by_id')
    op.drop_column('dataset_classes', 'deprecated_at')
    op.drop_column('dataset_classes', 'tier')
    op.drop_column('dataset_classes', 'state')
