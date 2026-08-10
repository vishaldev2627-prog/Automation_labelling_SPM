"""add decision/hard_fail/override_reason to model_promotions

Module 6 of the class-incremental promotion plan
(docs/mlflow_class_incremental_architecture.md §I): the enforcement step -
model_promotion_service.approve() now reads these columns to refuse a
REJECT by default (override-able, with a mandatory reason) and refuse a
hard_fail unconditionally (never override-able). Every pre-existing row
predates this enforcement and never went through it - backfilled to
decision="REJECT" (the fail-closed default, never an implicit PROMOTE) and
hard_fail=false, matching the column defaults exactly, since there is no
after-the-fact way to know what a pre-existing row's real verdict would
have been.

Revision ID: f3a8c1d92e47
Revises: e2f4a7b91c30
Create Date: 2026-08-11

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f3a8c1d92e47'
down_revision: Union[str, None] = 'e2f4a7b91c30'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('model_promotions', sa.Column('decision', sa.String(), server_default='REJECT', nullable=False))
    op.add_column('model_promotions', sa.Column('hard_fail', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('model_promotions', sa.Column('override_reason', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('model_promotions', 'override_reason')
    op.drop_column('model_promotions', 'hard_fail')
    op.drop_column('model_promotions', 'decision')
