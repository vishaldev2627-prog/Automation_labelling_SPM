"""add model_promotions

Auditable record of "MLflow says this version is Production, did a human
here approve or reject actually swapping the live-serving model to match
it" (M7.5). Deliberately a second, distinct gate from M7's own promotion-
recommendation tags in MLflow itself - curating what's eligible (a curator,
in MLflow's UI) and deciding what live annotators actually get suggestions
from (a model_reviewer, in this app) are different responsibilities, and
this table is the record of the second one.

Purely additive - nothing to backfill, since nothing before this milestone
ever proposed swapping the live-serving model automatically at all.

Revision ID: a1b2c3d4e5f6
Revises: d5a7c0f3e8b1
Create Date: 2026-08-05

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'd5a7c0f3e8b1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'model_promotions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('dataset_view', sa.String(), nullable=False),
        sa.Column('mlflow_model_name', sa.String(), nullable=False),
        sa.Column('mlflow_version', sa.String(), nullable=False),
        sa.Column('mlflow_run_id', sa.String(), nullable=False),
        sa.Column('promotion_recommendation', sa.String(), nullable=False),
        sa.Column('regressed_classes', sa.String(), nullable=True),
        sa.Column('status', sa.String(), server_default='pending', nullable=False),
        sa.Column('local_weights_path', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('decided_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('decided_by_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['decided_by_id'], ['annotators.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'dataset_view', 'mlflow_model_name', 'mlflow_version',
            name='uq_model_promotions_view_model_version',
        ),
    )
    op.create_index(op.f('ix_model_promotions_dataset_view'), 'model_promotions', ['dataset_view'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_model_promotions_dataset_view'), table_name='model_promotions')
    op.drop_table('model_promotions')
