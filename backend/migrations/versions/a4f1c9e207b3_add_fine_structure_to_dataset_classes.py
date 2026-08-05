"""add fine_structure to dataset_classes

Marks classes whose label quality depends on thin/branching detail surviving
export - crack, corrosion, wheel shelling. For these the pipeline scores on
Dice/IoU **plus length-recall** (docs/pipeline.md §5.4) and shelling length
(§5.5), and three things in our export path were quietly destroying exactly
that signal:

  1. polygon_service.mask_to_polygon() kept only the LARGEST external contour,
     so a crack that branches or breaks into segments lost every piece but one;
  2. Douglas-Peucker at epsilon = 0.002 * perimeter smoothed thin structure
     away;
  3. export wrote polygons only, never mask rasters.

A flagged class keeps every contour, skips simplification, and additionally
exports a binary mask PNG. Confirmed with the pipeline team 2026-08-05 (they
tile at training time, so we ship full frames + masks, not pre-tiled crops).

Seeded by keyword, exactly like safety_critical in dbc0c570a39a: an engineering
starting point for a domain expert to correct via the UI, not a certified list.

Revision ID: a4f1c9e207b3
Revises: dbc0c570a39a
Create Date: 2026-08-05

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a4f1c9e207b3'
down_revision: Union[str, None] = 'dbc0c570a39a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'dataset_classes',
        sa.Column('fine_structure', sa.Boolean(), server_default='false', nullable=False),
    )

    # Deliberately NARROWER than safety_critical's keyword set. This flag is
    # about defect morphology - thin, branching, length-measured features -
    # not about safety tier. A wheel or brake *component* outline is a blob and
    # simplifies fine; a crack across it does not. Matching on 'wheel' here
    # would flag every wheel component and make every export write pointless
    # mask rasters.
    op.execute(
        r"""
        UPDATE dataset_classes
        SET fine_structure = true
        WHERE name ~* '(crack|corrosion|shelling|scratch|flaking|spalling|tread)'
        """
    )


def downgrade() -> None:
    op.drop_column('dataset_classes', 'fine_structure')
