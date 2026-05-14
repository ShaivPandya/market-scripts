"""Add decision quality fields and expanded idea actions.

Revision ID: 20260514_0001
Revises: 20260513_0001
Create Date: 2026-05-14
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260514_0001"
down_revision: str | None = "20260513_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ACTIONS = (
    "buy",
    "add",
    "short",
    "sell",
    "trim",
    "reduce",
    "exit",
    "hedge",
    "rebalance",
    "hold",
    "watch",
    "research",
    "avoid",
    "do_nothing",
)
_LEGACY_IDEA_ACTIONS = ("buy", "watch", "avoid", "do_nothing")


def _action_check(actions: tuple[str, ...]) -> str:
    quoted = ", ".join(f"'{action}'" for action in actions)
    return f"action IN ({quoted})"


def upgrade() -> None:
    with op.batch_alter_table("idea_evaluations") as batch:
        batch.drop_constraint("ck_idea_evaluations_action", type_="check")
        batch.add_column(sa.Column("decision_quality_json", sa.Text, nullable=False, server_default="{}"))
        batch.add_column(sa.Column("decision_quality_gate_json", sa.Text, nullable=False, server_default="{}"))
        batch.create_check_constraint("ck_idea_evaluations_action", _action_check(_ACTIONS))

    with op.batch_alter_table("idea_comparison_rankings") as batch:
        batch.drop_constraint("ck_idea_comparison_action", type_="check")
        batch.create_check_constraint("ck_idea_comparison_action", _action_check(_ACTIONS))

    with op.batch_alter_table("recommendations") as batch:
        batch.add_column(sa.Column("decision_quality_json", sa.Text))
        batch.add_column(sa.Column("decision_quality_gate_json", sa.Text))


def downgrade() -> None:
    with op.batch_alter_table("recommendations") as batch:
        batch.drop_column("decision_quality_gate_json")
        batch.drop_column("decision_quality_json")

    with op.batch_alter_table("idea_comparison_rankings") as batch:
        batch.drop_constraint("ck_idea_comparison_action", type_="check")
        batch.create_check_constraint("ck_idea_comparison_action", _action_check(_LEGACY_IDEA_ACTIONS))

    with op.batch_alter_table("idea_evaluations") as batch:
        batch.drop_constraint("ck_idea_evaluations_action", type_="check")
        batch.drop_column("decision_quality_gate_json")
        batch.drop_column("decision_quality_json")
        batch.create_check_constraint("ck_idea_evaluations_action", _action_check(_LEGACY_IDEA_ACTIONS))
