"""Create GCP production state schema.

Revision ID: 20260429_0001
Revises:
Create Date: 2026-04-29
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260429_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "positions",
        sa.Column("ticker", sa.Text, primary_key=True),
        sa.Column("asset", sa.Text, nullable=False),
        sa.Column("direction", sa.Text, nullable=False),
        sa.Column("contrarian", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("conviction", sa.Integer, nullable=False, server_default=sa.text("3")),
        sa.Column("cost_basis", sa.Float),
        sa.Column("shares", sa.Float),
        sa.Column("role", sa.Text, nullable=False, server_default="position"),
    )

    op.create_table(
        "thesis_meta",
        sa.Column("ticker", sa.Text, primary_key=True),
        sa.Column("status", sa.Text, nullable=False, server_default="active"),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("updated_at", sa.Text, nullable=False),
    )
    op.create_table(
        "thesis_status_history",
        sa.Column("id", sa.Integer, sa.Identity(), primary_key=True),
        sa.Column("ticker", sa.Text, sa.ForeignKey("thesis_meta.ticker"), nullable=False),
        sa.Column("old_status", sa.Text),
        sa.Column("new_status", sa.Text, nullable=False),
        sa.Column("reason", sa.Text),
        sa.Column("changed_at", sa.Text, nullable=False),
    )
    op.create_table(
        "thesis_evaluations",
        sa.Column("id", sa.Integer, sa.Identity(), primary_key=True),
        sa.Column("ticker", sa.Text, sa.ForeignKey("thesis_meta.ticker"), nullable=False),
        sa.Column("evaluated_at", sa.Text, nullable=False),
        sa.Column("thesis_status", sa.Text, nullable=False),
        sa.Column("technical_read", sa.Text, nullable=False),
        sa.Column("fundamental_read", sa.Text, nullable=False),
        sa.Column("action", sa.Text, nullable=False),
        sa.Column("confidence", sa.Text, nullable=False),
        sa.Column("key_developments", sa.Text, nullable=False),
        sa.Column("earnings_note", sa.Text),
        sa.Column("risk_flag", sa.Text),
        sa.UniqueConstraint("ticker", "evaluated_at", name="uq_thesis_evaluations_ticker_evaluated_at"),
    )

    op.create_table(
        "catalysts",
        sa.Column("id", sa.Integer, sa.Identity(), primary_key=True),
        sa.Column("ticker", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("category", sa.Text, nullable=False, server_default="fundamental"),
        sa.Column("status", sa.Text, nullable=False, server_default="pending"),
        sa.Column("target_date", sa.Text),
        sa.Column("evidence", sa.Text),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("updated_at", sa.Text, nullable=False),
        sa.Column("created_by", sa.Text, nullable=False, server_default="user"),
    )
    op.create_index("idx_catalysts_ticker", "catalysts", ["ticker"])

    op.create_table(
        "kill_conditions",
        sa.Column("id", sa.Integer, sa.Identity(), primary_key=True),
        sa.Column("ticker", sa.Text, nullable=False),
        sa.Column("condition", sa.Text, nullable=False),
        sa.Column("metric", sa.Text),
        sa.Column("threshold", sa.Text),
        sa.Column("status", sa.Text, nullable=False, server_default="active"),
        sa.Column("triggered_at", sa.Text),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("updated_at", sa.Text, nullable=False),
        sa.Column("created_by", sa.Text, nullable=False, server_default="user"),
    )
    op.create_index("idx_kill_conditions_ticker", "kill_conditions", ["ticker"])

    op.create_table(
        "workflow_runs",
        sa.Column("run_id", sa.Text, primary_key=True),
        sa.Column("workflow_name", sa.Text, nullable=False),
        sa.Column("ticker", sa.Text),
        sa.Column("status", sa.Text, nullable=False, server_default="running"),
        sa.Column("started_at", sa.Text, nullable=False),
        sa.Column("completed_at", sa.Text),
        sa.Column("tool_sections", sa.Text),
        sa.Column("synthesis", sa.Text),
        sa.Column("artifacts", sa.Text),
        sa.Column("error", sa.Text),
    )
    op.create_index("idx_workflow_runs_name", "workflow_runs", ["workflow_name"])
    op.create_index("idx_workflow_runs_ticker", "workflow_runs", ["ticker"])
    op.create_index("idx_workflow_runs_started", "workflow_runs", ["started_at"])

    op.create_table(
        "action_items",
        sa.Column("id", sa.Integer, sa.Identity(), primary_key=True),
        sa.Column("ticker", sa.Text),
        sa.Column("action_type", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("urgency", sa.Text, nullable=False, server_default="normal"),
        sa.Column("status", sa.Text, nullable=False, server_default="open"),
        sa.Column("source_type", sa.Text, nullable=False, server_default="user"),
        sa.Column("source_id", sa.Text),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("completed_at", sa.Text),
        sa.Column("resolution_note", sa.Text),
    )
    op.create_index("idx_action_items_status", "action_items", ["status"])
    op.create_index("idx_action_items_ticker", "action_items", ["ticker"])

    op.create_table(
        "watch_triggers",
        sa.Column("id", sa.Integer, sa.Identity(), primary_key=True),
        sa.Column("ticker", sa.Text),
        sa.Column("trigger_type", sa.Text, nullable=False),
        sa.Column("condition", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False, server_default="active"),
        sa.Column("source_type", sa.Text, nullable=False, server_default="user"),
        sa.Column("source_id", sa.Text),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("fired_at", sa.Text),
        sa.Column("expires_at", sa.Text),
    )
    op.create_index("idx_watch_triggers_status", "watch_triggers", ["status"])
    op.create_index("idx_watch_triggers_ticker", "watch_triggers", ["ticker"])

    op.create_table(
        "research_notes",
        sa.Column("id", sa.Integer, sa.Identity(), primary_key=True),
        sa.Column("ticker", sa.Text),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("note_type", sa.Text, nullable=False, server_default="general"),
        sa.Column("source_type", sa.Text, nullable=False, server_default="user"),
        sa.Column("source_id", sa.Text),
        sa.Column("created_at", sa.Text, nullable=False),
    )
    op.create_index("idx_research_notes_ticker", "research_notes", ["ticker"])

    op.create_table(
        "pending_approvals",
        sa.Column("id", sa.Integer, sa.Identity(), primary_key=True),
        sa.Column("entity_type", sa.Text, nullable=False),
        sa.Column("entity_id", sa.Integer),
        sa.Column("ticker", sa.Text),
        sa.Column("proposed_change", sa.Text, nullable=False),
        sa.Column("reason", sa.Text),
        sa.Column("source_type", sa.Text, nullable=False, server_default="workflow"),
        sa.Column("source_id", sa.Text),
        sa.Column("status", sa.Text, nullable=False, server_default="pending"),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("resolved_at", sa.Text),
        sa.Column("resolved_note", sa.Text),
    )
    op.create_index("idx_pending_approvals_status", "pending_approvals", ["status"])
    op.create_index("idx_pending_approvals_ticker", "pending_approvals", ["ticker"])

    op.create_table(
        "conversation_sessions",
        sa.Column("session_id", sa.Text, primary_key=True),
        sa.Column("started_at", sa.Text, nullable=False),
        sa.Column("ended_at", sa.Text),
        sa.Column("message_count", sa.Integer, server_default=sa.text("0")),
        sa.Column("key_tickers", sa.Text),
        sa.Column("key_topics", sa.Text),
        sa.Column("summary", sa.Text),
        sa.Column("transcript", sa.Text, nullable=False),
        sa.Column("rolling_summary", sa.Text),
        sa.Column("server_messages", sa.Text, nullable=False, server_default="[]"),
    )
    op.create_index("idx_sessions_ended_at", "conversation_sessions", ["ended_at"])

    op.create_table(
        "ontology_nodes",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("type", sa.Text, nullable=False),
        sa.Column("label", sa.Text, nullable=False),
        sa.Column("properties_json", sa.Text, nullable=False),
        sa.Column("schema_name", sa.Text, nullable=False, server_default="legacy"),
        sa.Column("schema_version", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("updated_at", sa.Text, nullable=False),
    )
    op.create_index("idx_ontology_nodes_type", "ontology_nodes", ["type"])
    op.create_table(
        "ontology_edges",
        sa.Column("source_id", sa.Text, primary_key=True),
        sa.Column("target_id", sa.Text, primary_key=True),
        sa.Column("relation_type", sa.Text, primary_key=True),
        sa.Column("properties_json", sa.Text, nullable=False),
        sa.Column("schema_name", sa.Text, nullable=False, server_default="legacy"),
        sa.Column("schema_version", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("updated_at", sa.Text, nullable=False),
    )
    op.create_index("idx_ontology_edges_source", "ontology_edges", ["source_id"])
    op.create_index("idx_ontology_edges_target", "ontology_edges", ["target_id"])
    op.create_table(
        "ontology_runs",
        sa.Column("run_id", sa.Text, primary_key=True),
        sa.Column("as_of", sa.Text, nullable=False),
        sa.Column("source_status_json", sa.Text, nullable=False),
        sa.Column("required_modules_json", sa.Text, nullable=False),
        sa.Column("optional_modules_json", sa.Text, nullable=False),
        sa.Column("component_scores_json", sa.Text, nullable=False),
        sa.Column("created_at", sa.Text, nullable=False),
    )
    op.create_index("idx_ontology_runs_created_at", "ontology_runs", ["created_at"])
    op.create_table(
        "ontology_snapshot_nodes",
        sa.Column("run_id", sa.Text, sa.ForeignKey("ontology_runs.run_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("type", sa.Text, nullable=False),
        sa.Column("label", sa.Text, nullable=False),
        sa.Column("properties_json", sa.Text, nullable=False),
        sa.Column("schema_name", sa.Text, nullable=False, server_default="legacy"),
        sa.Column("schema_version", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("updated_at", sa.Text, nullable=False),
    )
    op.create_index("idx_ontology_snapshot_nodes_run_type", "ontology_snapshot_nodes", ["run_id", "type"])
    op.create_table(
        "ontology_snapshot_edges",
        sa.Column("run_id", sa.Text, sa.ForeignKey("ontology_runs.run_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("source_id", sa.Text, primary_key=True),
        sa.Column("target_id", sa.Text, primary_key=True),
        sa.Column("relation_type", sa.Text, primary_key=True),
        sa.Column("properties_json", sa.Text, nullable=False),
        sa.Column("schema_name", sa.Text, nullable=False, server_default="legacy"),
        sa.Column("schema_version", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("updated_at", sa.Text, nullable=False),
    )
    op.create_index("idx_ontology_snapshot_edges_run_source", "ontology_snapshot_edges", ["run_id", "source_id"])
    op.create_index("idx_ontology_snapshot_edges_run_target", "ontology_snapshot_edges", ["run_id", "target_id"])

    op.create_table(
        "retrieval_documents",
        sa.Column("doc_id", sa.Text, primary_key=True),
        sa.Column("doc_type", sa.Text, nullable=False),
        sa.Column("source_path", sa.Text),
        sa.Column("ticker", sa.Text),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_retrieval_documents_doc_type", "retrieval_documents", ["doc_type"])
    op.create_index("idx_retrieval_documents_ticker", "retrieval_documents", ["ticker"])
    op.execute(
        """
        CREATE TABLE retrieval_chunks (
            chunk_id text PRIMARY KEY,
            doc_id text NOT NULL REFERENCES retrieval_documents(doc_id) ON DELETE CASCADE,
            chunk_index integer NOT NULL,
            content text NOT NULL,
            heading text,
            embedding vector(384) NOT NULL
        )
        """
    )
    op.create_index("idx_retrieval_chunks_doc_id", "retrieval_chunks", ["doc_id"])
    op.execute(
        "CREATE INDEX idx_retrieval_chunks_embedding_hnsw ON retrieval_chunks USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "central_bank_items",
        sa.Column("guid", sa.Text, primary_key=True),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("published_at", sa.Text, nullable=False),
        sa.Column("content_sha256", sa.Text),
        sa.Column("content_text", sa.Text),
        sa.Column("summary_json", sa.Text),
        sa.Column("content_url", sa.Text),
    )

    op.create_table(
        "industry_transcripts",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("ticker", sa.Text, nullable=False),
        sa.Column("company_name", sa.Text, nullable=False),
        sa.Column("sector", sa.Text, nullable=False),
        sa.Column("sector_type", sa.Text, nullable=False),
        sa.Column("sub_sector", sa.Text, nullable=False),
        sa.Column("quarter", sa.Integer, nullable=False),
        sa.Column("year", sa.Integer, nullable=False),
        sa.Column("transcript_text", sa.Text),
        sa.Column("content_sha256", sa.Text),
        sa.Column("summary_json", sa.Text),
        sa.Column("fetched_at", sa.Text),
        sa.Column("summarized_at", sa.Text),
        sa.Column("transcript_date", sa.Text),
        sa.Column("is_stale", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("price_reaction_2d", sa.Float),
    )
    op.create_index("idx_industry_transcripts_ticker", "industry_transcripts", ["ticker"])
    op.create_index("idx_industry_transcripts_sector", "industry_transcripts", ["sector"])

    op.create_table(
        "async_jobs",
        sa.Column("job_id", sa.Text, primary_key=True),
        sa.Column("job_type", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("payload_json", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("result_json", postgresql.JSONB),
        sa.Column("cloud_run_job_name", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("error", sa.Text),
    )
    op.create_index("idx_async_jobs_status", "async_jobs", ["status"])
    op.create_index("idx_async_jobs_type_created", "async_jobs", ["job_type", "created_at"])

    op.create_table(
        "migration_runs",
        sa.Column("run_id", sa.Text, primary_key=True),
        sa.Column("source_manifest_sha256", sa.Text, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("status", sa.Text, nullable=False),
    )
    op.create_table(
        "migration_sources",
        sa.Column("run_id", sa.Text, sa.ForeignKey("migration_runs.run_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("source_name", sa.Text, primary_key=True),
        sa.Column("source_sha256", sa.Text, nullable=False),
        sa.Column("row_counts_json", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("object_manifest_json", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    for table in [
        "migration_sources",
        "migration_runs",
        "async_jobs",
        "industry_transcripts",
        "central_bank_items",
        "retrieval_chunks",
        "retrieval_documents",
        "ontology_snapshot_edges",
        "ontology_snapshot_nodes",
        "ontology_runs",
        "ontology_edges",
        "ontology_nodes",
        "conversation_sessions",
        "pending_approvals",
        "research_notes",
        "watch_triggers",
        "action_items",
        "workflow_runs",
        "kill_conditions",
        "catalysts",
        "thesis_evaluations",
        "thesis_status_history",
        "thesis_meta",
        "positions",
    ]:
        op.drop_table(table)
