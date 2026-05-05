from __future__ import annotations

from pathlib import Path

from ontology.read_model import TemporalReadModelRepository


class _FakeConnection:
    def __init__(self):
        self.execute_calls: list[tuple[str, object | None]] = []
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, sql: str, params: object | None = None):
        self.execute_calls.append((sql, params))
        return self

    def commit(self):
        self.commits += 1


def test_refresh_uses_security_definer_function():
    conn = _FakeConnection()
    repo = TemporalReadModelRepository(connection_factory=lambda: conn)

    repo.refresh()

    assert conn.execute_calls == [("SELECT refresh_ontology_temporal_read_models()", None)]
    assert conn.commits == 1


def test_temporal_read_model_migration_grants_runtime_roles():
    migration = Path("migrations/versions/20260505_0006_ontology_temporal_read_models.py").read_text(
        encoding="utf-8"
    )

    for view_name in (
        "ontology_current_position_risk_read_model",
        "ontology_current_position_signal_evidence_read_model",
        "ontology_current_position_thesis_context_read_model",
        "ontology_current_decision_lineage_read_model",
        "ontology_current_source_status_read_model",
        "ontology_current_computed_snapshot_read_model",
    ):
        assert view_name in migration

    assert "CREATE OR REPLACE FUNCTION refresh_ontology_temporal_read_models()" in migration
    assert "SECURITY DEFINER" in migration
    assert "GRANT SELECT ON TABLE %I TO talisman_app" in migration
    assert "GRANT SELECT ON TABLE %I TO talisman_worker" in migration
    assert "GRANT MAINTAIN ON TABLE %I TO talisman_app" in migration
    assert "GRANT MAINTAIN ON TABLE %I TO talisman_worker" in migration
    assert "GRANT EXECUTE ON FUNCTION refresh_ontology_temporal_read_models() TO talisman_app" in migration
    assert "GRANT EXECUTE ON FUNCTION refresh_ontology_temporal_read_models() TO talisman_worker" in migration
