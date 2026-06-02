import pytest


def _disable_postgres_state(monkeypatch):
    from api.optional_ontology_writes import optional_ontology_writes_available

    monkeypatch.setenv("STATE_DB_BACKEND", "sqlite")
    monkeypatch.setenv("TALISMAN_ALLOW_SQLITE_STATE", "true")
    monkeypatch.delenv("OPTIONAL_ONTOLOGY_WRITES_ENABLED", raising=False)
    optional_ontology_writes_available.cache_clear()


class _UnexpectedOntologyService:
    def __init__(self, *_args, **_kwargs):
        raise AssertionError("optional ontology write should be skipped")


class _FailingOntologyService:
    def __init__(self, *_args, **_kwargs):
        raise RuntimeError("ontology unavailable")


def test_optional_audit_write_is_skipped_when_postgres_state_disabled(monkeypatch):
    from api import audit

    _disable_postgres_state(monkeypatch)
    monkeypatch.setattr(audit, "OntologyObjectService", _UnexpectedOntologyService)

    assert audit.emit_audit_event("chat.test", "agent", "succeeded") is None


def test_fail_closed_audit_still_attempts_write(monkeypatch):
    from api import audit

    _disable_postgres_state(monkeypatch)
    monkeypatch.setattr(audit, "OntologyObjectService", _FailingOntologyService)

    with pytest.raises(audit.AuditWriteError):
        audit.emit_audit_event("chat.test", "agent", "succeeded", fail_closed=True)


def test_optional_provenance_write_is_skipped_when_postgres_state_disabled(monkeypatch):
    from api import provenance

    _disable_postgres_state(monkeypatch)
    monkeypatch.setattr(provenance, "OntologyObjectService", _UnexpectedOntologyService)

    assert provenance.start_event(event_type="agent_turn", event_name="agent_chat") is None


def test_fail_closed_provenance_still_attempts_write(monkeypatch):
    from api import provenance

    _disable_postgres_state(monkeypatch)
    monkeypatch.setattr(provenance, "OntologyObjectService", _FailingOntologyService)

    with pytest.raises(provenance.ProvenanceWriteError):
        provenance.start_event(event_type="agent_turn", event_name="agent_chat", fail_closed=True)
