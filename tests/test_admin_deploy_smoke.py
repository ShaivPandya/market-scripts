"""Tests for the GET /api/admin/deploy-smoke endpoint (SHA-34)."""

from __future__ import annotations


def test_deploy_smoke_requires_auth(client):
    resp = client.get("/api/admin/deploy-smoke")
    assert resp.status_code == 401


def test_deploy_smoke_all_pass(auth_client, monkeypatch):
    """When all backend checks pass, expect 200."""
    from api.routers import admin_jobs

    monkeypatch.setattr(admin_jobs, "_check_postgres", lambda: (True, "ok"))
    monkeypatch.setattr(admin_jobs, "_check_migration_head", lambda: (True, "not_configured"))
    monkeypatch.setattr(admin_jobs, "_check_read_model", lambda: (True, "ok"))
    monkeypatch.setattr(admin_jobs, "_check_action_approval_safety", lambda: (True, "pending_count=0"))

    resp = auth_client.get("/api/admin/deploy-smoke")
    assert resp.status_code == 200
    body = resp.json()
    assert "checks" in body
    assert all(c["passed"] for c in body["checks"].values())


def test_deploy_smoke_postgres_failure(auth_client, monkeypatch):
    """When Postgres fails, expect 503 with postgres in failed_checks."""
    from api.routers import admin_jobs

    monkeypatch.setattr(admin_jobs, "_check_postgres", lambda: (False, "connection refused"))
    monkeypatch.setattr(admin_jobs, "_check_migration_head", lambda: (True, "not_configured"))
    monkeypatch.setattr(admin_jobs, "_check_read_model", lambda: (True, "ok"))
    monkeypatch.setattr(admin_jobs, "_check_action_approval_safety", lambda: (True, "pending_count=0"))

    resp = auth_client.get("/api/admin/deploy-smoke")
    assert resp.status_code == 503
    body = resp.json()
    assert "postgres" in body["failed_checks"]
    assert body["checks"]["postgres"]["passed"] is False


def test_deploy_smoke_migration_mismatch(auth_client, monkeypatch):
    """When migration head mismatches, expect 503 with migration_head in failed_checks."""
    from api.routers import admin_jobs

    monkeypatch.setattr(admin_jobs, "_check_postgres", lambda: (True, "ok"))
    monkeypatch.setattr(
        admin_jobs,
        "_check_migration_head",
        lambda: (False, "mismatch: deployed=abc123 db=def456"),
    )
    monkeypatch.setattr(admin_jobs, "_check_read_model", lambda: (True, "ok"))
    monkeypatch.setattr(admin_jobs, "_check_action_approval_safety", lambda: (True, "pending_count=0"))

    resp = auth_client.get("/api/admin/deploy-smoke")
    assert resp.status_code == 503
    body = resp.json()
    assert "migration_head" in body["failed_checks"]


def test_deploy_smoke_read_model_failure(auth_client, monkeypatch):
    """When the read model fails, expect 503 with read_model in failed_checks."""
    from api.routers import admin_jobs

    monkeypatch.setattr(admin_jobs, "_check_postgres", lambda: (True, "ok"))
    monkeypatch.setattr(admin_jobs, "_check_migration_head", lambda: (True, "not_configured"))
    monkeypatch.setattr(admin_jobs, "_check_read_model", lambda: (False, "db not available"))
    monkeypatch.setattr(admin_jobs, "_check_action_approval_safety", lambda: (True, "pending_count=0"))

    resp = auth_client.get("/api/admin/deploy-smoke")
    assert resp.status_code == 503
    body = resp.json()
    assert "read_model" in body["failed_checks"]


def test_deploy_smoke_action_approval_failure(auth_client, monkeypatch):
    """When action/approval check fails, expect 503 with it in failed_checks."""
    from api.routers import admin_jobs

    monkeypatch.setattr(admin_jobs, "_check_postgres", lambda: (True, "ok"))
    monkeypatch.setattr(admin_jobs, "_check_migration_head", lambda: (True, "not_configured"))
    monkeypatch.setattr(admin_jobs, "_check_read_model", lambda: (True, "ok"))
    monkeypatch.setattr(
        admin_jobs,
        "_check_action_approval_safety",
        lambda: (False, "command service unavailable"),
    )

    resp = auth_client.get("/api/admin/deploy-smoke")
    assert resp.status_code == 503
    body = resp.json()
    assert "action_approval_safety" in body["failed_checks"]


def test_deploy_smoke_includes_release_info(auth_client, monkeypatch):
    """When TALISMAN_RELEASE_IMAGE_TAG is set, it appears in response."""
    from api.routers import admin_jobs

    monkeypatch.setenv("TALISMAN_RELEASE_IMAGE_TAG", "test-sha")
    monkeypatch.setattr(admin_jobs, "_check_postgres", lambda: (True, "ok"))
    monkeypatch.setattr(admin_jobs, "_check_migration_head", lambda: (True, "not_configured"))
    monkeypatch.setattr(admin_jobs, "_check_read_model", lambda: (True, "ok"))
    monkeypatch.setattr(admin_jobs, "_check_action_approval_safety", lambda: (True, "pending_count=0"))

    resp = auth_client.get("/api/admin/deploy-smoke")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("release", {}).get("image_tag") == "test-sha"


def test_deploy_smoke_multiple_failures(auth_client, monkeypatch):
    """When multiple checks fail, all appear in failed_checks."""
    from api.routers import admin_jobs

    monkeypatch.setattr(admin_jobs, "_check_postgres", lambda: (False, "error"))
    monkeypatch.setattr(admin_jobs, "_check_migration_head", lambda: (False, "error"))
    monkeypatch.setattr(admin_jobs, "_check_read_model", lambda: (True, "ok"))
    monkeypatch.setattr(admin_jobs, "_check_action_approval_safety", lambda: (False, "error"))

    resp = auth_client.get("/api/admin/deploy-smoke")
    assert resp.status_code == 503
    body = resp.json()
    assert len(body["failed_checks"]) == 3
    assert "postgres" in body["failed_checks"]
    assert "migration_head" in body["failed_checks"]
    assert "action_approval_safety" in body["failed_checks"]
