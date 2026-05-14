"""Tests for /api/health and /api/admin/health release metadata — SHA-33."""

import os

import pytest


@pytest.fixture(autouse=True)
def _set_release_env(monkeypatch):
    """Inject release env vars so the health endpoints have data to return."""
    monkeypatch.setenv("TALISMAN_RELEASE_GIT_SHA", "abc123def456abc123def456abc123def456abc1")
    monkeypatch.setenv("TALISMAN_RELEASE_GIT_SHA_SHORT", "abc123d")
    monkeypatch.setenv("TALISMAN_RELEASE_IMAGE_TAG", "abc123d")
    monkeypatch.setenv("TALISMAN_RELEASE_ENVIRONMENT", "production")

    # Force re-evaluation of the module-level _RELEASE_META dicts.
    import api.main as main_mod

    main_mod._RELEASE_META = {
        "git_sha": "abc123def456abc123def456abc123def456abc1",
        "git_sha_short": "abc123d",
        "image_tag": "abc123d",
        "environment": "production",
    }
    main_mod._RELEASE_META_SAFE = {
        "git_sha_short": "abc123d",
        "image_tag": "abc123d",
    }
    yield
    # Reset to empty so other tests aren't affected
    main_mod._RELEASE_META = {
        "git_sha": "",
        "git_sha_short": "",
        "image_tag": "",
        "environment": os.environ.get("ENVIRONMENT", "development"),
    }
    main_mod._RELEASE_META_SAFE = {}


class TestHealthReleaseMeta:
    """Public /api/health should expose only safe release subset."""

    def test_health_includes_release_block(self, client) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "release" in data

    def test_health_release_has_safe_fields(self, client) -> None:
        data = client.get("/api/health").json()
        release = data["release"]
        assert "git_sha_short" in release
        assert "image_tag" in release

    def test_health_release_does_not_expose_full_sha(self, client) -> None:
        data = client.get("/api/health").json()
        release = data["release"]
        assert "git_sha" not in release
        assert "environment" not in release

    def test_health_release_does_not_expose_secrets(self, client) -> None:
        data = client.get("/api/health").json()
        raw = str(data)
        for secret_key in ("DATABASE_URL", "API_KEY", "JWT_SECRET", "PASSWORD"):
            assert secret_key not in raw


class TestAdminHealthReleaseMeta:
    """Authenticated /api/admin/health should expose full release identity."""

    def test_admin_health_includes_release_block(self, auth_client) -> None:
        resp = auth_client.get("/api/admin/health")
        assert resp.status_code in (200, 503)  # may be unhealthy in test env
        data = resp.json()
        assert "release" in data

    def test_admin_health_release_has_full_fields(self, auth_client) -> None:
        data = auth_client.get("/api/admin/health").json()
        release = data["release"]
        assert "git_sha" in release
        assert "git_sha_short" in release
        assert "image_tag" in release
        assert "environment" in release

    def test_admin_health_release_values(self, auth_client) -> None:
        data = auth_client.get("/api/admin/health").json()
        release = data["release"]
        assert release["git_sha"] == "abc123def456abc123def456abc123def456abc1"
        assert release["git_sha_short"] == "abc123d"
        assert release["image_tag"] == "abc123d"
        assert release["environment"] == "production"

    def test_admin_health_does_not_expose_secrets(self, auth_client) -> None:
        data = auth_client.get("/api/admin/health").json()
        raw = str(data)
        for secret_key in ("DATABASE_URL", "API_KEY", "JWT_SECRET", "PASSWORD", "ANTHROPIC", "OPENAI"):
            assert secret_key not in raw


class TestHealthNoReleaseMeta:
    """When release env vars are empty, health should omit the release block."""

    @pytest.fixture(autouse=True)
    def _clear_release(self):
        import api.main as main_mod

        main_mod._RELEASE_META_SAFE = {}
        yield

    def test_health_omits_release_when_empty(self, client) -> None:
        data = client.get("/api/health").json()
        assert "release" not in data
        assert data["status"] == "ok"
