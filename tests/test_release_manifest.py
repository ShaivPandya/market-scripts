"""Tests for infra.gcp.release_manifest — SHA-33."""

import json
from pathlib import Path

import pytest

from infra.gcp.release_manifest import (
    REQUIRED_GIT_KEYS,
    REQUIRED_IMAGE_KEYS,
    REQUIRED_KEYS,
    SCHEMA_VERSION,
    _extract_rollback_refs,
    _parse_image_tag,
    build_manifest,
    validate_manifest,
    write_manifest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLEAN_GIT = {
    "sha": "abc123def456abc123def456abc123def456abc1",
    "sha_short": "abc123d",
    "branch": "main",
    "dirty": False,
}

_DIRTY_GIT = {**_CLEAN_GIT, "dirty": True}

_IMAGE_URI = "us-central1-docker.pkg.dev/my-project/talisman/api:abc123d"


def _base_manifest(**overrides):
    kwargs = {
        "image_uri": _IMAGE_URI,
        "git_metadata": _CLEAN_GIT,
        "migration_head": "aabbcc001122",
        "environment": "production",
        "config_profile": "default",
    }
    kwargs.update(overrides)
    return build_manifest(**kwargs)


# ---------------------------------------------------------------------------
# Schema + required fields
# ---------------------------------------------------------------------------


class TestManifestRequiredFields:
    def test_all_top_level_keys_present(self) -> None:
        manifest = _base_manifest()
        for key in REQUIRED_KEYS:
            assert key in manifest, f"missing top-level key: {key}"

    def test_git_keys_present(self) -> None:
        manifest = _base_manifest()
        for key in REQUIRED_GIT_KEYS:
            assert key in manifest["git"], f"missing git.{key}"

    def test_image_keys_present(self) -> None:
        manifest = _base_manifest()
        for key in REQUIRED_IMAGE_KEYS:
            assert key in manifest["image"], f"missing image.{key}"

    def test_schema_version(self) -> None:
        manifest = _base_manifest()
        assert manifest["schema_version"] == SCHEMA_VERSION

    def test_generated_at_is_iso(self) -> None:
        manifest = _base_manifest()
        from datetime import datetime

        # Should parse without error
        datetime.fromisoformat(manifest["generated_at"])

    def test_validation_passes_for_valid_manifest(self) -> None:
        manifest = _base_manifest()
        assert validate_manifest(manifest) == []


class TestManifestValidation:
    def test_missing_top_level_key(self) -> None:
        manifest = _base_manifest()
        del manifest["git"]
        errors = validate_manifest(manifest)
        assert any("git" in e for e in errors)

    def test_missing_git_subkey(self) -> None:
        manifest = _base_manifest()
        del manifest["git"]["sha"]
        errors = validate_manifest(manifest)
        assert any("git.sha" in e for e in errors)

    def test_missing_image_subkey(self) -> None:
        manifest = _base_manifest()
        del manifest["image"]["uri"]
        errors = validate_manifest(manifest)
        assert any("image.uri" in e for e in errors)


# ---------------------------------------------------------------------------
# Git metadata: clean vs dirty
# ---------------------------------------------------------------------------


class TestGitMetadata:
    def test_clean_tree(self) -> None:
        manifest = _base_manifest(git_metadata=_CLEAN_GIT)
        assert manifest["git"]["dirty"] is False

    def test_dirty_tree(self) -> None:
        manifest = _base_manifest(git_metadata=_DIRTY_GIT)
        assert manifest["git"]["dirty"] is True

    def test_git_sha_passthrough(self) -> None:
        manifest = _base_manifest(git_metadata=_CLEAN_GIT)
        assert manifest["git"]["sha"] == _CLEAN_GIT["sha"]
        assert manifest["git"]["sha_short"] == _CLEAN_GIT["sha_short"]
        assert manifest["git"]["branch"] == _CLEAN_GIT["branch"]


# ---------------------------------------------------------------------------
# Image metadata
# ---------------------------------------------------------------------------


class TestImageMetadata:
    def test_image_uri(self) -> None:
        manifest = _base_manifest()
        assert manifest["image"]["uri"] == _IMAGE_URI

    def test_image_tag_parsed(self) -> None:
        manifest = _base_manifest()
        assert manifest["image"]["tag"] == "abc123d"

    def test_image_digest_included_when_provided(self) -> None:
        digest = "sha256:deadbeef1234"
        manifest = _base_manifest(image_digest=digest)
        assert manifest["image"]["digest"] == digest

    def test_image_digest_absent_when_not_provided(self) -> None:
        manifest = _base_manifest()
        assert "digest" not in manifest["image"]


# ---------------------------------------------------------------------------
# Database + runtime
# ---------------------------------------------------------------------------


class TestDatabaseAndRuntime:
    def test_migration_head(self) -> None:
        manifest = _base_manifest(migration_head="rev123")
        assert manifest["database"]["migration_head"] == "rev123"

    def test_migration_head_none(self) -> None:
        manifest = _base_manifest(migration_head=None)
        assert manifest["database"]["migration_head"] is None

    def test_runtime_environment(self) -> None:
        manifest = _base_manifest(environment="staging")
        assert manifest["runtime"]["environment"] == "staging"

    def test_runtime_config_profile(self) -> None:
        manifest = _base_manifest(config_profile="custom")
        assert manifest["runtime"]["config_profile"] == "custom"


# ---------------------------------------------------------------------------
# Rollback refs
# ---------------------------------------------------------------------------


class TestRollbackRefs:
    def test_no_prior_manifest(self) -> None:
        manifest = _base_manifest(prior_manifest_path=None)
        assert manifest["rollback"] == {}

    def test_prior_manifest_file_missing(self, tmp_path: Path) -> None:
        manifest = _base_manifest(prior_manifest_path=tmp_path / "nope.json")
        assert manifest["rollback"] == {}

    def test_prior_manifest_extracted(self, tmp_path: Path) -> None:
        prior = {
            "schema_version": "1",
            "git": {"sha": "old_sha_full", "sha_short": "old_sha"},
            "image": {
                "uri": "us-central1-docker.pkg.dev/proj/repo/api:old_sha",
                "tag": "old_sha",
                "digest": "sha256:olddigest",
            },
        }
        prior_path = tmp_path / "prior.json"
        prior_path.write_text(json.dumps(prior))

        manifest = _base_manifest(prior_manifest_path=prior_path)
        rb = manifest["rollback"]
        assert rb["prior_git_sha"] == "old_sha_full"
        assert rb["prior_git_sha_short"] == "old_sha"
        assert rb["prior_image_uri"] == prior["image"]["uri"]
        assert rb["prior_image_tag"] == "old_sha"
        assert rb["prior_image_digest"] == "sha256:olddigest"

    def test_prior_manifest_invalid_json(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        manifest = _base_manifest(prior_manifest_path=bad)
        assert manifest["rollback"] == {}

    def test_prior_manifest_missing_schema_version(self, tmp_path: Path) -> None:
        bad = tmp_path / "nover.json"
        bad.write_text(json.dumps({"git": {"sha": "x"}}))
        manifest = _base_manifest(prior_manifest_path=bad)
        assert manifest["rollback"] == {}


# ---------------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------------


class TestImageTagParsing:
    def test_standard_tag(self) -> None:
        assert _parse_image_tag("registry/repo/img:v1.2.3") == "v1.2.3"

    def test_sha_tag(self) -> None:
        assert _parse_image_tag("registry/repo/img:abc123d") == "abc123d"

    def test_no_tag(self) -> None:
        assert _parse_image_tag("registry/repo/img") == "latest"


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


class TestWriteManifest:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        manifest = _base_manifest()
        out = write_manifest(manifest, tmp_path / "manifest.json")
        loaded = json.loads(out.read_text())
        assert loaded["schema_version"] == SCHEMA_VERSION
        assert loaded["git"]["sha"] == _CLEAN_GIT["sha"]

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "manifest.json"
        write_manifest(_base_manifest(), deep)
        assert deep.exists()

    def test_trailing_newline(self, tmp_path: Path) -> None:
        out = write_manifest(_base_manifest(), tmp_path / "m.json")
        assert out.read_text().endswith("\n")
