"""
Generate a structured release manifest for deployable builds.

The manifest captures git identity, container image metadata, database
migration state, runtime configuration, and rollback references so that
every deploy is fully traceable and reversible.

Usage from bash (deploy-backend.sh calls this):
    python -m infra.gcp.release_manifest \
        --image-uri "us-central1-docker.pkg.dev/…/api:abc123" \
        [--image-digest "sha256:…"] \
        [--migration-head "abc123def456"] \
        [--environment production] \
        [--config-profile default] \
        [--prior-manifest path/to/previous-manifest.json] \
        [--output path/to/release-manifest.json]

Usage from Python (for testing):
    from infra.gcp.release_manifest import build_manifest
    manifest = build_manifest(image_uri="…", ...)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _run_git(*args: str, cwd: str | Path | None = None) -> str | None:
    """Run a git command and return stripped stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def resolve_git_metadata(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Return a dict with git SHA, short SHA, branch, and dirty flag."""
    sha = _run_git("rev-parse", "HEAD", cwd=repo_root) or "unknown"
    sha_short = _run_git("rev-parse", "--short", "HEAD", cwd=repo_root) or "unknown"
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root) or "unknown"

    # Check dirty state
    dirty = False
    diff_index = _run_git("diff", "--quiet", cwd=repo_root)
    diff_cached = _run_git("diff", "--cached", "--quiet", cwd=repo_root)
    if diff_index is None or diff_cached is None:
        # git diff --quiet exits non-zero when there are changes
        dirty = True

    return {
        "sha": sha,
        "sha_short": sha_short,
        "branch": branch,
        "dirty": dirty,
    }


# ---------------------------------------------------------------------------
# Rollback helpers
# ---------------------------------------------------------------------------


def _load_prior_manifest(path: str | Path | None) -> dict[str, Any] | None:
    """Load a previously written manifest for rollback reference."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if isinstance(data, dict) and "schema_version" in data:
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _extract_rollback_refs(prior: dict[str, Any] | None) -> dict[str, Any]:
    """Extract rollback target refs from a prior manifest."""
    if prior is None:
        return {}
    refs: dict[str, Any] = {}
    git_info = prior.get("git")
    if isinstance(git_info, dict):
        if git_info.get("sha"):
            refs["prior_git_sha"] = git_info["sha"]
        if git_info.get("sha_short"):
            refs["prior_git_sha_short"] = git_info["sha_short"]
    image_info = prior.get("image")
    if isinstance(image_info, dict):
        if image_info.get("uri"):
            refs["prior_image_uri"] = image_info["uri"]
        if image_info.get("tag"):
            refs["prior_image_tag"] = image_info["tag"]
        if image_info.get("digest"):
            refs["prior_image_digest"] = image_info["digest"]
    return refs


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def _parse_image_tag(image_uri: str) -> str:
    """Extract the tag portion from an image URI like …/api:abc123."""
    if ":" in image_uri:
        return image_uri.rsplit(":", 1)[-1]
    return "latest"


def build_manifest(
    *,
    image_uri: str,
    image_digest: str | None = None,
    migration_head: str | None = None,
    environment: str = "production",
    config_profile: str = "default",
    repo_root: str | Path | None = None,
    prior_manifest_path: str | Path | None = None,
    git_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a release manifest dict.

    Parameters
    ----------
    image_uri:
        Full container image URI (e.g. us-central1-docker.pkg.dev/…/api:sha).
    image_digest:
        Image digest (sha256:…). Optional; resolved at deploy time.
    migration_head:
        Current Alembic migration head revision. Optional.
    environment:
        Target environment (production, staging, development).
    config_profile:
        Configuration profile name.
    repo_root:
        Path to repo root for git metadata resolution. Defaults to cwd.
    prior_manifest_path:
        Path to the previous manifest for rollback reference extraction.
    git_metadata:
        Pre-resolved git metadata dict (for testing). If None, resolved from repo.
    """
    if git_metadata is None:
        git_metadata = resolve_git_metadata(repo_root)
    else:
        git_metadata = dict(git_metadata)  # don't mutate caller's dict

    prior = _load_prior_manifest(prior_manifest_path)
    rollback = _extract_rollback_refs(prior)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "git": git_metadata,
        "image": {
            "uri": image_uri,
            "tag": _parse_image_tag(image_uri),
        },
        "database": {
            "migration_head": migration_head,
        },
        "runtime": {
            "environment": environment,
            "config_profile": config_profile,
        },
        "rollback": rollback,
    }

    if image_digest:
        manifest["image"]["digest"] = image_digest

    return manifest


def write_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    """Write the manifest to a JSON file and return the path."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2) + "\n")
    return p


# ---------------------------------------------------------------------------
# Required-field validation
# ---------------------------------------------------------------------------

REQUIRED_KEYS = ("schema_version", "generated_at", "git", "image", "database", "runtime", "rollback")
REQUIRED_GIT_KEYS = ("sha", "sha_short", "branch", "dirty")
REQUIRED_IMAGE_KEYS = ("uri", "tag")


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return a list of validation errors (empty = valid)."""
    errors: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in manifest:
            errors.append(f"missing top-level key: {key}")

    git = manifest.get("git")
    if isinstance(git, dict):
        for key in REQUIRED_GIT_KEYS:
            if key not in git:
                errors.append(f"missing git.{key}")
    elif git is not None:
        errors.append("git must be a dict")

    image = manifest.get("image")
    if isinstance(image, dict):
        for key in REQUIRED_IMAGE_KEYS:
            if key not in image:
                errors.append(f"missing image.{key}")
    elif image is not None:
        errors.append("image must be a dict")

    return errors


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a release manifest for a deployable build.",
    )
    parser.add_argument("--image-uri", required=True, help="Full container image URI")
    parser.add_argument("--image-digest", default=None, help="Image digest (sha256:…)")
    parser.add_argument("--migration-head", default=None, help="Alembic migration head revision")
    parser.add_argument("--environment", default="production", help="Target environment")
    parser.add_argument("--config-profile", default="default", help="Config profile name")
    parser.add_argument("--prior-manifest", default=None, help="Path to prior manifest for rollback refs")
    parser.add_argument(
        "--output",
        default="infra/gcp/release-manifest.json",
        help="Output path for the manifest JSON",
    )
    parser.add_argument("--repo-root", default=None, help="Path to repo root (default: auto-detect)")
    args = parser.parse_args(argv)

    manifest = build_manifest(
        image_uri=args.image_uri,
        image_digest=args.image_digest,
        migration_head=args.migration_head,
        environment=args.environment,
        config_profile=args.config_profile,
        repo_root=args.repo_root,
        prior_manifest_path=args.prior_manifest,
    )

    errors = validate_manifest(manifest)
    if errors:
        print(f"Manifest validation errors: {errors}", file=sys.stderr)
        sys.exit(1)

    out = write_manifest(manifest, args.output)
    print(f"Release manifest written to {out}")


if __name__ == "__main__":
    main()
