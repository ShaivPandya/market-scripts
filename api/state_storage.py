"""State object storage adapter.

Local development keeps using repository files.  Production defaults to Cloud
Storage and refuses to fall back to project-local writes.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from api.local_write_guard import assert_project_write_allowed


class StateStorageConfigError(RuntimeError):
    """Raised when a requested storage backend is not configured."""


def use_gcs_state() -> bool:
    backend = os.getenv("STATE_STORAGE_BACKEND", "").strip().lower()
    if backend:
        return backend == "gcs"
    return os.getenv("ENVIRONMENT", "development").strip().lower() == "production"


def _bucket_name() -> str:
    bucket = os.getenv("GCS_STATE_BUCKET", "").strip()
    if not bucket:
        raise StateStorageConfigError("GCS_STATE_BUCKET is required when state storage uses Cloud Storage.")
    return bucket


def _bucket():
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise StateStorageConfigError("google-cloud-storage is required for GCS state storage.") from exc
    return storage.Client().bucket(_bucket_name())


def _gs_uri(key: str) -> str:
    return f"gs://{_bucket_name()}/{key.lstrip('/')}"


def exists_text(local_path: Path, gcs_key: str) -> bool:
    if use_gcs_state():
        return bool(_bucket().blob(gcs_key).exists())
    return local_path.exists()


def read_text(local_path: Path, gcs_key: str, *, encoding: str = "utf-8") -> str:
    if use_gcs_state():
        return cast(str, _bucket().blob(gcs_key).download_as_text(encoding=encoding))
    return local_path.read_text(encoding=encoding)


def read_bytes(local_path: Path, gcs_key: str) -> bytes:
    if use_gcs_state():
        return cast(bytes, _bucket().blob(gcs_key).download_as_bytes())
    return local_path.read_bytes()


def object_updated(local_path: Path, gcs_key: str) -> datetime | None:
    """Return the last-modified time of the underlying object, or None if it doesn't exist."""
    if use_gcs_state():
        blob = _bucket().blob(gcs_key)
        if not blob.exists():
            return None
        blob.reload()
        return cast(datetime | None, blob.updated)
    if not local_path.exists():
        return None
    return datetime.fromtimestamp(local_path.stat().st_mtime, tz=UTC)


def write_text(
    local_path: Path,
    gcs_key: str,
    content: str,
    *,
    encoding: str = "utf-8",
    content_type: str = "text/plain; charset=utf-8",
    metadata: dict[str, str] | None = None,
) -> str:
    if use_gcs_state():
        blob = _bucket().blob(gcs_key)
        if metadata:
            blob.metadata = metadata
        blob.upload_from_string(content, content_type=content_type)
        return _gs_uri(gcs_key)

    assert_project_write_allowed(local_path.parent, operation="mkdir")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    assert_project_write_allowed(local_path, operation="write_text")
    local_path.write_text(content, encoding=encoding)
    return str(local_path)


def write_bytes(
    local_path: Path,
    gcs_key: str,
    content: bytes,
    *,
    content_type: str = "application/octet-stream",
    metadata: dict[str, str] | None = None,
) -> str:
    if use_gcs_state():
        blob = _bucket().blob(gcs_key)
        if metadata:
            blob.metadata = metadata
        blob.upload_from_string(content, content_type=content_type)
        return _gs_uri(gcs_key)

    assert_project_write_allowed(local_path.parent, operation="mkdir")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    assert_project_write_allowed(local_path, operation="write_bytes")
    local_path.write_bytes(content)
    return str(local_path)


def upload_file(
    local_path: Path,
    gcs_key: str,
    *,
    content_type: str | None = None,
    metadata: dict[str, str] | None = None,
) -> str:
    if not use_gcs_state():
        return str(local_path)
    blob = _bucket().blob(gcs_key)
    if metadata:
        blob.metadata = metadata
    blob.upload_from_filename(str(local_path), content_type=content_type)
    return _gs_uri(gcs_key)


def delete_file(local_path: Path, gcs_key: str) -> bool:
    """Delete an object from the configured state backend.

    Returns True when an object/file existed and was removed.
    """
    if use_gcs_state():
        blob = _bucket().blob(gcs_key)
        if not blob.exists():
            return False
        blob.delete()
        return True

    if not local_path.exists():
        return False
    assert_project_write_allowed(local_path, operation="unlink")
    local_path.unlink()
    return True


def object_metadata(gcs_key: str) -> dict[str, Any] | None:
    if not use_gcs_state():
        return None
    blob = _bucket().blob(gcs_key)
    if not blob.exists():
        return None
    blob.reload()
    return dict(blob.metadata or {})
