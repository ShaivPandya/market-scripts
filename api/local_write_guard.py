"""Production guardrails for project-local filesystem writes.

Production state must live in Cloud SQL and Cloud Storage.  This module
installs a small runtime guard that raises if code tries to mutate files under
the repository root while ``ENVIRONMENT=production``.
"""

from __future__ import annotations

import builtins
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PathInput = str | bytes | os.PathLike[Any]


class ProductionLocalWriteError(RuntimeError):
    """Raised when production code attempts to write inside PROJECT_ROOT."""


_lock = threading.Lock()
_installed = False
_guarded_roots: list[Path] = []
_allow_depth = 0

_original_open = builtins.open
_original_path_open = Path.open
_original_write_text = Path.write_text
_original_write_bytes = Path.write_bytes
_original_mkdir = Path.mkdir
_original_touch = Path.touch
_original_unlink = Path.unlink
_original_rename = Path.rename
_original_replace = Path.replace
_original_sqlite_connect = sqlite3.connect


def _is_production() -> bool:
    return os.getenv("ENVIRONMENT", "development").strip().lower() == "production"


def _resolve(path: PathInput) -> Path:
    return Path(os.fsdecode(path)).expanduser().resolve(strict=False)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def assert_project_write_allowed(path: PathInput, *, operation: str = "write") -> None:
    """Raise when *path* points under a guarded project root in production."""
    if _allow_depth > 0 or not _is_production():
        return
    resolved = _resolve(path)
    for root in _guarded_roots:
        if _is_under(resolved, root):
            raise ProductionLocalWriteError(
                f"Refusing to {operation} project-local path in production: {resolved}. "
                "Use the Cloud Storage or Postgres state adapter instead."
            )


def _is_write_mode(mode: str) -> bool:
    return any(flag in mode for flag in ("w", "a", "x", "+"))


def _guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any):
    if _is_write_mode(mode) and isinstance(file, str | bytes | os.PathLike):
        assert_project_write_allowed(file, operation=f"open({mode})")
    return _original_open(file, mode, *args, **kwargs)


def _guarded_path_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any):
    if _is_write_mode(mode):
        assert_project_write_allowed(self, operation=f"open({mode})")
    return _original_path_open(self, mode, *args, **kwargs)


def _guarded_write_text(self: Path, *args: Any, **kwargs: Any):
    assert_project_write_allowed(self, operation="write_text")
    return _original_write_text(self, *args, **kwargs)


def _guarded_write_bytes(self: Path, *args: Any, **kwargs: Any):
    assert_project_write_allowed(self, operation="write_bytes")
    return _original_write_bytes(self, *args, **kwargs)


def _guarded_mkdir(self: Path, *args: Any, **kwargs: Any):
    assert_project_write_allowed(self, operation="mkdir")
    return _original_mkdir(self, *args, **kwargs)


def _guarded_touch(self: Path, *args: Any, **kwargs: Any):
    assert_project_write_allowed(self, operation="touch")
    return _original_touch(self, *args, **kwargs)


def _guarded_unlink(self: Path, *args: Any, **kwargs: Any):
    assert_project_write_allowed(self, operation="unlink")
    return _original_unlink(self, *args, **kwargs)


def _guarded_rename(self: Path, target: str | os.PathLike[str], *args: Any, **kwargs: Any):
    assert_project_write_allowed(self, operation="rename source")
    assert_project_write_allowed(target, operation="rename target")
    return _original_rename(self, target, *args, **kwargs)


def _guarded_replace(self: Path, target: str | os.PathLike[str], *args: Any, **kwargs: Any):
    assert_project_write_allowed(self, operation="replace source")
    assert_project_write_allowed(target, operation="replace target")
    return _original_replace(self, target, *args, **kwargs)


def _sqlite_path(database: Any) -> PathInput | None:
    if database in (None, ":memory:"):
        return None
    if isinstance(database, str) and database.startswith("file:"):
        return database[5:].split("?", 1)[0]
    if isinstance(database, str | bytes | os.PathLike):
        return database
    return None


def _guarded_sqlite_connect(database: Any, *args: Any, **kwargs: Any):
    path = _sqlite_path(database)
    if path is not None:
        assert_project_write_allowed(path, operation="sqlite3.connect")
    return _original_sqlite_connect(database, *args, **kwargs)


def install_production_write_guard(project_root: os.PathLike[str] | str) -> None:
    """Install the guard and add *project_root* to the guarded roots."""
    global _installed
    root = _resolve(project_root)
    with _lock:
        if root not in _guarded_roots:
            _guarded_roots.append(root)
        if _installed:
            return
        builtins.open = _guarded_open
        Path.open = _guarded_path_open
        Path.write_text = _guarded_write_text
        Path.write_bytes = _guarded_write_bytes
        Path.mkdir = _guarded_mkdir
        Path.touch = _guarded_touch
        Path.unlink = _guarded_unlink
        Path.rename = _guarded_rename
        Path.replace = _guarded_replace
        sqlite3.connect = _guarded_sqlite_connect
        _installed = True


@contextmanager
def allow_project_writes():
    """Temporarily bypass the guard for tightly scoped maintenance code."""
    global _allow_depth
    _allow_depth += 1
    try:
        yield
    finally:
        _allow_depth -= 1
