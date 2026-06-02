"""Guards for best-effort ontology-backed operational writes."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from api.postgres import database_url, use_postgres_state

_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}
_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}


def _env_override() -> bool | None:
    raw = (os.getenv("OPTIONAL_ONTOLOGY_WRITES_ENABLED") or "").strip().lower()
    if not raw:
        return None
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    return None


def _cloud_sql_socket_available(url: str) -> bool:
    parsed = urlparse(url)
    host = (parse_qs(parsed.query).get("host") or [""])[0]
    if not host.startswith("/cloudsql/"):
        return True
    return Path(host).exists()


@lru_cache(maxsize=1)
def optional_ontology_writes_available() -> bool:
    """Return whether best-effort audit/provenance writes should be attempted."""

    override = _env_override()
    if override is not None:
        return override
    try:
        if not use_postgres_state():
            return False
    except RuntimeError:
        return False
    url = database_url()
    if not url:
        return False
    return _cloud_sql_socket_available(url)


def should_attempt_optional_ontology_write(*, fail_closed: bool) -> bool:
    """Fail-closed callers must still attempt the write and surface failures."""

    return fail_closed or optional_ontology_writes_available()
