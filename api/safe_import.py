"""
Utility for gracefully handling router import failures.

When an optional analysis module fails to import (missing env var, unavailable
dependency), this provides a stub router that returns a standard degraded
response instead of crashing the entire application.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger("api.safe_import")


def safe_import_router(
    module_path: str,
    router_attr: str = "router",
) -> tuple[APIRouter, bool]:
    """Import a router module, returning a stub on failure.

    Returns (router, is_healthy) so callers can track which modules loaded.
    """
    try:
        mod = importlib.import_module(module_path)
        router = getattr(mod, router_attr)
        return router, True
    except Exception as exc:
        logger.warning("Failed to import %s: %s — registering degraded stub", module_path, exc)
        return _make_stub_router(module_path, str(exc)), False


_degraded_modules: dict[str, str] = {}


def get_degraded_modules() -> dict[str, str]:
    """Return a mapping of module_path -> error for all degraded modules."""
    return dict(_degraded_modules)


def _make_stub_router(module_path: str, error: str) -> APIRouter:
    """Create a router whose endpoints all return a degraded status."""
    _degraded_modules[module_path] = error
    stub = APIRouter()
    return stub
