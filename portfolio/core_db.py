"""Removed domain-table state module."""

from __future__ import annotations

from typing import Any


class RemovedDomainStateError(RuntimeError):
    pass


def _removed(*_args: Any, **_kwargs: Any) -> Any:
    raise RemovedDomainStateError(
        "Domain-table state has been removed. Use OntologyCommandService for writes "
        "and OntologyRuntimeReadService for reads."
    )


def __getattr__(_name: str) -> Any:
    return _removed
