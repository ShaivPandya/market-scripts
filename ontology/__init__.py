"""Ontology package for cross-module semantic graph and query execution."""

from __future__ import annotations

from typing import Any

__all__ = ["OntologyObjectService", "OntologyQueryService"]


def __getattr__(name: str) -> Any:
    if name == "OntologyObjectService":
        from ontology.object_service import OntologyObjectService

        return OntologyObjectService
    if name == "OntologyQueryService":
        from ontology.service import OntologyQueryService

        return OntologyQueryService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
