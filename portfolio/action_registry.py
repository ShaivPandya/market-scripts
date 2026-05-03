"""Compatibility shim for the canonical ontology action registry."""

from ontology import action_registry as _canonical_registry
from ontology.action_registry import *  # noqa: F403

_ACTIONS = _canonical_registry._ACTIONS
