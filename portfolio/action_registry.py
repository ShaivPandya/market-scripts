"""Compatibility shim for the canonical ontology action registry."""

from ontology import action_registry as _canonical_registry

for _name, _value in _canonical_registry.__dict__.items():
    if _name in {"__name__", "__file__", "__package__", "__loader__", "__spec__", "__builtins__"}:
        continue
    globals()[_name] = _value
