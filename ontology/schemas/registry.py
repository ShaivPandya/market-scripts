from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ontology.models import EntityType, OntologyEdge, OntologyNode
from ontology.schemas.identity import (
    asset_id,
    catalyst_id,
    evaluation_id,
    macro_indicator_id,
    position_id,
    sector_id,
    signal_id,
    thesis_id,
)
from ontology.schemas.legacy import adapt_edge_payload, adapt_node_payload
from ontology.schemas.objects import (
    AssetV1,
    CatalystV1,
    EvaluationV1,
    MacroIndicatorV1,
    OntologyObjectV1,
    PositionV1,
    SectorV1,
    SignalV1,
    ThesisV1,
)
from ontology.schemas.relations import (
    ALLOWED_RELATIONS,
    OPTIONAL_RELATIONS,
    dump_edge_properties,
    edge_schema_for_relation,
    edge_schema_name,
)

NODE_SCHEMAS: dict[str, type] = {
    "Position": PositionV1,
    "Asset": AssetV1,
    "Sector": SectorV1,
    "MacroIndicator": MacroIndicatorV1,
    "Signal": SignalV1,
    "Thesis": ThesisV1,
    "Evaluation": EvaluationV1,
    "Catalyst": CatalystV1,
}
OPTIONAL_NODE_TYPES = {"Thesis", "Evaluation", "Catalyst"}


class OntologySchemaValidationError(ValueError):
    pass


@dataclass(slots=True)
class NormalizedGraph:
    nodes: list[OntologyNode]
    edges: list[OntologyEdge]
    node_id_map: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def normalize_node(
    node: OntologyNode,
    *,
    run_id: str | None = None,
    allow_legacy: bool = True,
) -> OntologyNode:
    try:
        schema_cls = NODE_SCHEMAS[node.type]
    except KeyError as exc:
        raise OntologySchemaValidationError(f"Unsupported node type: {node.type}") from exc

    legacy_payload = _is_legacy_payload(node.properties, node.schema_version)
    node_id = node.id
    label = node.label
    payload = dict(node.properties or {})
    if legacy_payload:
        if not allow_legacy:
            raise OntologySchemaValidationError(f"Legacy ontology node is not allowed: {node.id}")
        try:
            node_id, label, payload = adapt_node_payload(
                node_id=node.id,
                node_type=node.type,
                label=node.label,
                properties=payload,
                run_id=run_id,
            )
        except Exception as exc:
            raise OntologySchemaValidationError(f"Invalid legacy node {node.id}: {exc}") from exc

    try:
        model = schema_cls.model_validate(payload)
    except ValidationError as exc:
        raise OntologySchemaValidationError(f"Invalid {node.type} node {node.id}: {exc}") from exc

    expected_id = expected_node_id(node.type, model)
    if node_id != expected_id:
        if legacy_payload:
            node_id = expected_id
        else:
            raise OntologySchemaValidationError(f"Node {node.id} has non-canonical identity; expected {expected_id}")

    return OntologyNode(
        id=node_id,
        type=node.type,
        label=_label_for(node.type, label, model),
        properties=model.model_dump(mode="json"),
        schema_name=node.type,
        schema_version=1,
    )


def normalize_edge(
    edge: OntologyEdge,
    *,
    run_id: str | None = None,
    allow_legacy: bool = True,
    source_id: str | None = None,
    target_id: str | None = None,
) -> OntologyEdge:
    payload = dict(edge.properties or {})
    legacy_payload = _is_legacy_payload(payload, edge.schema_version)
    if legacy_payload:
        if not allow_legacy:
            raise OntologySchemaValidationError(
                f"Legacy ontology edge is not allowed: {edge.source_id}->{edge.target_id}:{edge.relation_type}"
            )
        try:
            payload = adapt_edge_payload(relation_type=edge.relation_type, properties=payload, run_id=run_id)
        except Exception as exc:
            raise OntologySchemaValidationError(f"Invalid legacy edge {edge.relation_type}: {exc}") from exc

    schema_cls = edge_schema_for_relation(edge.relation_type)
    try:
        model = schema_cls.model_validate(payload)
    except ValidationError as exc:
        raise OntologySchemaValidationError(
            f"Invalid {edge.relation_type} edge {edge.source_id}->{edge.target_id}: {exc}"
        ) from exc

    return OntologyEdge(
        source_id=source_id or edge.source_id,
        target_id=target_id or edge.target_id,
        relation_type=edge.relation_type,
        properties=dump_edge_properties(model),
        schema_name=edge_schema_name(edge.relation_type),
        schema_version=1,
    )


def normalize_graph(
    nodes: list[OntologyNode],
    edges: list[OntologyEdge],
    *,
    run_id: str | None = None,
    allow_legacy: bool = True,
    skip_optional_invalid: bool = False,
) -> NormalizedGraph:
    normalized_nodes: dict[str, OntologyNode] = {}
    id_map: dict[str, str] = {}
    skipped_old_ids: set[str] = set()
    warnings: list[str] = []

    for node in nodes:
        try:
            normalized = normalize_node(node, run_id=run_id, allow_legacy=allow_legacy)
        except OntologySchemaValidationError as exc:
            if skip_optional_invalid and node.type in OPTIONAL_NODE_TYPES:
                skipped_old_ids.add(node.id)
                warnings.append(str(exc))
                continue
            raise

        if normalized.id in normalized_nodes and normalized_nodes[normalized.id] != normalized:
            raise OntologySchemaValidationError(f"Duplicate canonical node id after normalization: {normalized.id}")
        normalized_nodes[normalized.id] = normalized
        id_map[node.id] = normalized.id

    normalized_edges: dict[tuple[str, str, str], OntologyEdge] = {}
    node_types = {node_id: node.type for node_id, node in normalized_nodes.items()}

    for edge in edges:
        if edge.source_id in skipped_old_ids or edge.target_id in skipped_old_ids:
            continue
        source_id = id_map.get(edge.source_id, edge.source_id)
        target_id = id_map.get(edge.target_id, edge.target_id)
        try:
            _validate_relation(edge.relation_type, source_id, target_id, node_types)
            normalized = normalize_edge(
                edge,
                run_id=run_id,
                allow_legacy=allow_legacy,
                source_id=source_id,
                target_id=target_id,
            )
        except OntologySchemaValidationError as exc:
            if skip_optional_invalid and edge.relation_type in OPTIONAL_RELATIONS:
                warnings.append(str(exc))
                continue
            raise

        normalized_edges[(normalized.source_id, normalized.target_id, normalized.relation_type)] = normalized

    return NormalizedGraph(
        nodes=list(normalized_nodes.values()),
        edges=list(normalized_edges.values()),
        node_id_map=id_map,
        warnings=warnings,
    )


def expected_node_id(node_type: str, model: OntologyObjectV1) -> str:
    if isinstance(model, PositionV1):
        return position_id(model.ticker)
    if isinstance(model, AssetV1):
        return asset_id(model.ticker)
    if isinstance(model, SectorV1):
        return sector_id(model.name)
    if isinstance(model, MacroIndicatorV1):
        return macro_indicator_id(model.indicator_key)
    if isinstance(model, SignalV1):
        return signal_id(model.source, model.name)
    if isinstance(model, ThesisV1):
        return thesis_id(model.ticker)
    if isinstance(model, EvaluationV1):
        return evaluation_id(model.ticker, model.evaluated_at)
    if isinstance(model, CatalystV1):
        return catalyst_id(model.ticker, model.name, model.description)
    raise OntologySchemaValidationError(f"Unsupported node schema for type {node_type}")


def node_from_schema(
    *,
    node_id: str,
    node_type: EntityType,
    label: str,
    model: OntologyObjectV1,
) -> OntologyNode:
    return normalize_node(
        OntologyNode(
            id=node_id,
            type=node_type,
            label=label,
            properties=model.model_dump(mode="json"),
            schema_name=node_type,
            schema_version=1,
        ),
        allow_legacy=False,
    )


def _validate_relation(
    relation_type: str,
    source_id: str,
    target_id: str,
    node_types: dict[str, str],
) -> None:
    expected = ALLOWED_RELATIONS.get(relation_type)
    if expected is None:
        raise OntologySchemaValidationError(f"Unsupported relation type: {relation_type}")
    source_type = node_types.get(source_id)
    target_type = node_types.get(target_id)
    if source_type is None:
        raise OntologySchemaValidationError(f"Edge {relation_type} has missing source node: {source_id}")
    if target_type is None:
        raise OntologySchemaValidationError(f"Edge {relation_type} has missing target node: {target_id}")
    if (source_type, target_type) != expected:
        raise OntologySchemaValidationError(
            f"Edge {relation_type} must connect {expected[0]}->{expected[1]}, got {source_type}->{target_type}"
        )


def _is_legacy_payload(properties: dict[str, Any], schema_version: int) -> bool:
    return schema_version != 1 and int(properties.get("schema_version") or 0) != 1


def _label_for(node_type: str, label: str, model: OntologyObjectV1) -> str:
    if isinstance(model, (PositionV1, AssetV1)):
        return model.ticker
    if isinstance(model, SectorV1):
        return model.name
    if isinstance(model, MacroIndicatorV1):
        return model.name
    if isinstance(model, SignalV1):
        return model.name
    if isinstance(model, ThesisV1):
        return f"Thesis: {model.ticker}"
    if isinstance(model, EvaluationV1):
        return f"Eval: {model.ticker}"
    if isinstance(model, CatalystV1):
        return model.name
    return label
