"""Read-side provenance graph assembly for trace and lineage APIs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from api import provenance
from api.postgres import connect, use_postgres_state
from ontology.runtime_read_service import OntologyRuntimeReadService
from ontology.schemas.relations import PROVENANCE_RELATION_TYPES

WARNING_EMPTY_TRACE = "empty_trace"
WARNING_SEED_NOT_FOUND = "seed_not_found"
WARNING_NODE_LIMIT_REACHED = "node_limit_reached"
WARNING_EDGE_LIMIT_REACHED = "edge_limit_reached"
WARNING_REDACTED_METADATA = "redacted_metadata"

VALID_WARNING_CODES = {
    WARNING_EMPTY_TRACE,
    WARNING_SEED_NOT_FOUND,
    WARNING_NODE_LIMIT_REACHED,
    WARNING_EDGE_LIMIT_REACHED,
    WARNING_REDACTED_METADATA,
}

DIRECTIONS = {"both", "upstream", "downstream"}
PROVENANCE_GRAPH_VIEW = "ontology_current_provenance_graph_edge_read_model"

_SELECTOR_REF_TYPES = {
    "recommendation_id": "recommendation",
    "workflow_run_id": provenance.REF_WORKFLOW_RUN,
    "ontology_run_id": provenance.REF_ONTOLOGY_RUN,
    "object_version_id": provenance.REF_ONTOLOGY_OBJECT_VERSION,
    "relation_version_id": provenance.REF_RELATION_VERSION,
    "source_record_id": provenance.REF_SOURCE_RECORD,
    "snapshot_id": provenance.REF_COMPUTED_SNAPSHOT_VERSION,
    "approval_id": provenance.REF_APPROVAL,
    "action_run_id": provenance.REF_ACTION_RUN,
    "agent_session_id": provenance.REF_AGENT_SESSION,
}

_RELATION_LINK_TYPES: dict[str, str] = {str(value): str(key) for key, value in provenance.LINK_RELATION_TYPES.items()}


@dataclass(frozen=True, slots=True)
class _Ref:
    ref_type: str
    ref_id: str
    ref_version: str | None = None
    object_uid: str | None = None

    @property
    def node_id(self) -> str:
        return _ref_node_id(self.ref_type, self.ref_id, self.ref_version)


class ProvenanceGraphService:
    """Build exact provenance traces from ontology provenance relation edges."""

    def __init__(
        self,
        *,
        reads: OntologyRuntimeReadService | None = None,
        connection_factory: Any = connect,
    ):
        self.reads = reads or OntologyRuntimeReadService()
        self.connection_factory = connection_factory

    def trace(
        self,
        *,
        selector: Mapping[str, Any],
        direction: str = "both",
        max_depth: int = 3,
        max_nodes: int = 250,
        max_edges: int = 500,
    ) -> dict[str, Any]:
        clean_selector = _clean_selector(selector)
        safe_direction = _safe_direction(direction)
        safe_depth = max(1, min(int(max_depth), 8))
        safe_max_nodes = max(1, min(int(max_nodes), 1000))
        safe_max_edges = max(1, min(int(max_edges), 2500))
        seed = _seed_from_selector(clean_selector)
        warnings: list[dict[str, Any]] = []
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}
        truncated = False

        frontier_refs: set[_Ref] = set()
        frontier_events: set[str] = set()
        if seed.get("seed_type") == "event":
            event_id = str(seed.get("event_id") or "")
            if event_id:
                frontier_events.add(event_id)
                event = self._load_event(event_id)
                if event:
                    nodes[_event_node_id(event_id)] = _event_node(event_id, event)
        elif seed.get("ref_type") and seed.get("ref_id"):
            frontier_refs.add(_make_ref(str(seed["ref_type"]), str(seed["ref_id"]), None))

        visited_refs: set[str] = set()
        visited_events: set[str] = set()
        use_postgres_state()
        edge_loader = self._query_read_model_edges

        for depth in range(safe_depth):
            if not frontier_refs and not frontier_events:
                break
            next_refs: set[_Ref] = set()
            next_events: set[str] = set()
            rows = edge_loader(
                refs=frontier_refs,
                event_ids=frontier_events,
                direction=safe_direction,
                limit=safe_max_edges + 1,
            )
            for row in rows:
                edge = _edge_from_row(row, depth=depth)
                if edge is None:
                    continue
                if len(edges) >= safe_max_edges and str(edge["id"]) not in edges:
                    truncated = True
                    warnings.append(_warning(WARNING_EDGE_LIMIT_REACHED))
                    break
                source_ref = _ref_from_edge_row(row, "source")
                target_ref = _ref_from_edge_row(row, "target")
                new_node_ids = {
                    source_ref.node_id,
                    target_ref.node_id,
                    *({_event_node_id(str(edge.get("event_id")))} if edge.get("event_id") else set()),
                }
                if len(nodes) + sum(1 for node_id in new_node_ids if node_id not in nodes) > safe_max_nodes:
                    truncated = True
                    warnings.append(_warning(WARNING_NODE_LIMIT_REACHED))
                    break
                nodes[source_ref.node_id] = _node_from_ref(source_ref)
                nodes[target_ref.node_id] = _node_from_ref(target_ref)
                event_id = str(edge.get("event_id") or "")
                if event_id:
                    event_node_id = _event_node_id(event_id)
                    nodes.setdefault(event_node_id, _event_node(event_id, self._load_event(event_id) or {}))
                edges[str(edge["id"])] = edge
                if safe_direction in {"both", "upstream"}:
                    next_refs.add(source_ref)
                if safe_direction in {"both", "downstream"}:
                    next_refs.add(target_ref)
            if truncated:
                break
            visited_refs.update(ref.node_id for ref in frontier_refs)
            visited_events.update(frontier_events)
            frontier_refs = {ref for ref in next_refs if ref.node_id not in visited_refs}
            frontier_events = {event_id for event_id in next_events if event_id not in visited_events}

        self._add_parent_event_edges(nodes, edges)

        if not nodes and not edges:
            warnings.append(_warning(WARNING_SEED_NOT_FOUND if seed else WARNING_EMPTY_TRACE))
        elif not edges:
            warnings.append(_warning(WARNING_EMPTY_TRACE))

        if any(edge.get("redaction_policy") for edge in edges.values()):
            warnings.append(_warning(WARNING_REDACTED_METADATA))

        return _graph_response(
            selector=clean_selector,
            seed=seed,
            direction=safe_direction,
            max_depth=safe_depth,
            nodes=nodes,
            edges=edges,
            warnings=warnings,
            truncated=truncated,
            lineage_state="ontology",
        )

    def _query_read_model_edges(
        self,
        *,
        refs: set[_Ref],
        event_ids: set[str],
        direction: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for ref in refs:
            ref_ids = _ref_id_candidates(ref.ref_type, ref.ref_id)
            if direction in {"both", "downstream"}:
                clauses.append("(source_object_uid = %s OR (source_ref_type = %s AND source_ref_id = ANY(%s)))")
                params.extend([ref.object_uid or _ref_object_uid(ref.ref_type, ref.ref_id), ref.ref_type, ref_ids])
            if direction in {"both", "upstream"}:
                clauses.append("(target_object_uid = %s OR (target_ref_type = %s AND target_ref_id = ANY(%s)))")
                params.extend([ref.object_uid or _ref_object_uid(ref.ref_type, ref.ref_id), ref.ref_type, ref_ids])
        for event_id in event_ids:
            clauses.append("event_id = %s")
            params.append(event_id)
        if not clauses:
            return []
        params.append(max(1, int(limit)))
        sql = f"""
        SELECT *
        FROM {PROVENANCE_GRAPH_VIEW}
        WHERE {" OR ".join(clauses)}
        ORDER BY valid_from ASC, relation_uid ASC
        LIMIT %s
        """
        with self.connection_factory() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def _query_direct_edges(
        self,
        *,
        refs: set[_Ref],
        event_ids: set[str],
        direction: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        matches: dict[str, dict[str, Any]] = {}
        frontier_node_ids = {ref.node_id for ref in refs}
        frontier_object_uids = {ref.object_uid or _ref_object_uid(ref.ref_type, ref.ref_id) for ref in refs}
        for relation_type in sorted(PROVENANCE_RELATION_TYPES):
            offset = 0
            while len(matches) <= limit:
                rows = self.reads.objects.query_relations(
                    relation_type=relation_type,
                    include_history=True,
                    limit=500,
                    offset=offset,
                )
                if not rows:
                    break
                for row in rows:
                    normalized = _normalize_edge_row(row)
                    source_ref = _ref_from_edge_row(normalized, "source")
                    target_ref = _ref_from_edge_row(normalized, "target")
                    event_id = str(_edge_properties(normalized).get("event_id") or "")
                    source_match = (
                        source_ref.node_id in frontier_node_ids or source_ref.object_uid in frontier_object_uids
                    )
                    target_match = (
                        target_ref.node_id in frontier_node_ids or target_ref.object_uid in frontier_object_uids
                    )
                    event_match = event_id in event_ids
                    directional_match = event_match or (
                        (direction == "both" and (source_match or target_match))
                        or (direction == "downstream" and source_match)
                        or (direction == "upstream" and target_match)
                    )
                    if directional_match:
                        matches[str(normalized.get("relation_uid") or len(matches))] = normalized
                        if len(matches) > limit:
                            break
                offset += 500
                if len(rows) < 500:
                    break
        return list(matches.values())

    def _load_event(self, event_id: str) -> dict[str, Any] | None:
        try:
            rows = self.reads.list_objects("ProvenanceEvent", filters={"event_id": event_id}, limit=1)
        except TypeError:
            rows = self.reads.list_objects("ProvenanceEvent", limit=500)
            rows = [row for row in rows if str(row.get("event_id") or row.get("id") or "") == event_id]
        except Exception:
            return None
        return dict(rows[0]) if rows else None

    def _add_parent_event_edges(self, nodes: dict[str, dict[str, Any]], edges: dict[str, dict[str, Any]]) -> None:
        event_nodes = [node for node in nodes.values() if node.get("node_type") == "event"]
        existing_keys = {
            (
                edge.get("source_node_id"),
                edge.get("target_node_id"),
                edge.get("edge_type"),
                edge.get("event_id"),
            )
            for edge in edges.values()
        }
        existing_parent_pairs = {
            (edge.get("source_node_id"), edge.get("target_node_id"))
            for edge in edges.values()
            if edge.get("edge_type") in {"event_parent", "triggered", "resolved_by"}
            or edge.get("relation_type") in {"provenance_triggered", "provenance_resolved_by"}
        }
        for node in event_nodes:
            raw_payload = node.get("payload")
            payload: Mapping[str, Any] = raw_payload if isinstance(raw_payload, Mapping) else {}
            parent_id = str(payload.get("parent_event_id") or "")
            child_id = str(node.get("event_id") or "")
            if not parent_id or not child_id:
                continue
            parent_node_id = _event_node_id(parent_id)
            child_node_id = _event_node_id(child_id)
            if parent_node_id not in nodes:
                nodes[parent_node_id] = _event_node(parent_id, self._load_event(parent_id) or {})
            key = (parent_node_id, child_node_id, "event_parent", child_id)
            if key in existing_keys or (parent_node_id, child_node_id) in existing_parent_pairs:
                continue
            edge_id = f"event_parent:{parent_id}->{child_id}"
            edges[edge_id] = {
                "id": edge_id,
                "source_node_id": parent_node_id,
                "target_node_id": child_node_id,
                "edge_type": "event_parent",
                "relation_type": None,
                "link_type": None,
                "event_id": child_id,
                "depth": 0,
                "timestamp": node.get("timestamp"),
                "retention_class": payload.get("retention_class"),
                "redaction_policy": payload.get("redaction_policy"),
                "metadata": {},
            }
            existing_keys.add(key)
            existing_parent_pairs.add((parent_node_id, child_node_id))


def _clean_selector(selector: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in selector.items() if value is not None}


def _safe_direction(direction: str | None) -> str:
    value = str(direction or "both").strip().lower()
    return value if value in DIRECTIONS else "both"


def _seed_from_selector(selector: Mapping[str, str]) -> dict[str, Any]:
    if not selector:
        return {}
    if selector.get("event_id"):
        return {"seed_type": "event", "event_id": selector["event_id"]}
    if selector.get("ref_type") and selector.get("ref_id"):
        ref = _make_ref(selector["ref_type"], selector["ref_id"], None)
        return {
            "seed_type": "ref",
            "ref_type": ref.ref_type,
            "ref_id": ref.ref_id,
            "object_uid": ref.object_uid,
            "node_id": ref.node_id,
        }
    for key, ref_type in _SELECTOR_REF_TYPES.items():
        if selector.get(key):
            ref = _make_ref(ref_type, selector[key], None)
            return {
                "seed_type": "ref",
                "selector_type": key,
                "selector_id": selector[key],
                "ref_type": ref.ref_type,
                "ref_id": ref.ref_id,
                "object_uid": ref.object_uid,
                "node_id": ref.node_id,
            }
    key, value = next(iter(selector.items()))
    return {"seed_type": "selector", "selector_type": key, "selector_id": value}


def _make_ref(ref_type: str, ref_id: str, ref_version: str | None) -> _Ref:
    return _Ref(ref_type=ref_type, ref_id=ref_id, ref_version=ref_version, object_uid=_ref_object_uid(ref_type, ref_id))


def _ref_object_uid(ref_type: str | None, ref_id: str | None) -> str | None:
    if not ref_type or not ref_id:
        return None
    try:
        return provenance.ref_object_uid_for(str(ref_type), str(ref_id))
    except Exception:
        return None


def _ref_id_candidates(ref_type: str, ref_id: str) -> list[str]:
    values = {str(ref_id)}
    uid = _ref_object_uid(ref_type, ref_id)
    if uid:
        values.add(uid)
        prefix = f"{ref_type}:"
        if uid.startswith(prefix):
            values.add(uid.split(":", 1)[1])
    prefix = f"{ref_type}:"
    if str(ref_id).startswith(prefix):
        values.add(str(ref_id).split(":", 1)[1])
    return sorted(values)


def _normalize_edge_row(row: Mapping[str, Any]) -> dict[str, Any]:
    properties = _edge_properties(row)
    return {
        "relation_uid": row.get("relation_uid") or properties.get("relation_uid") or properties.get("id"),
        "source_object_uid": row.get("source_object_uid") or row.get("source_uid"),
        "target_object_uid": row.get("target_object_uid") or row.get("target_uid"),
        "relation_type": row.get("relation_type"),
        "source_ref_type": row.get("source_ref_type") or properties.get("source_ref_type"),
        "source_ref_id": row.get("source_ref_id") or properties.get("source_ref_id"),
        "source_ref_version": row.get("source_ref_version") or properties.get("source_ref_version"),
        "target_ref_type": row.get("target_ref_type") or properties.get("target_ref_type"),
        "target_ref_id": row.get("target_ref_id") or properties.get("target_ref_id"),
        "target_ref_version": row.get("target_ref_version") or properties.get("target_ref_version"),
        "event_id": row.get("event_id") or properties.get("event_id"),
        "lineage_root_id": row.get("lineage_root_id") or properties.get("lineage_root_id"),
        "redaction_policy": row.get("redaction_policy") or properties.get("redaction_policy"),
        "retention_class": row.get("retention_class") or properties.get("retention_class"),
        "metadata": row.get("metadata") or row.get("metadata_json") or properties.get("metadata"),
        "valid_from": row.get("valid_from") or ((row.get("_meta") or {}).get("temporal") or {}).get("valid_from"),
        "tx_from": row.get("tx_from") or ((row.get("_meta") or {}).get("temporal") or {}).get("tx_from"),
        "properties": properties,
    }


def _edge_properties(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("properties") or row.get("properties_json") or {}
    return dict(value) if isinstance(value, Mapping) else {}


def _ref_from_edge_row(row: Mapping[str, Any], side: str) -> _Ref:
    normalized = _normalize_edge_row(row)
    ref_type = str(normalized.get(f"{side}_ref_type") or "")
    ref_id = str(normalized.get(f"{side}_ref_id") or normalized.get(f"{side}_object_uid") or "")
    ref_version = normalized.get(f"{side}_ref_version")
    object_uid = normalized.get(f"{side}_object_uid") or _ref_object_uid(ref_type, ref_id)
    return _Ref(
        ref_type=ref_type or "object",
        ref_id=ref_id,
        ref_version=str(ref_version) if ref_version else None,
        object_uid=object_uid,
    )


def _edge_from_row(row: Mapping[str, Any], *, depth: int) -> dict[str, Any] | None:
    normalized = _normalize_edge_row(row)
    source_ref = _ref_from_edge_row(normalized, "source")
    target_ref = _ref_from_edge_row(normalized, "target")
    relation_type = str(normalized.get("relation_type") or "")
    if not relation_type or not source_ref.ref_id or not target_ref.ref_id:
        return None
    link_type = _RELATION_LINK_TYPES.get(relation_type, relation_type.replace("provenance_", "", 1))
    edge_id = str(
        normalized.get("relation_uid")
        or f"{relation_type}:{normalized.get('event_id') or 'event'}:{source_ref.node_id}->{target_ref.node_id}"
    )
    return {
        "id": edge_id,
        "source_node_id": source_ref.node_id,
        "target_node_id": target_ref.node_id,
        "edge_type": link_type,
        "relation_type": relation_type,
        "link_type": link_type,
        "event_id": normalized.get("event_id"),
        "depth": depth,
        "timestamp": _to_iso(normalized.get("valid_from") or normalized.get("tx_from")),
        "retention_class": normalized.get("retention_class"),
        "redaction_policy": normalized.get("redaction_policy"),
        "metadata": _safe_payload(normalized.get("metadata")),
        "lineage_root_id": normalized.get("lineage_root_id"),
        "source_ref_type": source_ref.ref_type,
        "source_ref_id": source_ref.ref_id,
        "target_ref_type": target_ref.ref_type,
        "target_ref_id": target_ref.ref_id,
    }


def _node_from_ref(ref: _Ref) -> dict[str, Any]:
    return {
        "id": ref.node_id,
        "node_type": "reference",
        "ref_type": ref.ref_type,
        "ref_id": ref.ref_id,
        "ref_version": ref.ref_version,
        "object_uid": ref.object_uid,
        "label": f"{ref.ref_type}:{ref.ref_id}",
    }


def _event_node(event_id: str, event: Mapping[str, Any]) -> dict[str, Any]:
    payload = _safe_payload(event)
    return {
        "id": _event_node_id(event_id),
        "node_type": "event",
        "event_id": event_id,
        "event_type": event.get("event_type"),
        "event_name": event.get("event_name"),
        "status": event.get("status"),
        "label": event.get("event_name") or event.get("event_type") or event_id,
        "timestamp": event.get("started_at") or event.get("created_at"),
        "redaction_policy": event.get("redaction_policy"),
        "retention_class": event.get("retention_class"),
        "payload": payload,
    }


def _ref_node_from_edge(edge: Mapping[str, Any], side: str) -> dict[str, Any]:
    ref_type = str(edge.get(f"{side}_ref_type") or "")
    ref_id = str(edge.get(f"{side}_ref_id") or "")
    return _node_from_ref(_make_ref(ref_type, ref_id, None))


def _ref_node_id(ref_type: str, ref_id: str, ref_version: str | None) -> str:
    parts = ["ref", str(ref_type), str(ref_id)]
    if ref_version:
        parts.append(str(ref_version))
    return ":".join(parts)


def _event_node_id(event_id: str) -> str:
    return f"event:{event_id}"


def _safe_payload(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _as_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _warning(code: str, detail: str | None = None) -> dict[str, str]:
    payload = {"code": code}
    if detail:
        payload["detail"] = detail
    return payload


def _coerce_warning(value: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        code = str(value.get("code") or WARNING_EMPTY_TRACE)
        return {"code": code, **{str(k): v for k, v in value.items() if k != "code"}}
    return {"code": str(value)}


def _graph_response(
    *,
    selector: Mapping[str, str],
    seed: Mapping[str, Any],
    direction: str,
    max_depth: int,
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    warnings: Iterable[Mapping[str, Any]],
    truncated: bool,
    lineage_state: str,
) -> dict[str, Any]:
    node_list: list[dict[str, Any]] = sorted(
        (dict(node) for node in nodes.values()), key=lambda row: str(row.get("id") or "")
    )
    if isinstance(edges, Mapping):
        edge_list: list[dict[str, Any]] = sorted(
            (dict(edge) for edge in edges.values()), key=lambda row: str(row.get("id") or "")
        )
    else:
        edge_list = sorted((dict(edge) for edge in edges), key=lambda row: str(row.get("id") or ""))
    warning_list = _dedupe_warnings(warnings)
    return {
        "selector": dict(selector),
        "seed": dict(seed),
        "direction": direction,
        "max_depth": max_depth,
        "lineage_state": lineage_state,
        "nodes": node_list,
        "edges": edge_list,
        "timeline": _timeline(node_list, edge_list),
        "counts": _counts(node_list, edge_list, warning_list),
        "truncated": bool(truncated),
        "warnings": warning_list,
    }


def _dedupe_warnings(warnings: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for warning in warnings:
        payload = _coerce_warning(warning)
        code = str(payload.get("code") or "")
        if code not in VALID_WARNING_CODES:
            continue
        key = (code, str(payload.get("detail") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(payload)
    return out


def _counts(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    warnings: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    events = sum(1 for node in nodes if node.get("node_type") == "event")
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "events": events,
        "references": len(nodes) - events,
        "warnings": len(warnings),
    }


def _timeline(nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for node in nodes:
        timestamp = node.get("timestamp")
        if timestamp:
            items.append(
                {
                    "kind": "node",
                    "id": node.get("id"),
                    "node_type": node.get("node_type"),
                    "label": node.get("label"),
                    "status": node.get("status"),
                    "timestamp": _to_iso(timestamp),
                }
            )
    for edge in edges:
        timestamp = edge.get("timestamp")
        if timestamp:
            items.append(
                {
                    "kind": "edge",
                    "id": edge.get("id"),
                    "edge_type": edge.get("edge_type"),
                    "relation_type": edge.get("relation_type"),
                    "source_node_id": edge.get("source_node_id"),
                    "target_node_id": edge.get("target_node_id"),
                    "timestamp": _to_iso(timestamp),
                }
            )
    return sorted(items, key=lambda item: (str(item.get("timestamp") or ""), str(item.get("id") or "")))


def _to_iso(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)
