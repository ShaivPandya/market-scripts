from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

ActorType = Literal["user", "agent", "system"]


class OntologyAction:
    RUNS_LIST = "runs.list"
    QUERY = "query"
    GRAPH_READ = "graph.read"
    SNAPSHOTS_COMPARE = "snapshots.compare"
    SNAPSHOT_REFRESH = "snapshot.refresh"
    SNAPSHOT_SAVE = "snapshot.save"
    NODE_UPSERT = "node.upsert"
    EDGE_UPSERT = "edge.upsert"
    RUNS_PRUNE = "runs.prune"
    JOB_READ = "job.read"


@dataclass(frozen=True, slots=True)
class Actor:
    actor_id: str
    actor_type: ActorType
    roles: tuple[str, ...] = field(default_factory=tuple)
    source: str | None = None
    parent_actor_id: str | None = None


@dataclass(frozen=True, slots=True)
class NodeResource:
    id: str
    type: str
    label: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    schema_name: str | None = None
    schema_version: int | None = None


@dataclass(frozen=True, slots=True)
class EdgeResource:
    source_id: str
    target_id: str
    relation_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    schema_name: str | None = None
    schema_version: int | None = None


@dataclass(frozen=True, slots=True)
class FieldResource:
    owner_type: str
    owner_id: str
    field_name: str
    owner_kind: Literal["node", "edge", "result"] = "node"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    reason: str | None = None
    redaction: dict[str, Any] | None = None


class PolicyDenied(Exception):
    def __init__(self, reason: str = "Access denied"):
        super().__init__(reason)
        self.reason = reason


class OntologyPolicy(Protocol):
    def check_action(
        self,
        actor: Actor | None,
        action: str,
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision: ...

    def check_object(
        self,
        actor: Actor | None,
        node: NodeResource,
        action: str = "read",
    ) -> PolicyDecision: ...

    def check_relationship(
        self,
        actor: Actor | None,
        edge: EdgeResource,
        source: NodeResource | None = None,
        target: NodeResource | None = None,
        action: str = "read",
    ) -> PolicyDecision: ...

    def allowed_fields(
        self,
        actor: Actor | None,
        resource: NodeResource | EdgeResource | FieldResource,
    ) -> set[str] | None: ...


class DefaultOntologyPolicy:
    """V1 policy: authenticated admin and internal system actors retain full access."""

    def _is_allowed_actor(self, actor: Actor | None) -> bool:
        if actor is None:
            return False
        roles = {role.lower() for role in actor.roles}
        return actor.actor_type == "system" or "admin" in roles

    def check_action(
        self,
        actor: Actor | None,
        action: str,
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        if self._is_allowed_actor(actor):
            return PolicyDecision(True)
        return PolicyDecision(False, f"Actor is not allowed to perform ontology action '{action}'")

    def check_object(
        self,
        actor: Actor | None,
        node: NodeResource,
        action: str = "read",
    ) -> PolicyDecision:
        if self._is_allowed_actor(actor):
            return PolicyDecision(True)
        return PolicyDecision(False, f"Actor is not allowed to {action} ontology object '{node.id}'")

    def check_relationship(
        self,
        actor: Actor | None,
        edge: EdgeResource,
        source: NodeResource | None = None,
        target: NodeResource | None = None,
        action: str = "read",
    ) -> PolicyDecision:
        if self._is_allowed_actor(actor):
            return PolicyDecision(True)
        return PolicyDecision(
            False,
            f"Actor is not allowed to {action} ontology relationship '{edge.relation_type}'",
        )

    def allowed_fields(
        self,
        actor: Actor | None,
        resource: NodeResource | EdgeResource | FieldResource,
    ) -> set[str] | None:
        return None if self._is_allowed_actor(actor) else set()


DEFAULT_ONTOLOGY_POLICY = DefaultOntologyPolicy()


def admin_actor(subject: str = "admin", *, source: str = "api") -> Actor:
    actor_id = str(subject or "admin").strip() or "admin"
    return Actor(actor_id=actor_id, actor_type="user", roles=("admin",), source=source)


def system_actor(source: str) -> Actor:
    actor_id = f"system:{str(source or 'internal').strip() or 'internal'}"
    return Actor(actor_id=actor_id, actor_type="system", roles=("system",), source=source)


def agent_actor(parent_actor: Actor | None = None) -> Actor:
    parent = parent_actor or admin_actor(source="agent")
    return Actor(
        actor_id=f"agent:{parent.actor_id}",
        actor_type="agent",
        roles=tuple(parent.roles),
        source="agent",
        parent_actor_id=parent.actor_id,
    )


def actor_to_dict(actor: Actor) -> dict[str, Any]:
    return {
        "actor_id": actor.actor_id,
        "actor_type": actor.actor_type,
        "roles": list(actor.roles),
        "source": actor.source,
        "parent_actor_id": actor.parent_actor_id,
    }


def actor_from_dict(value: dict[str, Any] | Actor | None) -> Actor:
    if isinstance(value, Actor):
        return value
    if not isinstance(value, dict):
        return admin_actor()
    actor_type = str(value.get("actor_type") or "user").strip().lower()
    if actor_type not in {"user", "agent", "system"}:
        actor_type = "user"
    roles_raw = value.get("roles")
    roles = (
        tuple(str(role) for role in roles_raw if isinstance(role, str))
        if isinstance(roles_raw, (list, tuple, set))
        else ()
    )
    return Actor(
        actor_id=str(value.get("actor_id") or "admin"),
        actor_type=actor_type,  # type: ignore[arg-type]
        roles=roles,
        source=str(value["source"]) if value.get("source") is not None else None,
        parent_actor_id=str(value["parent_actor_id"]) if value.get("parent_actor_id") is not None else None,
    )


def actor_cache_key(actor: Actor | None) -> str:
    actor = actor or admin_actor()
    roles = ",".join(sorted(actor.roles))
    parent = actor.parent_actor_id or ""
    return f"{actor.actor_type}:{actor.actor_id}:roles={roles}:parent={parent}"


def require_allowed(decision: PolicyDecision) -> None:
    if not decision.allowed:
        raise PolicyDenied(decision.reason or "Access denied")


def node_resource_from_dict(node: dict[str, Any]) -> NodeResource:
    return NodeResource(
        id=str(node.get("id") or ""),
        type=str(node.get("type") or ""),
        label=str(node["label"]) if node.get("label") is not None else None,
        properties=dict(node.get("properties") or {}) if isinstance(node.get("properties"), dict) else {},
        schema_name=str(node["schema_name"]) if node.get("schema_name") is not None else None,
        schema_version=int(node["schema_version"]) if node.get("schema_version") is not None else None,
    )


def edge_resource_from_dict(edge: dict[str, Any]) -> EdgeResource:
    return EdgeResource(
        source_id=str(edge.get("source_id") or ""),
        target_id=str(edge.get("target_id") or ""),
        relation_type=str(edge.get("relation_type") or ""),
        properties=dict(edge.get("properties") or {}) if isinstance(edge.get("properties"), dict) else {},
        schema_name=str(edge["schema_name"]) if edge.get("schema_name") is not None else None,
        schema_version=int(edge["schema_version"]) if edge.get("schema_version") is not None else None,
    )


def redact_properties(
    actor: Actor | None,
    policy: OntologyPolicy,
    resource: NodeResource | EdgeResource,
    properties: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    allowed = policy.allowed_fields(actor, resource)
    if allowed is None:
        return dict(properties), 0
    redacted = {key: value for key, value in properties.items() if key in allowed}
    return redacted, max(0, len(properties) - len(redacted))


def _redact_dict_fields(
    actor: Actor | None,
    policy: OntologyPolicy,
    resource: NodeResource | EdgeResource | FieldResource,
    row: dict[str, Any],
) -> int:
    allowed = policy.allowed_fields(actor, resource)
    if allowed is None:
        return 0
    count = 0
    for key in list(row.keys()):
        if key.startswith("_"):
            continue
        if key not in allowed:
            row[key] = None
            count += 1
    return count


def filter_query_results(
    actor: Actor | None,
    policy: OntologyPolicy,
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out: list[dict[str, Any]] = []
    filtered_objects = 0
    redacted_fields = 0
    for result in results:
        resource = result.get("_resource")
        if isinstance(resource, NodeResource):
            if not policy.check_object(actor, resource).allowed:
                filtered_objects += 1
                continue
            row = {key: value for key, value in result.items() if key != "_resource"}
            redacted_fields += _redact_dict_fields(actor, policy, resource, row)
            out.append(row)
        else:
            out.append(dict(result))
    return out, {"filtered_objects": filtered_objects, "redacted_fields": redacted_fields}


def filter_graph(
    actor: Actor | None,
    policy: OntologyPolicy,
    graph: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    nodes = graph.get("nodes") if isinstance(graph, dict) else []
    edges = graph.get("edges") if isinstance(graph, dict) else []
    node_resources: dict[str, NodeResource] = {}
    filtered_nodes: list[dict[str, Any]] = []
    stats = {
        "filtered_objects": 0,
        "filtered_relationships": 0,
        "redacted_fields": 0,
    }

    for node in nodes if isinstance(nodes, list) else []:
        if not isinstance(node, dict):
            continue
        resource = node_resource_from_dict(node)
        if not policy.check_object(actor, resource).allowed:
            stats["filtered_objects"] += 1
            continue
        filtered = dict(node)
        props, redacted = redact_properties(actor, policy, resource, resource.properties)
        filtered["properties"] = props
        stats["redacted_fields"] += redacted
        node_resources[resource.id] = resource
        filtered_nodes.append(filtered)

    filtered_edges: list[dict[str, Any]] = []
    for edge in edges if isinstance(edges, list) else []:
        if not isinstance(edge, dict):
            continue
        resource = edge_resource_from_dict(edge)
        source = node_resources.get(resource.source_id)
        target = node_resources.get(resource.target_id)
        if source is None or target is None:
            stats["filtered_relationships"] += 1
            continue
        if not policy.check_relationship(actor, resource, source=source, target=target).allowed:
            stats["filtered_relationships"] += 1
            continue
        filtered = dict(edge)
        props, redacted = redact_properties(actor, policy, resource, resource.properties)
        filtered["properties"] = props
        stats["redacted_fields"] += redacted
        filtered_edges.append(filtered)

    return {"nodes": filtered_nodes, "edges": filtered_edges}, stats
