from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from uuid import uuid4

ActorType = Literal["user", "agent", "system"]
PolicyEffect = Literal["allow", "deny"]

_ANY = "*"


def _policy_decision_id(prefix: str = "policy") -> str:
    return f"{prefix}:{uuid4().hex[:20]}"


def _tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = [str(part).strip() for part in value]
    else:
        values = [str(value).strip()]
    return tuple(item for item in dict.fromkeys(values) if item)


def _lower_tuple(value: Any) -> tuple[str, ...]:
    return tuple(item.lower() for item in _tuple(value))


def _owner_actor_id() -> str:
    return (os.getenv("ONTOLOGY_OWNER_ACTOR_ID") or "admin").strip() or "admin"


def _owner_tenant_id() -> str:
    return (os.getenv("ONTOLOGY_OWNER_TENANT_ID") or "default").strip() or "default"


def _owner_account_id() -> str:
    return (os.getenv("ONTOLOGY_OWNER_ACCOUNT_ID") or "default").strip() or "default"


def _default_account_ids() -> tuple[str, ...]:
    return tuple(dict.fromkeys((_owner_account_id(), "default", "default-account")))


def _default_portfolio_ids() -> tuple[str, ...]:
    return ("default", "default-portfolio")


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
    purposes: tuple[str, ...] = field(default_factory=tuple)
    tenant_id: str | None = None
    account_ids: tuple[str, ...] = field(default_factory=tuple)
    portfolio_ids: tuple[str, ...] = field(default_factory=tuple)


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
    decision_id: str = field(default_factory=_policy_decision_id)
    matched_rule: str | None = None
    explanation: str | None = None
    audit: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    action: str
    actor: Actor | None = None
    purpose: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    tenant_id: str | None = None
    account_id: str | None = None
    portfolio_id: str | None = None
    data_markings: tuple[str, ...] = field(default_factory=tuple)
    data_sensitivity: str = "public_market"
    required_scopes: tuple[str, ...] = field(default_factory=tuple)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PolicyRule:
    rule_id: str
    effect: PolicyEffect = "allow"
    actions: tuple[str, ...] = field(default_factory=lambda: (_ANY,))
    actor_types: tuple[str, ...] = field(default_factory=tuple)
    actor_ids: tuple[str, ...] = field(default_factory=tuple)
    parent_actor_ids: tuple[str, ...] = field(default_factory=tuple)
    roles: tuple[str, ...] = field(default_factory=tuple)
    purposes: tuple[str, ...] = field(default_factory=tuple)
    tenant_ids: tuple[str, ...] = field(default_factory=tuple)
    account_ids: tuple[str, ...] = field(default_factory=tuple)
    portfolio_ids: tuple[str, ...] = field(default_factory=tuple)
    data_markings: tuple[str, ...] = field(default_factory=tuple)
    data_sensitivities: tuple[str, ...] = field(default_factory=tuple)
    required_scopes: tuple[str, ...] = field(default_factory=tuple)
    explanation: str = ""


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
    """Default single-user ABAC policy profile."""

    def __init__(self, engine: ABACPolicyEngine | None = None):
        self.engine = engine or ABACPolicyEngine(default_single_user_policy())

    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        return self.engine.evaluate(request)

    def check_action(
        self,
        actor: Actor | None,
        action: str,
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        return self.evaluate(policy_request_from_context(actor, action, context))

    def check_object(
        self,
        actor: Actor | None,
        node: NodeResource,
        action: str = "read",
    ) -> PolicyDecision:
        context = {
            **node.properties,
            "purpose": "ontology",
            "resource_type": node.type,
            "resource_id": node.id,
        }
        return self.evaluate(policy_request_from_context(actor, f"ontology.object.{action}", context))

    def check_relationship(
        self,
        actor: Actor | None,
        edge: EdgeResource,
        source: NodeResource | None = None,
        target: NodeResource | None = None,
        action: str = "read",
    ) -> PolicyDecision:
        context = {
            **edge.properties,
            "purpose": "ontology",
            "resource_type": edge.relation_type,
            "resource_id": f"{edge.source_id}:{edge.relation_type}:{edge.target_id}",
        }
        if source is not None:
            source_account = source.properties.get("account_id") or source.properties.get("owner_account_id")
            context.setdefault("account_id", source_account)
            context.setdefault("portfolio_id", source.properties.get("portfolio_id"))
        if target is not None:
            target_account = target.properties.get("account_id") or target.properties.get("owner_account_id")
            context.setdefault("account_id", target_account)
            context.setdefault("portfolio_id", target.properties.get("portfolio_id"))
        return self.evaluate(policy_request_from_context(actor, f"ontology.relationship.{action}", context))

    def allowed_fields(
        self,
        actor: Actor | None,
        resource: NodeResource | EdgeResource | FieldResource,
    ) -> set[str] | None:
        if isinstance(resource, NodeResource):
            decision = self.check_object(actor, resource)
        elif isinstance(resource, EdgeResource):
            decision = self.check_relationship(actor, resource)
        else:
            decision = self.check_action(
                actor,
                "ontology.field.read",
                {
                    "purpose": "ontology",
                    "resource_type": resource.owner_type,
                    "resource_id": resource.owner_id,
                },
            )
        return None if decision.allowed else set()


def _first_scope_value(context: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = context.get(key)
        if value is None and "." in key:
            value = _nested_context_value(context, key)
        text = str(value or "").strip()
        if text:
            return text
    return None


def _nested_context_value(context: dict[str, Any], path: str) -> Any:
    current: Any = context
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _data_markings(context: dict[str, Any]) -> tuple[str, ...]:
    markings = context.get("data_markings")
    if markings is None:
        markings = context.get("markings")
    return _lower_tuple(markings)


def _data_sensitivity(context: dict[str, Any]) -> str:
    value = context.get("data_sensitivity") or context.get("sensitivity") or "public_market"
    return str(value or "public_market").strip().lower() or "public_market"


def policy_request_from_context(
    actor: Actor | None,
    action: str,
    context: dict[str, Any] | None = None,
) -> PolicyRequest:
    raw = dict(context or {})
    return PolicyRequest(
        actor=actor,
        action=str(action or "").strip(),
        purpose=_first_scope_value(raw, "purpose", "intent", "use_case"),
        resource_type=_first_scope_value(raw, "resource_type", "object_type", "type"),
        resource_id=_first_scope_value(raw, "resource_id", "object_uid", "id"),
        tenant_id=_first_scope_value(raw, "tenant_id", "tenant"),
        account_id=_first_scope_value(raw, "account_id", "owner_account_id", "record.account_id"),
        portfolio_id=_first_scope_value(raw, "portfolio_id", "record.portfolio_id"),
        data_markings=_data_markings(raw),
        data_sensitivity=_data_sensitivity(raw),
        required_scopes=_tuple(raw.get("required_scopes")),
        context=raw,
    )


def default_single_user_policy() -> tuple[PolicyRule, ...]:
    account_ids = _default_account_ids()
    portfolio_ids = _default_portfolio_ids()
    return (
        PolicyRule(
            rule_id="default.system.allow",
            effect="allow",
            actor_types=("system",),
            explanation="System actor is allowed by the default single-user policy.",
        ),
        PolicyRule(
            rule_id="default.owner.allow",
            effect="allow",
            actor_types=("user", "agent"),
            roles=("owner",),
            tenant_ids=(_owner_tenant_id(),),
            account_ids=account_ids,
            portfolio_ids=portfolio_ids,
            explanation="Owner role is allowed within the default single-user scope.",
        ),
        PolicyRule(
            rule_id="default.admin_owner.allow",
            effect="allow",
            actor_types=("user",),
            actor_ids=(_owner_actor_id(),),
            roles=("admin",),
            tenant_ids=(_owner_tenant_id(),),
            account_ids=account_ids,
            portfolio_ids=portfolio_ids,
            explanation="Configured admin user is allowed within the default single-user scope.",
        ),
        PolicyRule(
            rule_id="default.delegated_admin_agent.allow",
            effect="allow",
            actor_types=("agent",),
            parent_actor_ids=(_owner_actor_id(),),
            roles=("admin",),
            tenant_ids=(_owner_tenant_id(),),
            account_ids=account_ids,
            portfolio_ids=portfolio_ids,
            explanation="Agent delegated by the configured admin is allowed within the default single-user scope.",
        ),
    )


class ABACPolicyEngine:
    def __init__(self, rules: tuple[PolicyRule, ...] | list[PolicyRule]):
        self.rules = tuple(rules)

    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        actor = request.actor
        if actor is None:
            return self._decision(
                False,
                request,
                reason=f"Actor is not allowed to perform policy action '{request.action}'",
                explanation="No actor was provided for ABAC evaluation.",
            )

        purpose_denial = _actor_purpose_denial(actor, request)
        if purpose_denial:
            return self._decision(False, request, reason=purpose_denial, explanation=purpose_denial)

        scope_denial = _actor_scope_denial(actor, request)
        if scope_denial:
            return self._decision(False, request, reason=scope_denial, explanation=scope_denial)

        matching = [rule for rule in self.rules if _rule_matches(rule, actor, request)]
        for rule in matching:
            if rule.effect == "deny":
                explanation = rule.explanation or f"ABAC deny rule '{rule.rule_id}' matched."
                return self._decision(
                    False,
                    request,
                    reason=explanation,
                    matched_rule=rule.rule_id,
                    explanation=explanation,
                )
        for rule in matching:
            if rule.effect == "allow":
                explanation = rule.explanation or f"ABAC allow rule '{rule.rule_id}' matched."
                return self._decision(
                    True,
                    request,
                    reason="allowed",
                    matched_rule=rule.rule_id,
                    explanation=explanation,
                )
        return self._decision(
            False,
            request,
            reason=f"Actor is not allowed to perform policy action '{request.action}'",
            explanation="No ABAC allow rule matched the request.",
        )

    def _decision(
        self,
        allowed: bool,
        request: PolicyRequest,
        *,
        reason: str,
        matched_rule: str | None = None,
        explanation: str | None = None,
    ) -> PolicyDecision:
        decision_id = _policy_decision_id("abac")
        audit = _audit_payload(request, decision_id, allowed, reason, matched_rule, explanation)
        return PolicyDecision(
            allowed=allowed,
            reason=reason,
            decision_id=decision_id,
            matched_rule=matched_rule,
            explanation=explanation,
            audit=audit,
        )


def _audit_payload(
    request: PolicyRequest,
    decision_id: str,
    allowed: bool,
    reason: str,
    matched_rule: str | None,
    explanation: str | None,
) -> dict[str, Any]:
    actor = request.actor
    return {
        "policy_decision_id": decision_id,
        "allowed": allowed,
        "reason": reason,
        "matched_rule": matched_rule,
        "explanation": explanation,
        "action": request.action,
        "purpose": request.purpose,
        "resource_type": request.resource_type,
        "resource_id": request.resource_id,
        "actor_id": actor.actor_id if actor else None,
        "actor_type": actor.actor_type if actor else None,
        "actor_roles": list(actor.roles) if actor else [],
        "parent_actor_id": actor.parent_actor_id if actor else None,
        "tenant_id": request.tenant_id,
        "account_id": request.account_id,
        "portfolio_id": request.portfolio_id,
        "data_markings": list(request.data_markings),
        "data_sensitivity": request.data_sensitivity,
        "required_scopes": list(request.required_scopes),
    }


def _actor_purpose_denial(actor: Actor, request: PolicyRequest) -> str | None:
    purposes = _lower_tuple(actor.purposes)
    if not purposes:
        return None
    purpose = str(request.purpose or "").strip().lower()
    if purpose and purpose in purposes:
        return None
    return f"Actor purpose does not allow policy action '{request.action}'"


def _actor_scope_denial(actor: Actor, request: PolicyRequest) -> str | None:
    if actor.actor_type == "system":
        return None
    tenant_id = str(request.tenant_id or "").strip()
    if tenant_id:
        actor_tenant = str(actor.tenant_id or _owner_tenant_id()).strip()
        if tenant_id != actor_tenant:
            return f"Requested tenant_id is outside the actor scope for policy action '{request.action}'"
    account_id = str(request.account_id or "").strip()
    if account_id:
        account_ids = set(_tuple(actor.account_ids) or _default_account_ids())
        if account_id not in account_ids:
            return f"Requested account_id is outside the actor scope for policy action '{request.action}'"
    portfolio_id = str(request.portfolio_id or "").strip()
    if portfolio_id:
        portfolio_ids = set(_tuple(actor.portfolio_ids) or _default_portfolio_ids())
        if portfolio_id not in portfolio_ids:
            return f"Requested portfolio_id is outside the actor scope for policy action '{request.action}'"
    return None


def _matches(patterns: tuple[str, ...], value: str | None, *, lower: bool = True) -> bool:
    normalized_patterns = _lower_tuple(patterns) if lower else _tuple(patterns)
    if not normalized_patterns or _ANY in normalized_patterns:
        return True
    normalized = str(value or "").strip()
    if lower:
        normalized = normalized.lower()
    if not normalized:
        return False
    return normalized in normalized_patterns


def _matches_any(patterns: tuple[str, ...], values: tuple[str, ...], *, lower: bool = True) -> bool:
    normalized_patterns = set(_lower_tuple(patterns) if lower else _tuple(patterns))
    if not normalized_patterns or _ANY in normalized_patterns:
        return True
    normalized_values = set(_lower_tuple(values) if lower else _tuple(values))
    return bool(normalized_patterns.intersection(normalized_values))


def _matches_all(patterns: tuple[str, ...], values: tuple[str, ...], *, lower: bool = True) -> bool:
    normalized_patterns = set(_lower_tuple(patterns) if lower else _tuple(patterns))
    if not normalized_patterns or _ANY in normalized_patterns:
        return True
    normalized_values = set(_lower_tuple(values) if lower else _tuple(values))
    return normalized_patterns.issubset(normalized_values)


def _matches_optional_scope(patterns: tuple[str, ...], value: str | None) -> bool:
    if not str(value or "").strip():
        return True
    return _matches(patterns, value, lower=False)


def _rule_matches(rule: PolicyRule, actor: Actor, request: PolicyRequest) -> bool:
    return (
        _matches(rule.actions, request.action)
        and _matches(rule.actor_types, actor.actor_type)
        and _matches(rule.actor_ids, actor.actor_id, lower=False)
        and _matches(rule.parent_actor_ids, actor.parent_actor_id, lower=False)
        and _matches_any(rule.roles, actor.roles)
        and _matches(rule.purposes, request.purpose)
        and _matches(rule.tenant_ids, request.tenant_id or _owner_tenant_id(), lower=False)
        and _matches_optional_scope(rule.account_ids, request.account_id)
        and _matches_optional_scope(rule.portfolio_ids, request.portfolio_id)
        and _matches_any(rule.data_markings, request.data_markings)
        and _matches(rule.data_sensitivities, request.data_sensitivity)
        and _matches_all(rule.required_scopes, request.required_scopes)
    )


DEFAULT_ONTOLOGY_POLICY = DefaultOntologyPolicy()


def admin_actor(subject: str = "admin", *, source: str = "api") -> Actor:
    actor_id = str(subject or "admin").strip() or "admin"
    return Actor(actor_id=actor_id, actor_type="user", roles=("owner", "admin"), source=source)


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
        purposes=tuple(parent.purposes),
        tenant_id=parent.tenant_id,
        account_ids=tuple(parent.account_ids),
        portfolio_ids=tuple(parent.portfolio_ids),
    )


def actor_to_dict(actor: Actor) -> dict[str, Any]:
    return {
        "actor_id": actor.actor_id,
        "actor_type": actor.actor_type,
        "roles": list(actor.roles),
        "source": actor.source,
        "parent_actor_id": actor.parent_actor_id,
        "purposes": list(actor.purposes),
        "tenant_id": actor.tenant_id,
        "account_ids": list(actor.account_ids),
        "portfolio_ids": list(actor.portfolio_ids),
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
    purposes_raw = value.get("purposes")
    purposes = (
        tuple(str(purpose) for purpose in purposes_raw if isinstance(purpose, str))
        if isinstance(purposes_raw, (list, tuple, set))
        else ()
    )
    account_ids_raw = value.get("account_ids")
    account_ids = (
        tuple(str(account_id) for account_id in account_ids_raw if isinstance(account_id, str))
        if isinstance(account_ids_raw, (list, tuple, set))
        else ()
    )
    portfolio_ids_raw = value.get("portfolio_ids")
    portfolio_ids = (
        tuple(str(portfolio_id) for portfolio_id in portfolio_ids_raw if isinstance(portfolio_id, str))
        if isinstance(portfolio_ids_raw, (list, tuple, set))
        else ()
    )
    return Actor(
        actor_id=str(value.get("actor_id") or "admin"),
        actor_type=actor_type,  # type: ignore[arg-type]
        roles=roles,
        source=str(value["source"]) if value.get("source") is not None else None,
        parent_actor_id=str(value["parent_actor_id"]) if value.get("parent_actor_id") is not None else None,
        purposes=purposes,
        tenant_id=str(value["tenant_id"]) if value.get("tenant_id") is not None else None,
        account_ids=account_ids,
        portfolio_ids=portfolio_ids,
    )


def actor_cache_key(actor: Actor | None) -> str:
    actor = actor or admin_actor()
    roles = ",".join(sorted(actor.roles))
    parent = actor.parent_actor_id or ""
    purposes = ",".join(sorted(actor.purposes))
    accounts = ",".join(sorted(actor.account_ids))
    portfolios = ",".join(sorted(actor.portfolio_ids))
    tenant = actor.tenant_id or ""
    return (
        f"{actor.actor_type}:{actor.actor_id}:roles={roles}:parent={parent}:purposes={purposes}:"
        f"tenant={tenant}:accounts={accounts}:portfolios={portfolios}"
    )


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
        edge_resource = edge_resource_from_dict(edge)
        source = node_resources.get(edge_resource.source_id)
        target = node_resources.get(edge_resource.target_id)
        if source is None or target is None:
            stats["filtered_relationships"] += 1
            continue
        if not policy.check_relationship(actor, edge_resource, source=source, target=target).allowed:
            stats["filtered_relationships"] += 1
            continue
        filtered = dict(edge)
        props, redacted = redact_properties(actor, policy, edge_resource, edge_resource.properties)
        filtered["properties"] = props
        stats["redacted_fields"] += redacted
        filtered_edges.append(filtered)

    return {"nodes": filtered_nodes, "edges": filtered_edges}, stats
