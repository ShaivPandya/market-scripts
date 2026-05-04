"""Governance controls for agent model egress and tool execution.

This module is intentionally small and deterministic: the product is currently
single-admin, but every agent decision still receives explicit actor, account,
portfolio, DLP, egress, budget, timeout, retry, and audit metadata.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any

from ontology.action_registry import ToolExposure
from ontology.policy import Actor, admin_actor

DEFAULT_ACCOUNT_SCOPE = "default-account"
DEFAULT_PORTFOLIO_SCOPE = "default-portfolio"
REDACTION_POLICY = "agent_dlp_v1"
EGRESS_POLICY_VERSION = "agent_provider_egress_v1"


class AgentGovernanceError(RuntimeError):
    """Governance failure with a stable code suitable for SSE/API payloads."""

    def __init__(self, message: str, *, code: str = "governance_denied"):
        super().__init__(message)
        self.message = message
        self.code = code


class AgentBudgetExceeded(AgentGovernanceError):
    def __init__(self, message: str):
        super().__init__(message, code="budget_exceeded")


class AgentDLPError(AgentGovernanceError):
    def __init__(self, message: str):
        super().__init__(message, code="dlp_denied")


class ToolPolicyDenied(AgentGovernanceError):
    def __init__(self, message: str):
        super().__init__(message, code="tool_policy_denied")


class ToolTimeoutError(AgentGovernanceError):
    def __init__(self, message: str):
        super().__init__(message, code="tool_timeout")


@dataclass(frozen=True)
class ExecutionContext:
    actor_id: str
    actor_type: str
    delegated_actor_id: str | None
    parent_actor_id: str | None
    roles: tuple[str, ...]
    account_id: str = DEFAULT_ACCOUNT_SCOPE
    portfolio_id: str = DEFAULT_PORTFOLIO_SCOPE

    @property
    def scope(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "delegated_actor_id": self.delegated_actor_id,
            "parent_actor_id": self.parent_actor_id,
            "account_id": self.account_id,
            "portfolio_id": self.portfolio_id,
        }


@dataclass(frozen=True)
class ToolPolicyDecision:
    decision_id: str
    allowed: bool
    reason: str
    scope: dict[str, Any]
    required_scopes: tuple[str, ...]
    data_sensitivity: str
    provider_egress: str
    audit_level: str


@dataclass
class AgentBudgetState:
    max_model_calls: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_MODEL_CALLS", "12")))
    max_tool_calls: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_TOOL_CALLS", "32")))
    max_high_cost_tools: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_HIGH_COST_TOOLS", "6")))
    max_input_tokens: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_INPUT_TOKENS", "200000")))
    max_output_tokens: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_OUTPUT_TOKENS", "50000")))
    max_cost_usd: float = field(default_factory=lambda: float(os.environ.get("AGENT_MAX_COST_USD", "25.0")))
    model_calls: int = 0
    tool_calls: int = 0
    high_cost_tool_calls: int = 0
    estimated_input_tokens: int = 0
    actual_input_tokens: int = 0
    actual_output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def check_model_call(self, *, estimated_input_tokens: int, estimated_cost_usd: float = 0.0) -> None:
        if self.model_calls + 1 > self.max_model_calls:
            raise AgentBudgetExceeded(f"Model call budget exceeded ({self.max_model_calls})")
        if self.estimated_input_tokens + estimated_input_tokens > self.max_input_tokens:
            raise AgentBudgetExceeded(f"Input token budget exceeded ({self.max_input_tokens})")
        if self.estimated_cost_usd + estimated_cost_usd > self.max_cost_usd:
            raise AgentBudgetExceeded(f"Cost budget exceeded (${self.max_cost_usd:.2f})")
        self.model_calls += 1
        self.estimated_input_tokens += estimated_input_tokens
        self.estimated_cost_usd += estimated_cost_usd

    def record_model_usage(self, usage: Mapping[str, Any] | None) -> None:
        if not usage:
            return
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if isinstance(input_tokens, int):
            self.actual_input_tokens += input_tokens
        if isinstance(output_tokens, int):
            self.actual_output_tokens += output_tokens
            if self.actual_output_tokens > self.max_output_tokens:
                raise AgentBudgetExceeded(f"Output token budget exceeded ({self.max_output_tokens})")

    def check_tool_call(self, tool: ToolExposure) -> None:
        if self.tool_calls + 1 > self.max_tool_calls:
            raise AgentBudgetExceeded(f"Tool call budget exceeded ({self.max_tool_calls})")
        high_cost = bool(tool.rate_limit.get("high_cost") or tool.tool_name in _HIGH_COST_TOOL_NAMES)
        if high_cost and self.high_cost_tool_calls + 1 > self.max_high_cost_tools:
            raise AgentBudgetExceeded(f"High-cost tool budget exceeded ({self.max_high_cost_tools})")
        self.tool_calls += 1
        if high_cost:
            self.high_cost_tool_calls += 1

    def to_meta(self) -> dict[str, Any]:
        return {
            "model_calls": self.model_calls,
            "max_model_calls": self.max_model_calls,
            "tool_calls": self.tool_calls,
            "max_tool_calls": self.max_tool_calls,
            "high_cost_tool_calls": self.high_cost_tool_calls,
            "max_high_cost_tools": self.max_high_cost_tools,
            "estimated_input_tokens": self.estimated_input_tokens,
            "max_input_tokens": self.max_input_tokens,
            "actual_input_tokens": self.actual_input_tokens,
            "actual_output_tokens": self.actual_output_tokens,
            "max_output_tokens": self.max_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "max_cost_usd": self.max_cost_usd,
        }


_HIGH_COST_TOOL_NAMES = {
    "query_ontology",
    "get_signal_aggregator",
    "get_sector_metrics",
    "run_dcf_valuation",
    "run_quality_screen",
    "run_short_screen",
    "run_long_screen",
    "run_fundamental_momentum",
    "run_portfolio_analyzer",
    "run_portfolio_sizer",
    "run_hedging_tool",
    "run_hedging_recommendation",
    "search_web",
}

_RATE_LOCK = threading.Lock()
_RATE_WINDOWS: dict[tuple[str, str], deque[float]] = defaultdict(deque)

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}\b")),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE)),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    (
        "named_secret",
        re.compile(
            r"(?i)\b(api[_-]?key|authorization|cookie|password|secret|session[_-]?token|access[_-]?token)\b"
            r"\s*[:=]\s*['\"]?[^'\"\s,;]{6,}"
        ),
    ),
)
_SENSITIVE_KEY_PARTS = {
    "authorization",
    "cookie",
    "password",
    "secret",
    "session",
    "token",
    "api_key",
    "apikey",
    "jwt",
}


def execution_context_for_actor(actor: Actor | None) -> ExecutionContext:
    resolved = actor or admin_actor(source="agent_governance")
    actor_id = str(resolved.actor_id or "admin")
    delegated_actor_id = actor_id if resolved.actor_type == "agent" else None
    return ExecutionContext(
        actor_id=actor_id,
        actor_type=str(resolved.actor_type or "user"),
        delegated_actor_id=delegated_actor_id,
        parent_actor_id=resolved.parent_actor_id,
        roles=tuple(resolved.roles or ()),
    )


def _decision_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4().hex[:20]}"


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return str(value)


def estimate_tokens(value: Any) -> int:
    return max(1, (len(_stable_json(value)) + 3) // 4)


def _is_sensitive_key(key: Any) -> bool:
    lowered = str(key or "").lower()
    if lowered in {
        "max_tokens",
        "max_output_tokens",
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "token_budget",
    } or lowered.endswith("_tokens"):
        return False
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def redact_secrets(value: Any) -> tuple[Any, list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []

    def redact_text(text: str, path: str) -> str:
        out = text
        for label, pattern in _SECRET_PATTERNS:
            matches = list(pattern.finditer(out))
            if not matches:
                continue
            findings.append({"type": label, "path": path, "count": len(matches)})
            out = pattern.sub(f"[REDACTED:{label}]", out)
        return out

    def walk(item: Any, path: str) -> Any:
        if isinstance(item, str):
            return redact_text(item, path)
        if isinstance(item, Mapping):
            out: dict[str, Any] = {}
            for key, child in item.items():
                child_path = f"{path}.{key}" if path else str(key)
                if _is_sensitive_key(key):
                    if child not in (None, ""):
                        findings.append({"type": "sensitive_key", "path": child_path, "count": 1})
                    out[str(key)] = "[REDACTED:sensitive_key]"
                else:
                    out[str(key)] = walk(child, child_path)
            return out
        if isinstance(item, list):
            return [walk(child, f"{path}[{idx}]") for idx, child in enumerate(item)]
        if isinstance(item, tuple):
            return tuple(walk(child, f"{path}[{idx}]") for idx, child in enumerate(item))
        return item

    return walk(value, "$"), findings


def classify_model_payload(payload: Mapping[str, Any]) -> str:
    raw = _stable_json(payload).lower()
    if any(term in raw for term in ("portfolio", "holding", "position", "p&l", "pnl", "thesis")):
        return "portfolio_private"
    if any(term in raw for term in ("research note", "retrieval", "knowledge base", "memo")):
        return "research_private"
    return "public_market"


def _estimate_model_cost_usd(provider: str, input_tokens: int, max_output_tokens: int) -> float:
    # Conservative estimate only; actual provider usage is recorded separately.
    provider_l = provider.lower()
    if provider_l == "anthropic":
        return ((input_tokens / 1_000_000.0) * 3.0) + ((max_output_tokens / 1_000_000.0) * 15.0)
    return ((input_tokens / 1_000_000.0) * 5.0) + ((max_output_tokens / 1_000_000.0) * 15.0)


def prepare_model_egress(
    *,
    provider: str,
    purpose: str,
    stream_kwargs: Mapping[str, Any],
    actor: Actor | None,
    budget: AgentBudgetState | None = None,
    parent_event_id: str | None = None,
    session_id: str | None = None,
    workflow_run_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """DLP-scan and record an egress manifest before a provider call."""

    sanitized_kwargs, findings = redact_secrets(dict(stream_kwargs))
    sensitivity = classify_model_payload(
        {
            "instructions": sanitized_kwargs.get("instructions") or sanitized_kwargs.get("system"),
            "conversation": sanitized_kwargs.get("input") or sanitized_kwargs.get("messages"),
        }
    )
    provider_egress = "external_allowed_raw_private" if sensitivity != "public_market" else "external_allowed"
    model = sanitized_kwargs.get("model")
    max_output_tokens = int(sanitized_kwargs.get("max_tokens") or sanitized_kwargs.get("max_output_tokens") or 0)
    estimated_input_tokens = estimate_tokens(
        {
            "instructions": sanitized_kwargs.get("instructions") or sanitized_kwargs.get("system"),
            "conversation": sanitized_kwargs.get("input") or sanitized_kwargs.get("messages"),
            "tools": sanitized_kwargs.get("tools"),
        }
    )
    estimated_cost_usd = _estimate_model_cost_usd(provider, estimated_input_tokens, max_output_tokens)
    if budget is not None:
        budget.check_model_call(estimated_input_tokens=estimated_input_tokens, estimated_cost_usd=estimated_cost_usd)

    context = execution_context_for_actor(actor)
    decision_id = _decision_id("egress")
    manifest = {
        "policy_decision_id": decision_id,
        "provider": provider,
        "model": model,
        "purpose": purpose,
        "data_sensitivity": sensitivity,
        "provider_egress": provider_egress,
        "redaction_policy": REDACTION_POLICY,
        "egress_policy_version": EGRESS_POLICY_VERSION,
        "dlp_findings": findings,
        "estimated_input_tokens": estimated_input_tokens,
        "max_output_tokens": max_output_tokens,
        "estimated_cost_usd": round(estimated_cost_usd, 6),
        "scope": context.scope,
        "session_id": session_id,
        "workflow_run_id": workflow_run_id,
        "parent_event_id": parent_event_id,
        "status": "allowed",
    }
    _record_audit(
        "agent.provider_egress",
        "agent_governance",
        "allowed",
        actor=actor,
        after_summary=manifest,
        metadata={"decision_id": decision_id},
    )
    return sanitized_kwargs, manifest


def evaluate_tool_call(
    tool: ToolExposure,
    *,
    actor: Actor | None,
    raw_args: Mapping[str, Any],
    budget: AgentBudgetState | None = None,
) -> ToolPolicyDecision:
    context = execution_context_for_actor(actor)
    roles = {role.lower() for role in context.roles}
    if "admin" not in roles and context.actor_type != "system":
        raise ToolPolicyDenied("Agent tool calls require the single-admin role")
    if tool.account_scope and tool.account_scope != context.account_id:
        raise ToolPolicyDenied(f"Tool requires account scope {tool.account_scope}")
    if tool.portfolio_scope and tool.portfolio_scope != context.portfolio_id:
        raise ToolPolicyDenied(f"Tool requires portfolio scope {tool.portfolio_scope}")
    arg_account = raw_args.get("account_id")
    arg_portfolio = raw_args.get("portfolio_id")
    if arg_account is not None and str(arg_account) != context.account_id:
        raise ToolPolicyDenied("Requested account_id is outside the delegated scope")
    if arg_portfolio is not None and str(arg_portfolio) != context.portfolio_id:
        raise ToolPolicyDenied("Requested portfolio_id is outside the delegated scope")
    _check_rate_limit(tool, context)
    if budget is not None:
        budget.check_tool_call(tool)
    decision = ToolPolicyDecision(
        decision_id=_decision_id("tool_policy"),
        allowed=True,
        reason="allowed",
        scope=context.scope,
        required_scopes=tuple(tool.required_scopes),
        data_sensitivity=tool.data_sensitivity,
        provider_egress=tool.provider_egress,
        audit_level=tool.audit_level,
    )
    _record_audit(
        "agent.tool_policy",
        "agent_governance",
        "allowed",
        actor=actor,
        object_refs=[{"type": "agent_tool", "id": tool.tool_name}],
        after_summary=tool_governance_meta(tool, decision),
        metadata={"decision_id": decision.decision_id},
        fail_closed=tool.audit_level == "financial_critical",
    )
    return decision


def _check_rate_limit(tool: ToolExposure, context: ExecutionContext) -> None:
    limit_raw = tool.rate_limit.get("limit") if isinstance(tool.rate_limit, Mapping) else None
    window_raw = tool.rate_limit.get("window_s") if isinstance(tool.rate_limit, Mapping) else None
    if limit_raw is None or window_raw is None:
        return
    try:
        limit = int(limit_raw)
        window_s = float(window_raw)
    except (TypeError, ValueError):
        return
    if limit <= 0 or window_s <= 0:
        return
    now = time.monotonic()
    key = (context.actor_id, tool.tool_name)
    with _RATE_LOCK:
        q = _RATE_WINDOWS[key]
        while q and now - q[0] > window_s:
            q.popleft()
        if len(q) >= limit:
            raise ToolPolicyDenied(
                f"Rate limit exceeded for {tool.tool_name} ({tool.rate_limit.get('label') or limit})"
            )
        q.append(now)


def tool_governance_meta(tool: ToolExposure, decision: ToolPolicyDecision | None = None) -> dict[str, Any]:
    meta = {
        "required_scopes": list(tool.required_scopes),
        "account_scope": tool.account_scope,
        "portfolio_scope": tool.portfolio_scope,
        "data_sensitivity": tool.data_sensitivity,
        "provider_egress": tool.provider_egress,
        "timeout_s": tool.timeout_s,
        "retry_policy": dict(tool.retry_policy),
        "token_budget": tool.token_budget,
        "cost_budget_usd": tool.cost_budget_usd,
        "rate_limit": dict(tool.rate_limit),
        "audit_level": tool.audit_level,
        "failure_mode": tool.failure_mode,
    }
    if decision is not None:
        meta.update(
            {
                "policy_decision_id": decision.decision_id,
                "scope": decision.scope,
                "policy_status": "allowed" if decision.allowed else "denied",
            }
        )
    return meta


def validate_tool_output(tool: ToolExposure, payload: Any) -> None:
    if not tool.output_spec.strict:
        return
    if not isinstance(payload, Mapping):
        raise AgentGovernanceError(f"Tool {tool.tool_name} returned non-object output", code="tool_output_validation")
    required = tool.output_spec.schema.get("required") if isinstance(tool.output_spec.schema, Mapping) else None
    for key in required or ():
        if key not in payload:
            raise AgentGovernanceError(
                f"Tool {tool.tool_name} output missing required field '{key}'",
                code="tool_output_validation",
            )


def run_with_timeout(fn: Callable[[], Any], *, timeout_s: float) -> Any:
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn)
    try:
        return future.result(timeout=max(0.001, timeout_s))
    except FutureTimeoutError as exc:
        future.cancel()
        raise ToolTimeoutError(f"Tool timed out after {timeout_s:.1f}s") from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def should_retry_tool_error(exc: Exception) -> bool:
    if isinstance(exc, (AgentBudgetExceeded, AgentDLPError, ToolPolicyDenied)):
        return False
    if isinstance(exc, ToolTimeoutError):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in (408, 409, 425, 429, 500, 502, 503, 504, 529):
        return True
    lowered = str(exc).lower()
    return any(term in lowered for term in ("timeout", "temporar", "overloaded", "rate_limit", "connection reset"))


def blocked_tool_payload(
    tool_name: str, exc: Exception, *, status: str = "blocked", meta: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    code = getattr(exc, "code", exc.__class__.__name__)
    payload = {
        "error": str(exc) or exc.__class__.__name__,
        "type": code,
        "_meta": {
            "tool": tool_name,
            "status": status,
            **dict(meta or {}),
        },
    }
    return payload


def _record_audit(
    action_name: str,
    action_category: str,
    status: str,
    *,
    actor: Actor | None = None,
    object_refs: list[dict[str, Any]] | None = None,
    after_summary: Any | None = None,
    metadata: Any | None = None,
    error: str | None = None,
    fail_closed: bool = False,
) -> None:
    try:
        from api.audit import emit_audit_event

        emit_audit_event(
            action_name,
            action_category,
            status,
            actor=actor,
            object_refs=object_refs,
            after_summary=after_summary,
            metadata=metadata,
            error=error,
            fail_closed=fail_closed,
            criticality="financial_critical" if fail_closed else "operational",
            producer_name="agent_governance",
            producer_version=EGRESS_POLICY_VERSION,
            redaction_policy=REDACTION_POLICY,
        )
    except Exception:
        if fail_closed:
            raise
