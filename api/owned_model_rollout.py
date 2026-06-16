"""Owned-model shadow, canary, fallback, and rollback controls (TL-92)."""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_utils import MODEL_MID, PROVIDER_TALISMAN, model_for_tier, resolve_model

logger = logging.getLogger(__name__)

OWNED_MODEL_ROLLOUT_VERSION = "owned_model_rollout_v1"
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
CANDIDATE_PROVIDER = PROVIDER_TALISMAN

DEFAULT_OWNED_MODEL_ROLLOUT: dict[str, Any] = {
    "enabled": False,
    "shadow_enabled": True,
    "canary_enabled": False,
    "canary_percent": 0,
    "min_confidence": DEFAULT_CONFIDENCE_THRESHOLD,
    "approved_task_classes": [
        "agent_turn",
        "synthesis",
        "routing",
        "routing_tool_use",
        "tool_use",
        "structured_output",
    ],
    "approved_candidate_id": None,
    "approved_model_ids": [],
    "candidate_provider": CANDIDATE_PROVIDER,
    "rule_version": OWNED_MODEL_ROLLOUT_VERSION,
}

FALLBACK_REASONS = frozenset(
    {
        "rollout_disabled",
        "kill_switch_active",
        "force_baseline_active",
        "task_class_not_eligible",
        "candidate_not_approved",
        "candidate_lifecycle_disabled",
        "provider_lifecycle_disabled",
        "confidence_below_threshold",
        "unsupported_capability",
        "endpoint_failure",
        "endpoint_timeout",
        "malformed_output",
        "schema_failure",
        "policy_denied",
        "gate_failure",
        "canary_not_selected",
        "candidate_unavailable",
        "model_lifecycle_disabled",
    }
)

_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE_VALUES


def owned_model_rollout_kill_switch() -> bool:
    return _env_flag("AGENT_OWNED_MODEL_ROLLOUT_KILL_SWITCH", default=False)


def owned_model_force_baseline() -> bool:
    return _env_flag("AGENT_OWNED_MODEL_FORCE_BASELINE", default=False)


def owned_model_shadow_mode_override() -> bool | None:
    raw = os.environ.get("AGENT_OWNED_MODEL_SHADOW_MODE")
    if raw is None or not str(raw).strip():
        return None
    return str(raw).strip().lower() not in _FALSE_VALUES


def owned_model_canary_enabled_override() -> bool | None:
    raw = os.environ.get("AGENT_OWNED_MODEL_CANARY_ENABLED")
    if raw is None or not str(raw).strip():
        return None
    return str(raw).strip().lower() not in _FALSE_VALUES


def normalize_owned_model_rollout(value: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(value or {})
    policy = dict(DEFAULT_OWNED_MODEL_ROLLOUT)
    policy["enabled"] = bool(raw.get("enabled", policy["enabled"]))
    policy["shadow_enabled"] = bool(raw.get("shadow_enabled", policy["shadow_enabled"]))
    policy["canary_enabled"] = bool(raw.get("canary_enabled", policy["canary_enabled"]))

    try:
        canary_percent = float(raw.get("canary_percent", policy["canary_percent"]))
    except (TypeError, ValueError):
        canary_percent = float(policy["canary_percent"])
    policy["canary_percent"] = max(0.0, min(100.0, canary_percent))

    try:
        min_confidence = float(raw.get("min_confidence", policy["min_confidence"]))
    except (TypeError, ValueError):
        min_confidence = float(policy["min_confidence"])
    policy["min_confidence"] = max(0.0, min(1.0, min_confidence))

    approved_task_classes: list[str] = []
    for item in list(raw.get("approved_task_classes") or policy["approved_task_classes"]):
        normalized = str(item or "").strip().lower()
        if normalized:
            approved_task_classes.append(normalized)
    if not approved_task_classes:
        raise ValueError("owned_model_rollout.approved_task_classes cannot be empty")
    policy["approved_task_classes"] = approved_task_classes

    candidate_id = raw.get("approved_candidate_id")
    policy["approved_candidate_id"] = str(candidate_id).strip() if candidate_id else None

    approved_model_ids: list[str] = []
    for item in list(raw.get("approved_model_ids") or []):
        normalized = str(item or "").strip()
        if normalized:
            approved_model_ids.append(normalized)
    policy["approved_model_ids"] = approved_model_ids

    candidate_provider = str(raw.get("candidate_provider") or policy["candidate_provider"]).strip().lower()
    if candidate_provider != CANDIDATE_PROVIDER:
        raise ValueError("owned_model_rollout.candidate_provider must be 'talisman'")
    policy["candidate_provider"] = candidate_provider

    rule_version = str(raw.get("rule_version") or policy["rule_version"]).strip()
    if not rule_version:
        raise ValueError("owned_model_rollout.rule_version cannot be empty")
    policy["rule_version"] = rule_version
    return policy


def rollout_policy_from_gateway(gateway_policy: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(gateway_policy or {})
    return normalize_owned_model_rollout(raw.get("owned_model_rollout"))


def _canary_bucket(*, session_id: str, client_turn_id: str | None) -> float:
    key = f"{session_id}:{client_turn_id or 'none'}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 10000) / 100.0


def _lifecycle_state(
    *,
    gateway_policy: dict[str, Any],
    provider: str,
    model: str | None,
) -> str:
    provider_lifecycle = dict(gateway_policy.get("provider_lifecycle") or {})
    model_lifecycle = dict(gateway_policy.get("model_lifecycle") or {})
    if model:
        return str(
            model_lifecycle.get(f"{provider}:{model}")
            or model_lifecycle.get(model)
            or provider_lifecycle.get(provider)
            or "enabled"
        ).lower()
    return str(provider_lifecycle.get(provider) or "enabled").lower()


def _registry_path() -> Path:
    from decision_quality.agent_model_training import DEFAULT_REGISTRY_PATH

    raw = os.environ.get("TALISMAN_MODEL_REGISTRY_PATH", "").strip()
    return Path(raw) if raw else DEFAULT_REGISTRY_PATH


def _load_registry_candidate(candidate_id: str | None) -> dict[str, Any] | None:
    if not candidate_id:
        return None
    try:
        from decision_quality.agent_model_training import load_registry

        registry = load_registry(_registry_path())
        entry = dict((registry.get("candidates") or {}).get(candidate_id) or {})
        if not entry:
            return None
        entry["candidate_id"] = candidate_id
        entry["active_candidate_id"] = registry.get("active_candidate_id")
        return entry
    except Exception:
        logger.exception("owned_model_rollout_registry_load_failed candidate_id=%s", candidate_id)
        return None


def _resolve_candidate_model(*, candidate_entry: dict[str, Any] | None, tier: str) -> str | None:
    if not candidate_entry:
        return None
    lifecycle = str(candidate_entry.get("lifecycle_state") or "").lower()
    if lifecycle not in {"approved"}:
        return None
    try:
        return model_for_tier(tier, CANDIDATE_PROVIDER)
    except Exception:
        return resolve_model(MODEL_MID, CANDIDATE_PROVIDER)


def classify_task_class(*, route_decision: Any | None = None, path: str = "agent_turn") -> str:
    normalized_path = str(path or "agent_turn").strip().lower()
    if normalized_path in {"decision_quality_chat", "portfolio_summary", "workflow", "workflow_synthesis"}:
        return "synthesis"
    if normalized_path in {"opportunity_candidate_chat_structured", "decision_quality_chat_structured"}:
        return "structured_output"
    if route_decision is not None:
        if bool(getattr(route_decision, "run_hidden_dq", False)) or bool(
            getattr(route_decision, "run_opportunity_preflight", False)
        ):
            return "structured_output"
        intent = str(getattr(route_decision, "intent_class", "") or "").strip().lower()
        if intent in {"thesis_review", "opportunity_discovery"}:
            return "structured_output"
        if intent in {"workflow_handoff"}:
            return "routing"
    if normalized_path == "agent_chat":
        return "agent_turn"
    return normalized_path or "agent_turn"


@dataclass(frozen=True)
class RolloutDecision:
    mode: str
    baseline_provider: str
    candidate_provider: str
    applied_provider: str
    task_class: str
    confidence: float
    canary_selected: bool
    candidate_id: str | None
    candidate_model: str | None
    rule_version: str
    fallback_reason: str | None = None

    def to_meta(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "baseline_provider": self.baseline_provider,
            "candidate_provider": self.candidate_provider,
            "applied_provider": self.applied_provider,
            "task_class": self.task_class,
            "confidence": self.confidence,
            "canary_selected": self.canary_selected,
            "candidate_id": self.candidate_id,
            "candidate_model": self.candidate_model,
            "rule_version": self.rule_version,
            "fallback_reason": self.fallback_reason,
            "final_response_source": self.applied_provider,
        }


@dataclass
class RolloutTelemetry:
    enabled: bool = False
    shadow_mode: bool = False
    canary_enabled: bool = False
    kill_switch_active: bool = False
    force_baseline_active: bool = False
    canary_percent: float = 0.0
    min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD
    canary_bucket: float | None = None
    candidate_entry: dict[str, Any] | None = None
    shadow_comparison: dict[str, Any] | None = None
    candidate_attempt: dict[str, Any] | None = None
    baseline_attempt: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_meta(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "enabled": self.enabled,
            "shadow_mode": self.shadow_mode,
            "canary_enabled": self.canary_enabled,
            "kill_switch_active": self.kill_switch_active,
            "force_baseline_active": self.force_baseline_active,
            "canary_percent": self.canary_percent,
            "min_confidence": self.min_confidence,
            "canary_bucket": self.canary_bucket,
            "shadow_comparison": self.shadow_comparison,
            "candidate_attempt": self.candidate_attempt,
            "baseline_attempt": self.baseline_attempt,
        }
        if self.candidate_entry:
            payload["candidate_entry"] = {
                "candidate_id": self.candidate_entry.get("candidate_id"),
                "lifecycle_state": self.candidate_entry.get("lifecycle_state"),
                "artifact_path": self.candidate_entry.get("artifact_path"),
            }
        payload.update(self.extra)
        return payload


def resolve_rollout_decision(
    *,
    task_class: str,
    baseline_provider: str,
    session_id: str,
    client_turn_id: str | None = None,
    confidence: float = 1.0,
    tier: str = MODEL_MID,
    gateway_policy: dict[str, Any] | None = None,
) -> tuple[RolloutDecision, RolloutTelemetry]:
    """Resolve whether this turn should use baseline, shadow, or canary owned-model routing."""
    from api.llm_settings import default_gateway_policy, get_gateway_policy_setting

    policy_gateway = gateway_policy or get_gateway_policy_setting() or default_gateway_policy()
    rollout_policy = rollout_policy_from_gateway(policy_gateway)
    telemetry = RolloutTelemetry(
        enabled=bool(rollout_policy.get("enabled")),
        shadow_mode=bool(rollout_policy.get("shadow_enabled")),
        canary_enabled=bool(rollout_policy.get("canary_enabled")),
        kill_switch_active=owned_model_rollout_kill_switch(),
        force_baseline_active=owned_model_force_baseline(),
        canary_percent=float(rollout_policy.get("canary_percent") or 0.0),
        min_confidence=float(rollout_policy.get("min_confidence") or DEFAULT_CONFIDENCE_THRESHOLD),
    )

    shadow_override = owned_model_shadow_mode_override()
    if shadow_override is not None:
        telemetry.shadow_mode = shadow_override
    canary_override = owned_model_canary_enabled_override()
    if canary_override is not None:
        telemetry.canary_enabled = canary_override

    normalized_task_class = str(task_class or "agent_turn").strip().lower()
    baseline = str(baseline_provider or "").strip().lower()
    candidate_provider = str(rollout_policy.get("candidate_provider") or CANDIDATE_PROVIDER).strip().lower()
    rule_version = str(rollout_policy.get("rule_version") or OWNED_MODEL_ROLLOUT_VERSION)

    def _off(reason: str) -> tuple[RolloutDecision, RolloutTelemetry]:
        telemetry.extra["eligibility_reason"] = reason
        decision = RolloutDecision(
            mode="off",
            baseline_provider=baseline,
            candidate_provider=candidate_provider,
            applied_provider=baseline,
            task_class=normalized_task_class,
            confidence=confidence,
            canary_selected=False,
            candidate_id=None,
            candidate_model=None,
            rule_version=rule_version,
            fallback_reason=reason,
        )
        return decision, telemetry

    if telemetry.kill_switch_active:
        return _off("kill_switch_active")
    if telemetry.force_baseline_active:
        return _off("force_baseline_active")
    if not telemetry.enabled:
        return _off("rollout_disabled")

    approved_task_classes = {str(item).lower() for item in rollout_policy.get("approved_task_classes") or []}
    if normalized_task_class not in approved_task_classes:
        return _off("task_class_not_eligible")

    candidate_id = rollout_policy.get("approved_candidate_id")
    candidate_entry = _load_registry_candidate(str(candidate_id) if candidate_id else None)
    telemetry.candidate_entry = candidate_entry
    if not candidate_entry:
        return _off("candidate_not_approved")
    if str(candidate_entry.get("lifecycle_state") or "").lower() != "approved":
        return _off("candidate_not_approved")

    candidate_model = _resolve_candidate_model(candidate_entry=candidate_entry, tier=tier)
    if not candidate_model:
        return _off("candidate_unavailable")

    approved_model_ids = {str(item) for item in rollout_policy.get("approved_model_ids") or []}
    if approved_model_ids and candidate_model not in approved_model_ids:
        return _off("candidate_not_approved")

    provider_state = _lifecycle_state(gateway_policy=policy_gateway, provider=candidate_provider, model=None)
    if provider_state == "disabled":
        return _off("provider_lifecycle_disabled")
    model_state = _lifecycle_state(
        gateway_policy=policy_gateway,
        provider=candidate_provider,
        model=candidate_model,
    )
    if model_state == "disabled":
        return _off("model_lifecycle_disabled")

    if confidence < telemetry.min_confidence:
        return _off("confidence_below_threshold")

    canary_bucket = _canary_bucket(session_id=session_id, client_turn_id=client_turn_id)
    telemetry.canary_bucket = canary_bucket
    canary_selected = telemetry.canary_enabled and canary_bucket < telemetry.canary_percent

    if canary_selected:
        decision = RolloutDecision(
            mode="canary",
            baseline_provider=baseline,
            candidate_provider=candidate_provider,
            applied_provider=candidate_provider,
            task_class=normalized_task_class,
            confidence=confidence,
            canary_selected=True,
            candidate_id=str(candidate_entry.get("candidate_id") or candidate_id or ""),
            candidate_model=candidate_model,
            rule_version=rule_version,
        )
        return decision, telemetry

    if telemetry.shadow_mode:
        decision = RolloutDecision(
            mode="shadow",
            baseline_provider=baseline,
            candidate_provider=candidate_provider,
            applied_provider=baseline,
            task_class=normalized_task_class,
            confidence=confidence,
            canary_selected=False,
            candidate_id=str(candidate_entry.get("candidate_id") or candidate_id or ""),
            candidate_model=candidate_model,
            rule_version=rule_version,
        )
        return decision, telemetry

    return _off("canary_not_selected")


def map_exception_to_fallback_reason(exc: Exception) -> str:
    if exc.__class__.__name__ == "ModelGatewayDenied":
        return "policy_denied"
    lowered = str(exc).lower()
    if "timeout" in lowered or exc.__class__.__name__ in {"TimeoutError", "ReadTimeout", "ConnectTimeout"}:
        return "endpoint_timeout"
    if "schema" in lowered or "validation" in lowered:
        return "schema_failure"
    if "malformed" in lowered or "parse" in lowered or "json" in lowered:
        return "malformed_output"
    if "unsupported" in lowered or "not supported" in lowered:
        return "unsupported_capability"
    if "503" in lowered or "502" in lowered or "connection" in lowered or "unavailable" in lowered:
        return "endpoint_failure"
    return "endpoint_failure"


def compare_model_outcomes(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_text = str(baseline.get("output_text") or "").strip()
    candidate_text = str(candidate.get("output_text") or "").strip()
    baseline_tools = sorted(str(item) for item in baseline.get("tool_names") or [])
    candidate_tools = sorted(str(item) for item in candidate.get("tool_names") or [])
    return {
        "output_text_match": baseline_text == candidate_text,
        "baseline_output_length": len(baseline_text),
        "candidate_output_length": len(candidate_text),
        "tool_overlap": sorted(set(baseline_tools) & set(candidate_tools)),
        "tool_only_in_baseline": sorted(set(baseline_tools) - set(candidate_tools)),
        "tool_only_in_candidate": sorted(set(candidate_tools) - set(baseline_tools)),
        "baseline_latency_ms": baseline.get("latency_ms"),
        "candidate_latency_ms": candidate.get("latency_ms"),
        "baseline_usage": baseline.get("usage") or {},
        "candidate_usage": candidate.get("usage") or {},
        "baseline_provider": baseline.get("provider"),
        "candidate_provider": candidate.get("provider"),
        "baseline_model": baseline.get("model"),
        "candidate_model": candidate.get("model"),
        "baseline_status": baseline.get("status"),
        "candidate_status": candidate.get("status"),
        "candidate_error": candidate.get("error"),
    }


def record_candidate_attempt(telemetry: RolloutTelemetry, *, attempt: dict[str, Any]) -> None:
    telemetry.candidate_attempt = attempt


def record_baseline_attempt(telemetry: RolloutTelemetry, *, attempt: dict[str, Any]) -> None:
    telemetry.baseline_attempt = attempt


def finalize_shadow_comparison(
    telemetry: RolloutTelemetry,
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    comparison = compare_model_outcomes(baseline=baseline, candidate=candidate)
    telemetry.shadow_comparison = comparison
    return comparison


def apply_canary_fallback(
    decision: RolloutDecision,
    telemetry: RolloutTelemetry,
    *,
    fallback_reason: str,
) -> RolloutDecision:
    telemetry.extra["canary_fallback_reason"] = fallback_reason
    return RolloutDecision(
        mode="canary",
        baseline_provider=decision.baseline_provider,
        candidate_provider=decision.candidate_provider,
        applied_provider=decision.baseline_provider,
        task_class=decision.task_class,
        confidence=decision.confidence,
        canary_selected=decision.canary_selected,
        candidate_id=decision.candidate_id,
        candidate_model=decision.candidate_model,
        rule_version=decision.rule_version,
        fallback_reason=fallback_reason,
    )


def rollout_reporting_summary(
    *,
    rollout_decision: RolloutDecision,
    rollout_meta: dict[str, Any],
    timings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    models = list((timings or {}).get("models") or [])
    tool_rows = list((timings or {}).get("tools") or [])
    fallback_count = 1 if rollout_decision.fallback_reason else 0
    candidate_calls = [row for row in models if str(row.get("provider")) == rollout_decision.candidate_provider]
    baseline_calls = [row for row in models if str(row.get("provider")) == rollout_decision.baseline_provider]
    return {
        "task_class": rollout_decision.task_class,
        "mode": rollout_decision.mode,
        "final_response_source": rollout_decision.applied_provider,
        "fallback_reason": rollout_decision.fallback_reason,
        "fallback_count": fallback_count,
        "candidate_model_calls": len(candidate_calls),
        "baseline_model_calls": len(baseline_calls),
        "tool_call_count": len(tool_rows),
        "shadow_comparison": rollout_meta.get("shadow_comparison"),
        "latency_ms_by_provider": {
            rollout_decision.baseline_provider: [
                row.get("duration_ms") for row in baseline_calls if row.get("duration_ms") is not None
            ],
            rollout_decision.candidate_provider: [
                row.get("duration_ms") for row in candidate_calls if row.get("duration_ms") is not None
            ],
        },
    }
