"""Persistent live-app LLM settings."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from api.postgres import use_postgres_state
from api.postgres_compat import PostgresCompatConnection

DB_PATH = Path(__file__).parent / "app_settings.db"

LLM_PROVIDER_KEY = "llm.provider"
LLM_REASONING_EFFORT_PREFIX = "llm.reasoning_effort"
LLM_GATEWAY_POLICY_KEY = "llm.gateway_policy"
ALLOWED_LLM_PROVIDERS = {"anthropic", "openai", "gemini", "local"}
MODEL_TIERS = {"low", "mid", "high"}
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
DEFAULT_REASONING_EFFORTS = {
    "anthropic": {"low": "high", "mid": "high", "high": "high"},
    "openai": {"low": "medium", "mid": "medium", "high": "medium"},
    "gemini": {"low": "minimal", "mid": "high", "high": "high"},
    "local": {"low": "none", "mid": "none", "high": "none"},
}
LIFECYCLE_STATES = {"draft", "enabled", "deprecated", "disabled"}
DATA_SENSITIVITIES = {
    "public_market",
    "portfolio_private",
    "research_private",
    "account_private",
    "operational_private",
}
DEFAULT_GATEWAY_POLICY = {
    "private_egress_mode": "allow_with_warning",
    "provider_lifecycle": {
        "anthropic": "enabled",
        "openai": "enabled",
        "gemini": "enabled",
        "local": "enabled",
    },
    "model_lifecycle": {},
    "denied_rules": [],
}

_lock = threading.Lock()
_conn: sqlite3.Connection | PostgresCompatConnection | None = None

_CREATE_APP_SETTINGS = """
CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _close_conn() -> None:
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None


def _get_conn(*, probe: bool = True) -> sqlite3.Connection | PostgresCompatConnection:
    global _conn
    if probe and _conn is not None:
        try:
            _conn.execute("SELECT 1")
        except Exception:
            _close_conn()
    if _conn is None:
        with _lock:
            if _conn is None:
                if use_postgres_state():
                    _conn = PostgresCompatConnection()
                else:
                    _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                    _conn.row_factory = sqlite3.Row
                    _init_db(_conn)
    return _conn


def _init_db(conn: sqlite3.Connection | PostgresCompatConnection) -> None:
    conn.execute(_CREATE_APP_SETTINGS)
    conn.commit()


def _row_to_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    return {key: row[key] for key in row.keys()}


def get_setting(key: str) -> dict[str, Any] | None:
    if not use_postgres_state() and not DB_PATH.exists():
        return None
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT key, value, updated_at FROM app_settings WHERE key = ?", (key,)).fetchone()
    return _row_to_dict(row) if row else None


def get_settings(keys: list[str] | tuple[str, ...]) -> dict[str, dict[str, Any]]:
    unique_keys = list(dict.fromkeys(keys))
    if not unique_keys:
        return {}
    if not use_postgres_state() and not DB_PATH.exists():
        return {}

    placeholders = ", ".join("?" for _ in unique_keys)
    sql = f"SELECT key, value, updated_at FROM app_settings WHERE key IN ({placeholders})"
    last_exc: Exception | None = None
    for attempt in range(2):
        conn = _get_conn(probe=False)
        try:
            with _lock:
                rows = conn.execute(sql, tuple(unique_keys)).fetchall()
            return {str(row["key"]): _row_to_dict(row) for row in rows}
        except Exception as exc:
            last_exc = exc
            _close_conn()
            if attempt == 1:
                raise
    if last_exc is not None:
        raise last_exc
    return {}


def set_setting(key: str, value: str) -> dict[str, Any]:
    conn = _get_conn()
    updated_at = _now()
    with _lock:
        conn.execute(
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, value, updated_at),
        )
        conn.commit()
        row = conn.execute("SELECT key, value, updated_at FROM app_settings WHERE key = ?", (key,)).fetchone()
    return _row_to_dict(row)


def default_gateway_policy() -> dict[str, Any]:
    return json.loads(json.dumps(DEFAULT_GATEWAY_POLICY))


def normalize_gateway_policy(value: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(value or {})
    policy = default_gateway_policy()
    private_mode = str(raw.get("private_egress_mode") or policy["private_egress_mode"]).strip().lower()
    if private_mode != "allow_with_warning":
        raise ValueError("private_egress_mode must be 'allow_with_warning'")
    policy["private_egress_mode"] = private_mode

    provider_lifecycle = dict(policy["provider_lifecycle"])
    for provider, state in dict(raw.get("provider_lifecycle") or {}).items():
        normalized_provider = str(provider or "").strip().lower()
        normalized_state = str(state or "").strip().lower()
        if normalized_provider not in ALLOWED_LLM_PROVIDERS:
            raise ValueError("Gateway provider lifecycle contains an unsupported provider")
        if normalized_state not in LIFECYCLE_STATES:
            raise ValueError("Gateway provider lifecycle contains an unsupported lifecycle state")
        provider_lifecycle[normalized_provider] = normalized_state
    policy["provider_lifecycle"] = provider_lifecycle

    model_lifecycle: dict[str, str] = {}
    for model, state in dict(raw.get("model_lifecycle") or {}).items():
        normalized_model = str(model or "").strip()
        normalized_state = str(state or "").strip().lower()
        if not normalized_model:
            raise ValueError("Gateway model lifecycle keys cannot be empty")
        if normalized_state not in LIFECYCLE_STATES:
            raise ValueError("Gateway model lifecycle contains an unsupported lifecycle state")
        model_lifecycle[normalized_model] = normalized_state
    policy["model_lifecycle"] = model_lifecycle

    denied_rules: list[dict[str, str]] = []
    for item in list(raw.get("denied_rules") or []):
        if not isinstance(item, dict):
            raise ValueError("Gateway denied_rules must contain objects")
        provider = str(item.get("provider") or "*").strip().lower()
        model = str(item.get("model") or "*").strip()
        sensitivity = str(item.get("data_sensitivity") or item.get("sensitivity") or "").strip().lower()
        if provider != "*" and provider not in ALLOWED_LLM_PROVIDERS:
            raise ValueError("Gateway denied rule contains an unsupported provider")
        if not model:
            raise ValueError("Gateway denied rule model cannot be empty")
        if sensitivity not in DATA_SENSITIVITIES:
            raise ValueError("Gateway denied rule contains an unsupported data_sensitivity")
        denied_rules.append({"provider": provider, "model": model, "data_sensitivity": sensitivity})
    policy["denied_rules"] = denied_rules
    return policy


def get_gateway_policy_setting(rows: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    row = (rows or {}).get(LLM_GATEWAY_POLICY_KEY) if rows is not None else get_setting(LLM_GATEWAY_POLICY_KEY)
    if not row:
        return default_gateway_policy()
    try:
        raw = json.loads(str(row.get("value") or "{}"))
        if not isinstance(raw, dict):
            return default_gateway_policy()
        return normalize_gateway_policy(raw)
    except (TypeError, json.JSONDecodeError, ValueError):
        return default_gateway_policy()


def set_gateway_policy_setting(policy: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_gateway_policy(policy)
    return set_setting(LLM_GATEWAY_POLICY_KEY, json.dumps(normalized, sort_keys=True))


def get_llm_provider_setting() -> str | None:
    row = get_setting(LLM_PROVIDER_KEY)
    if not row:
        return None
    provider = str(row.get("value") or "").strip().lower()
    return provider if provider in ALLOWED_LLM_PROVIDERS else None


def set_llm_provider_setting(provider: str) -> dict[str, Any]:
    normalized = (provider or "").strip().lower()
    if normalized not in ALLOWED_LLM_PROVIDERS:
        raise ValueError("LLM provider must be 'anthropic', 'openai', 'gemini', or 'local'")
    return set_setting(LLM_PROVIDER_KEY, normalized)


def _reasoning_key(provider: str, tier: str) -> str:
    return f"{LLM_REASONING_EFFORT_PREFIX}.{provider}.{tier}"


def _normalize_reasoning_provider(provider: str) -> str:
    normalized = (provider or "").strip().lower()
    if normalized not in ALLOWED_LLM_PROVIDERS:
        raise ValueError("LLM provider must be 'anthropic', 'openai', 'gemini', or 'local'")
    return normalized


def _normalize_reasoning_tier(tier: str) -> str:
    normalized = (tier or "").strip().lower()
    if normalized not in MODEL_TIERS:
        raise ValueError("model tier must be 'low', 'mid', or 'high'")
    return normalized


def _normalize_reasoning_effort(effort: str) -> str:
    normalized = (effort or "").strip().lower()
    if normalized not in REASONING_EFFORTS:
        raise ValueError("reasoning effort is not supported")
    return normalized


def get_llm_reasoning_effort_setting(provider: str, tier: str) -> str:
    normalized_provider = _normalize_reasoning_provider(provider)
    normalized_tier = _normalize_reasoning_tier(tier)
    default_effort = DEFAULT_REASONING_EFFORTS[normalized_provider][normalized_tier]
    row = get_setting(_reasoning_key(normalized_provider, normalized_tier))
    if not row:
        return default_effort

    effort = str(row.get("value") or "").strip().lower()
    return effort if effort in REASONING_EFFORTS else default_effort


def get_llm_reasoning_effort_settings(provider: str) -> dict[str, str]:
    normalized_provider = _normalize_reasoning_provider(provider)
    return {tier: get_llm_reasoning_effort_setting(normalized_provider, tier) for tier in ("low", "mid", "high")}


def set_llm_reasoning_effort_setting(provider: str, tier: str, effort: str) -> dict[str, Any]:
    normalized_provider = _normalize_reasoning_provider(provider)
    normalized_tier = _normalize_reasoning_tier(tier)
    normalized_effort = _normalize_reasoning_effort(effort)
    return set_setting(_reasoning_key(normalized_provider, normalized_tier), normalized_effort)


def set_llm_reasoning_effort_settings(provider: str, efforts: dict[str, str]) -> dict[str, str]:
    normalized_provider = _normalize_reasoning_provider(provider)
    saved: dict[str, str] = {}
    for tier in ("low", "mid", "high"):
        saved[tier] = set_llm_reasoning_effort_setting(
            normalized_provider,
            tier,
            efforts.get(tier, DEFAULT_REASONING_EFFORTS[normalized_provider][tier]),
        )["value"]
    return saved
