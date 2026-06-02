"""Optional Sentry observability with privacy-safe event scrubbing."""

from __future__ import annotations

import logging
import os
from typing import Any, Literal, cast
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

_INITIALIZED = False

# Keys never attached to Sentry scopes or breadcrumbs.
_DROP_CONTEXT_KEYS = frozenset(
    {
        "authorization",
        "cookie",
        "cookies",
        "set-cookie",
        "x-api-proxy-secret",
        "x-csrf-token",
        "password",
        "prompt",
        "prompts",
        "messages",
        "payload_json",
        "result_json",
        "payload",
        "result",
        "body",
        "request_body",
        "response_body",
        "content",
        "holdings",
        "positions",
        "portfolio",
        "thesis",
        "csrf",
        "csrf_token",
        "session",
        "__session",
    }
)


def _env_bool(name: str, *, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on", "enabled"}:
        return True
    if raw in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return default


def sentry_dsn() -> str:
    return (os.environ.get("SENTRY_DSN") or "").strip()


def sentry_enabled() -> bool:
    if not _env_bool("SENTRY_ENABLED", default=True):
        return False
    return bool(sentry_dsn())


def _release_identity() -> tuple[str, str]:
    environment = (
        (os.environ.get("SENTRY_ENVIRONMENT") or "").strip()
        or (os.environ.get("TALISMAN_RELEASE_ENVIRONMENT") or "").strip()
        or (os.environ.get("ENVIRONMENT") or "development").strip()
    )
    release = (
        (os.environ.get("SENTRY_RELEASE") or "").strip()
        or (os.environ.get("TALISMAN_RELEASE_GIT_SHA") or "").strip()
        or (os.environ.get("TALISMAN_RELEASE_IMAGE_TAG") or "").strip()
    )
    return environment, release


def _strip_query(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def scrub_event_payload(event: dict[str, Any]) -> dict[str, Any] | None:
    """Recursively scrub a Sentry event dict before transmission."""
    from api.agent_governance import redact_secrets

    redacted, _findings = redact_secrets(event)
    if not isinstance(redacted, dict):
        return None

    request = redacted.get("request")
    if isinstance(request, dict):
        request.pop("data", None)
        request.pop("cookies", None)
        headers = request.get("headers")
        if isinstance(headers, dict):
            request["headers"] = {
                str(key): "[REDACTED]"
                for key in headers
                if str(key).lower()
                in {
                    "authorization",
                    "cookie",
                    "set-cookie",
                    "x-api-proxy-secret",
                    "x-csrf-token",
                }
            }
        if isinstance(request.get("url"), str):
            request["url"] = _strip_query(str(request["url"]))

    breadcrumbs = redacted.get("breadcrumbs")
    if isinstance(breadcrumbs, dict):
        values = breadcrumbs.get("values")
        if isinstance(values, list):
            breadcrumbs["values"] = [_scrub_breadcrumb(item) for item in values if isinstance(item, dict)]

    extra = redacted.get("extra")
    if isinstance(extra, dict):
        redacted["extra"] = _scrub_mapping(extra)

    contexts = redacted.get("contexts")
    if isinstance(contexts, dict):
        redacted["contexts"] = _scrub_mapping(contexts)

    return redacted


def _scrub_breadcrumb(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    data = out.get("data")
    if isinstance(data, dict):
        out["data"] = _scrub_mapping(data)
    if isinstance(out.get("message"), str):
        from api.agent_governance import redact_secrets

        redacted, _ = redact_secrets(out["message"])
        out["message"] = redacted
    return out


def _scrub_mapping(value: dict[str, Any]) -> dict[str, Any]:
    from api.agent_governance import redact_secrets

    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        lowered = str(key).lower()
        if lowered in _DROP_CONTEXT_KEYS or any(part in lowered for part in ("password", "token", "secret", "prompt")):
            cleaned[str(key)] = "[REDACTED]"
            continue
        if isinstance(item, dict):
            cleaned[str(key)] = _scrub_mapping(item)
        elif isinstance(item, list):
            cleaned[str(key)] = item[:5]
        else:
            cleaned[str(key)] = item
    redacted, _ = redact_secrets(cleaned)
    return redacted if isinstance(redacted, dict) else cleaned


def _before_send(event: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any] | None:
    _ = hint
    return scrub_event_payload(event)


def init_sentry(*, component: str = "api") -> bool:
    """Initialize Sentry once per process. No-op when DSN is unset."""
    global _INITIALIZED
    if _INITIALIZED:
        return sentry_enabled()
    if not sentry_enabled():
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration
    except ImportError:
        logger.warning("sentry-sdk is not installed; observability disabled")
        return False

    environment, release = _release_identity()
    integrations: list[Any] = [
        LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
    ]
    if component == "api":
        try:
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            from sentry_sdk.integrations.starlette import StarletteIntegration

            integrations.extend(
                [
                    StarletteIntegration(transaction_style="url"),
                    FastApiIntegration(transaction_style="url"),
                ]
            )
        except ImportError:
            logger.debug("FastAPI Sentry integrations unavailable", exc_info=True)

    try:
        sentry_sdk.init(
            dsn=sentry_dsn(),
            environment=environment,
            release=release or None,
            traces_sample_rate=_env_float("SENTRY_TRACES_SAMPLE_RATE", 0.05),
            profiles_sample_rate=_env_float("SENTRY_PROFILES_SAMPLE_RATE", 0.0),
            send_default_pii=False,
            before_send=cast(Any, _before_send),
            integrations=integrations,
        )
        sentry_sdk.set_tag("component", component)
        _INITIALIZED = True
        logger.info("Sentry initialized component=%s environment=%s", component, environment)
        return True
    except Exception:
        logger.warning("Sentry initialization failed", exc_info=True)
        return False


def set_request_context(
    *,
    request_id: str | None = None,
    method: str | None = None,
    path: str | None = None,
    status_code: int | None = None,
) -> None:
    if not _INITIALIZED:
        return
    try:
        import sentry_sdk

        if request_id:
            sentry_sdk.set_tag("request_id", request_id)
        if method:
            sentry_sdk.set_tag("http.method", method)
        if path:
            sentry_sdk.set_tag("http.path", path)
        if status_code is not None:
            sentry_sdk.set_tag("http.status_code", status_code)
    except Exception:
        logger.debug("Failed to set Sentry request context", exc_info=True)


def capture_message(
    message: str,
    *,
    level: Literal["fatal", "critical", "error", "warning", "info", "debug"] = "error",
    context: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
) -> str | None:
    if not _INITIALIZED:
        return None
    try:
        import sentry_sdk

        from api.agent_governance import redact_secrets

        safe_message, _ = redact_secrets(message)
        with sentry_sdk.push_scope() as scope:
            if tags:
                for key, value in tags.items():
                    if value is not None:
                        scope.set_tag(str(key), str(value))
            if context:
                safe_context, _ = redact_secrets(
                    {k: v for k, v in context.items() if str(k).lower() not in _DROP_CONTEXT_KEYS}
                )
                if isinstance(safe_context, dict):
                    scope.set_context("talisman", safe_context)
            return sentry_sdk.capture_message(str(safe_message), level=level)
    except Exception:
        logger.debug("Sentry capture_message failed", exc_info=True)
        return None


def capture_exception(
    exc: BaseException,
    *,
    context: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
) -> str | None:
    if not _INITIALIZED:
        return None
    try:
        import sentry_sdk

        from api.agent_governance import redact_secrets

        with sentry_sdk.push_scope() as scope:
            if tags:
                for key, value in tags.items():
                    if value is not None:
                        scope.set_tag(str(key), str(value))
            if context:
                safe_context, _ = redact_secrets(
                    {k: v for k, v in context.items() if str(k).lower() not in _DROP_CONTEXT_KEYS}
                )
                if isinstance(safe_context, dict):
                    scope.set_context("talisman", safe_context)
            return sentry_sdk.capture_exception(exc)
    except Exception:
        logger.debug("Sentry capture_exception failed", exc_info=True)
        return None
