from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import unquote

from fastapi import Request
from fastapi.routing import APIRoute
from pydantic import TypeAdapter
from starlette.datastructures import Headers
from starlette.responses import JSONResponse, Response
from starlette.routing import Match

from ontology.schema_definitions import SCHEMA_KIND_API_REQUEST, SchemaDefinition

REQUEST_SCHEMA_VERSION = 1
REQUEST_SCHEMA_NAME_HEADER = "x-request-schema-name"
REQUEST_SCHEMA_VERSION_HEADER = "x-request-schema-version"

ApiRequestUpgradeAdapter = Callable[[dict[str, Any]], dict[str, Any]]
_API_REQUEST_UPGRADE_ADAPTERS: dict[tuple[str, int, int], ApiRequestUpgradeAdapter] = {}


@dataclass(frozen=True, slots=True)
class ApiRequestSchema:
    schema_name: str
    schema_version: int
    definition: dict[str, Any]


def register_api_request_upgrade_adapter(
    schema_name: str,
    from_version: int,
    to_version: int,
    adapter: ApiRequestUpgradeAdapter,
) -> None:
    _API_REQUEST_UPGRADE_ADAPTERS[(schema_name, int(from_version), int(to_version))] = adapter


def api_request_schema_name(route: APIRoute, method: str) -> str:
    return f"{method.lower()}:{route.path}"


def route_has_body(route: Any) -> bool:
    return isinstance(route, APIRoute) and getattr(route, "body_field", None) is not None


def collect_api_request_schema_definitions(routes: Iterable[Any]) -> list[SchemaDefinition]:
    definitions: list[SchemaDefinition] = []
    for route in routes:
        if not route_has_body(route):
            continue
        assert isinstance(route, APIRoute)
        for method in sorted(m for m in route.methods or [] if m not in {"HEAD", "OPTIONS"}):
            schema = schema_for_route(route, method)
            definitions.append(
                SchemaDefinition(
                    SCHEMA_KIND_API_REQUEST,
                    schema.schema_name,
                    schema.schema_version,
                    schema.definition,
                    compatibility={"route_path": route.path, "method": method},
                )
            )
    return definitions


def schema_for_route(route: APIRoute, method: str) -> ApiRequestSchema:
    return ApiRequestSchema(
        schema_name=api_request_schema_name(route, method),
        schema_version=REQUEST_SCHEMA_VERSION,
        definition=_body_field_schema(route),
    )


def schema_headers_for_path(app: Any, method: str, path: str) -> dict[str, str]:
    schema = schema_for_path(app, method, path)
    if schema is None:
        return {}
    return {
        "X-Request-Schema-Name": schema.schema_name,
        "X-Request-Schema-Version": str(schema.schema_version),
    }


def schema_for_path(app: Any, method: str, path: str) -> ApiRequestSchema | None:
    route = _match_route(app, method, path)
    if route is None or not route_has_body(route):
        return None
    return schema_for_route(route, method)


async def validate_and_upgrade_request_schema(
    app: Any,
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    route = _match_route(app, request.method, request.url.path)
    if route is None or not route_has_body(route):
        return await call_next(request)

    assert isinstance(route, APIRoute)
    expected = schema_for_route(route, request.method)
    supplied_name = _header(request.headers, REQUEST_SCHEMA_NAME_HEADER)
    supplied_version_raw = _header(request.headers, REQUEST_SCHEMA_VERSION_HEADER)
    if not supplied_name or not supplied_version_raw:
        return _bad_schema_response(
            "Missing request schema headers.",
            expected=expected,
            supplied_name=supplied_name,
            supplied_version=supplied_version_raw,
        )
    try:
        supplied_version = int(supplied_version_raw)
    except (TypeError, ValueError):
        return _bad_schema_response(
            "Invalid request schema version.",
            expected=expected,
            supplied_name=supplied_name,
            supplied_version=supplied_version_raw,
        )
    if not _schema_name_matches_endpoint(
        supplied_name,
        expected=expected,
        method=request.method,
        actual_path=request.url.path,
    ):
        return _bad_schema_response(
            "Request schema name does not match this endpoint.",
            expected=expected,
            supplied_name=supplied_name,
            supplied_version=supplied_version,
        )
    if supplied_version > expected.schema_version:
        return _bad_schema_response(
            "Unsupported future request schema version.",
            expected=expected,
            supplied_name=supplied_name,
            supplied_version=supplied_version,
        )
    request.state.request_schema_name = supplied_name
    request.state.request_schema_version = supplied_version

    if supplied_version < expected.schema_version:
        upgraded = await _upgrade_json_body(request, expected.schema_name, supplied_version, expected.schema_version)
        if isinstance(upgraded, Response):
            return upgraded

    return await call_next(request)


def _match_route(app: Any, method: str, path: str) -> APIRoute | None:
    scope = {"type": "http", "method": method.upper(), "path": path, "root_path": ""}
    for route in getattr(app.router, "routes", []):
        matches, _child_scope = route.matches(scope)
        if matches == Match.FULL and isinstance(route, APIRoute):
            return route
    return None


def _schema_name_matches_endpoint(
    supplied_name: str,
    *,
    expected: ApiRequestSchema,
    method: str,
    actual_path: str,
) -> bool:
    if supplied_name == expected.schema_name:
        return True

    prefix = f"{method.lower()}:"
    if not supplied_name.startswith(prefix):
        return False

    supplied_path = supplied_name[len(prefix) :]
    if supplied_path == actual_path:
        return True

    # Browser clients build concrete schema aliases from URL-encoded paths
    # (for example approval%3A...), while Starlette exposes request.url.path
    # decoded. Compare the decoded alias so path-parameter IDs with reserved
    # characters still bind to the matched endpoint.
    return unquote(supplied_path) == actual_path


def _body_field_schema(route: APIRoute) -> dict[str, Any]:
    field = getattr(route, "body_field", None)
    if field is None:
        return {}
    field_type = getattr(field, "type_", None)
    if field_type is None:
        return {}
    try:
        if hasattr(field_type, "model_json_schema"):
            return cast(dict[str, Any], field_type.model_json_schema())
        return cast(dict[str, Any], TypeAdapter(field_type).json_schema())
    except Exception:
        return {
            "title": getattr(field, "name", "RequestBody"),
            "type": str(field_type),
        }


def _header(headers: Headers, name: str) -> str | None:
    value = headers.get(name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bad_schema_response(
    detail: str,
    *,
    expected: ApiRequestSchema,
    supplied_name: str | None,
    supplied_version: str | int | None,
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "detail": detail,
            "expected_schema_name": expected.schema_name,
            "expected_schema_version": expected.schema_version,
            "supplied_schema_name": supplied_name,
            "supplied_schema_version": supplied_version,
        },
    )


async def _upgrade_json_body(
    request: Request,
    schema_name: str,
    from_version: int,
    to_version: int,
) -> Response | None:
    content_type = (request.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    if content_type and content_type != "application/json":
        return _bad_schema_response(
            "Only JSON request bodies can be upgraded across schema versions.",
            expected=ApiRequestSchema(schema_name, to_version, {}),
            supplied_name=schema_name,
            supplied_version=from_version,
        )

    body = await request.body()
    if not body:
        payload: dict[str, Any] = {}
    else:
        try:
            loaded = json.loads(body)
        except json.JSONDecodeError:
            return _bad_schema_response(
                "Invalid JSON request body.",
                expected=ApiRequestSchema(schema_name, to_version, {}),
                supplied_name=schema_name,
                supplied_version=from_version,
            )
        if not isinstance(loaded, dict):
            return _bad_schema_response(
                "Versioned request body must be a JSON object.",
                expected=ApiRequestSchema(schema_name, to_version, {}),
                supplied_name=schema_name,
                supplied_version=from_version,
            )
        payload = loaded

    current_version = int(from_version)
    while current_version < to_version:
        adapter = _API_REQUEST_UPGRADE_ADAPTERS.get((schema_name, current_version, current_version + 1))
        if adapter is None:
            return _bad_schema_response(
                f"Missing request schema upgrade adapter from {current_version} to {current_version + 1}.",
                expected=ApiRequestSchema(schema_name, to_version, {}),
                supplied_name=schema_name,
                supplied_version=from_version,
            )
        payload = adapter(payload)
        current_version += 1

    new_body = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": new_body, "more_body": False}

    request._body = new_body  # noqa: SLF001 - keep Starlette's body cache aligned with the upgraded payload.
    request._receive = receive  # noqa: SLF001 - middleware must replace body before FastAPI validation.
    return None
