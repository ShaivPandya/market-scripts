from __future__ import annotations

from api.request_schema import (
    collect_api_request_schema_definitions,
    register_api_request_upgrade_adapter,
    schema_headers_for_path,
)


def test_body_bearing_api_request_rejects_missing_schema_name(client):
    resp = client.post(
        "/api/v1/auth/login",
        json={"password": "testpass"},
        headers={"X-Request-Schema-Version": "1"},
    )

    assert resp.status_code == 400
    assert resp.json()["expected_schema_name"] == "post:/api/v1/auth/login"


def test_body_bearing_api_request_accepts_current_schema_headers(client):
    resp = client.post(
        "/api/v1/auth/login",
        json={"password": "testpass"},
        headers=schema_headers_for_path(client.app, "POST", "/api/v1/auth/login"),
    )

    assert resp.status_code == 200


def test_old_api_request_payload_is_upgraded_before_fastapi_validation(client):
    register_api_request_upgrade_adapter(
        "post:/api/v1/auth/login",
        0,
        1,
        lambda payload: {"password": payload["pwd"]},
    )
    headers = {
        "X-Request-Schema-Name": "post:/api/v1/auth/login",
        "X-Request-Schema-Version": "0",
    }

    resp = client.post("/api/v1/auth/login", json={"pwd": "testpass"}, headers=headers)

    assert resp.status_code == 200


def test_api_request_registry_covers_body_routes(client):
    definitions = collect_api_request_schema_definitions(client.app.router.routes)
    names = {definition.schema_name for definition in definitions}

    assert "post:/api/v1/auth/login" in names
    assert "post:/api/v1/ontology/query" in names
    assert "put:/api/v1/portfolio-positions" in names
