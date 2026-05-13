from __future__ import annotations

from api.request_schema import (
    ApiRequestSchema,
    _schema_name_matches_endpoint,
    collect_api_request_schema_definitions,
    register_api_request_upgrade_adapter,
    schema_headers_for_path,
)


def test_body_bearing_api_request_rejects_missing_schema_name(client):
    resp = client.post(
        "/api/auth/login",
        json={"password": "testpass"},
        headers={"X-Request-Schema-Version": "1"},
    )

    assert resp.status_code == 400
    assert resp.json()["expected_schema_name"] == "post:/api/auth/login"


def test_body_bearing_api_request_accepts_current_schema_headers(client):
    resp = client.post(
        "/api/auth/login",
        json={"password": "testpass"},
        headers=schema_headers_for_path(client.app, "POST", "/api/auth/login"),
    )

    assert resp.status_code == 200


def test_old_api_request_payload_is_upgraded_before_fastapi_validation(client):
    register_api_request_upgrade_adapter(
        "post:/api/auth/login",
        0,
        1,
        lambda payload: {"password": payload["pwd"]},
    )
    headers = {
        "X-Request-Schema-Name": "post:/api/auth/login",
        "X-Request-Schema-Version": "0",
    }

    resp = client.post("/api/auth/login", json={"pwd": "testpass"}, headers=headers)

    assert resp.status_code == 200


def test_percent_encoded_concrete_schema_alias_matches_decoded_route_path():
    expected = ApiRequestSchema(
        "post:/api/approvals/{approval_id}/approve",
        1,
        {},
    )

    assert _schema_name_matches_endpoint(
        "post:/api/approvals/approval%3Arecommendation_abc/approve",
        expected=expected,
        method="POST",
        actual_path="/api/approvals/approval:recommendation_abc/approve",
    )


def test_percent_encoded_concrete_schema_alias_passes_middleware(auth_client):
    resp = auth_client.post(
        "/api/approvals/approval%3Arecommendation_abc/approve",
        json={"note": "Apply"},
        headers={
            "X-Request-Schema-Name": "post:/api/approvals/approval%3Arecommendation_abc/approve",
            "X-Request-Schema-Version": "1",
        },
    )

    assert resp.status_code != 400
    assert resp.json().get("detail") != "Request schema name does not match this endpoint."


def test_api_request_registry_covers_body_routes(client):
    definitions = collect_api_request_schema_definitions(client.app.router.routes)
    names = {definition.schema_name for definition in definitions}

    assert "post:/api/auth/login" in names
    assert "post:/api/ontology/query" in names
    assert "put:/api/portfolio-positions" in names
