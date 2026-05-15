#!/usr/bin/env python3
"""Fetch live ontology portfolio positions from the deployed API.

This validates API access and writes a local JSON handoff for report generation.
The report sync step persists finished outputs back into the app.

Usage:
    python -m auto_report.fetch_state

Required env:
    TALISMAN_API_URL      — base URL of the deployed API (no trailing slash)
    API_PROXY_SECRET      — X-Api-Proxy-Secret header value (optional but usually required)
    TALISMAN_API_PASSWORD — password-mode login secret for CI/API automation (optional)

Optional GitHub Actions integration:
    GITHUB_ENV            — when present, TALISMAN_BOOK_SIZE is exported for later steps
    AUTO_REPORT_PORTFOLIO_STATE_PATH
                          — when present, portfolio state is written to this path
"""

from __future__ import annotations

import os
import sys

import requests

from auto_report.report_state import export_portfolio_state_path_for_github_actions, write_portfolio_state


def _build_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    proxy_secret = (os.getenv("API_PROXY_SECRET") or "").strip()
    if proxy_secret:
        headers["X-Api-Proxy-Secret"] = proxy_secret
    return headers


def _schema_headers(schema_name: str) -> dict[str, str]:
    return {
        "X-Request-Schema-Name": schema_name,
        "X-Request-Schema-Version": "1",
    }


def _login_if_needed(session: requests.Session, api_url: str, headers: dict[str, str]) -> None:
    password = (os.getenv("TALISMAN_API_PASSWORD") or "").strip()
    if not password:
        return

    response = session.post(
        f"{api_url}/api/auth/login",
        json={"password": password},
        headers={**headers, **_schema_headers("post:/api/auth/login")},
        timeout=30,
    )
    response.raise_for_status()


def _fetch_portfolio_book_size(session: requests.Session, api_url: str, headers: dict[str, str]) -> float | None:
    response = session.get(
        f"{api_url}/api/portfolio-settings",
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    if not isinstance(data, dict):
        return None

    value = data.get("book_size")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _export_book_size_for_github_actions(book_size: float) -> None:
    github_env = (os.getenv("GITHUB_ENV") or "").strip()
    if not github_env:
        return

    with open(github_env, "a", encoding="utf-8") as fh:
        fh.write(f"TALISMAN_BOOK_SIZE={book_size:.2f}\n")


def fetch_and_seed() -> int:
    api_url = (os.getenv("TALISMAN_API_URL") or "").strip().rstrip("/")
    if not api_url:
        print("ERROR: TALISMAN_API_URL is required.", file=sys.stderr)
        return 1

    headers = _build_headers()
    session = requests.Session()

    _login_if_needed(session, api_url, headers)

    response = session.get(
        f"{api_url}/api/portfolio-positions",
        params={"include_hedges": "true"},
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    if not isinstance(data, dict) or "positions" not in data:
        print("ERROR: Unexpected response shape from /portfolio-positions.", file=sys.stderr)
        return 1

    try:
        book_size = _fetch_portfolio_book_size(session, api_url, headers)
    except requests.RequestException as exc:
        book_size = None
        print(f"WARNING: Could not fetch portfolio book size: {exc}", file=sys.stderr)
    if book_size is not None:
        _export_book_size_for_github_actions(book_size)
        print(f"Fetched portfolio book size from app: ${book_size:,.0f}.")

    all_positions: list[dict] = data["positions"]
    state_path = write_portfolio_state({"positions": all_positions, "book_size": book_size})
    export_portfolio_state_path_for_github_actions(state_path)

    positions = [p for p in all_positions if p.get("role", "position") == "position"]
    hedges = [p for p in all_positions if p.get("role") == "hedge"]

    total = len(positions) + len(hedges)
    print(f"Fetched {len(positions)} position(s) and {len(hedges)} hedge(s) from ontology runtime.")
    print(f"Wrote report portfolio state to {state_path}.")

    if total == 0:
        print("WARNING: No positions returned from API; report generation will still block.", file=sys.stderr)

    return 0


def main(argv: list[str] | None = None) -> int:
    return fetch_and_seed()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.HTTPError as exc:
        print(f"ERROR: API request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
