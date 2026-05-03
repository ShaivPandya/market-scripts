#!/usr/bin/env python3
"""Fetch live portfolio positions from the deployed API and seed the local SQLite DB.

Run before auto_daily_report / auto_weekly_report in GitHub Actions so the local
portfolio.db contains real positions rather than an empty schema-only database.

Usage:
    python -m auto_report.fetch_state

Required env:
    TALISMAN_API_URL   — base URL of the deployed API (no trailing slash)
    API_PROXY_SECRET   — X-Api-Proxy-Secret header value (optional but usually required)
"""

from __future__ import annotations

import os
import sys

import requests


def _build_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    proxy_secret = (os.getenv("API_PROXY_SECRET") or "").strip()
    if proxy_secret:
        headers["X-Api-Proxy-Secret"] = proxy_secret
    return headers


def fetch_and_seed() -> int:
    api_url = (os.getenv("TALISMAN_API_URL") or "").strip().rstrip("/")
    if not api_url:
        print("ERROR: TALISMAN_API_URL is required.", file=sys.stderr)
        return 1

    headers = _build_headers()

    response = requests.get(
        f"{api_url}/api/v1/portfolio-positions",
        params={"include_hedges": "true"},
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    if not isinstance(data, dict) or "positions" not in data:
        print("ERROR: Unexpected response shape from /portfolio-positions.", file=sys.stderr)
        return 1

    all_positions: list[dict] = data["positions"]
    positions = [p for p in all_positions if p.get("role", "position") == "position"]
    hedges = [p for p in all_positions if p.get("role") == "hedge"]

    from portfolio.portfolio_db import save_positions

    save_positions(positions, role="position")
    save_positions(hedges, role="hedge")

    total = len(positions) + len(hedges)
    print(f"Seeded {len(positions)} position(s) and {len(hedges)} hedge(s) into local portfolio.db.")

    if total == 0:
        print("WARNING: No positions returned from API — report will still block.", file=sys.stderr)

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
