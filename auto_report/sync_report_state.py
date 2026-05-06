#!/usr/bin/env python3
"""Post GitHub Actions report outputs to the live app report-sync API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

from auto_report.recommendations import stable_hash

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _output_dir(report_type: str) -> Path:
    if report_type == "daily":
        return PROJECT_ROOT / "auto_report" / "outputs" / "daily"
    return PROJECT_ROOT / "auto_report" / "outputs"


def _source_url() -> str | None:
    server = os.getenv("GITHUB_SERVER_URL")
    repo = os.getenv("GITHUB_REPOSITORY")
    run_id = os.getenv("GITHUB_RUN_ID")
    if server and repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return None


def build_payload(report_type: str, output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or _output_dir(report_type)
    bundle_name = "daily_bundle.json" if report_type == "daily" else "weekly_bundle.json"

    report_md = _read_text(output_dir / "report.md") or ""
    commentary_md = _read_text(output_dir / "commentary.md") or ""
    recommendations_md = _read_text(output_dir / "recommendations.md") or ""
    recommendations = _read_json(output_dir / "recommendations.json")
    summary = _read_json(output_dir / "summary.json")
    bundle = _read_json(output_dir / bundle_name)
    report_metadata = _read_json(output_dir / "report_metadata.json")
    as_of = recommendations.get("as_of") or summary.get("as_of") or summary.get("date")
    if not as_of:
        raise RuntimeError(f"Could not determine as_of from {output_dir}")

    report_id = f"{report_type}:{as_of}"
    artifact_paths = {
        "report_md": str(output_dir / "report.md"),
        "commentary_md": str(output_dir / "commentary.md"),
        "recommendations_md": str(output_dir / "recommendations.md"),
        "recommendations_json": str(output_dir / "recommendations.json"),
        "summary_json": str(output_dir / "summary.json"),
        "bundle_json": str(output_dir / bundle_name),
        "report_metadata_json": str(output_dir / "report_metadata.json"),
    }
    return {
        "report_id": report_id,
        "as_of": as_of,
        "report_md": report_md,
        "commentary_md": commentary_md,
        "recommendations_md": recommendations_md,
        "recommendations": recommendations,
        "summary": summary,
        "bundle": bundle,
        "artifact_paths": artifact_paths,
        "metadata": {
            "source": "github_actions",
            "github_run_id": os.getenv("GITHUB_RUN_ID"),
            "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "github_workflow": os.getenv("GITHUB_WORKFLOW"),
            "github_sha": os.getenv("GITHUB_SHA"),
            "github_ref": os.getenv("GITHUB_REF"),
            "source_url": _source_url(),
            "issue_url": report_metadata.get("issue_url"),
        },
        "report_hash": stable_hash(report_md),
        "input_hash": stable_hash({"summary": summary, "bundle": bundle, "recommendations": recommendations}),
    }


def _response_error_detail(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = response.text
    if isinstance(body, dict) and "detail" in body:
        body = body["detail"]
    if not isinstance(body, str):
        body = json.dumps(body, default=str)
    return body[:2000]


def sync_payload(report_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    api_url = (os.getenv("TALISMAN_API_URL") or "").strip().rstrip("/")
    sync_secret = (os.getenv("REPORT_SYNC_SECRET") or "").strip()
    if not api_url:
        raise RuntimeError("TALISMAN_API_URL is required for report sync.")
    if not sync_secret:
        raise RuntimeError("REPORT_SYNC_SECRET is required for report sync.")

    headers = {
        "Content-Type": "application/json",
        "X-Report-Sync-Secret": sync_secret,
        "X-Request-Schema-Name": "post:/api/v1/report-sync/{report_type}",
        "X-Request-Schema-Version": "1",
    }
    proxy_secret = (os.getenv("API_PROXY_SECRET") or "").strip()
    if proxy_secret:
        headers["X-Api-Proxy-Secret"] = proxy_secret

    response = requests.post(
        f"{api_url}/api/v1/report-sync/{report_type}",
        headers=headers,
        json=payload,
        timeout=90,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = _response_error_detail(response)
        raise RuntimeError(f"Report sync API returned {response.status_code}: {detail}") from exc
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Report sync API returned a non-object response.")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync report outputs to the live app state.")
    parser.add_argument("report_type", choices=("daily", "weekly"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = build_payload(args.report_type, args.output_dir)
    result = sync_payload(args.report_type, payload)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Report sync failed: {exc}", file=sys.stderr)
        raise
