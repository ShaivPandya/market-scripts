"""
SHA-34: Post-deploy / post-rollback backend smoke runner.

Authenticates against the live Cloud Run API using the smoke password,
then probes health, auth, workspace, approvals, and the admin deploy-smoke
endpoint.  Retries transient failures and prints redacted results.

Usage (invoked by run-backend-smoke.sh):
    python -m infra.gcp.deploy_smoke \
        --service-url https://talisman-api-xxx.run.app \
        --mode post-deploy \
        [--expected-image-tag abc1234]

Exit codes:
    0  all required checks passed
    1  one or more required checks failed
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Check result model
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 2.0
_REQUEST_TIMEOUT_SECONDS = 15.0


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    required: bool = True


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _client(
    service_url: str,
    proxy_secret: str | None = None,
) -> httpx.Client:
    headers: dict[str, str] = {}
    if proxy_secret:
        headers["X-Api-Proxy-Secret"] = proxy_secret
    return httpx.Client(
        base_url=service_url.rstrip("/"),
        headers=headers,
        timeout=_REQUEST_TIMEOUT_SECONDS,
        follow_redirects=True,
    )


def _retryable_get(
    client: httpx.Client,
    path: str,
    *,
    expected_status: int = 200,
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.get(path)
            if resp.status_code == expected_status:
                return resp
            if resp.status_code >= 500:
                last_exc = RuntimeError(f"HTTP {resp.status_code}")
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
                continue
            return resp
        except httpx.TransportError as exc:
            last_exc = exc
            time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
    raise last_exc or RuntimeError("exhausted retries")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_health(
    client: httpx.Client,
    expected_tag: str | None,
) -> CheckResult:
    try:
        resp = _retryable_get(client, "/api/health")
        if resp.status_code != 200:
            return CheckResult("health", False, f"status={resp.status_code}")
        body = resp.json()
        if body.get("status") != "ok":
            return CheckResult("health", False, f"status_field={body.get('status')}")
        if expected_tag:
            release = body.get("release", {})
            actual_tag = release.get("image_tag", "")
            if actual_tag != expected_tag:
                return CheckResult(
                    "health",
                    False,
                    f"image_tag mismatch: expected={expected_tag} actual={actual_tag}",
                )
        return CheckResult("health", True, "ok")
    except Exception as exc:
        return CheckResult("health", False, f"error: {exc}")


def check_login(
    client: httpx.Client,
    smoke_password: str,
) -> tuple[CheckResult, dict[str, str], dict[str, str]]:
    """Returns (result, cookies_dict, extra_headers)."""
    try:
        username = (os.getenv("AUTH_DEFAULT_USERNAME") or "admin").strip() or "admin"
        resp = client.post(
            "/api/auth/login",
            json={"password": smoke_password, "username": username},
        )
        if resp.status_code != 200:
            return CheckResult("login", False, f"status={resp.status_code}"), {}, {}
        cookies = dict(resp.cookies)
        if "__session" not in cookies:
            return CheckResult("login", False, "no session cookie"), {}, {}
        body = resp.json()
        headers: dict[str, str] = {}
        if isinstance(body, dict) and body.get("csrfToken"):
            headers["X-CSRF-Token"] = str(body["csrfToken"])
        return CheckResult("login", True, "ok"), cookies, headers
    except Exception as exc:
        return CheckResult("login", False, f"error: {exc}"), {}, {}


def check_me(client: httpx.Client, cookies: dict[str, str]) -> CheckResult:
    try:
        resp = client.get("/api/auth/me", cookies=cookies)
        if resp.status_code != 200:
            return CheckResult("me", False, f"status={resp.status_code}")
        body = resp.json()
        if "username" not in body:
            return CheckResult("me", False, "missing username")
        return CheckResult("me", True, f"username={body['username']}")
    except Exception as exc:
        return CheckResult("me", False, f"error: {exc}")


def check_workspace(client: httpx.Client, cookies: dict[str, str]) -> CheckResult:
    try:
        resp = _retryable_get(client, "/api/workspace")
        # Must use cookies for auth
        resp = client.get("/api/workspace", cookies=cookies)
        if resp.status_code != 200:
            return CheckResult("workspace", False, f"status={resp.status_code}")
        body = resp.json()
        expected_keys = {"regime", "portfolio", "pending_approvals"}
        missing = expected_keys - set(body.keys())
        if missing:
            return CheckResult("workspace", False, f"missing keys: {missing}")
        return CheckResult("workspace", True, "ok")
    except Exception as exc:
        return CheckResult("workspace", False, f"error: {exc}")


def check_approvals_summary(
    client: httpx.Client,
    cookies: dict[str, str],
) -> CheckResult:
    try:
        resp = client.get("/api/approvals/summary?limit=1", cookies=cookies)
        if resp.status_code != 200:
            return CheckResult("approvals_summary", False, f"status={resp.status_code}")
        body = resp.json()
        if "count" not in body or "items" not in body:
            return CheckResult("approvals_summary", False, "missing expected keys")
        return CheckResult("approvals_summary", True, "ok")
    except Exception as exc:
        return CheckResult("approvals_summary", False, f"error: {exc}")


def check_admin_deploy_smoke(
    client: httpx.Client,
    cookies: dict[str, str],
) -> CheckResult:
    try:
        resp = client.get("/api/admin/deploy-smoke", cookies=cookies)
        if resp.status_code == 200:
            return CheckResult("admin_deploy_smoke", True, "ok")
        body: dict[str, Any] = {}
        try:
            body = resp.json()
        except Exception:
            pass
        failed = body.get("failed_checks", [])
        return CheckResult(
            "admin_deploy_smoke",
            False,
            f"status={resp.status_code} failed={failed}",
        )
    except Exception as exc:
        return CheckResult("admin_deploy_smoke", False, f"error: {exc}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_smoke(
    service_url: str,
    mode: str,
    expected_image_tag: str | None = None,
    proxy_secret: str | None = None,
    smoke_password: str | None = None,
) -> list[CheckResult]:
    if not smoke_password:
        raise ValueError("AUTH_SMOKE_PASSWORD is required")

    results: list[CheckResult] = []
    client = _client(service_url, proxy_secret)

    # 1. Health
    results.append(check_health(client, expected_image_tag))

    # 2. Login
    login_result, cookies, _csrf_headers = check_login(client, smoke_password)
    results.append(login_result)

    if not login_result.passed:
        # Can't continue authenticated checks
        results.append(CheckResult("me", False, "skipped: login failed"))
        results.append(CheckResult("workspace", False, "skipped: login failed"))
        results.append(CheckResult("approvals_summary", False, "skipped: login failed"))
        results.append(CheckResult("admin_deploy_smoke", False, "skipped: login failed"))
        return results

    # 3. /auth/me
    results.append(check_me(client, cookies))

    # 4. /workspace
    results.append(check_workspace(client, cookies))

    # 5. /approvals/summary
    results.append(check_approvals_summary(client, cookies))

    # 6. /admin/deploy-smoke
    results.append(check_admin_deploy_smoke(client, cookies))

    return results


def print_results(results: list[CheckResult], mode: str) -> bool:
    """Print redacted results; return True if all required checks passed."""
    print(f"\n{'=' * 60}")
    print(f"  Backend smoke — {mode}")
    print(f"{'=' * 60}")
    all_pass = True
    for r in results:
        icon = "✓" if r.passed else "✗"
        req = " [required]" if r.required else ""
        # Redact any secret-looking values from detail
        detail = r.detail
        for pattern in ("password", "secret", "token", "key"):
            if pattern in detail.lower():
                detail = "[REDACTED]"
                break
        print(f"  {icon} {r.name}: {detail}{req}")
        if not r.passed and r.required:
            all_pass = False
    print(f"{'=' * 60}")
    status = "PASSED" if all_pass else "FAILED"
    print(f"  Result: {status}")
    print(f"{'=' * 60}\n")
    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backend deploy smoke runner")
    parser.add_argument("--service-url", required=True, help="Cloud Run service URL")
    parser.add_argument(
        "--mode",
        choices=["post-deploy", "post-rollback"],
        default="post-deploy",
    )
    parser.add_argument(
        "--expected-image-tag",
        default=None,
        help="Expected release image tag (optional)",
    )
    args = parser.parse_args(argv)

    proxy_secret = os.environ.get("API_PROXY_SECRET", "")
    smoke_password = os.environ.get("AUTH_SMOKE_PASSWORD", "")

    if not smoke_password:
        print("ERROR: AUTH_SMOKE_PASSWORD environment variable is required", file=sys.stderr)
        return 1

    results = run_smoke(
        service_url=args.service_url,
        mode=args.mode,
        expected_image_tag=args.expected_image_tag,
        proxy_secret=proxy_secret or None,
        smoke_password=smoke_password,
    )

    passed = print_results(results, args.mode)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
