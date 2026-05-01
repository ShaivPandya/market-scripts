"""Shared utilities for auto report scripts (weekly & daily)."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

from llm_utils import MODEL_HIGH, call_llm_text, extract_citations, extract_text

log = logging.getLogger("auto_report.shared")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_prompt_file(path: Path, name: str) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Required prompt file missing: {path}\nCreate {name} with your content before running."
        )
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Prompt file is empty: {path}\nAdd content to {name} before running.")
    return content


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

from api.serializers import serialize_value  # noqa: E402


def serialize_bundle(raw: dict) -> dict:
    return {k: serialize_value(v) for k, v in raw.items()}


def write_bundle(bundle: dict, path: Path) -> Path:
    """Write a JSON bundle to *path* (full file path, not directory)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    log.info("Bundle written to %s (%d bytes)", path, path.stat().st_size)
    return path


# ---------------------------------------------------------------------------
# LLM meta-text stripping
# ---------------------------------------------------------------------------

_HR_RE = re.compile(r"^\s*([-*_]\s*){3,}\s*$")
_META_START_RES = [
    re.compile(r"^\s*if\s+you\s+(want|would\s+like|want\s+me|need)\b", re.IGNORECASE),
    re.compile(r"^\s*if\s+you['']d\s+like\b", re.IGNORECASE),
    re.compile(r"^\s*let\s+me\s+know\s+if\b", re.IGNORECASE),
    re.compile(r"^\s*i\s+can\s+(also\s+)?\b", re.IGNORECASE),
    re.compile(r"^\s*happy\s+to\b", re.IGNORECASE),
    re.compile(r"^\s*need\s+anything\s+else\b", re.IGNORECASE),
    re.compile(r"^\s*want\s+me\s+to\b", re.IGNORECASE),
]
_LEADING_PREAMBLE_RES = [
    re.compile(r"^\s*let\s+me\s+\w+", re.IGNORECASE),
    re.compile(r"^\s*i['']ll\s+\w+", re.IGNORECASE),
    re.compile(r"^\s*i\s+will\s+\w+", re.IGNORECASE),
]


def strip_llm_meta(report_md: str) -> str:
    """Strip leading preamble and trailing LLM meta-commentary from a generated report."""
    original = (report_md or "").strip()
    if not original:
        return original
    lines = original.splitlines()
    # Strip leading preamble lines (e.g. "Let me search for...", "I'll look...")
    while lines:
        line = lines[0].strip()
        if line and any(rx.match(line) for rx in _LEADING_PREAMBLE_RES):
            lines.pop(0)
        else:
            break
    # Strip leading blank lines left after preamble removal
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and _HR_RE.match(lines[-1]):
        lines.pop()
    lookback = 80
    start_idx = None
    scan_from = max(0, len(lines) - lookback)
    for i in range(scan_from, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        if any(rx.search(line) for rx in _META_START_RES):
            start_idx = i
            if i > 0 and _HR_RE.match(lines[i - 1]):
                start_idx = i - 1
            break
    if start_idx is not None:
        lines = lines[:start_idx]
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and _HR_RE.match(lines[-1]):
            lines.pop()
    cleaned = "\n".join(lines).strip()
    return cleaned or original


# ---------------------------------------------------------------------------
# Slim error helper
# ---------------------------------------------------------------------------


def slim_error(value):
    if not isinstance(value, dict):
        return value
    err = value.get("error")
    if not isinstance(err, str):
        return value
    first = err.strip().splitlines()[0] if err.strip() else "Unknown error"
    return {"error": first}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def call_claude(
    system_msg: str,
    user_msg: str,
    allowed_domains: list[str] | None = None,
    model: str = MODEL_HIGH,
    max_tokens: int = 16384,
) -> tuple[str, list[tuple[str, str]]]:
    """Call the configured LLM provider, optionally with web search.

    Returns (text, citations) where citations is a list of (title, url) tuples.
    """

    def _create_with_retry():
        """Call the configured LLM with automatic retry on rate-limit errors."""
        max_retries = 5
        for attempt in range(max_retries + 1):
            try:
                return call_llm_text(
                    prompt=user_msg,
                    model=model,
                    api_key=None,
                    max_tokens=max_tokens,
                    system=system_msg,
                    allowed_domains=allowed_domains,
                )
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code != 429 and "rate_limit" not in str(exc).lower():
                    raise
                if attempt == max_retries:
                    raise
                retry_after = None
                if hasattr(exc, "response") and exc.response is not None:
                    retry_after = exc.response.headers.get("retry-after")
                if retry_after:
                    wait = float(retry_after)
                else:
                    wait = min(2**attempt * 15, 120)  # 15s, 30s, 60s, 120s, 120s
                log.warning(
                    "Rate-limited (429) on attempt %d/%d — waiting %.0fs before retry",
                    attempt + 1,
                    max_retries + 1,
                    wait,
                )
                time.sleep(wait)

    t0 = time.perf_counter()
    text, citations, response = _create_with_retry()

    search_count = 0
    usage = getattr(response, "usage", None)
    if usage is not None and hasattr(usage, "server_tool_use") and usage.server_tool_use:
        search_count = getattr(usage.server_tool_use, "web_search_requests", 0)

    log.info(
        "LLM call completed in %.2fs (%d input tokens, %d output tokens, %d web searches)",
        time.perf_counter() - t0,
        getattr(usage, "input_tokens", 0) if usage is not None else 0,
        getattr(usage, "output_tokens", 0) if usage is not None else 0,
        search_count,
    )
    return text or extract_text(response), citations or extract_citations(response)


# ---------------------------------------------------------------------------
# GitHub Issue
# ---------------------------------------------------------------------------


def _detect_repo() -> str | None:
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        return repo
    try:
        url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            cwd=str(PROJECT_ROOT),
            text=True,
        ).strip()
        m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
        return m.group(1) if m else None
    except Exception:
        return None


def create_github_issue(title: str, body: str) -> str | None:
    from utils.retry import requests_post

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        log.warning("GITHUB_TOKEN not set — skipping issue creation")
        return None
    repo = _detect_repo()
    if not repo:
        log.warning("Could not detect repo owner/name — skipping issue creation")
        return None

    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    # GitHub body limit is 65536 chars
    if len(body) > 60000:
        body = body[:60000] + "\n\n... (truncated)"

    resp = requests_post(url, headers=headers, json={"title": title, "body": body}, timeout=30)
    if resp.status_code == 201:
        issue_url = resp.json().get("html_url", "")
        log.info("Created GitHub Issue: %s", issue_url)
        return issue_url
    else:
        log.error(
            "GitHub Issue creation failed (%d): %s",
            resp.status_code,
            resp.text[:500],
        )
        return None
