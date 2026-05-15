"""Local report state handoff for API-backed GitHub report jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORTFOLIO_STATE_PATH = PROJECT_ROOT / "auto_report" / "outputs" / "portfolio_state.json"
PORTFOLIO_STATE_ENV = "AUTO_REPORT_PORTFOLIO_STATE_PATH"
API_ONLY_ENV = "AUTO_REPORT_API_ONLY"

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}


def portfolio_state_path() -> Path:
    configured = (os.getenv(PORTFOLIO_STATE_ENV) or "").strip()
    return Path(configured) if configured else DEFAULT_PORTFOLIO_STATE_PATH


def env_flag(name: str, *, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in _TRUE_VALUES


def api_only_mode() -> bool:
    return env_flag(API_ONLY_ENV)


def missing_database_url_error(exc: BaseException) -> bool:
    return "DATABASE_URL is required for Postgres-backed state" in str(exc)


def write_portfolio_state(payload: dict[str, Any]) -> Path:
    path = portfolio_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def export_portfolio_state_path_for_github_actions(path: Path) -> None:
    github_env = (os.getenv("GITHUB_ENV") or "").strip()
    if not github_env:
        return
    with open(github_env, "a", encoding="utf-8") as fh:
        fh.write(f"{PORTFOLIO_STATE_ENV}={path}\n")


def load_portfolio_state() -> dict[str, Any] | None:
    path = portfolio_state_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_cached_positions(*, include_hedges: bool = False) -> list[dict[str, Any]] | None:
    state = load_portfolio_state()
    if not state:
        return None
    positions = state.get("positions")
    if not isinstance(positions, list):
        return None

    rows = [row for row in positions if isinstance(row, dict)]
    if include_hedges:
        return rows
    return [row for row in rows if str(row.get("role") or "position").lower() == "position"]


def load_cached_book_size() -> float | None:
    state = load_portfolio_state()
    if not state:
        return None
    raw_book_size = state.get("book_size")
    if raw_book_size is None:
        return None
    try:
        parsed = float(raw_book_size)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
