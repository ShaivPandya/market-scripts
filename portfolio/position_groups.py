from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from math import isnan
from typing import Any

CONVICTION_MIN = 1
CONVICTION_MAX = 5

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_group_name(value: Any) -> str | None:
    """Return canonical group display text, or None for ungrouped rows."""
    if value is None:
        return None
    if isinstance(value, float) and isnan(value):
        return None
    text = unicodedata.normalize("NFC", str(value))
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text or None


def group_key(value: Any) -> str | None:
    name = normalize_group_name(value)
    return name.casefold() if name else None


def normalize_group_conviction(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, float) and isnan(value):
        return None
    try:
        conviction = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Group conviction must be {CONVICTION_MIN}-{CONVICTION_MAX}.") from None
    if conviction < CONVICTION_MIN or conviction > CONVICTION_MAX:
        raise ValueError(f"Group conviction must be {CONVICTION_MIN}-{CONVICTION_MAX}, got {conviction}.")
    return conviction


def normalize_position_group_fields(row: Mapping[str, Any]) -> tuple[str | None, int | None]:
    name = normalize_group_name(row.get("group_name"))
    if not name:
        return None, None
    conviction = normalize_group_conviction(row.get("group_conviction"))
    if conviction is None:
        raise ValueError(f"Group '{name}' requires a group conviction.")
    return name, conviction


def validate_position_groups(rows: Sequence[Mapping[str, Any]]) -> None:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = normalize_group_name(row.get("group_name"))
        key = group_key(name)
        if not key:
            continue
        conviction = normalize_group_conviction(row.get("group_conviction"))
        if conviction is None:
            raise ValueError(f"Group '{name}' requires a group conviction.")
        direction = str(row.get("direction") or "").strip().lower()
        ticker = str(row.get("ticker") or "").strip().upper() or "position"
        existing = groups.setdefault(
            key,
            {"name": name, "conviction": conviction, "direction": direction, "members": []},
        )
        if existing["conviction"] != conviction:
            raise ValueError(
                f"Group '{existing['name']}' has inconsistent group convictions "
                f"({existing['conviction']} and {conviction})."
            )
        if existing["direction"] and direction and existing["direction"] != direction:
            raise ValueError(
                f"Group '{existing['name']}' cannot mix {existing['direction']} and {direction} positions."
            )
        if not existing["direction"] and direction:
            existing["direction"] = direction
        existing["members"].append(ticker)


def canonicalize_position_group_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize rows and apply the first display name used for each group key."""
    groups: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        name = normalize_group_name(item.get("group_name"))
        key = group_key(name)
        if not key:
            item["group_name"] = None
            item["group_conviction"] = None
            out.append(item)
            continue
        conviction = normalize_group_conviction(item.get("group_conviction"))
        if conviction is None:
            raise ValueError(f"Group '{name}' requires a group conviction.")
        direction = str(item.get("direction") or "").strip().lower()
        ticker = str(item.get("ticker") or "").strip().upper() or "position"
        group = groups.setdefault(
            key,
            {"name": name, "conviction": conviction, "direction": direction, "members": []},
        )
        if group["conviction"] != conviction:
            raise ValueError(
                f"Group '{group['name']}' has inconsistent group convictions ({group['conviction']} and {conviction})."
            )
        if group["direction"] and direction and group["direction"] != direction:
            raise ValueError(f"Group '{group['name']}' cannot mix {group['direction']} and {direction} positions.")
        if not group["direction"] and direction:
            group["direction"] = direction
        group["members"].append(ticker)
        item["group_name"] = group["name"]
        item["group_conviction"] = conviction
        out.append(item)
    return out


def group_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = normalize_group_name(row.get("group_name"))
        key = group_key(name)
        if not key:
            continue
        conviction = normalize_group_conviction(row.get("group_conviction"))
        direction = str(row.get("direction") or "").strip().lower()
        ticker = str(row.get("ticker") or "").strip().upper()
        group = groups.setdefault(
            key,
            {
                "group_name": name,
                "group_key": key,
                "group_conviction": conviction,
                "direction": direction,
                "members": [],
            },
        )
        if ticker:
            group["members"].append(ticker)
    return sorted(groups.values(), key=lambda item: str(item["group_name"]).casefold())
