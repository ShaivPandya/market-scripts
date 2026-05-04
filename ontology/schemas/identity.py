from __future__ import annotations

import hashlib
from datetime import datetime


def canonical_ticker(value: object) -> str:
    ticker = str(value or "").strip().upper()
    if not ticker or any(ch.isspace() for ch in ticker) or ":" in ticker:
        raise ValueError("ticker must be non-empty and may not contain whitespace or ':'")
    return ticker


def slug(text: object) -> str:
    value = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text or "").strip())
    value = "_".join(part for part in value.split("_") if part)
    return value or "unknown"


def short_hash(text: object, *, length: int = 10) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()[:length]


def canonical_timestamp_key(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "latest"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        text = parsed.isoformat()
    except ValueError:
        pass
    return slug(text)


def position_id(ticker: object) -> str:
    return f"position:{canonical_ticker(ticker)}"


def asset_id(ticker: object) -> str:
    return f"asset:{canonical_ticker(ticker)}"


def sector_id(name: object) -> str:
    return f"sector:{slug(name)}"


def macro_indicator_id(indicator_key: object) -> str:
    return f"macro_indicator:{slug(indicator_key)}"


def signal_id(source: object, name: object) -> str:
    return f"signal:{slug(source)}:{slug(name)}"


def thesis_id(ticker: object) -> str:
    return f"thesis:{canonical_ticker(ticker)}"


def evaluation_id(ticker: object, evaluated_at: object) -> str:
    return f"evaluation:{canonical_ticker(ticker)}:{canonical_timestamp_key(evaluated_at)}"


def catalyst_id(ticker: object, name: object, description: object) -> str:
    return f"catalyst:{canonical_ticker(ticker)}:{slug(name)}:{short_hash(description)}"


def hedge_position_id(ticker: object) -> str:
    return f"hedge_position:{canonical_ticker(ticker)}"


def kill_condition_id(ticker: object, identifier: object) -> str:
    return f"kill_condition:{canonical_ticker(ticker)}:{slug(identifier)}"


def thesis_claim_id(ticker: object, identifier: object) -> str:
    return f"thesis_claim:{canonical_ticker(ticker)}:{slug(identifier)}"


def action_item_id(identifier: object) -> str:
    return f"action_item:{slug(identifier)}"


def watch_trigger_id(identifier: object) -> str:
    return f"watch_trigger:{slug(identifier)}"


def research_note_id(identifier: object) -> str:
    return f"research_note:{slug(identifier)}"


def approval_id(identifier: object) -> str:
    return f"approval:{slug(identifier)}"


def action_run_id(identifier: object) -> str:
    return f"action_run:{slug(identifier)}"


def action_event_id(identifier: object) -> str:
    return f"action_event:{slug(identifier)}"


def workflow_run_id(identifier: object) -> str:
    return f"workflow_run:{slug(identifier)}"


def workflow_artifact_id(identifier: object) -> str:
    return f"workflow_artifact:{slug(identifier)}"


def recommendation_id(identifier: object) -> str:
    return f"recommendation:{slug(identifier)}"


def report_run_id(identifier: object) -> str:
    return f"report_run:{slug(identifier)}"


def document_artifact_id(kind: object, identifier: object) -> str:
    return f"document_artifact:{slug(kind)}:{slug(identifier)}"
