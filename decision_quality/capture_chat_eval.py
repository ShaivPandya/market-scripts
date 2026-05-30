"""Export a saved Stan chat turn into a draft decision-quality chat eval case."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_DIR = ROOT / "docs" / "decision_quality_chat_evals" / "cases"

SECRET_PATTERNS = (
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b(?:sk|pk|AIza)[A-Za-z0-9_\-]{16,}\b"),
)


def _redact_text(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact(item) for key, item in value.items()}
    return value


def _turn_pairs(transcript: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pending_user: dict[str, Any] | None = None
    for message in transcript:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            pending_user = message
        elif role == "assistant" and pending_user is not None:
            pairs.append((pending_user, message))
            pending_user = None
    return pairs


def build_case(
    *,
    session_id: str,
    turn_index: int,
    failure_tags: list[str],
) -> dict[str, Any]:
    from api import memory_db

    session = memory_db.get_session(session_id)
    if not session:
        raise ValueError(f"Unknown memory session: {session_id}")
    transcript = session.get("transcript")
    if not isinstance(transcript, list):
        raise ValueError(f"Session {session_id} has no transcript")
    pairs = _turn_pairs(transcript)
    if turn_index < 0 or turn_index >= len(pairs):
        raise ValueError(f"turn_index {turn_index} out of range; session has {len(pairs)} user/assistant turns")

    user_msg, assistant_msg = pairs[turn_index]
    user_text = str(user_msg.get("content") or "")
    assistant_text = str(assistant_msg.get("content") or "")
    tool_calls = assistant_msg.get("toolCalls") or assistant_msg.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        tool_calls = []
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    case_id = f"captured_{session_id}_{turn_index}_{timestamp}"
    return _redact(
        {
            "id": case_id,
            "status": "draft",
            "as_of_date": datetime.now(UTC).date().isoformat(),
            "user_message": user_text,
            "source_session_id": session_id,
            "source_turn_index": turn_index,
            "failure_tags": failure_tags,
            "bad_answer": assistant_text,
            "observed_tool_calls": tool_calls,
            "input_refs": [],
            "mock_tools": {},
            "expected_tool_names": [],
            "required_points": [],
            "required_decision_quality_dimensions": [
                "simple_thesis",
                "mispricing",
                "catalyst_or_reason_now",
                "evidence_for",
                "evidence_against",
                "price_action",
                "invalidation",
                "missing_inputs",
                "confidence_sizing",
                "trade_after_trade",
            ],
            "forbidden_patterns": ["could be a good buy", "depends on your risk tolerance"],
            "expected_stance": {"label": "human_to_fill", "any_terms": [], "forbidden_terms": []},
            "judge_min_score": 16,
            "human_notes": "Draft captured from a real chat failure. Fill required_points, expected_tool_names, mock_tools, and input_refs before moving to review.",
        }
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a saved Stan chat turn as a draft chat eval case.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--turn-index", type=int, required=True)
    parser.add_argument("--failure-tags", default="", help="Comma-separated failure tags.")
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tags = [tag.strip() for tag in args.failure_tags.split(",") if tag.strip()]
    case = build_case(session_id=args.session_id, turn_index=args.turn_index, failure_tags=tags)
    output_path = Path(args.output) if args.output else DEFAULT_CASES_DIR / f"{case['id']}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(case, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote draft decision-quality chat eval case: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
