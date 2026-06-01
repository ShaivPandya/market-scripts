"""Label contracts and provenance helpers for DQ synthesis supervised training."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any

from decision_quality.actions import ACTIONABLE_ACTIONS, normalize_action
from decision_quality.eval_corpus import (
    TRAINING_EXPORT_STATUSES,
    actionability_stance,
    case_corpus_tags,
    case_failure_tags,
    case_failure_type,
)
from decision_quality.opportunity_candidate import GRADUATE_ACTION

SUPERVISED_SCHEMA_VERSION = 1

SUPERVISED_TARGETS = frozenset(
    {
        "triage_next_action",
        "should_graduate",
        "missing_input_tags",
        "synthesis_stance",
    }
)

TRIAGE_ACTIONS = frozenset(
    {
        "watch",
        "research",
        "avoid",
        "do_nothing",
        GRADUATE_ACTION,
    }
)

SYNTHESIS_STANCES = frozenset(
    {
        "watch_only",
        "research_only",
        "actionable",
        "avoid",
        "do_nothing",
        "defer",
        "hold",
        "graduate",
        "unknown",
    }
)

MISSING_INPUT_TAGS = frozenset(
    {
        "entry",
        "valuation",
        "catalyst",
        "chart",
        "sizing",
        "source",
        "invalidation",
        "crowding",
        "price_confirmation",
        "portfolio",
        "thesis",
        "other",
    }
)

_MISSING_INPUT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "entry": ("entry", "pullback", "extended", "asymmetry", "new long", "new position"),
    "valuation": ("valuation", "multiple", "premium", "cheap", "expensive", "pe ", "p/e"),
    "catalyst": ("catalyst", "reason-now", "reason now", "earnings", "event"),
    "chart": ("chart", "technical", "moving average", "price action", "price confirmation"),
    "sizing": ("sizing", "position size", "portfolio exposure", "weight"),
    "source": ("source", "freshness", "stale", "evidence"),
    "invalidation": ("invalidation", "kill condition", "stop"),
    "crowding": ("crowding", "crowded", "positioning"),
    "price_confirmation": ("price confirmation", "confirm price", "current price"),
    "portfolio": ("portfolio", "exposure", "concentration"),
    "thesis": ("thesis", "variant", "consensus"),
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def normalize_missing_input_tag(text: object) -> str | None:
    blob = str(text or "").strip().lower()
    if not blob:
        return None
    for tag, keywords in _MISSING_INPUT_KEYWORDS.items():
        if any(keyword in blob for keyword in keywords):
            return tag
    return "other"


def normalize_missing_input_tags(values: list[str] | None) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        tag = normalize_missing_input_tag(raw)
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def normalize_synthesis_stance(value: object, *, recommended_action: object = None) -> str:
    text = str(value or "").strip().lower()
    if text in SYNTHESIS_STANCES:
        return text
    mapping = {
        "watch": "watch_only",
        "watch_only": "watch_only",
        "research": "research_only",
        "research_only": "research_only",
        "avoid": "avoid",
        "do_nothing": "do_nothing",
        "defer": "defer",
        "hold": "hold",
        "graduate_to_decision_quality": "graduate",
        GRADUATE_ACTION: "graduate",
    }
    if text in mapping:
        return mapping[text]
    action = normalize_action(recommended_action or text, fallback="")
    if action in ACTIONABLE_ACTIONS:
        return "actionable"
    if action == "watch":
        return "watch_only"
    if action == "research":
        return "research_only"
    if action == "avoid":
        return "avoid"
    if action == "do_nothing":
        return "do_nothing"
    return "unknown"


def normalize_triage_action(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if text in TRIAGE_ACTIONS:
        return text
    action = normalize_action(value, fallback="")
    if action in {"watch", "research", "avoid", "do_nothing"}:
        return action
    if action in ACTIONABLE_ACTIONS:
        return GRADUATE_ACTION
    return None


def infer_should_graduate(*, next_action: str | None, expected_graduation: bool | None = None) -> bool | None:
    if expected_graduation is not None:
        return bool(expected_graduation)
    if next_action is None:
        return None
    return next_action == GRADUATE_ACTION


def split_group_for_case(*, case_id: str, case_data: dict[str, Any]) -> str:
    source_session = case_data.get("source_session_id")
    if source_session:
        return f"session:{source_session}"
    ticker = ""
    screen = case_data.get("screen_context")
    if isinstance(screen, dict) and screen.get("ticker"):
        ticker = str(screen["ticker"]).upper()
    gold = case_data.get("gold_output")
    if isinstance(gold, dict) and gold.get("ticker"):
        ticker = str(gold["ticker"]).upper()
    as_of = str(case_data.get("as_of_date") or "")
    if ticker and as_of:
        return f"{ticker}:{as_of}"
    base = re.sub(r"_\d{4}(_chat)?$", "", case_id)
    base = re.sub(r"_(chat|eval)$", "", base)
    return base or case_id


def assign_split(split_group: str, *, train_ratio: float = 0.7, val_ratio: float = 0.15) -> str:
    digest = hashlib.sha256(split_group.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "validation"
    return "holdout"


def check_split_leakage(rows: list[dict[str, Any]]) -> list[str]:
    """Return leakage violations when one split_group appears in multiple splits."""
    groups: dict[str, set[str]] = {}
    for row in rows:
        group = str(row.get("split_group") or row.get("case_id") or "")
        split = str(row.get("split") or "")
        if not group or not split:
            continue
        groups.setdefault(group, set()).add(split)
    return sorted(group for group, splits in groups.items() if len(splits) > 1)


def build_row_provenance(
    *,
    case_id: str,
    source: str,
    source_path: str,
    case_data: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    return {
        "schema_version": SUPERVISED_SCHEMA_VERSION,
        "row_id": f"{source}:{case_id}",
        "source": source,
        "case_id": case_id,
        "source_path": source_path,
        "split_group": split_group_for_case(case_id=case_id, case_data=case_data),
        "split": split,
        "corpus_tags": case_corpus_tags(case_data),
        "failure_type": case_failure_type(case_data),
        "failure_tags": case_failure_tags(case_data),
        "eval_status": str(case_data.get("status") or "draft"),
        "labeled_at": _now_iso(),
        "label_reviewer": "eval_fixture",
    }


def labels_from_structured_dq_gold(gold: dict[str, Any]) -> dict[str, Any]:
    recommended = gold.get("recommended_action")
    actionability = gold.get("actionability") if isinstance(gold.get("actionability"), dict) else {}
    stance_source = actionability.get("status") or recommended
    missing_inputs = []
    if isinstance(actionability, dict):
        missing_inputs.extend(str(item) for item in actionability.get("missing_inputs") or [])
    price_action = gold.get("price_action_read")
    if isinstance(price_action, dict):
        missing_inputs.extend(str(item) for item in price_action.get("data_needed") or [])
    next_action = normalize_triage_action(recommended or stance_source)
    return {
        "label_next_action": next_action,
        "label_should_graduate": infer_should_graduate(next_action=next_action),
        "label_synthesis_stance": normalize_synthesis_stance(stance_source, recommended_action=recommended),
        "label_missing_input_tags": normalize_missing_input_tags(missing_inputs),
    }


def labels_from_opportunity_candidate_gold(
    gold: dict[str, Any], *, expected_graduation: bool | None = None
) -> dict[str, Any]:
    next_action = normalize_triage_action(gold.get("next_action"))
    return {
        "label_next_action": next_action,
        "label_should_graduate": infer_should_graduate(
            next_action=next_action,
            expected_graduation=expected_graduation,
        ),
        "label_synthesis_stance": normalize_synthesis_stance(next_action),
        "label_missing_input_tags": normalize_missing_input_tags(
            [str(item) for item in gold.get("missing_inputs") or []]
        ),
    }


def labels_from_chat_eval(case_data: dict[str, Any]) -> dict[str, Any]:
    expected_stance = case_data.get("expected_stance")
    stance_label = None
    if isinstance(expected_stance, dict):
        stance_label = expected_stance.get("label")
    context_pack = case_data.get("context_pack_expectations")
    missing_terms: list[str] = []
    if isinstance(context_pack, dict):
        missing_terms.extend(str(item) for item in context_pack.get("required_missing_input_terms") or [])
    next_action = None
    if isinstance(expected_stance, dict):
        terms = [str(item).lower() for item in expected_stance.get("any_terms") or []]
        if any("watch" in term for term in terms):
            next_action = "watch"
        elif any("research" in term for term in terms):
            next_action = "research"
        elif any("avoid" in term for term in terms):
            next_action = "avoid"
    routing = case_data.get("routing_expectations")
    if next_action is None and isinstance(routing, dict) and routing.get("run_opportunity_preflight"):
        next_action = "research"
    return {
        "label_next_action": next_action,
        "label_should_graduate": False if next_action in {"watch", "research", "avoid", "do_nothing"} else None,
        "label_synthesis_stance": normalize_synthesis_stance(stance_label or next_action),
        "label_missing_input_tags": normalize_missing_input_tags(missing_terms),
    }


def extract_labels_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("label_next_action") or row.get("label_synthesis_stance"):
        return {
            "next_action": row.get("label_next_action"),
            "should_graduate": row.get("label_should_graduate"),
            "synthesis_stance": row.get("label_synthesis_stance"),
            "missing_input_tags": row.get("label_missing_input_tags") or [],
        }
    return None


def row_is_training_eligible(row: dict[str, Any], *, statuses: frozenset[str] = TRAINING_EXPORT_STATUSES) -> bool:
    status = str(row.get("eval_status") or "draft")
    if status not in statuses:
        return False
    labels = extract_labels_from_row(row)
    return labels is not None and (
        labels.get("next_action") is not None or labels.get("synthesis_stance") not in {None, "unknown"}
    )
