"""Supervised intent-router baseline model loading and inference."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, cast

from decision_quality.intent_router import (
    INTENT_CLASSES,
    RouteContext,
    RouteDecision,
    _enforce_safety_floor,
)

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def _load_artifact(model_path: Path) -> dict[str, Any]:
    resolved = model_path.resolve()
    key = str(resolved)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cast(dict[str, Any], cached)

    import joblib

    artifact_raw = joblib.load(resolved)
    if not isinstance(artifact_raw, dict) or "pipeline" not in artifact_raw:
        raise ValueError(f"Invalid supervised router artifact: {resolved}")
    artifact = cast(dict[str, Any], artifact_raw)

    with _MODEL_LOCK:
        _MODEL_CACHE[key] = artifact
    return artifact


def featurize_training_row(row: dict[str, Any]) -> str:
    """Build a compact text feature for sklearn training/inference."""
    screen_raw = row.get("screen_context")
    screen = cast(dict[str, Any], screen_raw if isinstance(screen_raw, dict) else {})
    parts = [
        str(row.get("user_text") or ""),
        f"page={screen.get('page_name') or ''}",
        f"route={screen.get('route') or ''}",
        f"ticker={screen.get('ticker') or ''}",
        f"summary={screen.get('summary') or ''}",
    ]
    recent = row.get("recent_session_features") or []
    if isinstance(recent, list):
        for item in recent[-3:]:
            if isinstance(item, dict):
                parts.append(f"{item.get('role')}:{item.get('content')}")
    oc_meta = row.get("opportunity_candidate_metadata")
    if isinstance(oc_meta, dict):
        parts.append(f"oc_trigger={oc_meta.get('trigger') or ''}")
        parts.append(f"oc_type={oc_meta.get('opportunity_type') or ''}")
    return "\n".join(part for part in parts if part)


def extract_label_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve the gold label for one training row."""
    if row.get("label_intent_class"):
        return cast(
            dict[str, Any],
            {
                "intent_class": row.get("label_intent_class"),
                "run_hidden_dq": row.get("label_run_hidden_dq"),
                "run_opportunity_preflight": row.get("label_run_opportunity_preflight"),
                "workflow_name": row.get("label_workflow_name"),
                "tool_names": row.get("label_tool_names") or [],
            },
        )

    routing_expectations = row.get("routing_expectations")
    if isinstance(routing_expectations, dict):
        return {
            "intent_class": routing_expectations.get("intent_class"),
            "run_hidden_dq": routing_expectations.get("run_hidden_dq"),
            "run_opportunity_preflight": routing_expectations.get("run_opportunity_preflight"),
            "workflow_name": routing_expectations.get("workflow_name"),
            "tool_names": routing_expectations.get("required_tool_names") or [],
        }

    applied = row.get("applied_route")
    if isinstance(applied, dict) and applied.get("intent_class"):
        return {
            "intent_class": applied.get("intent_class"),
            "run_hidden_dq": applied.get("run_hidden_dq"),
            "run_opportunity_preflight": applied.get("run_opportunity_preflight"),
            "workflow_name": applied.get("workflow_name"),
            "tool_names": applied.get("tool_names") or [],
        }

    regex_baseline = row.get("regex_baseline")
    if isinstance(regex_baseline, dict) and regex_baseline.get("intent_class"):
        return {
            "intent_class": regex_baseline.get("intent_class"),
            "run_hidden_dq": regex_baseline.get("run_hidden_dq"),
            "run_opportunity_preflight": regex_baseline.get("run_opportunity_preflight"),
            "workflow_name": regex_baseline.get("workflow_name"),
            "tool_names": regex_baseline.get("tool_names") or [],
        }
    return None


def predict_route_decision(
    *,
    context: RouteContext,
    regex_baseline: RouteDecision,
    model_path: Path,
) -> RouteDecision | None:
    artifact = _load_artifact(model_path)
    pipeline = artifact["pipeline"]
    tool_vocab: list[str] = list(artifact.get("tool_vocab") or [])
    workflow_vocab: list[str] = list(artifact.get("workflow_vocab") or [])

    row = {
        "user_text": context.user_text,
        "screen_context": context.screen_context,
        "recent_session_features": context.recent_session_features,
        "opportunity_candidate_metadata": context.opportunity_candidate_metadata,
    }
    features = featurize_training_row(row)
    prediction_raw = pipeline.predict([features])[0]
    if not isinstance(prediction_raw, dict):
        return None
    prediction = cast(dict[str, Any], prediction_raw)

    intent_class = str(prediction.get("intent_class") or "general_research")
    if intent_class not in INTENT_CLASSES:
        intent_class = "general_research"

    predicted_tools = [str(item) for item in prediction.get("tool_names") or [] if str(item).strip()]
    if not predicted_tools:
        predicted_tools = list(regex_baseline.tool_names)

    workflow_name = prediction.get("workflow_name")
    wf_name = str(workflow_name).strip() if isinstance(workflow_name, str) and workflow_name.strip() else None
    if wf_name and workflow_vocab and wf_name not in workflow_vocab:
        wf_name = regex_baseline.workflow_name

    confidence = float(prediction.get("confidence") or artifact.get("default_confidence") or 0.82)
    decision = RouteDecision(
        intent_class=intent_class,
        run_hidden_dq=bool(prediction.get("run_hidden_dq")),
        run_opportunity_preflight=bool(prediction.get("run_opportunity_preflight")),
        workflow_name=wf_name,
        workflow_ticker=regex_baseline.workflow_ticker,
        tool_names=[name for name in predicted_tools if name in set(tool_vocab) or not tool_vocab]
        or list(regex_baseline.tool_names),
        confidence=max(0.0, min(1.0, confidence)),
        source="supervised",
        fallback_reason=None,
        tool_pack=str(prediction.get("tool_pack") or intent_class),
    )
    return _enforce_safety_floor(decision, regex_baseline=regex_baseline, user_text=context.user_text)


def write_model_card(model_dir: Path, *, metrics: dict[str, Any], dataset_manifest: dict[str, Any]) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    card_path = model_dir / "model_card.json"
    card_path.write_text(
        json.dumps(
            {
                "model_type": "intent_router_supervised_baseline",
                "metrics": metrics,
                "dataset_manifest": dataset_manifest,
            },
            indent=2,
            ensure_ascii=True,
            default=str,
        ),
        encoding="utf-8",
    )
    return card_path
