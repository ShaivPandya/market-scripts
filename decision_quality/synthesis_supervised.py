"""Supervised baseline loading and inference for DQ synthesis and opportunity triage."""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_quality.opportunity_candidate import GRADUATE_ACTION, OpportunityCandidate
from decision_quality.supervised_labels import (
    MISSING_INPUT_TAGS,
    SYNTHESIS_STANCES,
    TRIAGE_ACTIONS,
    normalize_missing_input_tags,
    normalize_synthesis_stance,
    normalize_triage_action,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.70
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def synthesis_supervised_enabled() -> bool:
    return _env_flag("AGENT_SYNTHESIS_SUPERVISED_ENABLED", default=False)


def synthesis_supervised_shadow_mode() -> bool:
    return _env_flag("AGENT_SYNTHESIS_SUPERVISED_SHADOW_MODE", default=True)


def synthesis_supervised_confidence_threshold() -> float:
    raw = os.environ.get("AGENT_SYNTHESIS_SUPERVISED_CONFIDENCE_THRESHOLD")
    if raw is None:
        return DEFAULT_CONFIDENCE_THRESHOLD
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE_THRESHOLD


def synthesis_supervised_model_path() -> Path | None:
    raw = os.environ.get("AGENT_SYNTHESIS_SUPERVISED_MODEL_PATH", "").strip()
    if not raw:
        return None
    return Path(raw)


def _load_artifact(model_path: Path) -> dict[str, Any]:
    resolved = model_path.resolve()
    key = str(resolved)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

    import joblib

    artifact = joblib.load(resolved)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError(f"Invalid synthesis supervised artifact: {resolved}")

    with _MODEL_LOCK:
        _MODEL_CACHE[key] = artifact
    return artifact


@dataclass(frozen=True)
class SupervisedTriagePrediction:
    next_action: str
    should_graduate: bool
    synthesis_stance: str
    missing_input_tags: list[str]
    confidence: float
    source: str = "supervised"


def featurize_context_row(row: dict[str, Any]) -> str:
    """Build compact text features from a training or runtime context row."""
    screen = row.get("screen_context") if isinstance(row.get("screen_context"), dict) else {}
    context_features = row.get("context_features") if isinstance(row.get("context_features"), dict) else {}
    parts = [
        str(row.get("user_text") or row.get("user_message") or row.get("user_question") or ""),
        f"page={screen.get('page_name') or ''}",
        f"route={screen.get('route') or ''}",
        f"ticker={screen.get('ticker') or ''}",
        f"summary={screen.get('summary') or ''}",
        f"opportunity_type={context_features.get('opportunity_type') or ''}",
        f"context_pack={context_features.get('context_pack_id') or ''}",
        f"pack_complete={context_features.get('context_pack_complete')}",
        f"missing_count={context_features.get('missing_input_count')}",
        f"data_quality={context_features.get('data_quality_tier') or ''}",
        f"failure_type={row.get('failure_type') or ''}",
    ]
    tags = row.get("corpus_tags") or context_features.get("corpus_tags") or []
    if isinstance(tags, list) and tags:
        parts.append("corpus_tags=" + ",".join(str(item) for item in tags))
    return "\n".join(part for part in parts if part and not part.endswith("="))


def build_context_features(context_bundle: dict[str, Any]) -> dict[str, Any]:
    context_pack = context_bundle.get("context_pack") if isinstance(context_bundle.get("context_pack"), dict) else {}
    data_quality = context_bundle.get("data_quality") if isinstance(context_bundle.get("data_quality"), dict) else {}
    missing_inputs = []
    if isinstance(context_pack, dict):
        missing_inputs.extend(str(item) for item in context_pack.get("missing_inputs") or [])
    return {
        "opportunity_type": (
            (context_pack.get("opportunity_types") or [None])[0]
            if isinstance(context_pack.get("opportunity_types"), list)
            else context_pack.get("opportunity_type")
        ),
        "context_pack_id": context_pack.get("pack_id"),
        "context_pack_complete": bool(context_pack.get("is_complete", True)),
        "missing_input_count": len(missing_inputs),
        "missing_input_tags": normalize_missing_input_tags(missing_inputs),
        "data_quality_tier": data_quality.get("critical_data_quality") or data_quality.get("source_health_status"),
    }


def build_runtime_row(*, user_text: str, context_bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_text": user_text,
        "screen_context": context_bundle.get("screen_context")
        if isinstance(context_bundle.get("screen_context"), dict)
        else {},
        "context_features": build_context_features(context_bundle),
    }


def _normalize_prediction(raw: dict[str, Any], *, default_confidence: float) -> SupervisedTriagePrediction:
    next_action = normalize_triage_action(raw.get("next_action")) or "research"
    if next_action not in TRIAGE_ACTIONS:
        next_action = "research"
    stance = normalize_synthesis_stance(raw.get("synthesis_stance") or next_action)
    if stance not in SYNTHESIS_STANCES:
        stance = "unknown"
    missing_tags = [
        tag
        for tag in normalize_missing_input_tags([str(item) for item in raw.get("missing_input_tags") or []])
        if tag in MISSING_INPUT_TAGS
    ]
    confidence = float(raw.get("confidence") or default_confidence)
    return SupervisedTriagePrediction(
        next_action=next_action,
        should_graduate=bool(raw.get("should_graduate"))
        if raw.get("should_graduate") is not None
        else next_action == GRADUATE_ACTION,
        synthesis_stance=stance,
        missing_input_tags=missing_tags,
        confidence=max(0.0, min(1.0, confidence)),
    )


def predict_triage_decision(
    *,
    context_bundle: dict[str, Any],
    user_text: str,
    model_path: Path | None = None,
) -> SupervisedTriagePrediction | None:
    resolved_path = model_path or synthesis_supervised_model_path()
    if resolved_path is None:
        return None
    try:
        artifact = _load_artifact(resolved_path)
    except Exception:
        logger.exception("synthesis_supervised_load_failed path=%s", resolved_path)
        return None

    pipeline = artifact["pipeline"]
    row = build_runtime_row(user_text=user_text, context_bundle=context_bundle)
    features = featurize_context_row(row)
    try:
        prediction = pipeline.predict([features])[0]
    except Exception:
        logger.exception("synthesis_supervised_predict_failed path=%s", resolved_path)
        return None
    if not isinstance(prediction, dict):
        return None
    return _normalize_prediction(prediction, default_confidence=float(artifact.get("default_confidence") or 0.82))


def apply_supervised_triage_overlay(
    *,
    opportunity_candidate: OpportunityCandidate,
    context_bundle: dict[str, Any],
    user_text: str,
    model_path: Path | None = None,
) -> tuple[OpportunityCandidate, dict[str, Any]]:
    """Run supervised triage in shadow or apply mode; gates must still run downstream."""
    meta: dict[str, Any] = {
        "enabled": synthesis_supervised_enabled(),
        "shadow_mode": synthesis_supervised_shadow_mode(),
        "confidence_threshold": synthesis_supervised_confidence_threshold(),
    }
    prediction = predict_triage_decision(
        context_bundle=context_bundle,
        user_text=user_text,
        model_path=model_path,
    )
    if prediction is None:
        meta["skipped"] = True
        meta["skip_reason"] = "model_unavailable"
        return opportunity_candidate, meta

    llm_next_action = opportunity_candidate.next_action
    meta["prediction"] = {
        "next_action": prediction.next_action,
        "should_graduate": prediction.should_graduate,
        "synthesis_stance": prediction.synthesis_stance,
        "missing_input_tags": prediction.missing_input_tags,
        "confidence": prediction.confidence,
        "source": prediction.source,
    }
    meta["shadow_comparison"] = {
        "next_action_match": llm_next_action == prediction.next_action,
        "should_graduate_match": (llm_next_action == GRADUATE_ACTION) == prediction.should_graduate,
        "llm_next_action": llm_next_action,
    }

    if synthesis_supervised_shadow_mode() or not synthesis_supervised_enabled():
        meta["applied"] = False
        meta["applied_source"] = "prompt_only"
        return opportunity_candidate, meta

    if prediction.confidence < synthesis_supervised_confidence_threshold():
        meta["applied"] = False
        meta["applied_source"] = "prompt_low_confidence_supervised"
        meta["fallback_reason"] = "confidence_below_threshold"
        return opportunity_candidate, meta

    updated = opportunity_candidate.model_copy(update={"next_action": prediction.next_action})
    meta["applied"] = True
    meta["applied_source"] = "supervised"
    return updated, meta


def write_model_card(model_dir: Path, *, metrics: dict[str, Any], dataset_manifest: dict[str, Any]) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    card_path = model_dir / "model_card.json"
    card_path.write_text(
        json.dumps(
            {
                "model_type": "synthesis_supervised_baseline",
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
