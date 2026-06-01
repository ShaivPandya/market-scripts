"""Export finalized recommendation outcomes into structured decision-quality eval cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decision_quality.eval_corpus import (
    STRUCTURED_DQ_DIMENSIONS,
    actionability_stance,
    confidence_bin,
    data_quality_tier,
    infer_process_attribution_tags,
)
from ontology.schemas.identity import course_of_action_id, decision_outcome_id, recommendation_id

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_DIR = ROOT / "docs" / "decision_quality_evals" / "cases"
DEFAULT_INPUTS_DIR = ROOT / "docs" / "decision_quality_evals" / "inputs"

FINALIZED_LABEL_STATUSES = frozenset({"confirmed", "corrected", "rejected"})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "outcome"


def _as_float(value: object, default: float | None = None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _load_parent_row(outcome_row: Mapping[str, Any], reads: Any) -> dict[str, Any] | None:
    source_kind = str(outcome_row.get("source_kind") or "")
    if source_kind == "recommendation":
        rec_key = outcome_row.get("recommendation_id")
        if not rec_key:
            return None
        uid = recommendation_id(str(rec_key))
        row = reads.get(uid)
        return row if isinstance(row, dict) else None
    if source_kind == "course_of_action":
        coa_key = outcome_row.get("course_of_action_id")
        if not coa_key:
            return None
        uid = course_of_action_id(str(coa_key))
        row = reads.get(uid)
        return row if isinstance(row, dict) else None
    return None


def build_as_of_input_snapshot(
    outcome_row: Mapping[str, Any],
    parent_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build an as-of recommendation snapshot with no realized outcome fields."""
    parent = parent_row or {}
    dq = outcome_row.get("decision_quality_snapshot")
    if not isinstance(dq, dict):
        dq = parent.get("decision_quality") if isinstance(parent.get("decision_quality"), dict) else {}
    gate = parent.get("decision_quality_gate") if isinstance(parent.get("decision_quality_gate"), dict) else {}
    return {
        "ticker": outcome_row.get("ticker") or parent.get("ticker"),
        "as_of_date": outcome_row.get("as_of") or parent.get("as_of"),
        "recommendation_action": parent.get("action"),
        "confidence": parent.get("confidence"),
        "source_quality": parent.get("source_quality"),
        "approval_status": parent.get("approval_status"),
        "horizon": outcome_row.get("horizon") or parent.get("horizon"),
        "decision_context": {
            "simple_thesis": dq.get("simple_thesis"),
            "opportunity_type": dq.get("opportunity_type"),
            "recommended_action": dq.get("recommended_action"),
            "actionability_status": (dq.get("actionability") or {}).get("status")
            if isinstance(dq.get("actionability"), dict)
            else None,
            "confidence": dq.get("confidence"),
        },
        "decision_quality_gate": {
            "status": gate.get("status"),
            "final_action": gate.get("final_action"),
        },
        "human_notes": "As-of recommendation snapshot exported from a finalized decision outcome.",
    }


def build_outcome_context(outcome_row: Mapping[str, Any], parent_row: Mapping[str, Any] | None) -> dict[str, Any]:
    """Authoring/grading-only outcome metadata. Must never appear in solver prompts."""
    metrics = outcome_row.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    parent = parent_row or {}
    return {
        "available_as_of_date": False,
        "decision_outcome_id": outcome_row.get("decision_outcome_id"),
        "recommendation_id": outcome_row.get("recommendation_id") or parent.get("recommendation_id"),
        "course_of_action_id": outcome_row.get("course_of_action_id") or parent.get("course_of_action_id"),
        "final_label_status": outcome_row.get("final_label_status"),
        "process_label": outcome_row.get("process_label") or metrics.get("process_label"),
        "metrics": metrics,
        "final_postmortem": outcome_row.get("final_postmortem"),
        "lessons_learned": outcome_row.get("lessons_learned"),
        "reviewed_lesson_tags": outcome_row.get("reviewed_lesson_tags") or [],
        "notes": (
            "Realized outcome metrics and post-mortem text are for grading and calibration only. "
            "They must not be treated as known at the decision date."
        ),
    }


def build_outcome_linkage(
    outcome_row: Mapping[str, Any],
    parent_row: Mapping[str, Any] | None,
    *,
    lesson_tags: list[str] | None = None,
) -> dict[str, Any]:
    parent = parent_row or {}
    dq = outcome_row.get("decision_quality_snapshot")
    if not isinstance(dq, dict):
        dq = parent.get("decision_quality") if isinstance(parent.get("decision_quality"), dict) else {}
    metrics = outcome_row.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    process_label = str(outcome_row.get("process_label") or metrics.get("process_label") or "")
    confidence = _as_float(parent.get("confidence"), _as_float(dq.get("confidence")))
    source_quality = str(parent.get("source_quality") or "")
    attribution_tags = infer_process_attribution_tags(outcome_row, parent_row, lesson_tags=lesson_tags)
    return {
        "decision_outcome_id": outcome_row.get("decision_outcome_id"),
        "recommendation_id": outcome_row.get("recommendation_id") or parent.get("recommendation_id"),
        "course_of_action_id": outcome_row.get("course_of_action_id") or parent.get("course_of_action_id"),
        "source_kind": outcome_row.get("source_kind") or parent.get("source_kind"),
        "process_label": process_label or None,
        "process_attribution_tags": attribution_tags,
        "reviewed_lesson_tags": lesson_tags or [],
        "confidence_bin": confidence_bin(confidence),
        "actionability_stance": actionability_stance(dq),
        "data_quality_tier": data_quality_tier(source_quality),
        "opportunity_type": dq.get("opportunity_type"),
        "final_label_status": outcome_row.get("final_label_status"),
    }


def build_case_from_outcome(
    outcome_row: Mapping[str, Any],
    parent_row: Mapping[str, Any] | None = None,
    *,
    input_snapshot_path: str | None = None,
    input_snapshot_sha256: str | None = None,
    lesson_tags: list[str] | None = None,
    status: str = "draft",
    user_question: str | None = None,
) -> dict[str, Any]:
    """Convert a finalized DecisionOutcome row into a structured eval case draft."""
    final_status = str(outcome_row.get("final_label_status") or "")
    if final_status not in FINALIZED_LABEL_STATUSES:
        raise ValueError(
            f"DecisionOutcome final_label_status must be one of {sorted(FINALIZED_LABEL_STATUSES)}; got {final_status!r}"
        )

    parent = parent_row or {}
    dq = outcome_row.get("decision_quality_snapshot")
    if not isinstance(dq, dict):
        dq = parent.get("decision_quality")
    if not isinstance(dq, dict):
        raise ValueError("DecisionOutcome must include decision_quality_snapshot or linked parent decision_quality")

    ticker = str(outcome_row.get("ticker") or parent.get("ticker") or "unknown").upper()
    as_of = str(outcome_row.get("as_of") or parent.get("as_of") or datetime.now(UTC).date().isoformat())
    outcome_key = str(outcome_row.get("decision_outcome_id") or "outcome")
    case_id = f"outcome_{_slug(ticker)}_{_slug(outcome_key)}_{as_of.replace('-', '')}"

    refs: list[dict[str, Any]] = []
    if input_snapshot_path:
        refs.append(
            {
                "type": "recommendation_snapshot",
                "path": input_snapshot_path,
                "sha256": input_snapshot_sha256,
                "description": "As-of recommendation and decision-quality snapshot without realized outcome fields.",
                "required": True,
            }
        )

    linkage = build_outcome_linkage(outcome_row, parent_row, lesson_tags=lesson_tags)
    question = user_question or (
        f"Using the {ticker} recommendation as of {as_of}, should this have been actionable, watch, research, or do_nothing?"
    )

    return {
        "id": case_id,
        "status": status,
        "decision_type": parent.get("decision_type") or "new_idea",
        "as_of_date": as_of,
        "user_question": question,
        "input_refs": refs,
        "assumptions": [
            "This case was exported from a finalized recommendation outcome.",
            "Solver prompts must use only as-of inputs; realized outcome metrics live in outcome_context.",
        ],
        "gold_output": dq,
        "rubric_scores": {dimension: None for dimension in STRUCTURED_DQ_DIMENSIONS} | {"total": None, "notes": ""},
        "human_notes": str(outcome_row.get("lessons_learned") or outcome_row.get("final_postmortem") or ""),
        "corpus_tags": ["structured_dq", "outcome_calibration"],
        "failure_type": None,
        "tool_pack": None,
        "required_dq_dimensions": list(STRUCTURED_DQ_DIMENSIONS),
        "outcome_linkage": linkage,
        "outcome_context": build_outcome_context(outcome_row, parent_row),
        "promotion_checklist": [
            "Verify input snapshot SHA-256 and as-of fields",
            "Confirm solver dry-run excludes outcome_context and outcome_linkage",
            "Fill rubric_scores after human review",
            "Move to approved only after leakage checks pass",
        ],
    }


def load_outcome_row(decision_outcome_key: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    from ontology.runtime_read_service import OntologyRuntimeReadService

    reads = OntologyRuntimeReadService()
    uid = decision_outcome_key
    if not uid.startswith("decision_outcome:"):
        uid = decision_outcome_id(uid)
    row = reads.get(uid)
    if not row:
        raise ValueError(f"DecisionOutcome not found: {decision_outcome_key}")
    parent = _load_parent_row(row, reads)
    return row, parent


def export_outcome_case(
    decision_outcome_key: str,
    *,
    lesson_tags: list[str] | None = None,
    status: str = "draft",
    cases_dir: Path = DEFAULT_CASES_DIR,
    inputs_dir: Path = DEFAULT_INPUTS_DIR,
    write_input_snapshot: bool = True,
) -> tuple[dict[str, Any], Path, Path | None]:
    outcome_row, parent_row = load_outcome_row(decision_outcome_key)
    input_path: Path | None = None
    input_rel: str | None = None
    input_sha: str | None = None

    if write_input_snapshot:
        ticker = str(outcome_row.get("ticker") or (parent_row or {}).get("ticker") or "unknown").lower()
        as_of = str(outcome_row.get("as_of") or (parent_row or {}).get("as_of") or "unknown")
        input_path = inputs_dir / f"{ticker}_outcome_recommendation_snapshot_{as_of}.json"
        snapshot = build_as_of_input_snapshot(outcome_row, parent_row)
        inputs_dir.mkdir(parents=True, exist_ok=True)
        input_path.write_text(json.dumps(snapshot, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        input_rel = str(input_path.relative_to(ROOT))
        input_sha = _sha256(input_path)

    case = build_case_from_outcome(
        outcome_row,
        parent_row,
        input_snapshot_path=input_rel,
        input_snapshot_sha256=input_sha,
        lesson_tags=lesson_tags,
        status=status,
    )
    cases_dir.mkdir(parents=True, exist_ok=True)
    case_path = cases_dir / f"{case['id']}.json"
    case_path.write_text(json.dumps(case, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return case, case_path, input_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a finalized DecisionOutcome into a draft structured decision-quality eval case."
    )
    parser.add_argument("--decision-outcome-id", required=True, help="DecisionOutcome id or object uid.")
    parser.add_argument(
        "--lesson-tags",
        default="",
        help="Comma-separated reviewed lesson tags (e.g. timing_wrong,risk_missed).",
    )
    parser.add_argument("--status", default="draft", choices=["draft", "review", "approved", "archived"])
    parser.add_argument("--output", default=None, help="Optional case output path.")
    parser.add_argument("--no-input-snapshot", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    lesson_tags = [tag.strip() for tag in args.lesson_tags.split(",") if tag.strip()]
    case, default_path, input_path = export_outcome_case(
        args.decision_outcome_id,
        lesson_tags=lesson_tags or None,
        status=args.status,
        write_input_snapshot=not args.no_input_snapshot,
    )
    output_path = Path(args.output) if args.output else default_path
    if args.output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(case, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote draft structured eval case: {output_path}")
    if input_path is not None:
        print(f"Wrote as-of input snapshot: {input_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
