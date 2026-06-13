"""Manual eval runner for decision-quality gold cases."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decision_quality.actions import ACTIONABLE_ACTIONS
from decision_quality.eval_corpus import (
    OUTCOME_AUTHORING_FIELDS,
    build_baseline_report,
    case_result_metadata,
    compare_reports,
    filter_cases,
    load_baseline,
    summarize_calibration,
    write_baseline,
)
from decision_quality.gates import apply_decision_quality_gates
from decision_quality.models import DecisionQuality, decision_quality_schema, parse_decision_quality
from llm_utils import MODEL_HIGH, call_llm_text, parse_json_text

ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = ROOT / "docs" / "decision_quality_evals" / "cases"
INPUTS_DIR = ROOT / "docs" / "decision_quality_evals" / "inputs"
PROMPTS_DIR = ROOT / "auto_report" / "prompts"
RUBRIC_PATH = ROOT / "docs" / "decision_quality_evals" / "rubric.md"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "decision_quality_evals"
DEFAULT_BASELINE_PATH = ROOT / "docs" / "decision_quality_evals" / "baselines" / "approved_corpus_baseline.json"

DEFAULT_STATUSES = ("review", "approved")
MODEL_BY_NAME = {"low": "low", "mid": "mid", "high": MODEL_HIGH}
RUBRIC_DIMENSIONS = (
    "thesis_clarity",
    "mispricing",
    "catalyst_reason_now",
    "evidence_quality",
    "disconfirming_evidence",
    "invalidation",
    "price_action_market_behavior",
    "actionability_discipline",
    "confidence_calibration",
    "sizing_risk_context",
)

ANSWER_KEY_FIELDS = {
    "gold_output",
    "rubric_scores",
    "human_notes",
    "future_outcome_context",
    "outcome_context",
    "confirmation_context",
    "exit_context",
    "correct_decision",
    "correct decision",
    "final_action",
    "final action",
    "recommended_action",
    "recommended action",
    "target_size",
    "target size",
    "raw_target_weight",
    "raw target weight",
} | OUTCOME_AUTHORING_FIELDS
ANSWER_KEY_PREFIXES = ("selected_later_", "post_confirmation_")
FUTURE_TEXT_MARKERS = (
    "future outcome",
    "outcome context",
    "later confirmed",
    "later confirmation",
    "later missed",
    "later re-rated",
    "not known on",
    "not available on",
    "by march 9",
    "by march 12",
    "by march 23",
    "by late march",
    "completed the hedge exit",
)


def _load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("AGENT_GOVERNANCE_AUDIT_ENABLED", "false")


@dataclass(frozen=True)
class EvalCase:
    path: Path
    data: dict[str, Any]

    @property
    def case_id(self) -> str:
        return str(self.data.get("id") or self.path.stem)

    @property
    def status(self) -> str:
        return str(self.data.get("status") or "draft")

    @property
    def gold_output(self) -> dict[str, Any]:
        value = self.data.get("gold_output")
        return value if isinstance(value, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_cases(
    *,
    case_selectors: list[str] | None = None,
    statuses: set[str] | None = None,
    corpus_tags: set[str] | None = None,
    failure_type: str | None = None,
    tool_pack: str | None = None,
    cases_dir: Path = CASES_DIR,
) -> list[EvalCase]:
    cases = [EvalCase(path=path, data=_read_json(path)) for path in sorted(cases_dir.glob("*.json"))]
    if case_selectors:
        selected: list[EvalCase] = []
        by_id = {case.case_id: case for case in cases}
        by_stem = {case.path.stem: case for case in cases}
        by_name = {case.path.name: case for case in cases}
        for selector in case_selectors:
            path = Path(selector)
            if path.exists():
                selected.append(EvalCase(path=path, data=_read_json(path)))
                continue
            match = by_id.get(selector) or by_stem.get(selector) or by_name.get(selector)
            if match is None:
                raise ValueError(f"Unknown decision-quality eval case: {selector}")
            selected.append(match)
        cases = selected
    resolved_statuses = statuses if statuses is not None else set(DEFAULT_STATUSES)
    cases = [case for case in cases if case.status in resolved_statuses]
    return filter_cases(
        cases,
        corpus_tags=corpus_tags,
        failure_type=failure_type,
        tool_pack=tool_pack,
    )


def validate_case_input_refs(case: EvalCase, *, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    refs = case.data.get("input_refs")
    if not isinstance(refs, list):
        return ["input_refs must be a list"]
    for idx, ref in enumerate(refs):
        if not isinstance(ref, dict):
            errors.append(f"input_refs[{idx}] must be an object")
            continue
        path_value = ref.get("path")
        expected_sha = ref.get("sha256")
        if path_value is None and expected_sha is None:
            continue
        if not isinstance(path_value, str) or not path_value:
            errors.append(f"input_refs[{idx}] has sha256 but no path")
            continue
        path = root / path_value
        if not path.exists():
            errors.append(f"input_refs[{idx}] path does not exist: {path_value}")
            continue
        if expected_sha is None:
            continue
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            errors.append(
                f"input_refs[{idx}] sha256 mismatch for {path_value}: expected {expected_sha}, got {actual_sha}"
            )
    return errors


def _is_answer_key(key: str) -> bool:
    normalized = key.strip().lower()
    return normalized in ANSWER_KEY_FIELDS or any(normalized.startswith(prefix) for prefix in ANSWER_KEY_PREFIXES)


def _looks_like_future_or_answer_text(value: str) -> bool:
    normalized = " ".join(value.lower().split())
    return any(marker in normalized for marker in FUTURE_TEXT_MARKERS)


def sanitize_case_input(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if _is_answer_key(str(key)):
                continue
            cleaned = sanitize_case_input(item)
            if cleaned not in (None, [], {}):
                sanitized[key] = cleaned
        return sanitized
    if isinstance(value, list):
        sanitized_items = [sanitize_case_input(item) for item in value]
        return [item for item in sanitized_items if item not in (None, [], {})]
    if isinstance(value, str) and _looks_like_future_or_answer_text(value):
        return None
    return value


def _load_input_content(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return _read_json(path)
    return path.read_text(encoding="utf-8")


def build_solver_payload(case: EvalCase, *, root: Path = ROOT) -> dict[str, Any]:
    refs: list[dict[str, Any]] = []
    for ref in case.data.get("input_refs") or []:
        if not isinstance(ref, dict):
            continue
        path_value = ref.get("path")
        sanitized_ref = sanitize_case_input(
            {
                "type": ref.get("type"),
                "description": ref.get("description"),
                "required": ref.get("required"),
            }
        )
        if not isinstance(sanitized_ref, dict):
            continue
        sanitized_ref["path"] = path_value
        if isinstance(path_value, str) and path_value:
            path = root / path_value
            sanitized_ref["content"] = sanitize_case_input(_load_input_content(path))
        refs.append(sanitized_ref)

    return {
        "id": case.case_id,
        "status": case.status,
        "as_of_date": case.data.get("as_of_date"),
        "decision_type": case.data.get("decision_type"),
        "user_question": case.data.get("user_question"),
        "input_refs": refs,
    }


def build_solver_prompt(payload: dict[str, Any]) -> str:
    return (
        "Solve this decision-quality eval case using only the supplied as-of inputs. "
        "Do not use later outcomes, outside knowledge, or facts not present in the payload. "
        "Return only the structured DecisionQuality JSON object, not a wrapper and not markdown. "
        "For recommended_action, classify the broad economic exposure: pressing an existing bearish trade is still "
        "`short`, and buying CDS protection or put options to profit from deterioration is `short`; put any add/press "
        "instruction in sizing_context.sizing_delta instead.\n\n"
        f"Sanitized case payload:\n{json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2)}"
    )


def _decision_quality_contract() -> str:
    return (PROMPTS_DIR / "decision_quality.md").read_text(encoding="utf-8")


def _nonempty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_nonempty(item) for item in value)
    return True


def _check(name: str, passed: bool, message: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "message": message}


def deterministic_score(
    *,
    case: EvalCase,
    candidate: DecisionQuality | None,
    gate: Any,
    parse_errors: list[str] | None = None,
) -> dict[str, Any]:
    gold = DecisionQuality.model_validate(case.gold_output)
    checks: list[dict[str, Any]] = []
    if candidate is None:
        message = "; ".join(parse_errors or []) or "candidate did not parse as DecisionQuality"
        checks.append(_check("schema_valid", False, message))
        return {"score": 0.0, "passed": False, "checks": checks}

    checks.append(_check("schema_valid", True, "candidate parsed as DecisionQuality"))
    checks.append(
        _check(
            "gate_not_downgraded_or_blocked",
            gate.status == "pass" and gate.final_action == candidate.recommended_action,
            f"gate status={gate.status}, final_action={gate.final_action}",
        )
    )
    checks.append(
        _check(
            "recommended_action",
            candidate.recommended_action == gold.recommended_action,
            f"expected {gold.recommended_action}, got {candidate.recommended_action}",
        )
    )
    checks.append(
        _check(
            "actionability_status",
            candidate.actionability.status == gold.actionability.status,
            f"expected {gold.actionability.status}, got {candidate.actionability.status}",
        )
    )
    gold_missing_inputs = gold.actionability.missing_inputs
    candidate_missing_inputs = candidate.actionability.missing_inputs
    missing_inputs_required = bool(gold_missing_inputs) or candidate.actionability.status == "missing_inputs"
    checks.append(
        _check(
            "missing_inputs_alignment",
            not missing_inputs_required or bool(candidate_missing_inputs),
            (
                "candidate must surface missing inputs when gold has missing inputs "
                "or actionability.status=missing_inputs"
            ),
        )
    )
    checks.append(
        _check(
            "conviction_level",
            candidate.conviction.level == gold.conviction.level,
            f"expected {gold.conviction.level}, got {candidate.conviction.level}",
        )
    )
    checks.append(
        _check(
            "confidence_calibration",
            candidate.confidence is not None
            and gold.confidence is not None
            and abs(candidate.confidence - gold.confidence) <= 0.25,
            f"expected within 0.25 of {gold.confidence}, got {candidate.confidence}",
        )
    )
    catalyst = candidate.catalyst_or_reason_now
    checks.append(
        _check(
            "catalyst_complete",
            _nonempty(catalyst.event_or_condition)
            and _nonempty(catalyst.expected_timeframe)
            and _nonempty(catalyst.why_now)
            and _nonempty(catalyst.source_evidence),
            "catalyst_or_reason_now must include event, timing, why_now, and evidence",
        )
    )
    invalidation = candidate.invalidation
    checks.append(
        _check(
            "invalidation_complete",
            _nonempty(invalidation.observable)
            and _nonempty(invalidation.metric_or_event)
            and _nonempty(invalidation.threshold)
            and _nonempty(invalidation.timeframe)
            and _nonempty(invalidation.implication),
            "invalidation must be observable, thresholded, time-bounded, and implication-linked",
        )
    )
    checks.append(
        _check(
            "evidence_against_present",
            bool(candidate.evidence_against),
            "candidate must include disconfirming evidence",
        )
    )
    sizing = candidate.sizing_context
    checks.append(
        _check(
            "sizing_context_complete",
            _nonempty(sizing.starting_size)
            and _nonempty(sizing.add_conditions)
            and _nonempty(sizing.liquidity_constraints)
            and _nonempty(sizing.portfolio_constraints)
            and _nonempty(sizing.sizing_delta.condition),
            "sizing context and sizing_delta.condition must be present",
        )
    )
    trade_after_trade = candidate.trade_after_trade
    checks.append(
        _check(
            "trade_after_trade_complete",
            _nonempty(trade_after_trade.if_right)
            and _nonempty(trade_after_trade.if_wrong)
            and _nonempty(trade_after_trade.next_review_trigger),
            "trade_after_trade must include if_right, if_wrong, and next_review_trigger",
        )
    )
    checks.append(
        _check(
            "actionability_action_consistency",
            (candidate.recommended_action in ACTIONABLE_ACTIONS) == (candidate.actionability.status == "actionable"),
            "actionable actions require actionability.status=actionable; non-actionable actions should not",
        )
    )

    passed_count = sum(1 for check in checks if check["passed"])
    score = round((passed_count / len(checks)) * 100, 2)
    return {"score": score, "passed": all(check["passed"] for check in checks), "checks": checks}


def _judge_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["scores", "total", "leakage_detected", "fatal_issues", "notes"],
        "properties": {
            "scores": {
                "type": "object",
                "additionalProperties": False,
                "required": list(RUBRIC_DIMENSIONS),
                "properties": {
                    dimension: {"type": "integer", "minimum": 0, "maximum": 2} for dimension in RUBRIC_DIMENSIONS
                },
            },
            "total": {"type": "number", "minimum": 0, "maximum": 20},
            "leakage_detected": {"type": "boolean"},
            "fatal_issues": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
        },
    }


def _normalize_judge_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "scores": {},
            "total": 0,
            "leakage_detected": True,
            "fatal_issues": ["judge returned non-JSON"],
            "notes": "",
        }
    scores_raw = value.get("scores")
    scores = scores_raw if isinstance(scores_raw, dict) else {}
    normalized_scores: dict[str, int] = {}
    for dimension in RUBRIC_DIMENSIONS:
        try:
            score = int(scores.get(dimension, 0))
        except (TypeError, ValueError):
            score = 0
        normalized_scores[dimension] = max(0, min(2, score))
    total_raw = value.get("total")
    try:
        total = float(str(total_raw))
    except (TypeError, ValueError):
        total = float(sum(normalized_scores.values()))
    fatal_raw = value.get("fatal_issues")
    fatal_issues = [str(item) for item in fatal_raw] if isinstance(fatal_raw, list) else []
    return {
        "scores": normalized_scores,
        "total": max(0.0, min(20.0, total)),
        "leakage_detected": bool(value.get("leakage_detected")),
        "fatal_issues": fatal_issues,
        "notes": str(value.get("notes") or ""),
    }


def _llm_error(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def run_judge(
    *,
    case: EvalCase,
    payload: dict[str, Any],
    candidate: DecisionQuality,
    model: str,
    provider: str | None,
    fail_under: float,
) -> dict[str, Any]:
    rubric = RUBRIC_PATH.read_text(encoding="utf-8")
    prompt = (
        "Grade the candidate decision_quality object against the gold output and rubric. "
        "Return only JSON with scores, total, leakage_detected, fatal_issues, and notes. "
        "Mark leakage_detected true if the candidate appears to use future outcomes not present in sanitized inputs.\n\n"
        f"Rubric:\n{rubric}\n\n"
        f"Sanitized inputs:\n{json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2)}\n\n"
        f"Candidate:\n{json.dumps(candidate.model_dump(mode='json'), ensure_ascii=True, sort_keys=True, indent=2)}\n\n"
        f"Gold output:\n{json.dumps(case.gold_output, ensure_ascii=True, sort_keys=True, indent=2)}"
    )
    text, _citations, _response = call_llm_text(
        prompt=prompt,
        model=model,
        max_tokens=3000,
        system="You are a strict investment decision-quality eval judge.",
        provider=provider,
        enable_web_search=False,
        json_schema=_judge_schema(),
        json_schema_name="decision_quality_eval_judge",
    )
    judge = _normalize_judge_result(parse_json_text(text))
    judge["passed"] = judge["total"] >= fail_under and not judge["leakage_detected"] and not judge["fatal_issues"]
    return judge


def run_case(
    case: EvalCase,
    *,
    model: str = MODEL_HIGH,
    provider: str | None = None,
    judge: bool = True,
    dry_run: bool = False,
    fail_under_judge: float = 14.0,
) -> dict[str, Any]:
    payload = build_solver_payload(case)
    prompt = build_solver_prompt(payload)
    result: dict[str, Any] = {
        "case_id": case.case_id,
        "case_path": str(case.path.relative_to(ROOT) if case.path.is_relative_to(ROOT) else case.path),
        "status": case.status,
        "as_of_date": case.data.get("as_of_date"),
        **case_result_metadata(case.data),
        "sanitized_payload": payload,
    }
    if dry_run:
        result.update({"dry_run": True, "solver_prompt": prompt})
        return result

    try:
        text, citations, _response = call_llm_text(
            prompt=prompt,
            model=model,
            max_tokens=6000,
            system=_decision_quality_contract(),
            provider=provider,
            enable_web_search=False,
            json_schema=decision_quality_schema(),
            json_schema_name="decision_quality_eval_solver",
        )
    except Exception as exc:
        result.update(
            {
                "dry_run": False,
                "raw_solver_output": None,
                "solver_error": _llm_error(exc),
                "citations": [],
                "candidate": None,
                "parse_errors": [f"solver call failed: {type(exc).__name__}: {exc}"],
                "decision_quality_gate": {
                    "status": "invalid",
                    "original_action": "watch",
                    "final_action": "watch",
                    "original_recommendation_status": "clear",
                    "final_recommendation_status": "review_required",
                    "confidence_cap": 0.0,
                    "reasons": [
                        {
                            "code": "INVALID_DECISION_QUALITY",
                            "severity": "blocker",
                            "message": "Solver LLM call failed before producing decision_quality.",
                        }
                    ],
                },
                "deterministic": {
                    "score": 0.0,
                    "passed": False,
                    "checks": [_check("solver_call", False, f"{type(exc).__name__}: {exc}")],
                },
            }
        )
        return result

    parsed = parse_json_text(text)
    raw_decision_quality = (
        parsed.get("decision_quality") if isinstance(parsed, dict) and "decision_quality" in parsed else parsed
    )
    candidate, parse_errors = parse_decision_quality(raw_decision_quality)
    gate = apply_decision_quality_gates(
        candidate,
        current_action=candidate.recommended_action if candidate else "watch",
        recommendation_status="clear",
        parse_errors=parse_errors,
    )
    deterministic = deterministic_score(case=case, candidate=candidate, gate=gate, parse_errors=parse_errors)
    result.update(
        {
            "dry_run": False,
            "raw_solver_output": text,
            "citations": [{"title": title, "url": url} for title, url in citations],
            "candidate": candidate.model_dump(mode="json") if candidate else None,
            "parse_errors": parse_errors,
            "decision_quality_gate": gate.model_dump(mode="json"),
            "deterministic": deterministic,
        }
    )
    if judge and candidate is not None:
        try:
            result["judge"] = run_judge(
                case=case,
                payload=payload,
                candidate=candidate,
                model=model,
                provider=provider,
                fail_under=fail_under_judge,
            )
        except Exception as exc:
            result["judge"] = {
                "passed": False,
                "error": _llm_error(exc),
                "total": 0.0,
                "leakage_detected": False,
                "fatal_issues": [f"judge call failed: {type(exc).__name__}: {exc}"],
                "notes": "Judge LLM call failed before producing rubric scores.",
            }
    return result


def _default_output_path() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_DIR / f"decision_quality_eval_{timestamp}.json"


def _parse_statuses(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def build_report(
    results: list[dict[str, Any]], *, fail_under_deterministic: float, fail_under_judge: float
) -> dict[str, Any]:
    deterministic_failures = [
        result
        for result in results
        if not result.get("dry_run")
        and (
            not (result.get("deterministic") or {}).get("passed")
            or (result.get("deterministic") or {}).get("score", 0) < fail_under_deterministic
        )
    ]
    judge_failures = [
        result
        for result in results
        if not result.get("dry_run") and "judge" in result and not (result.get("judge") or {}).get("passed")
    ]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "case_count": len(results),
            "deterministic_failures": [result["case_id"] for result in deterministic_failures],
            "judge_failures": [result["case_id"] for result in judge_failures],
            "fail_under_deterministic": fail_under_deterministic,
            "fail_under_judge": fail_under_judge,
            "calibration_summary": summarize_calibration(results),
        },
        "cases": results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run decision-quality eval cases against the configured LLM.")
    parser.add_argument("--case", action="append", default=[], help="Case id, filename, or path. Repeatable.")
    parser.add_argument("--status", default="review,approved", help="Comma-separated statuses to run.")
    parser.add_argument(
        "--approved-only",
        action="store_true",
        help="Run only approved cases (shortcut for --status approved).",
    )
    parser.add_argument(
        "--corpus-tag",
        action="append",
        default=[],
        help="Filter to cases containing this corpus tag. Repeatable.",
    )
    parser.add_argument("--failure-type", default=None, help="Filter to cases with this failure_type.")
    parser.add_argument("--tool-pack", default=None, help="Filter to cases with this tool_pack.")
    parser.add_argument("--model", choices=sorted(MODEL_BY_NAME), default="high")
    parser.add_argument("--provider", choices=["anthropic", "openai", "gemini"], default=None)
    parser.add_argument("--judge", dest="judge", action="store_true", default=True)
    parser.add_argument("--no-judge", dest="judge", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-under-deterministic", type=float, default=80.0)
    parser.add_argument("--fail-under-judge", type=float, default=14.0)
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline JSON path to compare against after the run.",
    )
    parser.add_argument(
        "--compare-to",
        default=None,
        help="Alias for --baseline.",
    )
    parser.add_argument(
        "--comparison-output",
        default=None,
        help="Optional path for the baseline comparison delta report.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write the current run summary to the baseline path.",
    )
    parser.add_argument(
        "--supervised-model",
        default=None,
        help="Optional supervised synthesis model artifact for offline label comparison.",
    )
    parser.add_argument(
        "--supervised-baseline-metrics",
        default=None,
        help="Optional baseline metrics JSON for supervised rollout gate comparison.",
    )
    return parser.parse_args(argv)


def _resolve_baseline_path(args: argparse.Namespace) -> Path:
    baseline_arg = args.compare_to or args.baseline
    if baseline_arg:
        return Path(baseline_arg)
    return DEFAULT_BASELINE_PATH


def main(argv: list[str] | None = None) -> int:
    _load_local_env()
    args = parse_args(argv)
    statuses = {"approved"} if args.approved_only else _parse_statuses(args.status)
    corpus_tags = set(args.corpus_tag or [])
    cases = load_cases(
        case_selectors=args.case or None,
        statuses=statuses,
        corpus_tags=corpus_tags or None,
        failure_type=args.failure_type,
        tool_pack=args.tool_pack,
    )
    model = MODEL_BY_NAME[args.model]
    results = [
        run_case(
            case,
            model=model,
            provider=args.provider,
            judge=args.judge,
            dry_run=args.dry_run,
            fail_under_judge=args.fail_under_judge,
        )
        for case in cases
    ]
    report = build_report(
        results,
        fail_under_deterministic=args.fail_under_deterministic,
        fail_under_judge=args.fail_under_judge,
    )
    if args.supervised_model:
        from decision_quality.synthesis_supervised_training import (
            build_supervised_eval_summary,
            rows_from_structured_cases,
        )

        baseline_metrics = None
        if args.supervised_baseline_metrics:
            baseline_metrics = json.loads(Path(args.supervised_baseline_metrics).read_text(encoding="utf-8"))
        supervised_rows = rows_from_structured_cases(cases)
        if supervised_rows:
            report["supervised_eval"] = build_supervised_eval_summary(
                rows=supervised_rows,
                model_path=Path(args.supervised_model),
                baseline_metrics=baseline_metrics,
            )
    output_path = Path(args.output) if args.output else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote decision-quality eval report: {output_path}")

    baseline_path = _resolve_baseline_path(args)
    comparison: dict[str, Any] | None = None
    if args.update_baseline:
        baseline = build_baseline_report(
            results,
            corpus_tags=corpus_tags or None,
            status_filter=statuses,
            notes="Structured decision-quality approved corpus baseline.",
        )
        write_baseline(baseline_path, baseline)
        print(f"Updated structured eval baseline: {baseline_path}")

    if (args.compare_to or args.baseline) and not args.dry_run:
        baseline = load_baseline(baseline_path)
        comparison = compare_reports(baseline, report)
        comparison_output = (
            Path(args.comparison_output)
            if args.comparison_output
            else output_path.with_name(output_path.stem + "_comparison.json")
        )
        comparison_output.parent.mkdir(parents=True, exist_ok=True)
        comparison_output.write_text(
            json.dumps(comparison, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Wrote baseline comparison: {comparison_output}")
        if comparison["summary"]["regression_detected"]:
            print(
                "Regression detected against baseline: "
                + ", ".join(comparison["summary"]["new_deterministic_failures"])
            )

    if args.dry_run:
        return 0
    failures = report["summary"]["deterministic_failures"] or report["summary"]["judge_failures"]
    if comparison and comparison["summary"]["regression_detected"]:
        return 1
    supervised_eval = report.get("supervised_eval")
    if isinstance(supervised_eval, dict) and not supervised_eval.get("rollout_gates", {}).get("passed", True):
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
