"""Manual eval runner for OpportunityCandidate gold cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decision_quality.actions import ACTIONABLE_ACTIONS
from decision_quality.candidate_gates import apply_opportunity_candidate_gates
from decision_quality.eval_corpus import (
    CORPUS_TAGS,
    build_baseline_report,
    case_result_metadata,
    compare_reports,
    filter_cases,
    load_baseline,
    summarize_calibration,
    validate_approved_case_metadata,
    write_baseline,
)
from decision_quality.opportunity_candidate import (
    OpportunityCandidate,
    OpportunityCandidateGate,
    opportunity_candidate_schema,
    parse_opportunity_candidate,
)
from llm_utils import MODEL_HIGH, call_llm_text, parse_json_text

ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = ROOT / "docs" / "opportunity_candidate_evals" / "cases"
PROMPTS_DIR = ROOT / "auto_report" / "prompts"
RUBRIC_PATH = ROOT / "docs" / "opportunity_candidate_evals" / "rubric.md"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "opportunity_candidate_evals"
DEFAULT_BASELINE_PATH = ROOT / "docs" / "opportunity_candidate_evals" / "baselines" / "approved_corpus_baseline.json"

DEFAULT_STATUSES = ("review", "approved")
MODEL_BY_NAME = {"low": "low", "mid": "mid", "high": MODEL_HIGH}
RUBRIC_DIMENSIONS = (
    "trigger_clarity",
    "why_now",
    "missing_inputs",
    "triage_discipline",
)

ANSWER_KEY_FIELDS = {
    "gold_output",
    "rubric_scores",
    "human_notes",
    "expected_graduation",
    "expected_final_action",
    "expected_gate_status",
    "expected_scout_status",
    "expected_skeptic_block_reasons",
    "expected_skeptic_status",
}


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
                raise ValueError(f"Unknown opportunity-candidate eval case: {selector}")
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


def validate_case_metadata(case: EvalCase) -> list[str]:
    errors = validate_approved_case_metadata(case.data)
    if case.status == "approved":
        return errors
    tags = case.data.get("corpus_tags")
    if isinstance(tags, list) and tags:
        unknown = sorted({str(tag) for tag in tags if str(tag)} - CORPUS_TAGS)
        if unknown:
            errors.append(f"unknown corpus_tags: {', '.join(unknown)}")
    return errors


def _is_answer_key(key: str) -> bool:
    return key.strip().lower() in ANSWER_KEY_FIELDS


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
    return value


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
            if path.exists():
                if path.suffix.lower() == ".json":
                    with path.open(encoding="utf-8") as handle:
                        content = json.load(handle)
                else:
                    content = path.read_text(encoding="utf-8")
                sanitized_ref["content"] = sanitize_case_input(content)
        refs.append(sanitized_ref)

    return {
        "id": case.case_id,
        "status": case.status,
        "as_of_date": case.data.get("as_of_date"),
        "decision_type": case.data.get("decision_type"),
        "user_question": case.data.get("user_question"),
        "assumptions": case.data.get("assumptions"),
        "input_refs": refs,
        "context_pack": sanitize_case_input(case.data.get("context_pack")),
        "data_quality": sanitize_case_input(case.data.get("data_quality")),
    }


def build_solver_prompt(payload: dict[str, Any]) -> str:
    return (
        "Solve this OpportunityCandidate eval case using only the supplied as-of inputs. "
        "Do not use later outcomes, outside knowledge, or facts not present in the payload. "
        "Return only the structured OpportunityCandidate JSON object, not a wrapper and not markdown. "
        "Never emit actionable buy/add/short/sell/trim/exit language in next_action.\n\n"
        f"Sanitized case payload:\n{json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2)}"
    )


def _opportunity_candidate_contract() -> str:
    return (PROMPTS_DIR / "opportunity_candidate.md").read_text(encoding="utf-8")


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


def _scout_status(candidate: OpportunityCandidate) -> str:
    missing_fields = [
        name
        for name, value in (
            ("trigger", candidate.trigger),
            ("variant_view", candidate.variant_view),
            ("why_now", candidate.why_now),
        )
        if not _nonempty(value)
    ]
    return "fail" if missing_fields else "pass"


def _skeptic_reason_codes(
    *,
    gate: OpportunityCandidateGate,
    has_decision_quality: bool,
) -> list[str]:
    codes = [reason.code for reason in gate.reasons if reason.severity in {"blocker", "warning"}]
    if not has_decision_quality:
        codes.append("MISSING_DECISION_QUALITY")
    return list(dict.fromkeys(codes))


def deterministic_score(
    *,
    case: EvalCase,
    candidate: OpportunityCandidate | None,
    gate: OpportunityCandidateGate,
    parse_errors: list[str] | None = None,
    has_decision_quality: bool = False,
) -> dict[str, Any]:
    gold = OpportunityCandidate.model_validate(case.gold_output)
    checks: list[dict[str, Any]] = []
    if candidate is None:
        message = "; ".join(parse_errors or []) or "candidate did not parse as OpportunityCandidate"
        checks.append(_check("schema_valid", False, message))
        return {"score": 0.0, "passed": False, "checks": checks}

    checks.append(_check("schema_valid", True, "candidate parsed as OpportunityCandidate"))
    checks.append(
        _check(
            "gate_not_downgraded_or_blocked",
            gate.status == "pass" and gate.final_action == candidate.next_action,
            f"gate status={gate.status}, final_action={gate.final_action}, candidate next_action={candidate.next_action}",
        )
    )
    checks.append(
        _check(
            "next_action",
            candidate.next_action == gold.next_action,
            f"expected {gold.next_action}, got {candidate.next_action}",
        )
    )
    checks.append(
        _check(
            "should_graduate",
            gate.should_graduate == bool(case.data.get("expected_graduation")),
            f"expected {case.data.get('expected_graduation')}, got {gate.should_graduate}",
        )
    )
    expected_final_action = case.data.get("expected_final_action")
    if expected_final_action is not None:
        checks.append(
            _check(
                "expected_final_action",
                gate.final_action == str(expected_final_action),
                f"expected {expected_final_action}, got {gate.final_action}",
            )
        )
    expected_gate_status = case.data.get("expected_gate_status")
    if expected_gate_status is not None:
        checks.append(
            _check(
                "expected_gate_status",
                gate.status == str(expected_gate_status),
                f"expected {expected_gate_status}, got {gate.status}",
            )
        )
    checks.append(
        _check(
            "non_actionable_next_action",
            candidate.next_action not in ACTIONABLE_ACTIONS,
            f"next_action must remain triage-only, got {candidate.next_action}",
        )
    )
    checks.append(
        _check(
            "trigger_present",
            _nonempty(candidate.trigger),
            "trigger is required for triage",
        )
    )

    expected_scout_status = case.data.get("expected_scout_status")
    if expected_scout_status is not None:
        scout_status = _scout_status(candidate)
        checks.append(
            _check(
                "expected_scout_status",
                scout_status == str(expected_scout_status),
                f"expected {expected_scout_status}, got {scout_status}",
            )
        )

    expected_skeptic_status = case.data.get("expected_skeptic_status")
    expected_skeptic_block_reasons = case.data.get("expected_skeptic_block_reasons")
    if expected_skeptic_status is not None or expected_skeptic_block_reasons is not None:
        reason_codes = _skeptic_reason_codes(gate=gate, has_decision_quality=has_decision_quality)
        if expected_skeptic_status is not None:
            skeptic_status = "pass" if not reason_codes else "fail"
            checks.append(
                _check(
                    "expected_skeptic_status",
                    skeptic_status == str(expected_skeptic_status),
                    f"expected {expected_skeptic_status}, got {skeptic_status}",
                )
            )
        if isinstance(expected_skeptic_block_reasons, list) and expected_skeptic_block_reasons:
            expected_codes = {str(item) for item in expected_skeptic_block_reasons}
            actual_codes = set(reason_codes)
            checks.append(
                _check(
                    "expected_skeptic_block_reasons",
                    expected_codes.issubset(actual_codes),
                    f"expected subset {sorted(expected_codes)}, got {sorted(actual_codes)}",
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
                    dimension: {"type": "integer", "minimum": 0, "maximum": 5} for dimension in RUBRIC_DIMENSIONS
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
        normalized_scores[dimension] = max(0, min(5, score))
    total_raw = value.get("total")
    try:
        total = float(total_raw)
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
    candidate: OpportunityCandidate,
    model: str,
    provider: str | None,
    fail_under: float,
) -> dict[str, Any]:
    rubric = RUBRIC_PATH.read_text(encoding="utf-8")
    prompt = (
        "Grade the candidate OpportunityCandidate object against the gold output and rubric. "
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
        system="You are a strict opportunity-triage eval judge.",
        provider=provider,
        enable_web_search=False,
        json_schema=_judge_schema(),
        json_schema_name="opportunity_candidate_eval_judge",
    )
    judge = _normalize_judge_result(parse_json_text(text))
    judge["passed"] = judge["total"] >= fail_under and not judge["leakage_detected"] and not judge["fatal_issues"]
    return judge


def _resolve_context(case: EvalCase) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    context_pack = case.data.get("context_pack")
    data_quality = case.data.get("data_quality")
    return (
        context_pack if isinstance(context_pack, dict) else None,
        data_quality if isinstance(data_quality, dict) else None,
    )


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
    context_pack, data_quality = _resolve_context(case)
    has_decision_quality = isinstance(case.data.get("decision_quality"), dict)
    result: dict[str, Any] = {
        "case_id": case.case_id,
        "case_path": str(case.path.relative_to(ROOT) if case.path.is_relative_to(ROOT) else case.path),
        "status": case.status,
        "as_of_date": case.data.get("as_of_date"),
        **case_result_metadata(case.data),
        "sanitized_payload": payload,
        "metadata_errors": validate_case_metadata(case),
        "input_ref_errors": validate_case_input_refs(case),
    }
    if dry_run:
        gold_candidate = OpportunityCandidate.model_validate(case.gold_output)
        gold_gate = apply_opportunity_candidate_gates(
            gold_candidate,
            context_pack=context_pack,
            data_quality=data_quality,
        )
        gold_deterministic = deterministic_score(
            case=case,
            candidate=gold_candidate,
            gate=gold_gate,
            has_decision_quality=has_decision_quality,
        )
        result.update(
            {
                "dry_run": True,
                "solver_prompt": prompt,
                "gold_deterministic": gold_deterministic,
            }
        )
        return result

    try:
        text, citations, _response = call_llm_text(
            prompt=prompt,
            model=model,
            max_tokens=6000,
            system=_opportunity_candidate_contract(),
            provider=provider,
            enable_web_search=False,
            json_schema=opportunity_candidate_schema(),
            json_schema_name="opportunity_candidate_eval_solver",
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
                "opportunity_candidate_gate": {
                    "status": "invalid",
                    "original_action": "research",
                    "final_action": "research",
                    "should_graduate": False,
                    "reasons": [
                        {
                            "code": "INVALID_OPPORTUNITY_CANDIDATE",
                            "severity": "blocker",
                            "message": "Solver LLM call failed before producing opportunity_candidate.",
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
    raw_candidate = (
        parsed.get("opportunity_candidate")
        if isinstance(parsed, dict) and "opportunity_candidate" in parsed
        else parsed
    )
    candidate, parse_errors = parse_opportunity_candidate(raw_candidate)
    gate = apply_opportunity_candidate_gates(
        candidate,
        parse_errors=parse_errors,
        context_pack=context_pack,
        data_quality=data_quality,
    )
    deterministic = deterministic_score(
        case=case,
        candidate=candidate,
        gate=gate,
        parse_errors=parse_errors,
        has_decision_quality=has_decision_quality,
    )
    result.update(
        {
            "dry_run": False,
            "raw_solver_output": text,
            "citations": [{"title": title, "url": url} for title, url in citations],
            "candidate": candidate.model_dump(mode="json") if candidate else None,
            "parse_errors": parse_errors,
            "opportunity_candidate_gate": gate.model_dump(mode="json"),
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
    return DEFAULT_OUTPUT_DIR / f"opportunity_candidate_eval_{timestamp}.json"


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
    dry_run_failures = [
        result
        for result in results
        if result.get("dry_run")
        and (
            result.get("metadata_errors")
            or result.get("input_ref_errors")
            or not (result.get("gold_deterministic") or {}).get("passed")
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
            "dry_run_failures": [result["case_id"] for result in dry_run_failures],
            "judge_failures": [result["case_id"] for result in judge_failures],
            "fail_under_deterministic": fail_under_deterministic,
            "fail_under_judge": fail_under_judge,
            "calibration_summary": summarize_calibration(results),
        },
        "cases": results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpportunityCandidate eval cases against the configured LLM.")
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
    return parser.parse_args(argv)


def _resolve_baseline_path(args: argparse.Namespace) -> Path:
    baseline_arg = args.compare_to or args.baseline
    if baseline_arg:
        return Path(baseline_arg)
    return DEFAULT_BASELINE_PATH


def _dry_run_exit_code(results: list[dict[str, Any]]) -> int:
    for result in results:
        if result.get("metadata_errors"):
            return 1
        if result.get("input_ref_errors"):
            return 1
        gold = result.get("gold_deterministic") or {}
        if not gold.get("passed"):
            return 1
    return 0


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
    output_path = Path(args.output) if args.output else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote opportunity-candidate eval report: {output_path}")

    baseline_path = _resolve_baseline_path(args)
    comparison: dict[str, Any] | None = None
    if args.update_baseline:
        baseline_results = [result for result in results if not result.get("dry_run")]
        baseline = build_baseline_report(
            baseline_results,
            corpus_tags=corpus_tags or None,
            status_filter=statuses,
            notes="OpportunityCandidate approved corpus baseline.",
        )
        write_baseline(baseline_path, baseline)
        print(f"Updated opportunity-candidate eval baseline: {baseline_path}")

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
        return _dry_run_exit_code(results)
    failures = report["summary"]["deterministic_failures"] or report["summary"]["judge_failures"]
    if comparison and comparison["summary"]["regression_detected"]:
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
