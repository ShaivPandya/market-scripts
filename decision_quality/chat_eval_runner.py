"""Live chat eval runner for Stan decision-quality thesis answers."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decision_quality.eval_corpus import (
    build_baseline_report,
    case_result_metadata,
    compare_reports,
    filter_cases,
    load_baseline,
    write_baseline,
)
from llm_utils import MODEL_HIGH, call_llm_json

ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = ROOT / "docs" / "decision_quality_chat_evals" / "cases"
INPUTS_DIR = ROOT / "docs" / "decision_quality_chat_evals" / "inputs"
RUBRIC_PATH = ROOT / "docs" / "decision_quality_chat_evals" / "rubric.md"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "decision_quality_chat_evals"
DEFAULT_BASELINE_PATH = ROOT / "docs" / "decision_quality_chat_evals" / "baselines" / "approved_corpus_baseline.json"

DEFAULT_STATUSES = ("review", "approved")
MODEL_BY_NAME = {"low": "low", "mid": "mid", "high": MODEL_HIGH}
DEFAULT_JUDGE_MIN_SCORE = 16.0
ARTIFACTS_PATTERN = re.compile(r"```artifacts\s*\n(.*?)```", re.DOTALL)
APPROVAL_BOUNDARY_PATTERNS = (
    r"\bpending approval\b",
    r"\bapproval[- ]gated\b",
    r"\bmust be approved\b",
    r"\bneeds approval\b",
    r"\breview(?:ed)? in workspace\b",
    r"\bproposal\b",
    r"\bproposed\b",
)

DIMENSION_PATTERNS: dict[str, tuple[str, ...]] = {
    "simple_thesis": (r"\bthesis\b", r"\bthe idea\b", r"\byou'?re saying\b"),
    "mispricing": (r"\bmispricing\b", r"\bmarket (?:is )?pricing\b", r"\bdiscount\b", r"\bbear case\b"),
    "catalyst_or_reason_now": (r"\bcatalyst\b", r"\breason[- ]now\b", r"\bwhy now\b", r"\bnext\b"),
    "evidence_for": (r"\bevidence for\b", r"\bfor the thesis\b", r"\bsupports\b", r"\bwhat works\b"),
    "evidence_against": (r"\bevidence against\b", r"\bagainst the thesis\b", r"\brisk\b", r"\bhole\b"),
    "price_action": (r"\bprice action\b", r"\bchart\b", r"\btechnical\b", r"\bvolume\b"),
    "invalidation": (r"\binvalidation\b", r"\bkill\b", r"\bwrong\b", r"\bthreshold\b"),
    "missing_inputs": (r"\bmissing\b", r"\bneed\b", r"\bbefore\b", r"\binput\b"),
    "confidence_sizing": (r"\bconfidence\b", r"\bconviction\b", r"\bsize\b", r"\bsizing\b", r"\bstarter\b"),
    "trade_after_trade": (r"\bif right\b", r"\bif wrong\b", r"\breview\b", r"\bnext review\b"),
}


@dataclass(frozen=True)
class ChatEvalCase:
    path: Path
    data: dict[str, Any]

    @property
    def case_id(self) -> str:
        return str(self.data.get("id") or self.path.stem)

    @property
    def status(self) -> str:
        return str(self.data.get("status") or "draft")

    @property
    def judge_min_score(self) -> float:
        try:
            return float(self.data.get("judge_min_score", DEFAULT_JUDGE_MIN_SCORE))
        except (TypeError, ValueError):
            return DEFAULT_JUDGE_MIN_SCORE


@dataclass(frozen=True)
class AgentChatRun:
    final_text: str
    events: list[tuple[str, dict[str, Any]]]
    tool_names: list[str]
    done_payload: dict[str, Any]
    status_code: int = 200
    elapsed_ms: float | None = None
    error: str | None = None


def _load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("AGENT_GOVERNANCE_AUDIT_ENABLED", "false")


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
) -> list[ChatEvalCase]:
    cases = [ChatEvalCase(path=path, data=_read_json(path)) for path in sorted(cases_dir.glob("*.json"))]
    if case_selectors:
        by_id = {case.case_id: case for case in cases}
        by_stem = {case.path.stem: case for case in cases}
        by_name = {case.path.name: case for case in cases}
        selected: list[ChatEvalCase] = []
        for selector in case_selectors:
            path = Path(selector)
            if path.exists():
                selected.append(ChatEvalCase(path=path, data=_read_json(path)))
                continue
            prefix_matches = [
                case for case in cases if case.case_id.startswith(selector) or case.path.stem.startswith(selector)
            ]
            if prefix_matches:
                selected.extend(prefix_matches)
                continue
            match = by_id.get(selector) or by_stem.get(selector) or by_name.get(selector)
            if match is None:
                raise ValueError(f"Unknown decision-quality chat eval case: {selector}")
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


def validate_case_input_refs(case: ChatEvalCase, *, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    refs = case.data.get("input_refs")
    if refs is None:
        return []
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


def load_input_ref_content(case: ChatEvalCase, *, root: Path = ROOT) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for ref in case.data.get("input_refs") or []:
        if not isinstance(ref, dict):
            continue
        path_value = ref.get("path")
        if not isinstance(path_value, str) or not path_value:
            continue
        path = root / path_value
        if not path.exists():
            continue
        if path.suffix.lower() == ".json":
            content: Any = _read_json(path)
        else:
            content = path.read_text(encoding="utf-8")
        loaded.append(
            {
                "type": ref.get("type"),
                "description": ref.get("description"),
                "path": path_value,
                "content": content,
            }
        )
    return loaded


def parse_sse_events(raw: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for chunk in raw.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        event_name: str | None = None
        data_lines: list[str] = []
        for line in chunk.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: ") :])
        if not event_name or not data_lines:
            continue
        try:
            payload = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append((event_name, payload))
    return events


def run_from_sse_text(raw: str, *, status_code: int = 200, elapsed_ms: float | None = None) -> AgentChatRun:
    events = parse_sse_events(raw)
    final_text = "".join(payload.get("text", "") for event, payload in events if event == "delta")
    tool_names: list[str] = []
    for event, payload in events:
        if event in {"tool_call", "tool_result"} and isinstance(payload.get("name"), str):
            name = str(payload["name"])
            if name not in tool_names:
                tool_names.append(name)
    done_payload = next((payload for event, payload in reversed(events) if event == "done"), {})
    error_payload = next((payload for event, payload in events if event == "error"), None)
    return AgentChatRun(
        final_text=final_text,
        events=events,
        tool_names=tool_names,
        done_payload=done_payload,
        status_code=status_code,
        elapsed_ms=elapsed_ms,
        error=str(error_payload.get("message")) if isinstance(error_payload, dict) else None,
    )


def mocked_tool_executor(mock_tools: dict[str, Any], *, strict: bool = True) -> Callable[..., str]:
    state: dict[str, int] = {}

    def execute(name: str, arguments: dict[str, Any], **_kwargs: Any) -> str:
        if name not in mock_tools:
            if strict:
                return json.dumps({"error": f"No mocked tool payload for {name}", "_meta": {"status": "failed_closed"}})
            return json.dumps({"name": name, "arguments": arguments, "mock": "not_configured"})
        payload = mock_tools[name]
        if isinstance(payload, list):
            idx = state.get(name, 0)
            state[name] = idx + 1
            payload = payload[min(idx, len(payload) - 1)] if payload else {}
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)

    return execute


@contextlib.contextmanager
def _patched_agent_tools(mock_tools: dict[str, Any] | None) -> Iterator[None]:
    if not mock_tools:
        yield
        return
    import api.routers.agent as agent_router

    original = agent_router.execute_tool
    agent_router.execute_tool = mocked_tool_executor(mock_tools)  # type: ignore[assignment]
    try:
        yield
    finally:
        agent_router.execute_tool = original  # type: ignore[assignment]


def run_agent_chat_in_process(case: ChatEvalCase, *, auth_password: str | None = None) -> AgentChatRun:
    from fastapi.testclient import TestClient

    from api.main import app
    from api.request_schema import schema_headers_for_path

    body: dict[str, Any] = {"message": str(case.data.get("user_message") or ""), "finalize_synchronously": True}
    screen_context = case.data.get("screen_context")
    if isinstance(screen_context, dict):
        body["screen_context"] = screen_context

    started = time.perf_counter()
    with _patched_agent_tools(case.data.get("mock_tools") if isinstance(case.data.get("mock_tools"), dict) else None):
        with TestClient(app) as client:
            password = auth_password or os.environ.get("AUTH_PASSWORD")
            csrf_headers: dict[str, str] = {}
            if password:
                login = client.post(
                    "/api/auth/login",
                    json={
                        "password": password,
                        "username": (os.environ.get("AUTH_DEFAULT_USERNAME") or "admin").strip() or "admin",
                    },
                    headers=schema_headers_for_path(app, "POST", "/api/auth/login"),
                )
                if login.status_code == 200:
                    token = login.json().get("csrfToken")
                    if token:
                        csrf_headers = {"X-CSRF-Token": token}
            response = client.post(
                "/api/agent/chat",
                json=body,
                headers={**schema_headers_for_path(app, "POST", "/api/agent/chat"), **csrf_headers},
            )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    if response.status_code != 200:
        return AgentChatRun(
            final_text="",
            events=[],
            tool_names=[],
            done_payload={},
            status_code=response.status_code,
            elapsed_ms=elapsed_ms,
            error=response.text[:1000],
        )
    return run_from_sse_text(response.text, status_code=response.status_code, elapsed_ms=elapsed_ms)


def _contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(str(term).lower() in lowered for term in terms if str(term).strip())


def _contains_all(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return all(str(term).lower() in lowered for term in terms if str(term).strip())


def _point_passed(text: str, point: Any) -> tuple[bool, str]:
    if isinstance(point, str):
        return (point.lower() in text.lower(), point)
    if not isinstance(point, dict):
        return False, "required point must be a string or object"
    label = str(point.get("label") or "required point")
    patterns = [str(item) for item in point.get("patterns") or []]
    any_terms = [str(item) for item in point.get("any_terms") or []]
    all_terms = [str(item) for item in point.get("all_terms") or []]
    passed = True
    details: list[str] = []
    if patterns:
        pattern_passed = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
        passed = passed and pattern_passed
        details.append(f"patterns={pattern_passed}")
    if any_terms:
        any_passed = _contains_any(text, any_terms)
        passed = passed and any_passed
        details.append(f"any_terms={any_passed}")
    if all_terms:
        all_passed = _contains_all(text, all_terms)
        passed = passed and all_passed
        details.append(f"all_terms={all_passed}")
    if not patterns and not any_terms and not all_terms:
        return False, f"{label}: no patterns or terms configured"
    return passed, f"{label}: {', '.join(details)}"


def _check(name: str, passed: bool, message: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "message": message}


def _parse_artifacts_block(text: str) -> tuple[dict[str, Any] | None, str | None]:
    match = ARTIFACTS_PATTERN.search(text)
    if not match:
        return None, "missing artifacts block"
    raw = match.group(1).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid artifacts JSON: {exc}"
    if not isinstance(value, dict):
        return None, "artifacts block must be a JSON object"
    return value, None


def _workflow_tool_calls(done_payload: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = done_payload.get("tool_calls") if isinstance(done_payload, dict) else None
    return [call for call in tool_calls if isinstance(call, dict)] if isinstance(tool_calls, list) else []


def _routing_expectation_checks(
    checks: list[dict[str, Any]],
    *,
    case: ChatEvalCase,
    run: AgentChatRun,
) -> None:
    expectations = case.data.get("routing_expectations")
    if not isinstance(expectations, dict):
        return

    router_meta = run.done_payload.get("intent_router") if isinstance(run.done_payload, dict) else None
    applied = router_meta.get("applied") if isinstance(router_meta, dict) else None
    if not isinstance(applied, dict):
        checks.append(_check("intent_router_meta_present", False, "done payload missing intent_router.applied"))
        return

    checks.append(_check("intent_router_meta_present", True, "intent_router metadata present"))

    expected_intent = expectations.get("intent_class")
    if expected_intent is not None:
        actual_intent = str(applied.get("intent_class") or "")
        checks.append(
            _check(
                "routing_intent_class",
                actual_intent == str(expected_intent),
                f"expected={expected_intent!r}, actual={actual_intent!r}",
            )
        )

    if expectations.get("run_hidden_dq") is not None:
        expected = bool(expectations.get("run_hidden_dq"))
        actual = bool(applied.get("run_hidden_dq"))
        checks.append(
            _check(
                "routing_run_hidden_dq",
                actual == expected,
                f"expected={expected}, actual={actual}",
            )
        )

    if expectations.get("run_opportunity_preflight") is not None:
        expected = bool(expectations.get("run_opportunity_preflight"))
        actual = bool(applied.get("run_opportunity_preflight"))
        checks.append(
            _check(
                "routing_run_opportunity_preflight",
                actual == expected,
                f"expected={expected}, actual={actual}",
            )
        )

    expected_workflow = expectations.get("workflow_name")
    if expected_workflow is not None:
        actual_workflow = applied.get("workflow_name")
        checks.append(
            _check(
                "routing_workflow_name",
                str(actual_workflow or "") == str(expected_workflow),
                f"expected={expected_workflow!r}, actual={actual_workflow!r}",
            )
        )

    required_tools = [str(item) for item in expectations.get("required_tool_names") or []]
    if required_tools:
        applied_tools = [str(item) for item in applied.get("tool_names") or []]
        missing = [name for name in required_tools if name not in applied_tools]
        checks.append(
            _check(
                "routing_required_tool_names",
                not missing,
                f"missing={missing}; applied={applied_tools}",
            )
        )

    allowed_fallback_reasons = expectations.get("allowed_fallback_reasons")
    if allowed_fallback_reasons is not None:
        fallback_reason = applied.get("fallback_reason")
        allowed = {str(item) for item in allowed_fallback_reasons}
        if fallback_reason is None:
            checks.append(_check("routing_fallback_reason", True, "no fallback applied"))
        else:
            checks.append(
                _check(
                    "routing_fallback_reason",
                    str(fallback_reason) in allowed,
                    f"fallback_reason={fallback_reason!r}, allowed={sorted(allowed)}",
                )
            )

    min_confidence = expectations.get("min_confidence")
    if min_confidence is not None:
        try:
            threshold = float(min_confidence)
        except (TypeError, ValueError):
            threshold = 0.0
        try:
            confidence = float(str(applied.get("confidence")))
        except (TypeError, ValueError):
            confidence = 0.0
        checks.append(
            _check(
                "routing_min_confidence",
                confidence >= threshold,
                f"confidence={confidence}, min={threshold}",
            )
        )


def _append_workflow_expectation_checks(
    checks: list[dict[str, Any]],
    *,
    case: ChatEvalCase,
    run: AgentChatRun,
    text: str,
) -> None:
    expectations = case.data.get("workflow_expectations")
    if not isinstance(expectations, dict):
        return

    if expectations.get("requires_workflow_run_id"):
        workflow_run_id = run.done_payload.get("workflow_run_id") if isinstance(run.done_payload, dict) else None
        checks.append(
            _check(
                "workflow_run_id_present",
                isinstance(workflow_run_id, str) and bool(workflow_run_id.strip()),
                f"workflow_run_id={workflow_run_id!r}",
            )
        )

    expected_tool_names = [str(item) for item in case.data.get("expected_tool_names") or []]
    if expected_tool_names:
        tool_calls = _workflow_tool_calls(run.done_payload)
        tool_status_by_name = {
            str(call.get("name")): str(call.get("status") or "").lower()
            for call in tool_calls
            if isinstance(call.get("name"), str)
        }
        missing = [name for name in expected_tool_names if name not in tool_status_by_name]
        bad_status = [
            name for name in expected_tool_names if name in tool_status_by_name and tool_status_by_name[name] != "ok"
        ]
        checks.append(
            _check(
                "workflow_tool_metadata",
                not missing and not bad_status,
                f"missing={missing}; bad_status={bad_status}; seen={tool_status_by_name}",
            )
        )

    expected_artifact_keys = [str(item) for item in expectations.get("expected_artifact_keys") or []]
    if expected_artifact_keys:
        artifacts, error = _parse_artifacts_block(text)
        checks.append(
            _check(
                "workflow_artifacts_parseable",
                artifacts is not None,
                error or f"artifact_keys={sorted(artifacts.keys()) if artifacts else []}",
            )
        )
        if artifacts is not None:
            missing_keys = [key for key in expected_artifact_keys if key not in artifacts]
            checks.append(
                _check(
                    "workflow_artifact_keys",
                    not missing_keys,
                    f"missing={missing_keys}; seen={sorted(artifacts.keys())}",
                )
            )

    if expectations.get("requires_pending_approval_language"):
        checks.append(
            _check(
                "workflow_pending_approval_boundary",
                any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in APPROVAL_BOUNDARY_PATTERNS),
                "workflow answer must describe generated actions as proposals or pending approvals",
            )
        )


def _tool_quality_expectation_checks(
    checks: list[dict[str, Any]],
    *,
    case: ChatEvalCase,
    run: AgentChatRun,
    text: str,
) -> None:
    expectations = case.data.get("tool_quality_expectations")
    if not isinstance(expectations, dict):
        return

    dq_meta = run.done_payload.get("decision_quality_chat") if isinstance(run.done_payload, dict) else None
    tool_quality = dq_meta.get("tool_quality") if isinstance(dq_meta, dict) else None
    if expectations.get("requires_tool_quality_meta"):
        checks.append(
            _check(
                "tool_quality_meta_present",
                isinstance(tool_quality, dict),
                "done payload missing decision_quality_chat.tool_quality",
            )
        )
        if not isinstance(tool_quality, dict):
            return

    if not isinstance(tool_quality, dict):
        return

    if expectations.get("min_blocker_count") is not None:
        expected = int(expectations["min_blocker_count"])
        actual = int(tool_quality.get("blocker_count") or 0)
        checks.append(
            _check(
                "tool_quality_min_blocker_count",
                actual >= expected,
                f"expected>={expected}, actual={actual}",
            )
        )

    if expectations.get("max_blocker_count") is not None:
        expected = int(expectations["max_blocker_count"])
        actual = int(tool_quality.get("blocker_count") or 0)
        checks.append(
            _check(
                "tool_quality_max_blocker_count",
                actual <= expected,
                f"expected<={expected}, actual={actual}",
            )
        )

    for field in (
        "expected_price_confirmation_status",
        "expected_source_health_status",
        "expected_critical_data_quality",
    ):
        expected_status = expectations.get(field)
        if expected_status is None:
            continue
        key = field.removeprefix("expected_")
        actual_status = str(tool_quality.get(key) or "")
        checks.append(
            _check(
                f"tool_quality_{key}",
                actual_status == str(expected_status),
                f"expected={expected_status!r}, actual={actual_status!r}",
            )
        )

    required_codes = [str(item) for item in expectations.get("required_blocking_reason_codes") or []]
    if required_codes:
        actual_codes = {str(item) for item in tool_quality.get("blocking_reason_codes") or []}
        missing = [code for code in required_codes if code not in actual_codes]
        checks.append(
            _check(
                "tool_quality_blocking_reason_codes",
                not missing,
                f"missing={missing}; actual={sorted(actual_codes)}",
            )
        )

    if expectations.get("forbid_actionable_language_when_blocked"):
        blocked = int(tool_quality.get("blocker_count") or 0) > 0 or str(
            tool_quality.get("critical_data_quality") or ""
        ).lower() in {"stale", "failed"}
        if blocked:
            checks.append(
                _check(
                    "tool_quality_no_actionable_language",
                    not re.search(r"\b(buy now|add now|strong buy|short now|sell now)\b", text, flags=re.IGNORECASE),
                    "actionable language must be suppressed when tool quality blocks",
                )
            )

    missing_input_terms = [str(item) for item in expectations.get("required_missing_input_terms") or []]
    if missing_input_terms:
        checks.append(
            _check(
                "tool_quality_missing_input_terms",
                _contains_any(text, missing_input_terms),
                f"expected any of {missing_input_terms}",
            )
        )


def _context_pack_expectation_checks(
    checks: list[dict[str, Any]],
    *,
    case: ChatEvalCase,
    run: AgentChatRun,
    text: str,
) -> None:
    expectations = case.data.get("context_pack_expectations")
    if not isinstance(expectations, dict):
        return

    context_pack = run.done_payload.get("context_pack") if isinstance(run.done_payload, dict) else None
    oc_meta = run.done_payload.get("opportunity_candidate_preflight") if isinstance(run.done_payload, dict) else None
    if expectations.get("requires_context_pack_meta"):
        checks.append(
            _check(
                "context_pack_meta_present",
                isinstance(context_pack, dict),
                "done payload missing context_pack metadata",
            )
        )
        if not isinstance(context_pack, dict):
            return

    if not isinstance(context_pack, dict):
        context_pack = oc_meta.get("context_pack") if isinstance(oc_meta, dict) else None
    if not isinstance(context_pack, dict):
        checks.append(_check("context_pack_meta_present", False, "context_pack metadata missing"))
        return

    checks.append(_check("context_pack_meta_present", True, "context_pack metadata present"))

    expected_pack = expectations.get("expected_context_pack")
    if expected_pack is not None:
        actual_pack = str(context_pack.get("pack_id") or "")
        checks.append(
            _check(
                "context_pack_id",
                actual_pack == str(expected_pack),
                f"expected={expected_pack!r}, actual={actual_pack!r}",
            )
        )

    expected_opportunity_type = expectations.get("expected_opportunity_type")
    if expected_opportunity_type is not None:
        actual_types = {str(item) for item in context_pack.get("opportunity_types") or []}
        checks.append(
            _check(
                "context_pack_opportunity_type",
                str(expected_opportunity_type) in actual_types,
                f"expected={expected_opportunity_type!r}, actual_types={sorted(actual_types)}",
            )
        )

    required_tools = [str(item) for item in expectations.get("required_tool_names") or []]
    if required_tools:
        seen_tools = [str(item) for item in run.tool_names]
        missing = [name for name in required_tools if name not in seen_tools]
        checks.append(
            _check(
                "context_pack_required_tool_names",
                not missing,
                f"missing={missing}; seen={seen_tools}",
            )
        )

    if expectations.get("expect_complete") is True:
        checks.append(
            _check(
                "context_pack_complete",
                bool(context_pack.get("is_complete")),
                f"is_complete={context_pack.get('is_complete')}",
            )
        )
    if expectations.get("expect_complete") is False:
        checks.append(
            _check(
                "context_pack_incomplete",
                not bool(context_pack.get("is_complete")),
                f"is_complete={context_pack.get('is_complete')}",
            )
        )

    missing_input_terms = [str(item) for item in expectations.get("required_missing_input_terms") or []]
    if missing_input_terms:
        checks.append(
            _check(
                "context_pack_missing_input_terms",
                _contains_any(text, missing_input_terms),
                f"expected any of {missing_input_terms}",
            )
        )

    if expectations.get("forbid_actionable_when_incomplete"):
        incomplete = not bool(context_pack.get("is_complete"))
        if incomplete:
            checks.append(
                _check(
                    "context_pack_no_actionable_language",
                    not re.search(r"\b(buy now|add now|strong buy|short now|sell now)\b", text, flags=re.IGNORECASE),
                    "actionable language must be suppressed when context pack is incomplete",
                )
            )


def _scout_skeptic_sizer_expectation_checks(
    checks: list[dict[str, Any]],
    *,
    case: ChatEvalCase,
    run: AgentChatRun,
    text: str,
) -> None:
    expectations = case.data.get("scout_skeptic_sizer_expectations")
    if not isinstance(expectations, dict):
        return

    gate_meta = run.done_payload.get("scout_skeptic_sizer_gate") if isinstance(run.done_payload, dict) else None
    checks.append(
        _check(
            "scout_skeptic_sizer_gate_present",
            isinstance(gate_meta, dict),
            "done payload missing scout_skeptic_sizer_gate trace",
        )
    )
    if not isinstance(gate_meta, dict):
        return

    raw_sizer_meta = gate_meta.get("sizer")
    sizer_meta: dict[str, Any] = raw_sizer_meta if isinstance(raw_sizer_meta, dict) else {}
    for pass_name in ("scout", "skeptic", "sizer"):
        expected = expectations.get(f"{pass_name}_pass")
        if expected is None:
            continue
        raw_pass_meta = gate_meta.get(pass_name)
        pass_meta: dict[str, Any] = raw_pass_meta if isinstance(raw_pass_meta, dict) else {}
        actual = str(pass_meta.get("status") or "").lower() == "pass"
        checks.append(
            _check(
                f"scout_skeptic_sizer_{pass_name}_pass",
                actual is bool(expected) and actual == expected,
                f"expected {pass_name}_pass={expected}, status={pass_meta.get('status')!r}",
            )
        )

    expected_final_action = expectations.get("expected_final_action")
    if expected_final_action is not None:
        actual_final = str(gate_meta.get("final_action_type") or sizer_meta.get("final_action") or "")
        checks.append(
            _check(
                "scout_skeptic_sizer_final_action",
                actual_final.lower() == str(expected_final_action).lower(),
                f"expected={expected_final_action!r}, actual={actual_final!r}",
            )
        )

    if expectations.get("no_actionable_when_skeptic_failed") and expectations.get("skeptic_pass") is False:
        checks.append(
            _check(
                "scout_skeptic_sizer_no_actionable_when_skeptic_failed",
                not re.search(r"\b(buy now|add now|strong buy|short now|sell now)\b", text, flags=re.IGNORECASE),
                "actionable language must be suppressed when skeptic pass fails",
            )
        )

    max_bps = expectations.get("max_sizing_delta_bps")
    if max_bps is not None:
        try:
            actual_bps = int(str(sizer_meta.get("max_sizing_delta_bps")))
            checks.append(
                _check(
                    "scout_skeptic_sizer_max_bps",
                    actual_bps <= int(max_bps),
                    f"expected<={max_bps}, actual={actual_bps}",
                )
            )
        except (TypeError, ValueError):
            checks.append(
                _check(
                    "scout_skeptic_sizer_max_bps",
                    False,
                    f"expected<={max_bps}, actual={sizer_meta.get('max_sizing_delta_bps')!r}",
                )
            )


def deterministic_score(case: ChatEvalCase, run: AgentChatRun) -> dict[str, Any]:
    text = run.final_text or ""
    checks: list[dict[str, Any]] = []

    checks.append(_check("http_ok", run.status_code == 200, f"status_code={run.status_code}, error={run.error or ''}"))
    checks.append(_check("nonempty_answer", bool(text.strip()), "final assistant text must be nonempty"))
    checks.append(
        _check(
            "no_raw_json",
            not re.search(r'"\s*(simple_thesis|recommended_action|decision_quality)\s*"\s*:', text),
            "final answer must not expose raw decision_quality JSON",
        )
    )

    expected_tool_names = [str(item) for item in case.data.get("expected_tool_names") or []]
    missing_tools = [name for name in expected_tool_names if name not in run.tool_names]
    checks.append(
        _check(
            "expected_tool_coverage",
            not missing_tools,
            f"missing tools: {missing_tools}; seen: {run.tool_names}",
        )
    )

    for idx, point in enumerate(case.data.get("required_points") or []):
        passed, message = _point_passed(text, point)
        checks.append(_check(f"required_point_{idx + 1}", passed, message))

    for dimension in case.data.get("required_decision_quality_dimensions") or []:
        name = str(dimension)
        patterns = DIMENSION_PATTERNS.get(name, ())
        passed = bool(patterns) and any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
        checks.append(_check(f"dimension_{name}", passed, f"patterns={list(patterns)}"))

    for pattern in case.data.get("forbidden_patterns") or []:
        pattern_text = str(pattern)
        checks.append(
            _check(
                f"forbidden_{pattern_text[:32]}",
                not re.search(pattern_text, text, flags=re.IGNORECASE),
                f"forbidden pattern: {pattern_text}",
            )
        )

    stance = case.data.get("expected_stance")
    if isinstance(stance, dict):
        label = str(stance.get("label") or "expected_stance")
        any_terms = [str(item) for item in stance.get("any_terms") or []]
        forbidden_terms = [str(item) for item in stance.get("forbidden_terms") or []]
        checks.append(_check(f"stance_{label}", not any_terms or _contains_any(text, any_terms), str(any_terms)))
        for term in forbidden_terms:
            checks.append(
                _check(
                    f"stance_forbidden_{term[:32]}",
                    term.lower() not in text.lower(),
                    f"forbidden stance term: {term}",
                )
            )
    elif isinstance(stance, str) and stance.strip():
        checks.append(_check("expected_stance", stance.lower() in text.lower(), stance))

    dq_meta = run.done_payload.get("decision_quality_chat") if isinstance(run.done_payload, dict) else None
    if isinstance(dq_meta, dict):
        final_action = str(dq_meta.get("final_action") or "").lower()
        gate_status = str(dq_meta.get("gate_status") or "").lower()
        if final_action in {"watch", "research", "avoid", "do_nothing"}:
            checks.append(
                _check(
                    "gate_action_consistency",
                    not re.search(r"\b(buy now|add now|strong buy)\b", text, flags=re.IGNORECASE),
                    f"gate_status={gate_status}, final_action={final_action}",
                )
            )

    _append_workflow_expectation_checks(checks, case=case, run=run, text=text)
    _routing_expectation_checks(checks, case=case, run=run)
    _tool_quality_expectation_checks(checks, case=case, run=run, text=text)
    _context_pack_expectation_checks(checks, case=case, run=run, text=text)
    _scout_skeptic_sizer_expectation_checks(checks, case=case, run=run, text=text)

    passed_count = sum(1 for check in checks if check["passed"])
    score = round((passed_count / len(checks)) * 100, 2) if checks else 0.0
    return {"score": score, "passed": all(check["passed"] for check in checks), "checks": checks}


def _judge_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["scores", "total", "fatal_issues", "notes"],
        "properties": {
            "scores": {
                "type": "object",
                "additionalProperties": {"type": "integer", "minimum": 0, "maximum": 2},
            },
            "total": {"type": "number", "minimum": 0, "maximum": 20},
            "fatal_issues": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
        },
    }


def _normalize_judge(value: Any, *, min_score: float) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"scores": {}, "total": 0.0, "fatal_issues": ["judge returned non-JSON"], "notes": "", "passed": False}
    try:
        total = float(str(value.get("total")))
    except (TypeError, ValueError):
        total = 0.0
    fatal = value.get("fatal_issues")
    fatal_issues = [str(item) for item in fatal] if isinstance(fatal, list) else []
    judge: dict[str, Any] = {
        "scores": value.get("scores") if isinstance(value.get("scores"), dict) else {},
        "total": max(0.0, min(20.0, total)),
        "fatal_issues": fatal_issues,
        "notes": str(value.get("notes") or ""),
    }
    judge["passed"] = judge["total"] >= min_score and not fatal_issues
    return judge


def run_judge(
    *,
    case: ChatEvalCase,
    run: AgentChatRun,
    model: str,
    provider: str | None,
    min_score: float,
) -> dict[str, Any]:
    rubric = RUBRIC_PATH.read_text(encoding="utf-8")
    prompt = (
        "Grade this live Stan chat answer against the rubric and case expectations. "
        "Return only JSON with scores, total, fatal_issues, and notes.\n\n"
        f"Rubric:\n{rubric}\n\n"
        f"Case:\n{json.dumps(case.data, ensure_ascii=True, sort_keys=True, indent=2)}\n\n"
        f"Loaded input refs:\n{json.dumps(load_input_ref_content(case), ensure_ascii=True, sort_keys=True, indent=2)}\n\n"
        f"Tool names used:\n{json.dumps(run.tool_names, ensure_ascii=True)}\n\n"
        f"Final answer:\n{run.final_text}"
    )
    parsed, _citations, _response, _diagnostics = call_llm_json(
        prompt=prompt,
        model=model,
        max_tokens=3000,
        system="You are a strict live-chat investment decision-quality judge.",
        provider=provider,
        enable_web_search=False,
        json_schema=_judge_schema(),
        json_schema_name="decision_quality_chat_eval_judge",
    )
    return _normalize_judge(parsed, min_score=min_score)


def run_case(
    case: ChatEvalCase,
    *,
    agent_runner: Callable[[ChatEvalCase], AgentChatRun] | None = None,
    judge: bool = False,
    model: str = MODEL_HIGH,
    provider: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    ref_errors = validate_case_input_refs(case)
    result: dict[str, Any] = {
        "case_id": case.case_id,
        "case_path": str(case.path.relative_to(ROOT) if case.path.is_relative_to(ROOT) else case.path),
        "status": case.status,
        "as_of_date": case.data.get("as_of_date"),
        **case_result_metadata(case.data),
        "input_ref_errors": ref_errors,
        "loaded_input_refs": load_input_ref_content(case),
    }
    if dry_run:
        result.update({"dry_run": True, "request": {"message": case.data.get("user_message")}})
        return result
    if ref_errors:
        result.update(
            {
                "dry_run": False,
                "final_text": "",
                "tool_names": [],
                "deterministic": {
                    "score": 0.0,
                    "passed": False,
                    "checks": [_check("input_refs", False, "; ".join(ref_errors))],
                },
            }
        )
        return result

    runner = agent_runner or (lambda eval_case: run_agent_chat_in_process(eval_case))
    run = runner(case)
    deterministic = deterministic_score(case, run)
    result.update(
        {
            "dry_run": False,
            "status_code": run.status_code,
            "elapsed_ms": run.elapsed_ms,
            "error": run.error,
            "final_text": run.final_text,
            "tool_names": run.tool_names,
            "done_payload": run.done_payload,
            "events": [{"event": event, "payload": payload} for event, payload in run.events],
            "deterministic": deterministic,
        }
    )
    if judge and deterministic.get("passed"):
        try:
            result["judge"] = run_judge(
                case=case,
                run=run,
                model=model,
                provider=provider,
                min_score=case.judge_min_score,
            )
        except Exception as exc:
            result["judge"] = {
                "passed": False,
                "total": 0.0,
                "fatal_issues": [f"judge call failed: {type(exc).__name__}: {exc}"],
                "notes": "",
            }
    return result


def build_report(results: list[dict[str, Any]], *, fail_under_deterministic: float) -> dict[str, Any]:
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
        },
        "cases": results,
    }


def _default_output_path() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_DIR / f"decision_quality_chat_eval_{timestamp}.json"


def _parse_statuses(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live decision-quality chat eval cases against Stan.")
    parser.add_argument("--case", action="append", default=[], help="Case id, filename, path, or prefix. Repeatable.")
    parser.add_argument(
        "--routing-only",
        action="store_true",
        help="Only run routing_* cases (shortcut for --case routing_).",
    )
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
    parser.add_argument("--judge", dest="judge", action="store_true", default=False)
    parser.add_argument("--no-judge", dest="judge", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--auth-password", default=None)
    parser.add_argument("--fail-under-deterministic", type=float, default=100.0)
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
    case_selectors = list(args.case or [])
    if args.routing_only and "routing_" not in case_selectors:
        case_selectors.append("routing_")
    statuses = {"approved"} if args.approved_only else _parse_statuses(args.status)
    corpus_tags = set(args.corpus_tag or [])
    cases = load_cases(
        case_selectors=case_selectors or None,
        statuses=statuses,
        corpus_tags=corpus_tags or None,
        failure_type=args.failure_type,
        tool_pack=args.tool_pack,
    )
    model = MODEL_BY_NAME[args.model]

    def runner(case: ChatEvalCase) -> AgentChatRun:
        return run_agent_chat_in_process(case, auth_password=args.auth_password)

    results = [
        run_case(case, agent_runner=runner, judge=args.judge, model=model, provider=args.provider, dry_run=args.dry_run)
        for case in cases
    ]
    report = build_report(results, fail_under_deterministic=args.fail_under_deterministic)
    if args.supervised_model:
        from decision_quality.synthesis_supervised_training import (
            build_supervised_eval_summary,
            rows_from_chat_cases,
        )

        baseline_metrics = None
        if args.supervised_baseline_metrics:
            baseline_metrics = json.loads(Path(args.supervised_baseline_metrics).read_text(encoding="utf-8"))
        supervised_rows = rows_from_chat_cases(cases)
        if supervised_rows:
            report["supervised_eval"] = build_supervised_eval_summary(
                rows=supervised_rows,
                model_path=Path(args.supervised_model),
                baseline_metrics=baseline_metrics,
            )
    output_path = Path(args.output) if args.output else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote decision-quality chat eval report: {output_path}")

    baseline_path = _resolve_baseline_path(args)
    comparison: dict[str, Any] | None = None
    if args.update_baseline:
        baseline = build_baseline_report(
            results,
            corpus_tags=corpus_tags or None,
            status_filter=statuses,
            notes="Decision-quality chat approved corpus baseline.",
        )
        write_baseline(baseline_path, baseline)
        print(f"Updated chat eval baseline: {baseline_path}")

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
