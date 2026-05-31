from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from decision_quality.actions import ACTIONABLE_ACTIONS
from decision_quality.candidate_gates import apply_opportunity_candidate_gates
from decision_quality.opportunity_candidate import (
    OpportunityCandidate,
    parse_opportunity_candidate,
)

CASES_DIR = Path("docs/opportunity_candidate_evals/cases")


def _case_paths() -> list[Path]:
    return sorted(CASES_DIR.glob("*.json"))


def _load_case(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_candidate(**overrides: object) -> dict:
    base = _load_case(CASES_DIR / "opportunity_candidate_graduate_nvda_2026.json")["gold_output"]
    base.update(overrides)
    return base


@pytest.mark.parametrize("case_path", _case_paths(), ids=lambda path: path.name)
def test_eval_gold_outputs_parse_as_opportunity_candidate(case_path: Path):
    candidate = OpportunityCandidate.model_validate(_load_case(case_path)["gold_output"])
    assert candidate.trigger


@pytest.mark.parametrize("case_path", _case_paths(), ids=lambda path: path.name)
def test_eval_gold_outputs_pass_candidate_gates(case_path: Path):
    case = _load_case(case_path)
    candidate = OpportunityCandidate.model_validate(case["gold_output"])
    gate = apply_opportunity_candidate_gates(candidate)
    assert gate.final_action == candidate.next_action
    assert gate.should_graduate == case["expected_graduation"]


def test_actionable_next_action_is_coerced_to_graduation():
    candidate, errors = parse_opportunity_candidate(_valid_candidate(next_action="buy"))
    assert errors == []
    assert candidate is not None
    assert candidate.next_action == "graduate_to_decision_quality"
    gate = apply_opportunity_candidate_gates(candidate)
    assert gate.final_action == "graduate_to_decision_quality"
    assert gate.should_graduate is True


def test_missing_trigger_invalidates_graduation():
    candidate = OpportunityCandidate.model_validate(
        _valid_candidate(trigger="", next_action="graduate_to_decision_quality")
    )
    gate = apply_opportunity_candidate_gates(candidate)
    assert gate.final_action == "research"
    assert gate.should_graduate is False


def test_parse_opportunity_candidate_coerces_aliases():
    candidate, errors = parse_opportunity_candidate(
        {
            "source": "agent_chat",
            "trigger": "Pullback scan",
            "consensus": "Crowded",
            "variant_view": "Better entry",
            "why_now": "User asked now",
            "price_confirmation": "Needs chart",
            "next_action": "graduate_to_dq",
            "source_refs": ["deck excerpt"],
        }
    )
    assert errors == []
    assert candidate is not None
    assert candidate.next_action == "graduate_to_decision_quality"
    assert candidate.source_refs[0].label == "deck excerpt"


def test_strict_model_rejects_unknown_fields():
    payload = _valid_candidate()
    payload["recommended_action"] = "buy"
    with pytest.raises(ValidationError):
        OpportunityCandidate.model_validate(payload)


def test_non_actionable_actions_never_map_to_actionable_set():
    for case_path in _case_paths():
        candidate = OpportunityCandidate.model_validate(_load_case(case_path)["gold_output"])
        assert candidate.next_action not in ACTIONABLE_ACTIONS
