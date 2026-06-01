from __future__ import annotations

from decision_quality.candidate_gates import apply_opportunity_candidate_gates
from decision_quality.context_packs import (
    assess_context_pack,
    build_context_pack_tool_calls,
    get_context_pack,
    resolve_context_pack,
)
from decision_quality.opportunity_candidate import OpportunityCandidate


def test_resolve_context_pack_from_tool_pack_and_intent():
    pack = resolve_context_pack(
        user_text="Here is my Meta thesis?",
        intent_class="thesis_review",
        tool_pack="thesis_review",
        screen_context={"ticker": "META"},
    )
    assert pack.pack_id == "quality_entry"

    catalyst = resolve_context_pack(
        user_text="Has the NVDA export-license catalyst played out yet?",
        intent_class="catalyst_status",
        tool_pack="catalyst_status",
        screen_context={"ticker": "NVDA"},
    )
    assert catalyst.pack_id == "catalyst"


def test_resolve_context_pack_from_opportunity_type_metadata():
    pack = resolve_context_pack(
        user_text="Pressure-test this idea",
        opportunity_candidate_metadata={"opportunity_type": "forced_liquidation"},
    )
    assert pack.pack_id == "credit_liquidity"


def test_build_context_pack_tool_calls_for_quality_entry():
    pack = get_context_pack("quality_entry")
    calls = build_context_pack_tool_calls(
        pack=pack,
        user_text="What do you think about NVDA as a long?",
        screen_context={"ticker": "NVDA"},
        allowed_tool_names={
            "get_portfolio",
            "get_dossier",
            "get_thesis",
            "get_position_valuation",
            "run_chart",
            "get_thesis_evaluations",
            "search_knowledge_base",
        },
    )
    names = [call["name"] for call in calls]
    assert "get_thesis" in names
    assert "run_chart" in names
    assert "get_position_valuation" in names


def test_assess_context_pack_marks_missing_price_confirmation():
    pack = get_context_pack("quality_entry")
    assessment = assess_context_pack(
        pack=pack,
        tool_results=[
            {"name": "get_portfolio", "status": "ok", "result": {"summary": {"position_count": 0}}},
            {"name": "get_dossier", "status": "ok", "result": {"ticker": "META"}},
            {"name": "get_thesis", "status": "ok", "result": {"content": "thesis"}},
            {"name": "get_position_valuation", "status": "ok", "result": {"price": 100}},
            {
                "name": "run_chart",
                "status": "ok",
                "result": {"technical_read": "", "data_needed": ["current chart"]},
            },
        ],
        data_quality={"blocking_reason_codes": ["MISSING_PRICE_CONFIRMATION"], "critical_data_quality": "ok"},
    )
    assert assessment["is_complete"] is False
    assert "run_chart" in assessment["missing_tools"]
    assert any("price" in item.lower() for item in assessment["missing_inputs"])


def test_candidate_gate_downgrades_on_incomplete_context_pack():
    candidate = OpportunityCandidate.model_validate(
        {
            "trigger": "User pasted thesis",
            "consensus": "Market is cautious",
            "variant_view": "AI monetization can offset capex",
            "why_now": "Annual meeting soon",
            "price_confirmation": "Needs chart",
            "next_action": "graduate_to_decision_quality",
        }
    )
    gate = apply_opportunity_candidate_gates(
        candidate,
        context_pack={
            "pack_id": "quality_entry",
            "is_complete": False,
            "missing_tools": ["run_chart"],
            "missing_inputs": ["price-action confirmation"],
        },
        data_quality={"blocking_reason_codes": ["MISSING_PRICE_CONFIRMATION"]},
    )
    assert gate.final_action == "research"
    assert gate.should_graduate is False
    assert any(reason.code == "CONTEXT_PACK_INCOMPLETE" for reason in gate.reasons)


def test_same_ticker_resolves_different_packs():
    quality = resolve_context_pack(
        user_text="what do you think about nvidia as a long?",
        intent_class="thesis_review",
        tool_pack="thesis_review",
        screen_context={"ticker": "NVDA"},
    )
    catalyst = resolve_context_pack(
        user_text="Has the NVDA export-license catalyst played out yet?",
        intent_class="catalyst_status",
        tool_pack="catalyst_status",
        screen_context={"ticker": "NVDA"},
    )
    assert quality.pack_id == "quality_entry"
    assert catalyst.pack_id == "catalyst"
    assert quality.required_tools != catalyst.required_tools
