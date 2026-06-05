from __future__ import annotations

import hashlib
from datetime import datetime


def canonical_ticker(value: object) -> str:
    ticker = str(value or "").strip().upper()
    if not ticker or any(ch.isspace() for ch in ticker) or ":" in ticker:
        raise ValueError("ticker must be non-empty and may not contain whitespace or ':'")
    return ticker


def slug(text: object) -> str:
    value = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text or "").strip())
    value = "_".join(part for part in value.split("_") if part)
    return value or "unknown"


def short_hash(text: object, *, length: int = 10) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()[:length]


def canonical_timestamp_key(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "latest"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        text = parsed.isoformat()
    except ValueError:
        pass
    return slug(text)


def position_id(ticker: object) -> str:
    return f"position:{canonical_ticker(ticker)}"


def portfolio_position_uid(row: object) -> str:
    from portfolio.instruments import position_row_id

    if isinstance(row, dict):
        key = position_row_id(row)
    else:
        key = str(row or "").strip().upper()
    if not key:
        raise ValueError("position uid requires a non-empty position key.")
    if key.startswith("position:"):
        return key
    return position_id(key)


def hedge_position_uid(row: object) -> str:
    from portfolio.instruments import position_row_id

    if isinstance(row, dict):
        key = position_row_id(row)
    else:
        key = str(row or "").strip().upper()
    if not key:
        raise ValueError("hedge position uid requires a non-empty position key.")
    prefix, sep, suffix = key.partition(":")
    if sep and prefix.lower() in {"hedge_position", "position"}:
        key = suffix
    return hedge_position_id(key)


def asset_id(ticker: object) -> str:
    return f"asset:{canonical_ticker(ticker)}"


def instrument_id(identifier: object) -> str:
    return f"instrument:{slug(identifier)}"


def issuer_id(identifier: object) -> str:
    return f"issuer:{slug(identifier)}"


def investor_id(identifier: object) -> str:
    return f"investor:{slug(identifier)}"


def account_id(identifier: object) -> str:
    return f"account:{slug(identifier)}"


def portfolio_id(identifier: object) -> str:
    return f"portfolio:{slug(identifier)}"


def investment_policy_id(identifier: object) -> str:
    return f"investment_policy:{slug(identifier)}"


def risk_limit_id(identifier: object) -> str:
    return f"risk_limit:{slug(identifier)}"


def risk_metric_id(identifier: object) -> str:
    return f"risk_metric:{slug(identifier)}"


def scenario_id(identifier: object) -> str:
    return f"scenario:{slug(identifier)}"


def policy_gate_result_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("policy_gate_result:"):
        text = text.split(":", 1)[1]
    return f"policy_gate_result:{slug(text)}"


def trade_proposal_id(identifier: object) -> str:
    return f"trade_proposal:{slug(identifier)}"


def source_record_object_id(identifier: object) -> str:
    return f"source_record:{slug(identifier)}"


def object_version_ref_id(identifier: object) -> str:
    return f"object_version_ref:{slug(identifier)}"


def relation_version_ref_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("relation_version_ref:"):
        text = text.split(":", 1)[1]
    return f"relation_version_ref:{slug(text)}"


def schema_definition_ref_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("schema_definition_ref:"):
        text = text.split(":", 1)[1]
    return f"schema_definition_ref:{slug(text)}"


def ontology_run_ref_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("ontology_run_ref:"):
        text = text.split(":", 1)[1]
    return f"ontology_run_ref:{slug(text)}"


def agent_session_ref_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("agent_session_ref:"):
        text = text.split(":", 1)[1]
    return f"agent_session_ref:{slug(text)}"


def model_call_ref_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("model_call_ref:"):
        text = text.split(":", 1)[1]
    return f"model_call_ref:{slug(text)}"


def tool_call_ref_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("tool_call_ref:"):
        text = text.split(":", 1)[1]
    return f"tool_call_ref:{slug(text)}"


def computed_snapshot_ref_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("computed_snapshot_ref:"):
        text = text.split(":", 1)[1]
    return f"computed_snapshot_ref:{slug(text)}"


def market_regime_snapshot_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("market_regime_snapshot:"):
        text = text.split(":", 1)[1]
    return f"market_regime_snapshot:{slug(text)}"


def signal_factor_score_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("signal_factor_score:"):
        text = text.split(":", 1)[1]
    return f"signal_factor_score:{slug(text)}"


def forward_outlook_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("forward_outlook:"):
        text = text.split(":", 1)[1]
    return f"forward_outlook:{slug(text)}"


def regime_episode_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("regime_episode:"):
        text = text.split(":", 1)[1]
    return f"regime_episode:{slug(text)}"


def position_risk_snapshot_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("position_risk_snapshot:"):
        text = text.split(":", 1)[1]
    return f"position_risk_snapshot:{slug(text)}"


def portfolio_risk_snapshot_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("portfolio_risk_snapshot:"):
        text = text.split(":", 1)[1]
    return f"portfolio_risk_snapshot:{slug(text)}"


def executed_action_id(identifier: object) -> str:
    return f"executed_action:{slug(identifier)}"


def executed_decision_record_id(identifier: object) -> str:
    return f"executed_decision_record:{slug(identifier)}"


def audit_event_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("audit_event:"):
        text = text.split(":", 1)[1]
    return f"audit_event:{slug(text)}"


def sector_id(name: object) -> str:
    return f"sector:{slug(name)}"


def macro_indicator_id(indicator_key: object) -> str:
    return f"macro_indicator:{slug(indicator_key)}"


def signal_id(source: object, name: object) -> str:
    return f"signal:{slug(source)}:{slug(name)}"


def thesis_id(ticker: object) -> str:
    return f"thesis:{canonical_ticker(ticker)}"


def evaluation_id(ticker: object, evaluated_at: object) -> str:
    return f"evaluation:{canonical_ticker(ticker)}:{canonical_timestamp_key(evaluated_at)}"


def catalyst_id(ticker: object, name: object, description: object) -> str:
    return f"catalyst:{canonical_ticker(ticker)}:{slug(name)}:{short_hash(description)}"


def hedge_position_id(ticker: object) -> str:
    return f"hedge_position:{canonical_ticker(ticker)}"


def kill_condition_id(ticker: object, identifier: object) -> str:
    return f"kill_condition:{canonical_ticker(ticker)}:{slug(identifier)}"


def thesis_claim_id(ticker: object, identifier: object) -> str:
    return f"thesis_claim:{canonical_ticker(ticker)}:{slug(identifier)}"


def evidence_id(identifier: object) -> str:
    return f"evidence:{slug(identifier)}"


def citation_id(identifier: object) -> str:
    return f"citation:{slug(identifier)}"


def action_item_id(identifier: object) -> str:
    return f"action_item:{slug(identifier)}"


def watch_trigger_id(identifier: object) -> str:
    return f"watch_trigger:{slug(identifier)}"


def monitor_hit_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("monitor_hit:"):
        text = text.split(":", 1)[1]
    return f"monitor_hit:{slug(text)}"


def monitor_definition_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("monitor_definition:"):
        text = text.split(":", 1)[1]
    return f"monitor_definition:{slug(text)}"


def mission_definition_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("mission_definition:"):
        text = text.split(":", 1)[1]
    return f"mission_definition:{slug(text)}"


def opportunity_candidate_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("opportunity_candidate:"):
        text = text.split(":", 1)[1]
    return f"opportunity_candidate:{slug(text)}"


def approval_id(identifier: object) -> str:
    return f"approval:{slug(identifier)}"


def action_run_id(identifier: object) -> str:
    return f"action_run:{slug(identifier)}"


def action_event_id(identifier: object) -> str:
    return f"action_event:{slug(identifier)}"


def provenance_event_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("provenance_event:"):
        text = text.split(":", 1)[1]
    return f"provenance_event:{slug(text)}"


def workflow_run_id(identifier: object) -> str:
    return f"workflow_run:{slug(identifier)}"


def workflow_artifact_id(identifier: object) -> str:
    return f"workflow_artifact:{slug(identifier)}"


def recommendation_id(identifier: object) -> str:
    return f"recommendation:{slug(identifier)}"


def course_of_action_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("course_of_action:"):
        text = text.split(":", 1)[1]
    return f"course_of_action:{slug(text)}"


def course_of_action_comparison_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("course_of_action_comparison:"):
        text = text.split(":", 1)[1]
    return f"course_of_action_comparison:{slug(text)}"


def scenario_assumption_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("scenario_assumption:"):
        text = text.split(":", 1)[1]
    return f"scenario_assumption:{slug(text)}"


def simulated_outcome_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("simulated_outcome:"):
        text = text.split(":", 1)[1]
    return f"simulated_outcome:{slug(text)}"


def decision_outcome_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("decision_outcome:"):
        text = text.split(":", 1)[1]
    return f"decision_outcome:{slug(text)}"


def course_of_action_rationale_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("course_of_action_rationale:"):
        text = text.split(":", 1)[1]
    return f"course_of_action_rationale:{slug(text)}"


def course_of_action_dissent_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("course_of_action_dissent:"):
        text = text.split(":", 1)[1]
    return f"course_of_action_dissent:{slug(text)}"


def report_run_id(identifier: object) -> str:
    return f"report_run:{slug(identifier)}"


def document_artifact_id(kind: object, identifier: object) -> str:
    return f"document_artifact:{slug(kind)}:{slug(identifier)}"


def source_manifest_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("source_manifest:"):
        text = text.split(":", 1)[1]
    return f"source_manifest:{slug(text)}"


def media_artifact_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("media_artifact:"):
        text = text.split(":", 1)[1]
    return f"media_artifact:{slug(text)}"


def extraction_run_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("extraction_run:"):
        text = text.split(":", 1)[1]
    return f"extraction_run:{slug(text)}"


def observation_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("observation:"):
        text = text.split(":", 1)[1]
    return f"observation:{slug(text)}"


def classification_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("classification:"):
        text = text.split(":", 1)[1]
    return f"classification:{slug(text)}"


def pattern_detection_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("pattern_detection:"):
        text = text.split(":", 1)[1]
    return f"pattern_detection:{slug(text)}"


def analyst_feedback_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("analyst_feedback:"):
        text = text.split(":", 1)[1]
    return f"analyst_feedback:{slug(text)}"


def equity_overview_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("equity_overview:"):
        text = text.split(":", 1)[1]
    return f"equity_overview:{slug(text)}"


def company_financial_profile_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("company_financial_profile:"):
        text = text.split(":", 1)[1]
    return f"company_financial_profile:{slug(text)}"


def extrinsic_sensitivity_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("extrinsic_sensitivity:"):
        text = text.split(":", 1)[1]
    return f"extrinsic_sensitivity:{slug(text)}"


def industry_force_assessment_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("industry_force_assessment:"):
        text = text.split(":", 1)[1]
    return f"industry_force_assessment:{slug(text)}"


def supply_demand_outlook_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("supply_demand_outlook:"):
        text = text.split(":", 1)[1]
    return f"supply_demand_outlook:{slug(text)}"


def supply_chain_relationship_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("supply_chain_relationship:"):
        text = text.split(":", 1)[1]
    return f"supply_chain_relationship:{slug(text)}"


def thesis_document_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("thesis_document:"):
        text = text.split(":", 1)[1]
    return f"thesis_document:{slug(text)}"


def thesis_section_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("thesis_section:"):
        text = text.split(":", 1)[1]
    return f"thesis_section:{slug(text)}"


def investment_idea_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("investment_idea:"):
        text = text.split(":", 1)[1]
    return f"investment_idea:{slug(text)}"


def idea_evaluation_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("idea_evaluation:"):
        text = text.split(":", 1)[1]
    return f"idea_evaluation:{slug(text)}"


def idea_lifecycle_event_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("idea_lifecycle_event:"):
        text = text.split(":", 1)[1]
    return f"idea_lifecycle_event:{slug(text)}"


def idea_comparison_run_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("idea_comparison_run:"):
        text = text.split(":", 1)[1]
    return f"idea_comparison_run:{slug(text)}"


def idea_comparison_ranking_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("idea_comparison_ranking:"):
        text = text.split(":", 1)[1]
    return f"idea_comparison_ranking:{slug(text)}"


def factor_score_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("factor_score:"):
        text = text.split(":", 1)[1]
    return f"factor_score:{slug(text)}"


def missing_information_requirement_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("missing_information_requirement:"):
        text = text.split(":", 1)[1]
    return f"missing_information_requirement:{slug(text)}"


def optimization_mission_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("optimization_mission:"):
        text = text.split(":", 1)[1]
    return f"optimization_mission:{slug(text)}"


def optimization_run_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("optimization_run:"):
        text = text.split(":", 1)[1]
    return f"optimization_run:{slug(text)}"


def optimization_action_snapshot_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("optimization_action_snapshot:"):
        text = text.split(":", 1)[1]
    return f"optimization_action_snapshot:{slug(text)}"


def optimization_alert_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("optimization_alert:"):
        text = text.split(":", 1)[1]
    return f"optimization_alert:{slug(text)}"


def source_freshness_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("source_freshness:"):
        text = text.split(":", 1)[1]
    return f"source_freshness:{slug(text)}"


def management_quality_assessment_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("management_quality_assessment:"):
        text = text.split(":", 1)[1]
    return f"management_quality_assessment:{slug(text)}"


def management_quality_scorecard_row_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("management_quality_scorecard_row:"):
        text = text.split(":", 1)[1]
    return f"management_quality_scorecard_row:{slug(text)}"


def management_quality_accomplishment_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("management_quality_accomplishment:"):
        text = text.split(":", 1)[1]
    return f"management_quality_accomplishment:{slug(text)}"


def management_quality_setback_id(identifier: object) -> str:
    text = str(identifier or "").strip()
    if text.startswith("management_quality_setback:"):
        text = text.split(":", 1)[1]
    return f"management_quality_setback:{slug(text)}"
