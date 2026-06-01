import type {
  ApprovalRecord,
  ApprovalSourceHealthReview,
  CourseOfActionRecord,
  DecisionOutcomeRecord,
  DecisionQualityGate,
  OpportunityCandidateRecord,
  PolicyGateResult,
  ProvenanceSelector,
  RecommendationRecord,
  SourceHealth,
} from "@/lib/api"
import type { AgentMessage, EgressRecord, ToolCall } from "@/hooks/agentChatShared"
import type { DecisionStateFields } from "@/lib/decisionState"
import { humanizeDecisionValue } from "@/lib/decisionState"

export type DecisionTraceEntityKind =
  | "approval"
  | "recommendation"
  | "course_of_action"
  | "opportunity_candidate"
  | "decision_outcome"
  | "workflow_run"
  | "monitor_hit"
  | "source_record"
  | "agent_session"
  | "agent_message"

export interface DecisionTraceBlocker {
  code?: string
  message: string
  severity?: "info" | "warning" | "blocker" | string
}

export interface DecisionTraceGate {
  label: string
  status: string
  originalAction?: string
  finalAction?: string
  reasons: DecisionTraceBlocker[]
}

export interface DecisionTraceSource {
  id?: string
  name: string
  domain?: string
  status?: string
  qualityState?: string
  stale?: boolean
  detail?: string
  reason?: string
  required?: boolean
}

export interface DecisionTraceTool {
  name: string
  status?: string
  message?: string
  blocksActionable?: boolean
}

export interface DecisionTraceSummary {
  title: string
  subtitle?: string
  entityKind: DecisionTraceEntityKind
  decisionState?: string | null
  effectScope?: string | null
  policyState?: string | null
  qualityState?: string | null
  lineageState?: string | null
  ticker?: string | null
  asOf?: string | null
}

export interface AgentTraceSnapshot {
  decisionQualityChat?: Record<string, unknown> | null
  scoutSkepticSizerGate?: Record<string, unknown> | null
  opportunityCandidatePreflight?: Record<string, unknown> | null
}

export interface DecisionTraceModel {
  summary: DecisionTraceSummary
  blockers: DecisionTraceBlocker[]
  gates: DecisionTraceGate[]
  sources: DecisionTraceSource[]
  tools: DecisionTraceTool[]
  provenanceSelector: ProvenanceSelector | null
  notes: string[]
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

function gateReasons(gate: DecisionQualityGate | null | undefined): DecisionTraceBlocker[] {
  if (!gate || !Array.isArray(gate.reasons)) return []
  return gate.reasons
    .filter(reason => reason?.message || reason?.code)
    .map(reason => ({
      code: reason.code,
      message: reason.message || reason.code || "Gate reason",
      severity: reason.severity,
    }))
}

function policyGateReasons(gate: PolicyGateResult | null | undefined): DecisionTraceBlocker[] {
  if (!gate) return []
  const reasons = [
    ...(Array.isArray(gate.failure_reasons) ? gate.failure_reasons : []),
    ...(Array.isArray(gate.warnings) ? gate.warnings : []),
  ]
  return reasons
    .filter(reason => reason?.message || reason?.code || reason?.check)
    .map(reason => ({
      code: reason.code || reason.check,
      message: reason.message || reason.code || reason.check || "Policy gate reason",
      severity: gate.failure_reasons?.includes(reason) ? "blocker" : "warning",
    }))
}

function decisionQualityGateTrace(gate: DecisionQualityGate | null | undefined, label = "Decision Quality"): DecisionTraceGate | null {
  if (!gate) return null
  return {
    label,
    status: String(gate.status || "unknown"),
    originalAction: gate.original_action,
    finalAction: gate.final_action,
    reasons: gateReasons(gate),
  }
}

function policyGateTrace(gate: PolicyGateResult | null | undefined): DecisionTraceGate | null {
  if (!gate) return null
  return {
    label: "Policy Gate",
    status: String(gate.decision || "unknown"),
    reasons: policyGateReasons(gate),
  }
}

function summaryFromFields(
  fields: DecisionStateFields,
  kind: DecisionTraceEntityKind,
  title: string,
  extras?: Partial<DecisionTraceSummary>,
): DecisionTraceSummary {
  return {
    title,
    entityKind: kind,
    decisionState: fields.decision_state,
    effectScope: fields.effect_scope,
    policyState: fields.policy_state,
    qualityState: fields.quality_state,
    lineageState: fields.lineage_state,
    ...extras,
  }
}

function sourceHealthReviewSources(review: ApprovalSourceHealthReview | null | undefined): DecisionTraceSource[] {
  if (!review) return []
  const mapIssue = (issue: (typeof review.blockers)[number], required = true): DecisionTraceSource => ({
    id: issue.id ?? undefined,
    name: issue.source_name || issue.domain || "Source",
    domain: issue.domain ?? undefined,
    status: issue.status ?? undefined,
    qualityState: issue.quality_state ?? undefined,
    detail: issue.detail ?? undefined,
    reason: issue.reason ?? undefined,
    required,
  })
  return [
    ...review.blockers.map(issue => mapIssue(issue, true)),
    ...review.warnings.map(issue => mapIssue(issue, false)),
  ]
}

function workspaceSourceHealthSlice(sourceHealth: SourceHealth | null | undefined, domains?: string[]): DecisionTraceSource[] {
  if (!sourceHealth?.domains?.length) return []
  const selectedDomains = domains?.length
    ? sourceHealth.domains.filter(domain => domains.includes(domain.domain))
    : sourceHealth.domains
  return selectedDomains.flatMap(domain =>
    domain.sources.map(source => ({
      id: source.id,
      name: source.source_name,
      domain: source.domain,
      status: source.status,
      qualityState: source.quality_state,
      stale: source.stale,
      detail: source.detail ?? undefined,
      required: source.required,
    })),
  )
}

function missingInputBlockers(values: string[] | null | undefined): DecisionTraceBlocker[] {
  return (values ?? [])
    .filter(Boolean)
    .map(value => ({ code: "MISSING_INPUT", message: value, severity: "blocker" }))
}

function scoutSkepticSizerGateTrace(payload: Record<string, unknown> | null | undefined): DecisionTraceGate[] {
  if (!payload || payload.enabled === false) return []
  const gates: DecisionTraceGate[] = []
  for (const pass of ["scout", "skeptic", "sizer"] as const) {
    const section = asRecord(payload[pass])
    if (!section.ran && section.status == null) continue
    const reasonCodes = Array.isArray(section.reason_codes) ? section.reason_codes as string[] : []
    gates.push({
      label: humanizeDecisionValue(pass),
      status: String(section.status || "unknown"),
      reasons: reasonCodes.map(code => ({ code, message: code.replace(/_/g, " "), severity: "warning" })),
    })
  }
  if (payload.gate_status) {
    gates.unshift({
      label: "Scout / Skeptic / Sizer",
      status: String(payload.gate_status),
      originalAction: payload.original_action_type ? String(payload.original_action_type) : undefined,
      finalAction: payload.final_action_type ? String(payload.final_action_type) : undefined,
      reasons: [],
    })
  }
  return gates
}

function toolQualityTools(payload: Record<string, unknown> | null | undefined): DecisionTraceTool[] {
  const summaries = Array.isArray(payload?.tool_summaries) ? payload.tool_summaries as Record<string, unknown>[] : []
  return summaries.map(summary => ({
    name: String(summary.name || "tool"),
    status: summary.tool_status ? String(summary.tool_status) : summary.source_status ? String(summary.source_status) : undefined,
    message: summary.reason ? String(summary.reason) : undefined,
    blocksActionable: summary.blocks_actionable === true,
  }))
}

function safeToolCalls(toolCalls: ToolCall[] | undefined): DecisionTraceTool[] {
  return (toolCalls ?? []).map(tool => ({
    name: tool.name,
    status: tool.status,
    message: tool.message,
  }))
}

function safeEgressNotes(records: EgressRecord[] | undefined): string[] {
  return (records ?? []).map(record => {
    const parts = [
      record.decision ? `Egress ${record.decision}` : null,
      record.decisionReason,
      record.dataSensitivity ? `Sensitivity ${record.dataSensitivity}` : null,
    ].filter(Boolean)
    return parts.join(" · ")
  })
}

export function provenanceSelectorForRecord(
  kind: DecisionTraceEntityKind,
  record: Record<string, unknown>,
): ProvenanceSelector | null {
  switch (kind) {
    case "approval": {
      const id = String(record.id ?? "").trim()
      return id ? { approval_id: id } : null
    }
    case "recommendation": {
      const id = String(record.id ?? "").trim()
      return id ? { recommendation_id: id } : null
    }
    case "course_of_action": {
      const refId = String(record.course_of_action_id ?? record.id ?? "").trim()
      return refId ? { ref_type: "CourseOfAction", ref_id: refId } : null
    }
    case "opportunity_candidate": {
      const refId = String(record.candidate_id ?? record.id ?? "").trim()
      return refId ? { ref_type: "OpportunityCandidate", ref_id: refId } : null
    }
    case "decision_outcome": {
      const recommendationId = String(record.recommendation_id ?? "").trim()
      if (recommendationId) return { recommendation_id: recommendationId }
      const refId = String(record.decision_outcome_id ?? record.object_uid ?? record.id ?? "").trim()
      return refId ? { ref_type: "DecisionOutcome", ref_id: refId } : null
    }
    case "workflow_run": {
      const runId = String(record.run_id ?? "").trim()
      return runId ? { workflow_run_id: runId } : null
    }
    case "monitor_hit": {
      const approvalId = String(record.approval_id ?? "").trim()
      if (approvalId) return { approval_id: approvalId }
      const refId = String(record.id ?? "").trim()
      return refId ? { ref_type: "MonitorHit", ref_id: refId } : null
    }
    case "source_record": {
      const sourceRecordId = String(record.source_record_id ?? record.id ?? "").trim()
      return sourceRecordId ? { source_record_id: sourceRecordId } : null
    }
    case "agent_session":
    case "agent_message": {
      const sessionId = String(record.session_id ?? record.agent_session_id ?? "").trim()
      return sessionId ? { agent_session_id: sessionId } : null
    }
    default:
      return null
  }
}

export function buildApprovalTrace(
  approval: ApprovalRecord,
): DecisionTraceModel {
  const gates = [policyGateTrace(approval.policy_gate)].filter(Boolean) as DecisionTraceGate[]
  const blockers: DecisionTraceBlocker[] = [
    ...sourceHealthReviewSources(approval.source_health_review)
      .filter(source => source.reason || source.detail)
      .map(source => ({
        code: source.status ? String(source.status).toUpperCase() : "SOURCE_HEALTH",
        message: source.reason || source.detail || `${source.name} blocked review`,
        severity: "blocker" as const,
      })),
    ...policyGateReasons(approval.policy_gate).filter(reason => reason.severity === "blocker"),
  ]
  if (approval.base_state_status === "stale") {
    blockers.unshift({
      code: "BASE_STATE_STALE",
      message: approval.base_state_message || "Underlying state changed after proposal creation.",
      severity: "blocker",
    })
  }
  return {
    summary: summaryFromFields(approval, "approval", traceApprovalLabel(approval.id), {
      subtitle: approval.reason || approval.entity_type,
      ticker: approval.ticker,
    }),
    blockers,
    gates,
    sources: sourceHealthReviewSources(approval.source_health_review),
    tools: [],
    provenanceSelector: provenanceSelectorForRecord("approval", approval as unknown as Record<string, unknown>),
    notes: approval.application_error ? [`Application error: ${approval.application_error}`] : [],
  }
}

function traceApprovalLabel(id: string): string {
  const trimmed = String(id || "").trim()
  if (!trimmed) return "Approval"
  return trimmed.length > 24 ? `Approval ${trimmed.slice(0, 8)}…` : `Approval ${trimmed}`
}

export function buildRecommendationTrace(rec: RecommendationRecord): DecisionTraceModel {
  const gates = [
    policyGateTrace(rec.policy_gate),
    decisionQualityGateTrace(rec.decision_quality_gate),
  ].filter(Boolean) as DecisionTraceGate[]
  return {
    summary: summaryFromFields(rec, "recommendation", rec.action.replace(/_/g, " "), {
      subtitle: rec.report_type,
      ticker: rec.instrument,
      asOf: rec.as_of,
    }),
    blockers: gateReasons(rec.decision_quality_gate).filter(reason => reason.severity === "blocker"),
    gates,
    sources: [],
    tools: [],
    provenanceSelector: provenanceSelectorForRecord("recommendation", rec as unknown as Record<string, unknown>),
    notes: rec.rationale ? [rec.rationale] : [],
  }
}

export function buildCourseOfActionTrace(coa: CourseOfActionRecord): DecisionTraceModel {
  const gates = [
    policyGateTrace(coa.policy_gate),
    decisionQualityGateTrace(coa.decision_quality_gate),
    ...scoutSkepticSizerGateTrace(asRecord(coa.payload?.scout_skeptic_sizer_gate)),
  ].filter(Boolean) as DecisionTraceGate[]
  return {
    summary: summaryFromFields(coa, "course_of_action", coa.action.replace(/_/g, " "), {
      subtitle: coa.actionability?.replace(/_/g, " "),
      ticker: coa.ticker,
      asOf: coa.as_of,
    }),
    blockers: gateReasons(coa.decision_quality_gate).filter(reason => reason.severity === "blocker"),
    gates,
    sources: [],
    tools: [],
    provenanceSelector: provenanceSelectorForRecord("course_of_action", coa as unknown as Record<string, unknown>),
    notes: coa.rationale_summary ? [coa.rationale_summary] : [],
  }
}

export function buildOpportunityCandidateTrace(candidate: OpportunityCandidateRecord): DecisionTraceModel {
  const gateStatus = String(candidate.gate_status || "unknown")
  const gates: DecisionTraceGate[] = [{
    label: "Opportunity Candidate Gate",
    status: gateStatus,
    finalAction: candidate.gate_final_action,
    reasons: missingInputBlockers(candidate.missing_inputs),
  }]
  const sourceRefs = (candidate.source_refs ?? []).slice(0, 8).map((ref, index) => ({
    id: String(ref.source_record_id ?? ref.id ?? index),
    name: String(ref.source_name ?? ref.title ?? "Source reference"),
    domain: ref.domain ? String(ref.domain) : undefined,
    status: ref.status ? String(ref.status) : undefined,
    detail: ref.summary ? String(ref.summary) : undefined,
  }))
  return {
    summary: {
      title: candidate.ticker || "Sector / Thematic",
      subtitle: candidate.opportunity_type.replace(/_/g, " "),
      entityKind: "opportunity_candidate",
      decisionState: candidate.decision_state ?? candidate.status,
      ticker: candidate.ticker,
      asOf: candidate.updated_at,
    },
    blockers: missingInputBlockers(candidate.missing_inputs),
    gates,
    sources: sourceRefs,
    tools: [],
    provenanceSelector: provenanceSelectorForRecord("opportunity_candidate", candidate as unknown as Record<string, unknown>),
    notes: [
      candidate.trigger,
      candidate.why_now ? `Why now: ${candidate.why_now}` : "",
      candidate.price_confirmation ? `Price confirmation: ${candidate.price_confirmation}` : "",
    ].filter(Boolean),
  }
}

export function buildDecisionOutcomeTrace(outcome: DecisionOutcomeRecord): DecisionTraceModel {
  return {
    summary: {
      title: outcome.ticker || "Decision outcome",
      subtitle: outcome.process_label?.replace(/_/g, " "),
      entityKind: "decision_outcome",
      decisionState: outcome.decision_state ?? outcome.learning_state,
      ticker: outcome.ticker,
      asOf: outcome.as_of,
      lineageState: outcome.lineage_state,
    },
    blockers: [],
    gates: [],
    sources: [],
    tools: [],
    provenanceSelector: provenanceSelectorForRecord("decision_outcome", outcome as unknown as Record<string, unknown>),
    notes: [
      outcome.draft_postmortem ? String(outcome.draft_postmortem) : "",
      outcome.final_postmortem ? String(outcome.final_postmortem) : "",
    ].filter(Boolean),
  }
}

export function buildWorkflowRunTrace(run: Record<string, unknown>): DecisionTraceModel {
  const runId = String(run.run_id ?? "")
  return {
    summary: {
      title: String(run.workflow_name ?? "Workflow run").replace(/_/g, " "),
      subtitle: String(run.status ?? "unknown"),
      entityKind: "workflow_run",
      ticker: run.ticker ? String(run.ticker) : null,
      asOf: run.completed_at ? String(run.completed_at) : run.started_at ? String(run.started_at) : null,
    },
    blockers: run.error ? [{ code: "WORKFLOW_ERROR", message: String(run.error), severity: "blocker" }] : [],
    gates: [],
    sources: [],
    tools: [],
    provenanceSelector: provenanceSelectorForRecord("workflow_run", run),
    notes: run.synthesis ? [String(run.synthesis).slice(0, 500)] : [],
  }
}

export function buildMonitorHitTrace(hit: Record<string, unknown>): DecisionTraceModel {
  return {
    summary: {
      title: hit.ticker ? String(hit.ticker) : "Monitor hit",
      subtitle: String(hit.hit_type ?? "monitor hit").replace(/_/g, " "),
      entityKind: "monitor_hit",
      ticker: hit.ticker ? String(hit.ticker) : null,
      asOf: hit.detected_at ? String(hit.detected_at) : null,
    },
    blockers: [],
    gates: [],
    sources: [],
    tools: [],
    provenanceSelector: provenanceSelectorForRecord("monitor_hit", hit),
    notes: [
      hit.entity_label ? String(hit.entity_label) : "",
      hit.evidence ? String(hit.evidence) : "",
    ].filter(Boolean),
  }
}

export function buildSourceRecordTrace(sourceRecordId: string, label?: string): DecisionTraceModel {
  return {
    summary: {
      title: label || "Source record",
      entityKind: "source_record",
    },
    blockers: [],
    gates: [],
    sources: [],
    tools: [],
    provenanceSelector: { source_record_id: sourceRecordId },
    notes: [],
  }
}

export function buildAgentMessageTrace(
  sessionId: string | null | undefined,
  message: AgentMessage,
): DecisionTraceModel {
  const snapshot = message.traceSnapshot
  const dqChat = asRecord(snapshot?.decisionQualityChat)
  const preflight = asRecord(snapshot?.opportunityCandidatePreflight)
  const scoutGate = asRecord(snapshot?.scoutSkepticSizerGate)
  const toolQuality = asRecord(dqChat.tool_quality)
  const gates: DecisionTraceGate[] = []
  if (dqChat.ran || dqChat.gate_status) {
    gates.push({
      label: "Decision Quality Chat",
      status: String(dqChat.gate_status || "unknown"),
      finalAction: dqChat.final_action ? String(dqChat.final_action) : undefined,
      reasons: Number(dqChat.missing_inputs_count || 0) > 0
        ? [{ code: "MISSING_INPUTS", message: `${dqChat.missing_inputs_count} missing inputs`, severity: "warning" }]
        : [],
    })
  }
  if (preflight.ran || preflight.gate_status) {
    gates.push({
      label: "Opportunity Candidate Preflight",
      status: String(preflight.gate_status || (preflight.ran ? "ran" : "skipped")),
      finalAction: preflight.final_action ? String(preflight.final_action) : undefined,
      reasons: Number(preflight.missing_inputs_count || 0) > 0
        ? [{ code: "MISSING_INPUTS", message: `${preflight.missing_inputs_count} missing inputs`, severity: "warning" }]
        : [],
    })
  }
  gates.push(...scoutSkepticSizerGateTrace(scoutGate))

  const blockers = Array.isArray(toolQuality.blocking_reason_codes)
    ? (toolQuality.blocking_reason_codes as string[]).map(code => ({
        code,
        message: code.replace(/_/g, " "),
        severity: "blocker",
      }))
    : []

  return {
    summary: {
      title: "Stan response",
      subtitle: message.clientTurnId ? `Turn ${message.clientTurnId}` : undefined,
      entityKind: "agent_message",
      decisionState: message.toolCalls?.some(tool => tool.status === "blocked") ? "blocked" : "analysis",
    },
    blockers,
    gates,
    sources: [],
    tools: [
      ...toolQualityTools(toolQuality),
      ...safeToolCalls(message.toolCalls),
    ],
    provenanceSelector: sessionId ? { agent_session_id: sessionId } : null,
    notes: safeEgressNotes(message.egressRecords),
  }
}

export function buildProvenanceOnlyTrace(selector: ProvenanceSelector, title = "Lineage trace"): DecisionTraceModel {
  return {
    summary: {
      title,
      entityKind: "source_record",
    },
    blockers: [],
    gates: [],
    sources: [],
    tools: [],
    provenanceSelector: selector,
    notes: [],
  }
}

export function buildDecisionTrace(
  kind: DecisionTraceEntityKind,
  record: Record<string, unknown>,
  options?: { sourceHealth?: SourceHealth | null; sessionId?: string | null; message?: AgentMessage },
): DecisionTraceModel {
  switch (kind) {
    case "approval":
      return buildApprovalTrace(record as unknown as ApprovalRecord)
    case "recommendation":
      return buildRecommendationTrace(record as unknown as RecommendationRecord)
    case "course_of_action":
      return buildCourseOfActionTrace(record as unknown as CourseOfActionRecord)
    case "opportunity_candidate":
      return buildOpportunityCandidateTrace(record as unknown as OpportunityCandidateRecord)
    case "decision_outcome":
      return buildDecisionOutcomeTrace(record as unknown as DecisionOutcomeRecord)
    case "workflow_run":
      return buildWorkflowRunTrace(record)
    case "monitor_hit":
      return buildMonitorHitTrace(record)
    case "source_record":
      return buildSourceRecordTrace(String(record.source_record_id ?? record.id ?? ""), String(record.label ?? ""))
    case "agent_message":
      if (!options?.message) {
        return buildProvenanceOnlyTrace(
          provenanceSelectorForRecord("agent_session", { session_id: options?.sessionId }) ?? {},
          "Stan response",
        )
      }
      return buildAgentMessageTrace(options.sessionId, options.message)
    case "agent_session":
      return buildProvenanceOnlyTrace(
        provenanceSelectorForRecord("agent_session", { session_id: options?.sessionId ?? record.session_id }) ?? {},
        "Stan session",
      )
    default:
      return buildProvenanceOnlyTrace(provenanceSelectorForRecord(kind, record) ?? {}, "Decision trace")
  }
}

export function extractAgentTraceSnapshot(payload: Record<string, unknown>): AgentTraceSnapshot {
  return {
    decisionQualityChat: asRecord(payload.decision_quality_chat),
    scoutSkepticSizerGate: asRecord(payload.scout_skeptic_sizer_gate),
    opportunityCandidatePreflight: asRecord(payload.opportunity_candidate_preflight),
  }
}

export { workspaceSourceHealthSlice, traceApprovalLabel }
