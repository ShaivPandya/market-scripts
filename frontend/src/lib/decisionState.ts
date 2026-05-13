import type { StatusTone } from "@/components/shared/StatusBadge"

export type DecisionState =
  | "draft"
  | "analysis"
  | "recommendation"
  | "proposal"
  | "pending_approval"
  | "approved"
  | "rejected"
  | "failed"
  | "applied"
  | "executed"

export type DecisionKind =
  | "draft"
  | "analysis"
  | "recommendation"
  | "proposal"
  | "approval"
  | "internal_state_change"
  | "external_execution"

export type EffectScope = "read_only" | "internal_state" | "external_execution"
export type PolicyState = "pass" | "warn" | "review_required" | "blocked" | "error" | "missing" | string
export type QualityState = "ok" | "degraded" | "stale" | "failed" | "missing" | string
export type LineageState = "complete" | "partial" | "partial" | "retry_pending" | "missing" | string
export type BaseStateStatus = "valid" | "stale" | "untracked" | "unknown" | string

export interface DecisionStateFields {
  decision_state?: DecisionState | string | null
  decision_kind?: DecisionKind | string | null
  effect_scope?: EffectScope | string | null
  execution_capability?: string | null
  policy_state?: PolicyState | null
  quality_state?: QualityState | null
  lineage_state?: LineageState | null
  base_state_status?: BaseStateStatus | null
  base_state_valid?: boolean | null
  base_state_message?: string | null
  application_status?: string | null
  approval_state?: string | null
  outcome_state?: string | null
  confidence?: number | null
  as_of?: string | null
}

export function humanizeDecisionValue(value: string | null | undefined): string {
  if (!value) return "Unknown"
  return value.replace(/_/g, " ").replace(/\b\w/g, ch => ch.toUpperCase())
}

export function decisionStateLabel(state: string | null | undefined): string {
  switch (state) {
    case "draft":
      return "Draft"
    case "analysis":
      return "Analysis"
    case "recommendation":
      return "Recommendation"
    case "proposal":
      return "Proposal"
    case "pending_approval":
      return "Pending Approval"
    case "approved":
      return "Approved"
    case "rejected":
      return "Rejected"
    case "failed":
      return "Failed"
    case "applied":
      return "Applied Internal Change"
    case "executed":
      return "External Execution"
    default:
      return humanizeDecisionValue(state)
  }
}

export function decisionStateTone(state: string | null | undefined): StatusTone {
  switch (state) {
    case "analysis":
    case "recommendation":
      return "info"
    case "pending_approval":
    case "proposal":
    case "approved":
      return "warning"
    case "applied":
    case "executed":
      return "success"
    case "rejected":
      return "neutral"
    case "failed":
      return "error"
    case "draft":
    default:
      return "neutral"
  }
}

export function policyTone(state: string | null | undefined): StatusTone {
  switch (state) {
    case "pass":
      return "success"
    case "warn":
    case "review_required":
      return "warning"
    case "blocked":
    case "error":
      return "error"
    default:
      return "neutral"
  }
}

export function qualityTone(state: string | null | undefined): StatusTone {
  switch (state) {
    case "ok":
      return "success"
    case "degraded":
    case "stale":
      return "warning"
    case "failed":
      return "error"
    default:
      return "neutral"
  }
}

export function approvalDecisionState(value: {
  status?: string | null
  application_status?: string | null
  decision_state?: string | null
}): DecisionState {
  if (value.decision_state) return value.decision_state as DecisionState
  const status = String(value.status ?? "pending")
  const app = String(value.application_status ?? "pending")
  if (status === "rejected") return "rejected"
  if (app === "failed") return "failed"
  if (status === "approved" && app === "applied") return "applied"
  if (status === "approved") return "approved"
  if (status === "pending") return "pending_approval"
  return "proposal"
}

export function recommendationDecisionState(value: {
  recommendation_status?: string | null
  critical_data_quality?: string | null
  approval_status?: string | null
  decision_state?: string | null
}): DecisionState {
  if (value.decision_state) return value.decision_state as DecisionState
  const approval = String(value.approval_status ?? "none")
  if (approval === "pending") return "pending_approval"
  if (approval === "approved") return "approved"
  if (approval === "rejected") return "rejected"
  if (value.recommendation_status === "error" || value.critical_data_quality === "failed") return "failed"
  return "recommendation"
}
