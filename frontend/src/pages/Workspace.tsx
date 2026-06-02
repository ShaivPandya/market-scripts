import { Link, useSearchParams } from "react-router-dom"
import { useQueryClient } from "@tanstack/react-query"
import { Bell, CheckCircle, AlertTriangle, Play, Clock, GitBranch, Database, FileText, X, ChevronDown, Shield, Flag, RefreshCw } from "lucide-react"
import { useApiQuery } from "@/hooks/useApiQuery"
import {
  fetchWorkspace,
  fetchApprovals,
  fetchApproval,
  fetchApprovalSummary,
  approveItem,
  rejectItem,
  rejectAndRestageApproval,
  replaceApprovalProposal,
  bulkReject,
  completeAction,
  cancelTrigger,
  dismissAction,
  dismissWorkspaceThesisPressure,
  dismissOptimizationAlert,
  replaceTrigger,
  refreshWorkspaceSources,
  type ApprovalRecord,
  type CourseOfActionComparisonRecord,
  type CourseOfActionRecord,
  type DecisionOutcomeRecord,
  type OptimizationAlert,
  type ThesisClaim,
  type PolicyGateReason,
  type PolicyGateResult,
  type RecommendationRecord,
  type SourceHealth,
  type TriggerMutationBody,
} from "@/lib/api"
import {
  approvalSummaryQueryKey,
  invalidateAfterApprovalResolution,
  invalidateApprovalSummaries,
  formatApprovalResolutionError,
  patchResolvedApprovalSummaries,
  shouldRefetchApprovalSummariesAfterError,
} from "@/lib/approvalQueries"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { PostMortemReviewDialog } from "@/components/shared/PostMortemReviewDialog"
import { TraceTriggerButton } from "@/components/shared/TraceTriggerButton"
import { useDecisionTrace } from "@/contexts/DecisionTraceContext"
import { Dialog } from "@/components/shared/Dialog"
import { ApprovalChangeSummary } from "@/components/shared/ApprovalChangeSummary"
import { ApprovalProgressSummary } from "@/components/shared/ApprovalProgressSummary"
import { approvalActionLabel } from "@/components/shared/approvalProgress"
import { ActionButton } from "@/components/shared/FormControls"
import { formatApprovalDisplayLabel } from "@/components/shared/StagedProposalNotice"
import { WhatChangedPanel, type WhatChangedSummary } from "@/components/shared/WhatChangedPanel"
import { OpportunityScoutQueuePanel } from "@/components/shared/OpportunityScoutQueuePanel"
import type { OpportunityCandidateRecord } from "@/lib/api"
import { WatchTriggerEditDialog, type EditableWatchTrigger } from "@/components/shared/WatchTriggerEditDialog"
import {
  DecisionStateBadge,
  EffectScopeBadge,
  BaseStateBadge,
  PolicyStateBadge,
  QualityStateBadge,
} from "@/components/shared/DecisionStateBadge"
import { approvalDecisionState, courseOfActionDecisionState, recommendationDecisionState } from "@/lib/decisionState"
import { cn } from "@/lib/utils"
import { useCallback, useEffect, useRef, useState } from "react"

interface WorkspaceData {
  regime: {
    regime: string
    composite_score: number
    signal: string
    snapshot?: {
      as_of?: string | null
      stale?: boolean
      refresh_status?: string
      error?: string | null
    } | null
  } | null
  portfolio: {
    position_count: number
    total_pnl: number | null
    total_pnl_pct: number | null
    risk?: {
      result_id?: string | null
      as_of?: string | null
      computed_at?: string | null
      quality?: string | null
      confidence?: number | null
      average_risk_score?: number | null
      max_risk_score?: number | null
      risk_level?: string | null
      risk_buckets?: { high?: number; medium?: number; low?: number } | null
      top_contributors?: Array<Record<string, unknown>>
    } | null
  } | null
  source_health?: SourceHealth | null
  what_changed?: WhatChangedSummary | null
  thesis_pressure: {
    ticker: string
    status: string
    action: string
    confidence: string
    risk_flag: string | null
    evaluated_at: string
    pressure_key: string
  }[]
  pending_approvals: { count: number; items: ApprovalRecord[] }
  recommendations: {
    latest_daily: RecommendationRecord | null
    latest_weekly: RecommendationRecord | null
    pending_actionable: { count: number; items: RecommendationRecord[] }
    blocked_warnings: {
      report_type: string
      as_of: string
      critical_data_quality: string
      blocked_reasons: string[]
    }[]
    pending_approval_count: number
  }
  course_of_actions?: {
    pending: { count: number; items: CourseOfActionRecord[] }
    recent: { count: number; items: CourseOfActionRecord[] }
    comparisons: { count: number; items: CourseOfActionComparisonRecord[] }
    pending_approval_count: number
  }
  open_actions: { count: number; items: ActionItem[] }
  active_triggers: { count: number; items: Trigger[] }
  monitor_hits: { count: number; items: MonitorHit[] }
  opportunity_candidates?: { count: number; items: OpportunityCandidateRecord[] }
  recent_workflow_runs: WorkflowRun[]
  continuous_optimization?: {
    open_alert_count: number
    open_alerts: OptimizationAlert[]
  }
  monitor_builder?: {
    active_monitor_count: number
    active_mission_count: number
    active_monitors: BuilderDefinition[]
    active_missions: BuilderDefinition[]
  }
  thesis_claims?: {
    challenged_count: number
    items: ThesisClaim[]
  }
  decision_learning?: {
    pending_review: { count: number; items: DecisionOutcomeRecord[] }
    recent_finalized: { count: number; items: DecisionOutcomeRecord[] }
  }
  recent_report_runs: ReportRun[]
}

interface ActionItem {
  id: number | string
  ticker: string | null
  description: string
  action_type: string
  urgency: string
  created_at: string
}

interface Trigger {
  id: number | string
  ticker: string | null
  condition: string
  trigger_type: string
  status: string
  created_at: string
  expires_at?: string | null
  definition?: Record<string, unknown> | null
  last_checked_at: string | null
  last_evidence: string | null
}

interface MonitorHit {
  id: number | string
  ticker: string | null
  entity_type: string
  entity_id: string
  entity_label?: string | null
  hit_type: string
  severity?: string | null
  status: string
  confidence?: number | null
  evidence?: string | null
  detected_at?: string | null
  approval_id?: string | null
}

interface BuilderDefinition {
  id?: string | number
  object_uid?: string
  monitor_id?: string
  mission_id?: string
  name: string
  status: string
  condition?: string | null
  trigger_type?: string | null
  mission_type?: string | null
  cadence?: Record<string, unknown> | null
  schedule?: Record<string, unknown> | null
  updated_at?: string | null
}

interface WorkflowRun {
  run_id: string | null
  workflow_name: string | null
  ticker: string | null
  status: string | null
  started_at: string | null
  completed_at: string | null
  synthesis?: string | null
  artifacts?: Record<string, unknown> | null
}

interface ReportRun {
  id?: string | number
  report_id?: string
  report_type?: string
  as_of?: string
  status?: string
  source?: string | null
  synced_at?: string | null
  created_at?: string | null
  error?: string | null
  issue_url?: string | null
}

const REGIME_SIGNAL_MAP: Record<string, { signal: "success" | "warning" | "error"; label: string }> = {
  bullish: { signal: "success", label: "Bullish" },
  neutral: { signal: "warning", label: "Neutral" },
  transitional: { signal: "warning", label: "Transitional" },
  bearish: { signal: "error", label: "Bearish" },
  "risk-off": { signal: "error", label: "Risk-Off" },
  "risk-on": { signal: "success", label: "Risk-On" },
}

const CLAIM_STATUS_COLORS: Record<string, string> = {
  challenged: "text-amber-700 bg-amber-50 dark:text-amber-400 dark:bg-amber-950",
  disconfirmed: "text-red-700 bg-red-50 dark:text-red-400 dark:bg-red-950",
}

const ALERT_SEVERITY_COLORS: Record<string, string> = {
  urgent: "text-red-600 bg-red-50 dark:text-red-400 dark:bg-red-950",
  high: "text-orange-600 bg-orange-50 dark:text-orange-400 dark:bg-orange-950",
  normal: "text-blue-600 bg-blue-50 dark:text-blue-400 dark:bg-blue-950",
  low: "text-gray-600 bg-gray-50 dark:text-gray-400 dark:bg-gray-800",
}

const URGENCY_COLORS: Record<string, string> = {
  urgent: "text-red-600 bg-red-50 dark:text-red-400 dark:bg-red-950",
  high: "text-orange-600 bg-orange-50 dark:text-orange-400 dark:bg-orange-950",
  normal: "text-blue-600 bg-blue-50 dark:text-blue-400 dark:bg-blue-950",
  low: "text-gray-600 bg-gray-50 dark:text-gray-400 dark:bg-gray-800",
}

const ACTIONABLE_RECOMMENDATION_ACTIONS = new Set(["buy", "add", "short", "sell", "trim", "reduce", "exit", "hedge", "rebalance"])
const FINANCIAL_ACTION_ITEM_TYPES = new Set(["enter", "exit", "resize", "hedge"])
const WORKSPACE_APPROVAL_LIMIT = 50
const BULK_DISMISS_APPROVAL_NOTE = "Dismissed from Workspace bulk action."
type ApprovalDialogAction = "approve" | "reject" | "restage"
type TriggerEditState =
  | { kind: "active"; trigger: Trigger }
  | { kind: "approval"; approval: ApprovalRecord; trigger: EditableWatchTrigger }

function formatRiskScore(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "--"
  return value.toFixed(2)
}

function formatTime(iso: string | null | undefined): string {
  const value = String(iso ?? "").trim()
  if (!value) return "Unknown time"
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return value
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })
}

const KNOWN_WORKFLOW_NAMES = [
  "morning_brief",
  "thesis_review",
  "pre_earnings",
  "post_earnings_review",
  "weekly_portfolio_review",
  "thesis_invalidation_check",
]

function workflowNameFromRunId(runId: string | null | undefined): string | null {
  const value = String(runId ?? "").trim()
  if (!value) return null
  if (value.startsWith("workflow:")) {
    const [, workflowName] = value.split(":")
    if (workflowName) return workflowName
  }
  const slug = value.replace(/_/g, "-").toLowerCase()
  return KNOWN_WORKFLOW_NAMES.find(name => slug.includes(name.replace(/_/g, "-"))) ?? null
}

function workflowRunLabel(run: WorkflowRun): string {
  const raw = String(run.workflow_name ?? "").trim()
  const workflowName = raw && raw.toLowerCase() !== "unknown"
    ? raw
    : workflowNameFromRunId(run.run_id) ?? "workflow run"
  return workflowName.replace(/_/g, " ")
}

function workflowRunTicker(run: WorkflowRun): string | null {
  const ticker = String(run.ticker ?? "").trim().toUpperCase()
  if (ticker) return ticker
  const artifactTicker = (run.artifacts?.evaluation_draft as { ticker?: unknown } | undefined)?.ticker
  const artifactValue = String(artifactTicker ?? "").trim().toUpperCase()
  if (artifactValue) return artifactValue
  const titleTicker = String(run.synthesis ?? "").match(/\b([A-Z]{1,6})\s+Thesis Review\b/)?.[1]
  return titleTicker ?? null
}

function workflowStatusClass(status: string | null | undefined): string {
  const value = String(status ?? "").toLowerCase()
  if (["completed", "succeeded", "success", "ok"].includes(value)) return "bg-green-500"
  if (["running", "started", "queued"].includes(value)) return "bg-blue-500 animate-pulse"
  if (["failed", "error"].includes(value)) return "bg-red-500"
  return "bg-gray-400"
}

function workflowRunTime(run: WorkflowRun): string {
  return formatTime(run.started_at ?? run.completed_at)
}

function reportRunLabel(run: ReportRun): string {
  const reportType = cleanText(run.report_type) ?? "report"
  return titleCase(reportType)
}

function reportRunTime(run: ReportRun): string {
  return formatTime(run.as_of ?? run.synced_at ?? run.created_at)
}

function reportRunKey(run: ReportRun, index: number): string {
  return String(run.report_id ?? run.id ?? `report-run-${index}`)
}

function claimStatusClass(status: string | null | undefined): string {
  return CLAIM_STATUS_COLORS[String(status || "").toLowerCase()] ?? "text-gray-600 bg-gray-50 dark:text-gray-400 dark:bg-gray-800"
}

function alertSeverityClass(severity: string | null | undefined): string {
  return ALERT_SEVERITY_COLORS[String(severity || "normal").toLowerCase()] ?? ALERT_SEVERITY_COLORS.normal
}

function ThesisClaimsPanel({ claims, claimCount }: { claims: ThesisClaim[]; claimCount: number }) {
  if (claimCount <= 0) return null
  return (
    <section className="theme-surface flex min-h-0 max-h-[min(56rem,calc(100dvh-8rem))] flex-col overflow-hidden rounded-xl p-4">
      <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
        <AlertTriangle size={14} className="text-amber-500" />
        Thesis Claim Issues
        <span className="ml-auto text-xs text-subtle">{claimCount} challenged or disconfirmed</span>
      </h2>
      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
        {claims.map(claim => (
          <div key={claim.id} className="rounded-lg border border-app px-3 py-2 text-sm">
            <div className="flex flex-wrap items-center gap-2">
              <Link to={`/dossier/${encodeURIComponent(claim.ticker)}`} state={{ from: "workspace" }} className="font-semibold text-app hover:underline">
                {claim.ticker}
              </Link>
              <span className={cn("rounded px-1.5 py-0.5 text-xs font-medium", claimStatusClass(claim.status))}>
                {claim.status}
              </span>
            </div>
            <p className="mt-1 text-xs text-muted line-clamp-2">{claim.claim}</p>
            {claim.disconfirming_evidence && (
              <p className="mt-1 text-xs text-red-600 dark:text-red-400 line-clamp-2">{claim.disconfirming_evidence}</p>
            )}
          </div>
        ))}
      </div>
    </section>
  )
}

function sourceHealthLabel(value: string | null | undefined): string {
  const normalized = String(value || "missing").replace(/_/g, " ")
  return normalized.charAt(0).toUpperCase() + normalized.slice(1)
}

function sourceHealthOverallClass(quality: string | null | undefined): string {
  const value = String(quality || "missing")
  if (value === "ok") return "theme-badge-success"
  if (value === "failed" || value === "missing") return "theme-badge-error"
  if (value === "stale" || value === "degraded") return "theme-badge-warning"
  return "theme-badge-neutral"
}

function reliabilityTierLabel(tier: string | null | undefined): string {
  const value = String(tier || "standard").replace(/_/g, " ")
  return value.charAt(0).toUpperCase() + value.slice(1)
}

function reliabilityTierBadgeClass(tier: string | null | undefined): string {
  const value = String(tier || "standard").toLowerCase()
  if (value === "critical") return "theme-badge-error"
  if (value === "standard") return "theme-badge-warning"
  if (value === "supplemental") return "theme-badge-neutral"
  return "theme-badge-neutral"
}

type ApprovalSourceHealthReview = NonNullable<ApprovalRecord["source_health_review"]>
type ApprovalSourceHealthIssue = ApprovalSourceHealthReview["blockers"][number]

function approvalSourceHealthClass(status: string | null | undefined): string {
  const value = String(status || "ok")
  if (value === "blocked") return "theme-badge-error"
  if (value === "warning") return "theme-badge-warning"
  return "theme-badge-success"
}

function approvalSourceHealthLabel(review: ApprovalSourceHealthReview): string {
  const blockers = review.blockers?.length ?? 0
  const warnings = review.warnings?.length ?? 0
  if (blockers > 0) return `${blockers} source blocker${blockers === 1 ? "" : "s"}`
  if (warnings > 0) return `${warnings} source warning${warnings === 1 ? "" : "s"}`
  return "Sources ok"
}

function ApprovalSourceHealthBadge({ review }: { review?: ApprovalRecord["source_health_review"] }) {
  if (!review || review.status === "ok") return null
  return (
    <span className={cn("theme-badge inline-flex items-center gap-1", approvalSourceHealthClass(review.status))}>
      <Database size={12} aria-hidden="true" />
      {approvalSourceHealthLabel(review)}
    </span>
  )
}

function approvalSourceIssueLabel(issue: ApprovalSourceHealthIssue): string {
  return String(issue.source_name || issue.id || "source").replace(/_/g, " ")
}

function ApprovalSourceHealthPanel({ review }: { review?: ApprovalRecord["source_health_review"] }) {
  if (!review || review.status === "ok") return null
  const rows = [
    ...(review.blockers ?? []).map(issue => ({ ...issue, severity: "blocked" })),
    ...(review.warnings ?? []).map(issue => ({ ...issue, severity: "warning" })),
  ]
  return (
    <div className={cn("rounded-lg border px-3 py-2 text-sm", review.status === "blocked" ? "border-red-200 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200" : "border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200")}>
      <div className="mb-2 flex flex-wrap items-center gap-2 font-medium">
        <Database size={14} aria-hidden="true" />
        <span>{approvalSourceHealthLabel(review)}</span>
      </div>
      <div className="space-y-1">
        {rows.map((issue, idx) => (
          <div key={`${issue.id || issue.source_name || "source"}-${issue.severity}-${idx}`} className="grid gap-1 text-xs sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center">
            <div className="min-w-0">
              <span className="font-medium capitalize">{approvalSourceIssueLabel(issue)}</span>
              {issue.reason && <span className="ml-2 text-current/75">{issue.reason}</span>}
              {issue.detail && <span className="ml-2 text-current/75">{issue.detail}</span>}
            </div>
            <div className="flex flex-wrap gap-2 text-current/75 sm:justify-end">
              <span>{sourceHealthLabel(issue.status)}</span>
              {issue.reliability_tier && (
                <span className={cn("theme-badge", reliabilityTierBadgeClass(issue.reliability_tier))}>
                  {reliabilityTierLabel(issue.reliability_tier)}
                </span>
              )}
              {issue.sla_breach && <span>SLA breach</span>}
              <span>{issue.required ? "required" : "optional"}</span>
              <span>{formatTime(issue.freshness_timestamp ?? issue.as_of ?? issue.fetched_at)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function policyGateFromRecommendation(rec: RecommendationRecord): PolicyGateResult | null {
  if (rec.policy_gate) return rec.policy_gate
  if (!rec.policy_gate_decision) return null
  return {
    decision: rec.policy_gate_decision,
    review_required: Boolean(rec.policy_gate_review_required),
    failure_reasons: rec.policy_gate_failures_json ?? [],
    warnings: rec.policy_gate_warnings_json ?? [],
    disclosures: rec.policy_gate_disclosures_json ?? [],
  }
}

function recommendationNeedsPolicyGate(rec: RecommendationRecord): boolean {
  const action = String(rec.action || "").trim().toLowerCase()
  const effectScope = String(rec.effect_scope || "read_only").trim().toLowerCase()
  if (ACTIONABLE_RECOMMENDATION_ACTIONS.has(action)) return true
  if (effectScope !== "read_only") return true
  if (policyGateFromRecommendation(rec)) return true
  const policyState = String(rec.policy_state || "").trim().toLowerCase()
  return Boolean(policyState && policyState !== "missing")
}

function policyGateFromApproval(approval: ApprovalRecord): PolicyGateResult | null {
  if (approval.policy_gate) return approval.policy_gate
  const proposed = approval.proposed_change
  const direct = proposed.policy_gate_result
  if (isPolicyGateResult(direct)) return direct
  const record = proposed.record
  if (record && typeof record === "object" && !Array.isArray(record)) {
    const nested = (record as Record<string, unknown>).policy_gate_result
    if (isPolicyGateResult(nested)) return nested
  }
  return null
}

function approvalNeedsPolicyGate(approval: ApprovalRecord, gate: PolicyGateResult | null): boolean {
  if (gate) return true
  if (approval.action_id === "update_portfolio_positions" || approval.action_id === "update_hedge_positions") return true
  if (approval.action_id === "create_action_item") {
    return FINANCIAL_ACTION_ITEM_TYPES.has(String(approval.proposed_change.action_type || ""))
  }
  if (approval.action_id === "create_recommendation") {
    const record = approval.proposed_change.record
    if (record && typeof record === "object" && !Array.isArray(record)) {
      return ACTIONABLE_RECOMMENDATION_ACTIONS.has(String((record as Record<string, unknown>).action || ""))
    }
  }
  return false
}

function isPolicyGateResult(value: unknown): value is PolicyGateResult {
  return Boolean(value && typeof value === "object" && !Array.isArray(value) && "decision" in value)
}

function gateTone(decision?: string): string {
  if (decision === "pass") return "text-green-700 bg-green-50 dark:text-green-400 dark:bg-green-950"
  if (decision === "warn") return "text-amber-700 bg-amber-50 dark:text-amber-400 dark:bg-amber-950"
  if (decision === "review_required") return "text-orange-700 bg-orange-50 dark:text-orange-400 dark:bg-orange-950"
  return "text-red-700 bg-red-50 dark:text-red-400 dark:bg-red-950"
}

function applicationLabel(a: ApprovalRecord): string {
  if (a.base_state_status === "stale") return "state changed"
  const app = String(a.application_status || "pending").replace(/_/g, " ")
  if (a.can_retry_apply) return `failed application · retry available`
  return app
}

function approvalSubjectLabel(approval: ApprovalRecord): string {
  return String(approval.entity_type || "proposal").replace(/_/g, " ")
}

function asPlainRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null
}

function cleanText(value: unknown): string | null {
  const text = String(value ?? "").trim()
  return text ? text.replace(/_/g, " ") : null
}

function cleanTicker(value: unknown): string | null {
  const text = String(value ?? "").trim()
  return text ? text.toUpperCase() : null
}

function titleCase(value: string): string {
  return value.replace(/\b\w/g, char => char.toUpperCase())
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

function approvalTickerLabel(approval: ApprovalRecord): string | null {
  const change = approval.proposed_change
  const record = asPlainRecord(change.record)
  return cleanTicker(approval.ticker) ?? cleanTicker(change.ticker) ?? cleanTicker(record?.ticker)
}

function recommendationTargetLabel(approval: ApprovalRecord): string | null {
  const change = approval.proposed_change
  const record = asPlainRecord(change.record)
  const ticker = approvalTickerLabel(approval)
  const instrument = cleanText(record?.instrument ?? change.instrument)
  if (ticker && instrument && instrument.toUpperCase() !== ticker) return `${ticker} ${instrument}`
  return ticker ?? instrument
}

function approvalReasonLabel(approval: ApprovalRecord): string | null {
  const reason = cleanText(approval.reason)
  if (approval.action_id !== "create_recommendation") return reason

  const target = recommendationTargetLabel(approval)
  const ticker = approvalTickerLabel(approval)
  if (!target) return reason
  if (reason) {
    if (ticker && reason.toUpperCase().includes(ticker)) return reason

    const record = asPlainRecord(approval.proposed_change.record)
    const instrument = cleanText(record?.instrument ?? approval.proposed_change.instrument)
    if (instrument) {
      const instrumentPattern = new RegExp(`(recommendation\\s+for\\s+)${escapeRegExp(instrument)}\\b`, "i")
      if (instrumentPattern.test(reason)) return reason.replace(instrumentPattern, `$1${target}`)
    }
    return `${reason} (${ticker ?? target})`
  }

  const record = asPlainRecord(approval.proposed_change.record)
  const reportType = cleanText(record?.report_type)
  return `${reportType ? `${titleCase(reportType)} ` : ""}recommendation for ${target}`
}

function watchTriggerProposalFromApproval(approval: ApprovalRecord): EditableWatchTrigger | null {
  if (approval.action_id !== "create_watch_trigger") return null
  const change = approval.proposed_change
  return {
    id: approval.id,
    condition: String(change.condition ?? ""),
    trigger_type: String(change.trigger_type || "custom"),
    ticker: cleanTicker(change.ticker),
    expires_at: typeof change.expires_at === "string" ? change.expires_at : null,
    definition: asPlainRecord(change.definition),
  }
}

function reasonText(reason: PolicyGateReason): string {
  return reason.message || reason.code || reason.check || "Policy gate issue"
}

function PolicyGatePanel({ gate }: { gate: PolicyGateResult | null }) {
  if (!gate) {
    return (
      <div className="mt-2 border-t border-app pt-2 text-[11px] text-amber-700 dark:text-amber-300">
        Policy gate missing. Actionable recommendations require a stored gate result before approval.
      </div>
    )
  }
  const failures = gate.failure_reasons ?? []
  const warnings = gate.warnings ?? []
  const topItems = [...failures, ...warnings].slice(0, 3)
  return (
    <div className="mt-2 border-t border-app pt-2 text-[11px]">
      <div className="flex flex-wrap items-center gap-2">
        <span className={cn("rounded px-1.5 py-0.5 font-medium", gateTone(gate.decision))}>
          Policy {String(gate.decision || "unknown").replace(/_/g, " ")}
        </span>
        {gate.review_required && <span className="text-orange-700 dark:text-orange-300">approval note review required</span>}
        <span className="text-subtle">decision support only; human approval required</span>
      </div>
      {topItems.length > 0 && (
        <ul className="mt-1 space-y-0.5 text-muted">
          {topItems.map((item, idx) => <li key={`${item.code || item.check || "gate"}-${idx}`}>{reasonText(item)}</li>)}
        </ul>
      )}
    </div>
  )
}

function RiskBindingLine({ record }: { record: RecommendationRecord | Record<string, unknown> }) {
  const riskQuality = String(record.risk_quality || "")
  const riskSnapshot = String(record.risk_snapshot_id || "")
  const portfolioSnapshot = String(record.portfolio_risk_snapshot_id || "")
  if (!riskQuality && !riskSnapshot && !portfolioSnapshot) return null
  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-subtle">
      {riskQuality && <QualityStateBadge state={riskQuality} />}
      {riskSnapshot && <span>Risk {riskSnapshot}</span>}
      {portfolioSnapshot && <span>Portfolio {portfolioSnapshot}</span>}
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────
   Workspace redesign — calm "Daily Briefing" information architecture.
   What to do first (summary + action queue), status second (risk,
   triggers, timeline, source health). Plain-language over raw diffs.
   ──────────────────────────────────────────────────────────────── */

function greeting(date = new Date()): string {
  const h = date.getHours()
  if (h < 12) return "Good morning"
  if (h < 18) return "Good afternoon"
  return "Good evening"
}

function todayLabel(date = new Date()): string {
  return date.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric" })
}

const NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
function numWord(n: number): string {
  return n >= 0 && n <= 10 ? NUM_WORDS[n] : String(n)
}
function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

interface BriefingFacts {
  regime: string | null
  regimeTone: "success" | "warning" | "error" | null
  regimeScore: number | null
  positions: number | null
  avgRisk: number | null
  riskBand: string | null
  riskQuality: string | null
  pendingApprovals: number
  optimizerAlerts: number
  degradedSources: number
  failedReports: string[]
}

function buildBriefingFacts(
  data: WorkspaceData,
  approvalCount: number,
  optimizerAlertCount: number,
): BriefingFacts {
  const regime = data.regime
  const regimeInfo = regime?.signal ? REGIME_SIGNAL_MAP[regime.signal.toLowerCase()] : null
  const risk = data.portfolio?.risk
  const counts = data.source_health?.counts ?? {}
  const degraded = (counts.optional_degraded ?? 0) + (counts.sla_breach ?? 0)
  const failedReports = data.recommendations.blocked_warnings.map(w => String(w.report_type || "").toLowerCase()).filter(Boolean)
  const band = risk?.risk_level ? String(risk.risk_level).toLowerCase() : null
  return {
    regime: regimeInfo?.label ?? (regime?.regime ?? null),
    regimeTone: regimeInfo?.signal ?? null,
    regimeScore: regime?.composite_score ?? null,
    positions: data.portfolio?.position_count ?? null,
    avgRisk: typeof risk?.average_risk_score === "number" ? risk.average_risk_score : null,
    riskBand: band,
    riskQuality: risk?.quality ?? null,
    pendingApprovals: approvalCount,
    optimizerAlerts: optimizerAlertCount,
    degradedSources: degraded,
    failedReports,
  }
}

function templateBriefingSummary(f: BriefingFacts): string {
  const parts: string[] = []
  if (f.regime && f.positions != null) {
    const bandClause = f.riskBand ? ` carrying ${f.riskBand} average risk` : ""
    parts.push(`The market is in a ${f.regime} regime and your ${numWord(f.positions)} position${f.positions === 1 ? "" : "s"} are${bandClause || " in line with target"}.`)
  } else if (f.regime) {
    parts.push(`The market is in a ${f.regime} regime.`)
  } else if (f.positions != null) {
    parts.push(`You are holding ${numWord(f.positions)} position${f.positions === 1 ? "" : "s"}.`)
  }

  const approvals = f.pendingApprovals === 0
    ? "Nothing is waiting on your approval"
    : `${f.pendingApprovals === 1 ? "One approval is" : `${capitalize(numWord(f.pendingApprovals))} approvals are`} waiting on you`
  const alerts = f.optimizerAlerts > 0
    ? `, and the optimizer flagged ${numWord(f.optimizerAlerts)} change${f.optimizerAlerts === 1 ? "" : "s"} worth a look`
    : ""
  parts.push(`${approvals}${alerts}.`)

  if (f.degradedSources > 0) {
    const failed = f.failedReports.length > 0
      ? ` ${capitalize(f.failedReports.join(" & "))} recommendation run${f.failedReports.length === 1 ? "" : "s"} did not complete as a result.`
      : " Nothing critical has failed."
    parts.push(`${capitalize(numWord(f.degradedSources))} data source${f.degradedSources === 1 ? " is" : "s are"} running degraded.${failed}`)
  }

  return parts.join(" ")
}

function StatInline({ tone, value, label }: { tone: "success" | "warning" | "error" | "neutral"; value: string; label: string }) {
  const dot = tone === "neutral" ? "bg-[hsl(var(--foreground-quaternary))]"
    : tone === "success" ? "bg-[hsl(var(--success))]"
    : tone === "warning" ? "bg-[hsl(var(--warning))]"
    : "bg-[hsl(var(--destructive))]"
  return (
    <span className="inline-flex items-baseline gap-2">
      <span className={cn("h-2 w-2 self-center rounded-full", dot)} />
      <strong className="font-semibold text-app">{value}</strong>
      <span className="text-sm text-subtle">{label}</span>
    </span>
  )
}

function BriefingSummaryCard({ facts }: { facts: BriefingFacts }) {
  const [generatedAt, setGeneratedAt] = useState(() => new Date())
  const summary = templateBriefingSummary(facts)
  const regimeStat = facts.regime
    ? <StatInline tone={facts.regimeTone ?? "neutral"} value={facts.regime} label={facts.regimeScore != null ? `regime · score ${facts.regimeScore}` : "regime"} />
    : null
  const riskStat = facts.avgRisk != null
    ? <StatInline tone={facts.riskQuality && facts.riskQuality !== "ok" ? "warning" : "success"} value={formatRiskScore(facts.avgRisk)} label={`avg risk${facts.positions != null ? ` · ${facts.positions} position${facts.positions === 1 ? "" : "s"}` : ""}`} />
    : null
  const approvalStat = <StatInline tone={facts.pendingApprovals > 0 ? "warning" : "success"} value={String(facts.pendingApprovals)} label={`approval${facts.pendingApprovals === 1 ? "" : "s"} pending`} />
  const sourceStat = facts.degradedSources > 0
    ? <StatInline tone="warning" value="Degraded" label={`${facts.degradedSources} source${facts.degradedSources === 1 ? "" : "s"}`} />
    : <StatInline tone="success" value="Healthy" label="sources" />

  return (
    <section className="theme-surface mb-6 rounded-xl p-5 sm:p-6">
      <div className="mb-4 flex items-center gap-2">
        <span className="text-xs font-bold uppercase tracking-[0.13em] text-subtle">Today's read</span>
        <span className="ml-auto inline-flex items-center gap-1.5 theme-badge theme-badge-neutral">
          <FileText size={12} aria-hidden="true" /> Template
        </span>
        <button
          type="button"
          onClick={() => setGeneratedAt(new Date())}
          title="Regenerate summary"
          className="theme-icon-button h-8 w-8"
        >
          <RefreshCw size={15} />
        </button>
      </div>
      <p className="m-0 text-lg leading-relaxed text-muted sm:text-xl">{summary}</p>
      <div className="mt-5 flex flex-wrap gap-x-7 gap-y-3 border-t border-app pt-4 text-sm">
        {regimeStat}
        {riskStat}
        {approvalStat}
        {sourceStat}
      </div>
      <p className="mt-3 text-[11px] text-subtle">
        Computed from your live data · as of {generatedAt.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })}
      </p>
    </section>
  )
}

type PortfolioRisk = NonNullable<NonNullable<WorkspaceData["portfolio"]>["risk"]>

function WorkspaceRiskCard({ risk }: { risk: PortfolioRisk }) {
  const contributors = Array.isArray(risk.top_contributors) ? risk.top_contributors : []
  const degraded = Boolean(risk.quality && risk.quality !== "ok")
  return (
    <section className="theme-surface flex flex-col rounded-xl p-4">
      <div className="mb-4 flex items-center gap-2">
        <Shield size={15} className={degraded ? "text-amber-500" : "text-blue-500"} />
        <h2 className="text-sm font-semibold text-app">Portfolio risk</h2>
        <span className="ml-auto"><QualityStateBadge state={risk.quality || "missing"} /></span>
      </div>
      <div className="mb-4 flex items-end gap-6">
        <div>
          <p className="text-2xl font-bold tracking-tight text-app">{formatRiskScore(risk.average_risk_score)}</p>
          <p className="mt-1 text-xs text-subtle">Avg risk</p>
        </div>
        <div>
          <p className="text-2xl font-bold tracking-tight text-app">{formatRiskScore(risk.max_risk_score)}</p>
          <p className="mt-1 text-xs text-subtle">Max risk</p>
        </div>
        <div className="ml-auto text-right">
          <p className="text-sm font-semibold text-app">
            H {risk.risk_buckets?.high ?? 0} · M {risk.risk_buckets?.medium ?? 0} · L {risk.risk_buckets?.low ?? 0}
          </p>
          <p className="mt-1 text-xs text-subtle">Buckets · {risk.as_of || risk.computed_at || "—"}</p>
        </div>
      </div>
      {contributors.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {contributors.slice(0, 5).map((row, idx) => {
            const score = typeof row.risk_score === "number" ? row.risk_score : null
            return (
              <span key={`${String(row.ticker || "risk")}-${idx}`} className="inline-flex items-center gap-1.5 rounded-full border border-app bg-card-muted px-2.5 py-1 text-xs">
                <span className="font-bold tabular-nums text-app">{String(row.ticker || "Portfolio")}</span>
                <span className={cn("tabular-nums font-semibold", score != null && score >= 0.36 ? "text-amber-600 dark:text-amber-400" : "text-subtle")}>{formatRiskScore(score)}</span>
              </span>
            )
          })}
        </div>
      )}
    </section>
  )
}

function WorkspaceTriggersCard({
  triggers,
  count,
  processingIds,
  onEdit,
  onCancel,
}: {
  triggers: Trigger[]
  count: number
  processingIds: Set<number | string>
  onEdit: (t: Trigger) => void
  onCancel: (id: number | string) => void
}) {
  return (
    <section className="theme-surface flex min-h-0 flex-col rounded-xl p-4">
      <div className="mb-3 flex items-center gap-2">
        <Flag size={15} className="text-cyan-500" />
        <h2 className="text-sm font-semibold text-app">Active triggers</h2>
        <span className="ml-auto text-xs text-subtle">{count} active</span>
      </div>
      {triggers.length === 0 ? (
        <p className="text-xs text-subtle">No active triggers.</p>
      ) : (
        <div className="-mx-1 max-h-[22rem] space-y-1 overflow-y-auto px-1">
          {triggers.slice(0, 6).map(t => (
            <div key={t.id} className="rounded-lg px-2 py-2 text-sm transition-colors hover:bg-[hsl(var(--background-card-muted))]">
              <div className="flex items-center gap-2">
                {t.ticker && (
                  <Link to={`/dossier/${encodeURIComponent(t.ticker)}`} state={{ from: "workspace" }} className="shrink-0 font-bold text-app hover:underline">
                    {t.ticker}
                  </Link>
                )}
                <span className="min-w-0 flex-1 truncate text-muted">{t.condition}</span>
                <button
                  type="button"
                  onClick={() => onEdit(t)}
                  disabled={processingIds.has(t.id)}
                  className="shrink-0 text-xs font-semibold text-link hover:underline disabled:opacity-50"
                >
                  Edit
                </button>
                <button
                  type="button"
                  onClick={() => onCancel(t.id)}
                  disabled={processingIds.has(t.id)}
                  aria-label={`Cancel trigger ${t.id}`}
                  className="theme-icon-button h-7 w-7 shrink-0 disabled:opacity-50"
                >
                  <X size={13} />
                </button>
              </div>
              <div className="mt-0.5 flex flex-wrap items-center gap-x-3 text-[11px] text-subtle">
                <span>{t.trigger_type.replace(/_/g, " ")}</span>
                {t.last_checked_at && <span>Checked {formatTime(t.last_checked_at)}</span>}
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}

type QueueTab = "all" | "approvals" | "alerts"

function ActionQueueCard({
  approvalItems,
  approvalCount,
  approvalLoading,
  approvalError,
  optimizerAlerts,
  optimizerCount,
  optimizerDismissError,
  pressures,
  pressureDismissError,
  processingIds,
  bulkDismissSubmitting,
  onReviewApproval,
  onBulkDismiss,
  onDismissOptimizer,
  onDismissPressure,
  onOpenApprovalTrace,
}: {
  approvalItems: ApprovalRecord[]
  approvalCount: number
  approvalLoading: boolean
  approvalError: unknown
  optimizerAlerts: OptimizationAlert[]
  optimizerCount: number
  optimizerDismissError: string | null
  pressures: WorkspaceData["thesis_pressure"]
  pressureDismissError: string | null
  processingIds: Set<number | string>
  bulkDismissSubmitting: boolean
  onReviewApproval: (a: ApprovalRecord) => void
  onBulkDismiss: () => void
  onDismissOptimizer: (a: OptimizationAlert) => void
  onDismissPressure: (tp: WorkspaceData["thesis_pressure"][number]) => void
  onOpenApprovalTrace: (a: ApprovalRecord) => void
}) {
  const [tab, setTab] = useState<QueueTab>("all")
  const alertCount = optimizerCount + pressures.length
  const counts: Record<QueueTab, number> = { all: approvalCount + alertCount, approvals: approvalCount, alerts: alertCount }
  const showApprovals = tab === "all" || tab === "approvals"
  const showAlerts = tab === "all" || tab === "alerts"
  const canBulkDismiss = approvalCount > 0 && !approvalLoading && !approvalError && showApprovals
  const isEmpty =
    (!showApprovals || (approvalCount === 0 && !approvalLoading && !approvalError)) &&
    (!showAlerts || alertCount === 0)

  const tabs: [QueueTab, string][] = [["all", "All"], ["approvals", "Approvals"], ["alerts", "Alerts"]]

  return (
    <section className="theme-surface flex min-h-0 flex-col overflow-hidden rounded-xl">
      <div className="border-b border-app p-4">
        <div className="mb-3 flex items-center gap-2">
          <Bell size={16} className="text-blue-500" />
          <h2 className="text-sm font-semibold text-app">Action queue</h2>
          {canBulkDismiss ? (
            <button
              type="button"
              onClick={onBulkDismiss}
              disabled={bulkDismissSubmitting}
              className="ml-auto rounded px-2 py-1 text-xs font-medium text-red-700 bg-red-50 hover:bg-red-100 dark:text-red-300 dark:bg-red-950 dark:hover:bg-red-900 disabled:opacity-50"
            >
              Dismiss all
            </button>
          ) : (
            <span className="ml-auto text-xs text-subtle">What needs you, in order</span>
          )}
        </div>
        <div className="inline-flex gap-1 rounded-full border border-app bg-card-muted p-1">
          {tabs.map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setTab(id)}
              className={cn(
                "inline-flex h-7 items-center gap-1.5 rounded-full px-3 text-xs font-semibold transition-colors",
                tab === id ? "bg-elevated text-app shadow-sm" : "text-subtle hover:text-app",
              )}
            >
              {label}
              <span className={cn(
                "inline-flex h-[17px] min-w-[17px] items-center justify-center rounded-full px-1 text-[11px] font-bold tabular-nums",
                tab === id ? "bg-[hsl(var(--accent-muted))] text-[hsl(var(--accent))]" : "bg-card-muted text-subtle",
              )}>
                {counts[id]}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="min-h-0 flex-1 space-y-2.5 overflow-y-auto p-3.5">
        {(optimizerDismissError || pressureDismissError) && showAlerts && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
            {optimizerDismissError || pressureDismissError}
          </div>
        )}

        {showApprovals && approvalLoading && (
          <div className="rounded-lg border border-app px-3 py-2 text-sm text-muted">Loading approvals…</div>
        )}
        {showApprovals && !!approvalError && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            Failed to load approvals: {String(approvalError)}
          </div>
        )}
        {showApprovals && !approvalLoading && !approvalError && approvalItems.map(a => {
          const gate = policyGateFromApproval(a)
          const displayTicker = approvalTickerLabel(a)
          const displayReason = approvalReasonLabel(a)
          return (
            <div key={a.id} className="overflow-hidden rounded-xl border border-amber-200 bg-amber-50/60 dark:border-amber-900/60 dark:bg-amber-950/30">
              <div className="flex items-center gap-2 border-b border-amber-200/70 px-3.5 py-2 dark:border-amber-900/50">
                <AlertTriangle size={14} className="text-amber-600 dark:text-amber-400" />
                <span className="text-[11px] font-bold uppercase tracking-[0.06em] text-amber-700 dark:text-amber-300">Pending approval</span>
                {displayTicker && <span className="font-bold tabular-nums text-app">{displayTicker}</span>}
                <span className="text-xs text-subtle">{approvalSubjectLabel(a)}</span>
              </div>
              <div className="bg-card px-3.5 py-3">
                <div className="mb-2 flex flex-wrap gap-1.5">
                  <BaseStateBadge state={a.base_state_status} message={a.base_state_message} />
                  <EffectScopeBadge scope={a.effect_scope ?? "internal_state"} />
                  <PolicyStateBadge state={a.policy_state ?? gate?.decision ?? "missing"} />
                  <QualityStateBadge state={a.quality_state ?? "missing"} />
                  <ApprovalSourceHealthBadge review={a.source_health_review} />
                </div>
                {displayReason && <p className="mb-2 line-clamp-2 text-sm text-muted">{displayReason}</p>}
                {approvalNeedsPolicyGate(a, gate) && <PolicyGatePanel gate={gate} />}
                <ApprovalProgressSummary approval={a} compact />
                <div className="mt-3 flex gap-2">
                  <button
                    type="button"
                    onClick={() => onReviewApproval(a)}
                    disabled={processingIds.has(a.id)}
                    className="theme-button-base theme-button-primary min-h-9 flex-1 px-4 text-xs"
                  >
                    Review
                  </button>
                  <TraceTriggerButton compact label={`View approval ${a.id} trace`} onClick={() => onOpenApprovalTrace(a)} />
                </div>
              </div>
            </div>
          )
        })}

        {showAlerts && optimizerAlerts.map(alert => (
          <div key={`opt-${alert.id}`} className="flex gap-3 rounded-xl border border-app bg-card px-3.5 py-3 transition-colors hover:bg-[hsl(var(--background-card-muted))]">
            <span className="w-[3px] shrink-0 self-stretch rounded-full bg-amber-400" />
            <div className="min-w-0 flex-1">
              <div className="mb-1 flex flex-wrap items-center gap-2">
                {alert.ticker ? (
                  <Link to={`/dossier/${encodeURIComponent(alert.ticker)}`} state={{ from: "workspace" }} className="font-bold tabular-nums text-app hover:underline">{alert.ticker}</Link>
                ) : (
                  <span className="font-bold tabular-nums text-app">PORTFOLIO</span>
                )}
                <span className={cn("rounded px-1.5 py-0.5 text-[11px] font-semibold", alertSeverityClass(alert.severity))}>{alert.severity}</span>
                <span className="text-[11px] text-subtle">{alert.alert_type.replace(/_/g, " ")}</span>
              </div>
              <p className="line-clamp-2 text-sm leading-snug text-muted">{alert.change_summary}</p>
            </div>
            <button
              type="button"
              onClick={() => onDismissOptimizer(alert)}
              disabled={processingIds.has(`optimizer-alert-${alert.id}`)}
              aria-label="Dismiss alert"
              className="theme-icon-button h-7 w-7 shrink-0 disabled:opacity-50"
            >
              <X size={14} />
            </button>
          </div>
        ))}

        {showAlerts && pressures.map(tp => (
          <div key={`pressure-${tp.pressure_key}`} className="flex gap-3 rounded-xl border border-app bg-card px-3.5 py-3 transition-colors hover:bg-[hsl(var(--background-card-muted))]">
            <span className={cn("w-[3px] shrink-0 self-stretch rounded-full", tp.action === "exit" || tp.action === "reduce" ? "bg-red-400" : "bg-amber-400")} />
            <div className="min-w-0 flex-1">
              <div className="mb-1 flex flex-wrap items-center gap-2">
                <Link to={`/dossier/${encodeURIComponent(tp.ticker)}`} state={{ from: "workspace" }} className="font-bold tabular-nums text-app hover:underline">{tp.ticker}</Link>
                <span className={cn("rounded px-1.5 py-0.5 text-[11px] font-semibold", tp.action === "exit" || tp.action === "reduce" ? "text-red-600 bg-red-50 dark:text-red-400 dark:bg-red-950" : "text-amber-600 bg-amber-50 dark:text-amber-400 dark:bg-amber-950")}>Evaluation: {tp.action}</span>
                <span className="text-[11px] text-subtle">{tp.confidence}</span>
              </div>
              <p className="text-sm leading-snug text-muted">
                Thesis pressure flagged a possible {tp.action}{tp.risk_flag ? ` — ${tp.risk_flag}` : ""}.
              </p>
            </div>
            <button
              type="button"
              onClick={() => onDismissPressure(tp)}
              disabled={processingIds.has(`pressure-${tp.pressure_key}`)}
              aria-label={`Clear ${tp.ticker} pressure row`}
              className="theme-icon-button h-7 w-7 shrink-0 disabled:opacity-50"
            >
              <X size={14} />
            </button>
          </div>
        ))}

        {isEmpty && (
          <div className="py-12 text-center">
            <span className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-green-50 text-green-600 dark:bg-green-950 dark:text-green-400">
              <CheckCircle size={24} />
            </span>
            <p className="text-sm font-semibold text-app">You're all caught up</p>
            <p className="mt-1 text-xs text-subtle">Nothing in this lane needs action right now.</p>
          </div>
        )}
      </div>
    </section>
  )
}

interface TimelineEntry {
  id: string
  kind: "workflow" | "report"
  title: string
  detail: string | null
  ticker: string | null
  status: string | null
  time: string
  sortKey: number
  trace?: () => void
  traceLabel?: string
}

function WorkspaceTimeline({
  workflowRuns,
  reportRuns,
  onViewWorkflowTrace,
}: {
  workflowRuns: WorkflowRun[]
  reportRuns: ReportRun[]
  onViewWorkflowTrace?: (run: WorkflowRun) => void
}) {
  const [filter, setFilter] = useState<"all" | "workflow" | "report">("all")
  const sortKey = (iso: string | null | undefined) => {
    const t = new Date(String(iso ?? "")).getTime()
    return Number.isNaN(t) ? 0 : t
  }
  const entries: TimelineEntry[] = [
    ...workflowRuns.map((run, i): TimelineEntry => {
      const ticker = workflowRunTicker(run)
      const runId = String(run.run_id ?? "").trim()
      return {
        id: runId || `workflow-${i}`,
        kind: "workflow",
        title: `${capitalize(workflowRunLabel(run))} ${ticker ? "review" : "run"}`,
        detail: run.synthesis ? String(run.synthesis).slice(0, 140) : `Workflow run ${run.status ?? ""}`.trim(),
        ticker,
        status: run.status ?? null,
        time: workflowRunTime(run),
        sortKey: sortKey(run.started_at ?? run.completed_at),
        trace: runId && onViewWorkflowTrace ? () => onViewWorkflowTrace(run) : undefined,
        traceLabel: runId ? `View workflow ${runId} trace` : undefined,
      }
    }),
    ...reportRuns.map((run, i): TimelineEntry => ({
      id: reportRunKey(run, i),
      kind: "report",
      title: `${reportRunLabel(run)} report ${String(run.status ?? "").toLowerCase() === "failed" ? "did not complete" : "generated"}`,
      detail: run.error ? String(run.error) : (run.status ? capitalize(String(run.status)) : null),
      ticker: null,
      status: run.status ?? null,
      time: reportRunTime(run),
      sortKey: sortKey(run.as_of ?? run.synced_at ?? run.created_at),
    })),
  ].sort((a, b) => b.sortKey - a.sortKey)

  const counts = {
    all: entries.length,
    workflow: entries.filter(e => e.kind === "workflow").length,
    report: entries.filter(e => e.kind === "report").length,
  }
  const shown = entries.filter(e => filter === "all" || e.kind === filter)
  const filters: ["all" | "workflow" | "report", string][] = [["all", "All"], ["workflow", "Workflows"], ["report", "Reports"]]

  if (entries.length === 0) return null

  return (
    <section className="theme-surface mb-6 rounded-xl p-5">
      <div className="mb-5 flex flex-wrap items-center gap-2">
        <Clock size={16} className="text-gray-500" />
        <h2 className="text-sm font-semibold text-app">Timeline</h2>
        <span className="text-xs text-subtle">everything that moved, newest first</span>
        <div className="ml-auto inline-flex gap-1.5">
          {filters.map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setFilter(id)}
              className={cn(
                "inline-flex h-7 items-center gap-1 rounded-full border px-3 text-xs font-semibold transition-colors",
                filter === id ? "border-transparent bg-[hsl(var(--accent))] text-[hsl(var(--accent-foreground))]" : "border-app text-subtle hover:text-app",
              )}
            >
              {label} <span className="tabular-nums opacity-75">{counts[id]}</span>
            </button>
          ))}
        </div>
      </div>
      {shown.length === 0 ? (
        <p className="py-8 text-center text-sm text-subtle">Nothing in this filter.</p>
      ) : (
        <div className="pl-1">
          {shown.map((e, i) => {
            const last = i === shown.length - 1
            return (
              <div key={e.id} className="flex gap-4">
                <div className="w-16 shrink-0 pt-0.5 text-right">
                  <div className="text-xs font-semibold text-muted">{e.time.split(",")[0]}</div>
                  <div className="text-[11px] tabular-nums text-subtle">{e.time.split(",").slice(1).join(",").trim()}</div>
                </div>
                <div className="flex w-[18px] shrink-0 flex-col items-center">
                  <span className={cn("mt-1.5 h-3 w-3 shrink-0 rounded-full ring-[1.5px] ring-[hsl(var(--separator))]", workflowStatusClass(e.status), "border-2 border-[hsl(var(--background-card))]")} />
                  {!last && <span className="mt-1 w-0.5 flex-1 bg-[hsl(var(--separator))]" />}
                </div>
                <div className={cn("flex min-w-0 flex-1 gap-3", last ? "pb-0" : "pb-5")}>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-sm font-semibold text-app">{e.title}</span>
                      {e.ticker && (
                        <Link to={`/dossier/${encodeURIComponent(e.ticker)}`} state={{ from: "workspace" }} className="text-xs font-bold text-link hover:underline">{e.ticker}</Link>
                      )}
                    </div>
                    {e.detail && <p className="mt-0.5 line-clamp-1 text-xs leading-snug text-subtle">{e.detail}</p>}
                  </div>
                  {e.trace && <TraceTriggerButton compact label={e.traceLabel ?? `Trace ${e.title}`} onClick={e.trace} />}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}

function CollapsibleSourceHealth({ sourceHealth }: { sourceHealth: SourceHealth }) {
  const [open, setOpen] = useState(false)
  const counts = sourceHealth.counts ?? {}
  const domains = sourceHealth.domains ?? []
  return (
    <section className="theme-surface overflow-hidden rounded-xl">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="flex w-full flex-wrap items-center gap-4 px-5 py-3.5 text-left"
      >
        <Database size={16} className="text-subtle" />
        <span className="text-sm font-semibold text-app">Source health</span>
        <div className="flex flex-wrap gap-1.5">
          <span className="theme-badge theme-badge-success">{counts.ok ?? 0} ok</span>
          {(counts.sla_breach ?? 0) > 0 && <span className="theme-badge theme-badge-warning">{counts.sla_breach} SLA breach</span>}
          {(counts.optional_degraded ?? 0) > 0 && <span className="theme-badge theme-badge-neutral">{counts.optional_degraded} degraded</span>}
        </div>
        <div className="ml-auto flex items-center gap-3">
          <span className="text-xs text-subtle">Updated {formatTime(sourceHealth.generated_at)}</span>
          <span className={cn("theme-badge", sourceHealthOverallClass(sourceHealth.overall_quality))}>{sourceHealthLabel(sourceHealth.overall_quality)}</span>
          <ChevronDown size={18} className={cn("text-subtle transition-transform", open && "rotate-180")} />
        </div>
      </button>
      {open && (
        <div className="grid grid-cols-1 gap-6 border-t border-app px-5 py-5 xl:grid-cols-2">
          {domains.length === 0 ? (
            <p className="text-sm text-muted">No source freshness records are available yet.</p>
          ) : domains.map(domain => (
            <div key={domain.domain}>
              <div className="mb-2.5 flex items-center justify-between">
                <span className="text-sm font-semibold text-app">{domain.label}</span>
                <span className={cn("theme-badge", sourceHealthOverallClass(domain.overall_quality))}>{sourceHealthLabel(domain.overall_quality)}</span>
              </div>
              <div>
                {domain.sources.map((source, i) => (
                  <div key={source.id} className={cn("flex items-center gap-2.5 py-2", i > 0 && "border-t border-app")}>
                    <span className="text-sm text-muted">{source.source_name.replace(/_/g, " ")}</span>
                    <span className="text-[11px] font-semibold text-subtle">
                      {reliabilityTierLabel(source.reliability_tier)} · {source.required ? "required" : "optional"}
                    </span>
                    <span className={cn("ml-auto text-xs font-semibold", source.status === "ok" ? "text-green-600 dark:text-green-400" : "text-amber-600 dark:text-amber-400")}>
                      {sourceHealthLabel(source.status)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}

export function Workspace() {
  const qc = useQueryClient()
  const [searchParams, setSearchParams] = useSearchParams()
  const deepLinkApprovalId = String(searchParams.get("approval_id") ?? "").trim()
  const deepLinkAttemptRef = useRef<string | null>(null)
  const { data, isPending, error } = useApiQuery<WorkspaceData>(
    ["workspace"],
    fetchWorkspace,
    60_000,
  )
  const approvalSummary = useApiQuery(
    approvalSummaryQueryKey({ status: "pending", limit: WORKSPACE_APPROVAL_LIMIT }),
    () => fetchApprovalSummary({ status: "pending", limit: WORKSPACE_APPROVAL_LIMIT }),
    30_000,
  )

  const [processingIds, setProcessingIds] = useState<Set<number | string>>(new Set())
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())
  const [refreshError, setRefreshError] = useState<string | null>(null)
  const { openDecisionTrace } = useDecisionTrace()
  const [postMortemReview, setPostMortemReview] = useState<DecisionOutcomeRecord | null>(null)
  const [approvalReview, setApprovalReview] = useState<ApprovalRecord | null>(null)
  const [approvalNote, setApprovalNote] = useState("")
  const [approvalError, setApprovalError] = useState<string | null>(null)
  const [approvalDialogAction, setApprovalDialogAction] = useState<ApprovalDialogAction | null>(null)
  const [bulkDismissOpen, setBulkDismissOpen] = useState(false)
  const [bulkDismissSubmitting, setBulkDismissSubmitting] = useState(false)
  const [bulkDismissError, setBulkDismissError] = useState<string | null>(null)
  const [pressureDismissError, setPressureDismissError] = useState<string | null>(null)
  const [triggerEdit, setTriggerEdit] = useState<TriggerEditState | null>(null)
  const [triggerEditError, setTriggerEditError] = useState<string | null>(null)
  const [triggerEditSubmitting, setTriggerEditSubmitting] = useState(false)
  const [optimizerDismissError, setOptimizerDismissError] = useState<string | null>(null)

  const clearApprovalDeepLink = useCallback(() => {
    if (!searchParams.has("approval_id")) return
    const next = new URLSearchParams(searchParams)
    next.delete("approval_id")
    setSearchParams(next, { replace: true })
  }, [searchParams, setSearchParams])

  function openApprovalReview(approval: ApprovalRecord) {
    setApprovalReview(approval)
    setApprovalNote("")
    setApprovalError(null)
    setApprovalDialogAction(null)
  }

  function toggleExpanded(key: string) {
    setExpandedIds(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  async function handleApproval(approval: ApprovalRecord, action: "approve" | "reject", note?: string) {
    setApprovalError(null)
    if (action === "approve" && !String(note || "").trim()) {
      setApprovalError("Approval note is required before applying an internal state change.")
      return
    }
    setApprovalDialogAction(action)
    setProcessingIds(prev => new Set(prev).add(approval.id))
    try {
      let resolved: ApprovalRecord
      if (action === "approve") {
        resolved = await approveItem(approval.id, String(note || "").trim())
      } else {
        resolved = await rejectItem(approval.id, note?.trim() || undefined)
      }
      patchResolvedApprovalSummaries(qc, resolved, approval)
      setApprovalReview(null)
      setApprovalNote("")
      invalidateAfterApprovalResolution(qc, approval)
    } catch (err) {
      setApprovalError(formatApprovalResolutionError(err))
      if (shouldRefetchApprovalSummariesAfterError(err)) void invalidateApprovalSummaries(qc)
    } finally {
      setProcessingIds(prev => {
        const next = new Set(prev)
        next.delete(approval.id)
        return next
      })
      setApprovalDialogAction(null)
    }
  }

  async function handleRejectAndRestage(approval: ApprovalRecord, note?: string) {
    setProcessingIds(prev => new Set(prev).add(approval.id))
    setApprovalError(null)
    setApprovalDialogAction("restage")
    try {
      const result = await rejectAndRestageApproval(approval.id, note?.trim() || undefined)
      patchResolvedApprovalSummaries(qc, result.original, approval)
      setApprovalReview(null)
      setApprovalNote("")
      invalidateAfterApprovalResolution(qc, approval)
    } catch (err) {
      setApprovalError(formatApprovalResolutionError(err))
      if (shouldRefetchApprovalSummariesAfterError(err)) void invalidateApprovalSummaries(qc)
    } finally {
      setProcessingIds(prev => {
        const next = new Set(prev)
        next.delete(approval.id)
        return next
      })
      setApprovalDialogAction(null)
    }
  }

  async function handleBulkDismissApprovals() {
    setBulkDismissSubmitting(true)
    setBulkDismissError(null)
    try {
      const pending = await fetchApprovals("pending")
      const ids = pending.approvals.map(approval => approval.id).filter(Boolean)
      if (ids.length === 0) {
        setBulkDismissOpen(false)
        void invalidateApprovalSummaries(qc)
        void qc.invalidateQueries({ queryKey: ["workspace"] })
        return
      }

      const result = await bulkReject(ids, BULK_DISMISS_APPROVAL_NOTE)
      const failures = result.results.filter(row => row.status !== "rejected")
      void invalidateApprovalSummaries(qc)
      void qc.invalidateQueries({ queryKey: ["workspace"] })

      if (failures.length > 0) {
        const failedLabels = failures.slice(0, 3).map(row => row.message ? `${row.id}: ${row.message}` : row.id).join("; ")
        const suffix = failures.length > 3 ? `; ${failures.length - 3} more` : ""
        setBulkDismissError(`Dismissed ${result.results.length - failures.length} of ${ids.length} approvals. Failed: ${failedLabels}${suffix}`)
        return
      }

      setBulkDismissOpen(false)
    } catch (err) {
      setBulkDismissError(formatApprovalResolutionError(err))
      void invalidateApprovalSummaries(qc)
      void qc.invalidateQueries({ queryKey: ["workspace"] })
    } finally {
      setBulkDismissSubmitting(false)
    }
  }

  async function handleDismissPressure(tp: WorkspaceData["thesis_pressure"][number]) {
    const processingKey = `pressure-${tp.pressure_key}`
    const previous = qc.getQueryData<WorkspaceData>(["workspace"])
    setPressureDismissError(null)
    setProcessingIds(prev => new Set(prev).add(processingKey))
    qc.setQueryData<WorkspaceData>(["workspace"], current => current
      ? {
          ...current,
          thesis_pressure: current.thesis_pressure.filter(row => row.pressure_key !== tp.pressure_key),
        }
      : current)
    try {
      await dismissWorkspaceThesisPressure({ ticker: tp.ticker, pressure_key: tp.pressure_key })
      void qc.invalidateQueries({ queryKey: ["workspace"] })
    } catch (err) {
      if (previous) qc.setQueryData(["workspace"], previous)
      setPressureDismissError(err instanceof Error ? err.message : String(err))
    } finally {
      setProcessingIds(prev => {
        const next = new Set(prev)
        next.delete(processingKey)
        return next
      })
    }
  }

  async function handleActionItem(id: number | string, action: "complete" | "dismiss") {
    setProcessingIds(prev => new Set(prev).add(id))
    try {
      if (action === "complete") await completeAction(id)
      else await dismissAction(id)
      void invalidateApprovalSummaries(qc)
      void qc.invalidateQueries({ queryKey: ["workspace"] })
    } finally {
      setProcessingIds(prev => { const n = new Set(prev); n.delete(id); return n })
    }
  }

  async function handleCancelTrigger(id: number | string) {
    setProcessingIds(prev => new Set(prev).add(id))
    try {
      await cancelTrigger(id)
      void invalidateApprovalSummaries(qc)
      void qc.invalidateQueries({ queryKey: ["workspace"] })
    } finally {
      setProcessingIds(prev => { const n = new Set(prev); n.delete(id); return n })
    }
  }

  async function handleSubmitTriggerEdit(body: TriggerMutationBody) {
    if (!triggerEdit) return
    setTriggerEditSubmitting(true)
    setTriggerEditError(null)
    try {
      if (triggerEdit.kind === "active") {
        await replaceTrigger(triggerEdit.trigger.id, {
          ...body,
          reason: `Replace watch trigger ${triggerEdit.trigger.id}`,
        })
        void invalidateApprovalSummaries(qc)
        void qc.invalidateQueries({ queryKey: ["workspace"] })
      } else {
        const result = await replaceApprovalProposal(triggerEdit.approval.id, {
          ...body,
          reason: triggerEdit.approval.reason || "Edit watch trigger proposal",
        })
        patchResolvedApprovalSummaries(qc, result.original, triggerEdit.approval)
        invalidateAfterApprovalResolution(qc, triggerEdit.approval)
        setApprovalReview(null)
        setApprovalNote("")
      }
      setTriggerEdit(null)
    } catch (err) {
      setTriggerEditError(formatApprovalResolutionError(err))
      if (shouldRefetchApprovalSummariesAfterError(err)) void invalidateApprovalSummaries(qc)
    } finally {
      setTriggerEditSubmitting(false)
    }
  }

  async function handleDismissOptimizerAlert(alert: OptimizationAlert) {
    const processingKey = `optimizer-alert-${alert.id}`
    setOptimizerDismissError(null)
    setProcessingIds(prev => new Set(prev).add(processingKey))
    try {
      await dismissOptimizationAlert(alert.id)
      void qc.invalidateQueries({ queryKey: ["workspace"] })
    } catch (err) {
      setOptimizerDismissError(err instanceof Error ? err.message : String(err))
    } finally {
      setProcessingIds(prev => {
        const next = new Set(prev)
        next.delete(processingKey)
        return next
      })
    }
  }

  useEffect(() => {
    if (!deepLinkApprovalId || approvalReview?.id === deepLinkApprovalId) return
    if (deepLinkAttemptRef.current === deepLinkApprovalId) return

    const embeddedItems = [
      ...(approvalSummary.data?.items ?? []),
      ...(data?.pending_approvals.items ?? []),
    ]
    const fromList = embeddedItems.find(item => String(item.id) === deepLinkApprovalId)
    if (fromList) {
      deepLinkAttemptRef.current = deepLinkApprovalId
      openApprovalReview(fromList)
      clearApprovalDeepLink()
      return
    }

    if (isPending || (approvalSummary.isPending && !approvalSummary.data)) return

    let cancelled = false
    deepLinkAttemptRef.current = deepLinkApprovalId
    void fetchApproval(deepLinkApprovalId)
      .then(approval => {
        if (cancelled) return
        openApprovalReview(approval)
        clearApprovalDeepLink()
      })
      .catch(() => {
        if (!cancelled) deepLinkAttemptRef.current = null
      })

    return () => {
      cancelled = true
    }
  }, [
    approvalReview?.id,
    approvalSummary.data,
    approvalSummary.isPending,
    clearApprovalDeepLink,
    data?.pending_approvals.items,
    deepLinkApprovalId,
    isPending,
  ])

  if (isPending) return <LoadingSpinner message="Loading portfolio commander..." />
  if (error) return <ErrorMessage message={String(error)} />
  if (!data) return null

  const approvalSummaryData = approvalSummary.data
  const approvalCount = approvalSummaryData?.count ?? data.pending_approvals.count
  const approvalItems = approvalSummaryData?.items ?? data.pending_approvals.items
  const approvalSummaryInitialLoading = approvalSummary.isPending && !approvalSummaryData
  const courseOfActions = data.course_of_actions ?? {
    pending: { count: 0, items: [] },
    recent: { count: 0, items: [] },
    comparisons: { count: 0, items: [] },
    pending_approval_count: 0,
  }
  const approvalSummaryError = approvalSummary.error
  const portfolioRisk = data.portfolio?.risk
  const optimizerAlerts = data.continuous_optimization?.open_alerts ?? []
  const optimizerAlertCount = data.continuous_optimization?.open_alert_count ?? optimizerAlerts.length
  const briefingFacts = buildBriefingFacts(data, approvalCount, optimizerAlertCount)
  const builderMonitors = data.monitor_builder?.active_monitors ?? []
  const builderMissions = data.monitor_builder?.active_missions ?? []
  const builderDefinitionCount =
    (data.monitor_builder?.active_monitor_count ?? builderMonitors.length) +
    (data.monitor_builder?.active_mission_count ?? builderMissions.length)
  const thesisClaimItems = data.thesis_claims?.items ?? []
  const thesisClaimCount = data.thesis_claims?.challenged_count ?? thesisClaimItems.length
  const recentReportRuns = data.recent_report_runs ?? []

  return (
    <div>
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div className="min-w-0">
          <p className="text-xs font-bold uppercase tracking-[0.13em] text-subtle">{todayLabel()}</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-[-0.03em] text-app">{greeting()}.</h1>
          <p className="mt-1.5 text-sm text-subtle">Here's where your book stands and the few things worth your attention.</p>
        </div>
        <RefreshButton
          queryKeys={[["workspace"]]}
          beforeRefetch={refreshWorkspaceSources}
          onSuccess={() => setRefreshError(null)}
          onError={err => setRefreshError(err instanceof Error ? err.message : String(err))}
        />
      </div>
      {refreshError && (
        <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          Refresh failed: {refreshError}
        </div>
      )}

      {/* Calm summary — plain-language read of the book */}
      <BriefingSummaryCard facts={briefingFacts} />

      {/* What changed — elevated, concise */}
      <WhatChangedPanel summary={data.what_changed} className="mb-6" from="workspace" />

      {(data.opportunity_candidates?.count ?? 0) > 0 && (
        <OpportunityScoutQueuePanel
          items={data.opportunity_candidates?.items ?? []}
          onUpdated={() => void qc.invalidateQueries({ queryKey: ["workspace"] })}
          onOpenTrace={candidate =>
            openDecisionTrace({
              kind: "opportunity_candidate",
              record: candidate as unknown as Record<string, unknown>,
            })
          }
        />
      )}

      {/* What to do first — action queue beside key status */}
      <div className="mb-6 grid grid-cols-1 items-start gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <ActionQueueCard
          approvalItems={approvalItems}
          approvalCount={approvalCount}
          approvalLoading={approvalSummaryInitialLoading}
          approvalError={approvalSummaryError}
          optimizerAlerts={optimizerAlerts}
          optimizerCount={optimizerAlertCount}
          optimizerDismissError={optimizerDismissError}
          pressures={data.thesis_pressure}
          pressureDismissError={pressureDismissError}
          processingIds={processingIds}
          bulkDismissSubmitting={bulkDismissSubmitting}
          onReviewApproval={openApprovalReview}
          onBulkDismiss={() => {
            setBulkDismissOpen(true)
            setBulkDismissError(null)
          }}
          onDismissOptimizer={handleDismissOptimizerAlert}
          onDismissPressure={handleDismissPressure}
          onOpenApprovalTrace={a =>
            openDecisionTrace({ kind: "approval", record: a as unknown as Record<string, unknown> })
          }
        />
        <div className="flex flex-col gap-6">
          {portfolioRisk && <WorkspaceRiskCard risk={portfolioRisk} />}
          {data.active_triggers.count > 0 && (
            <WorkspaceTriggersCard
              triggers={data.active_triggers.items}
              count={data.active_triggers.count}
              processingIds={processingIds}
              onEdit={t => {
                setTriggerEdit({ kind: "active", trigger: t })
                setTriggerEditError(null)
              }}
              onCancel={handleCancelTrigger}
            />
          )}
        </div>
      </div>

      {/* Timeline — everything that moved, pulled into its own band */}
      <WorkspaceTimeline
        workflowRuns={data.recent_workflow_runs}
        reportRuns={recentReportRuns}
        onViewWorkflowTrace={run =>
          openDecisionTrace({ kind: "workflow_run", record: run as unknown as Record<string, unknown> })
        }
      />

      {/* Needs review & secondary status */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Course of Action Review */}
        {(courseOfActions.pending.count > 0 || courseOfActions.recent.count > 0 || courseOfActions.comparisons.count > 0) && (
          <section className="theme-surface rounded-xl p-4 lg:col-span-2">
            <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
              <GitBranch size={14} className="text-blue-500" />
              Course of Action Review
              <span className="ml-auto text-xs text-subtle">
                {courseOfActions.pending.count} pending · {courseOfActions.comparisons.count} comparison{courseOfActions.comparisons.count !== 1 ? "s" : ""}
              </span>
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {(courseOfActions.pending.items.length ? courseOfActions.pending.items : courseOfActions.recent.items).slice(0, 6).map(coa => (
                <div key={`coa-${coa.id}`} className="rounded-lg border border-app px-3 py-2 text-sm">
                  <div className="flex flex-wrap items-center gap-2">
                    <DecisionStateBadge state={courseOfActionDecisionState(coa)} />
                    <span className="font-semibold text-app">{coa.action.replace(/_/g, " ")}</span>
                    {coa.ticker && (
                      <Link to={`/dossier/${encodeURIComponent(coa.ticker)}`} state={{ from: "workspace" }} className="text-xs font-semibold text-blue-600 hover:underline dark:text-blue-400">
                        {coa.ticker}
                      </Link>
                    )}
                    <span className="ml-auto text-xs text-subtle">{coa.actionability?.replace(/_/g, " ") ?? "course"}</span>
                  </div>
                  <div className="mt-1 flex flex-wrap gap-2">
                    <EffectScopeBadge scope={coa.effect_scope ?? "internal_state"} />
                    <QualityStateBadge state={coa.quality_state ?? coa.source_quality ?? "missing"} />
                    <PolicyStateBadge state={coa.policy_state ?? coa.policy_gate_decision ?? "missing"} />
                  </div>
                  <p className="mt-2 text-xs text-muted line-clamp-2">
                    {coa.rationale_summary || "No rationale summary captured yet."}
                  </p>
                  {coa.comparison_id && (
                    <p className="mt-1 text-[11px] text-subtle">Comparison: {coa.comparison_id}</p>
                  )}
                  <div className="mt-2">
                    <TraceTriggerButton
                      compact
                      label={`Trace course of action ${coa.action}`}
                      onClick={() =>
                        openDecisionTrace({
                          kind: "course_of_action",
                          record: coa as unknown as Record<string, unknown>,
                        })
                      }
                    />
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        <ThesisClaimsPanel claims={thesisClaimItems} claimCount={thesisClaimCount} />

        {/* Recommendation Summary */}
        {(data.recommendations.latest_daily || data.recommendations.latest_weekly || data.recommendations.blocked_warnings.length > 0) && (
          <section className="theme-surface rounded-xl p-4 lg:col-span-2">
            <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
              <AlertTriangle size={14} className={data.recommendations.blocked_warnings.length ? "text-amber-500" : "text-blue-500"} />
              Recommendation Review
              <span className="ml-auto text-xs text-subtle">{data.recommendations.pending_actionable.count} pending approval{data.recommendations.pending_actionable.count !== 1 ? "s" : ""}</span>
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[data.recommendations.latest_daily, data.recommendations.latest_weekly].filter(Boolean).map(rec => (
                <div key={`${rec!.report_type}-${rec!.id}`} className="rounded-lg border border-app px-3 py-2 text-sm">
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-medium text-app capitalize">{rec!.report_type}</span>
                    <span className="text-xs text-subtle">{rec!.as_of}</span>
                  </div>
                  <div className="mt-1 flex flex-wrap items-center gap-2">
                    <DecisionStateBadge state={recommendationDecisionState(rec!)} />
                    <EffectScopeBadge scope={rec!.effect_scope ?? "read_only"} />
                    <span className="text-xs px-1.5 py-0.5 rounded bg-[hsl(var(--muted-2))] text-muted">{rec!.stance}</span>
                    <span className="text-xs px-1.5 py-0.5 rounded bg-[hsl(var(--muted-2))] text-muted">{rec!.action.replace(/_/g, " ")}</span>
                    <QualityStateBadge state={rec!.quality_state ?? rec!.critical_data_quality} />
                    {recommendationNeedsPolicyGate(rec!) && (
                      <PolicyStateBadge state={rec!.policy_state ?? rec!.policy_gate_decision ?? "missing"} />
                    )}
                  </div>
                  <p className="mt-2 text-xs text-muted line-clamp-2">{rec!.rationale}</p>
                  <RiskBindingLine record={rec!} />
                  {recommendationNeedsPolicyGate(rec!) && <PolicyGatePanel gate={policyGateFromRecommendation(rec!)} />}
                  <div className="mt-2">
                    <TraceTriggerButton
                      compact
                      label={`Trace ${rec!.report_type} recommendation`}
                      onClick={() =>
                        openDecisionTrace({
                          kind: "recommendation",
                          record: rec! as unknown as Record<string, unknown>,
                        })
                      }
                    />
                  </div>
                </div>
              ))}
            </div>
            {data.recommendations.blocked_warnings.length > 0 && (
              <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300">
                {data.recommendations.blocked_warnings.map(w => (
                  <div key={`${w.report_type}-${w.as_of}`}>
                    <span className="font-medium capitalize">{w.report_type}</span> recommendations blocked by {w.critical_data_quality}: {w.blocked_reasons.join("; ") || "critical data unavailable"}
                  </div>
                ))}
              </div>
            )}
            {data.recommendations.pending_actionable.items.length > 0 && (
              <div className="mt-3 space-y-2">
                {data.recommendations.pending_actionable.items.map(rec => (
                  <div key={`pending-rec-${rec.id}`} className="rounded-lg border border-app px-3 py-2 text-sm">
                    <div className="flex flex-wrap items-center gap-2">
                      <DecisionStateBadge state={recommendationDecisionState(rec)} />
                      <span className="font-semibold text-app">{rec.action.replace(/_/g, " ")}</span>
                      <span className="text-xs text-subtle">{rec.instrument}</span>
                      {rec.ticker && (
                        <Link to={`/dossier/${encodeURIComponent(rec.ticker)}`} state={{ from: "workspace" }} className="text-xs font-semibold text-blue-600 hover:underline dark:text-blue-400">
                          {rec.ticker}
                        </Link>
                      )}
                      <span className="ml-auto text-xs text-subtle">approval {rec.approval_status}</span>
                    </div>
                    <div className="mt-1 flex flex-wrap gap-2">
                      <EffectScopeBadge scope={rec.effect_scope ?? "internal_state"} />
                      <QualityStateBadge state={rec.quality_state ?? rec.critical_data_quality} />
                      {recommendationNeedsPolicyGate(rec) && (
                        <PolicyStateBadge state={rec.policy_state ?? rec.policy_gate_decision ?? "missing"} />
                      )}
                    </div>
                    <p className="mt-1 text-xs text-muted line-clamp-2">{rec.rationale}</p>
                    <RiskBindingLine record={rec} />
                    {recommendationNeedsPolicyGate(rec) && <PolicyGatePanel gate={policyGateFromRecommendation(rec)} />}
                    <div className="mt-2">
                      <TraceTriggerButton
                        compact
                        label={`Trace pending recommendation ${rec.action}`}
                        onClick={() =>
                          openDecisionTrace({
                            kind: "recommendation",
                            record: rec as unknown as Record<string, unknown>,
                          })
                        }
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        {(data.decision_learning?.pending_review.count || data.decision_learning?.recent_finalized.count) ? (
          <section className="theme-surface rounded-xl p-4 lg:col-span-2">
            <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
              <GitBranch size={14} className="text-violet-500" />
              Decision Learning
              <span className="ml-auto text-xs text-subtle">
                {data.decision_learning?.pending_review.count ?? 0} draft review
                {(data.decision_learning?.pending_review.count ?? 0) !== 1 ? "s" : ""}
              </span>
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {(data.decision_learning?.pending_review.items ?? []).map(outcome => (
                <div key={`pending-outcome-${outcome.decision_outcome_id || outcome.object_uid || outcome.id}`} className="rounded-lg border border-app px-3 py-2 text-sm">
                  <div className="flex flex-wrap items-center gap-2">
                    {outcome.ticker && (
                      <Link to={`/dossier/${encodeURIComponent(outcome.ticker)}`} state={{ from: "workspace" }} className="font-semibold text-app hover:underline">
                        {outcome.ticker}
                      </Link>
                    )}
                    <span className="text-xs text-subtle">{outcome.as_of}</span>
                    {outcome.process_label && (
                      <span className="text-xs px-1.5 py-0.5 rounded bg-[hsl(var(--muted-2))] text-muted">
                        {outcome.process_label.replace(/_/g, " ")}
                      </span>
                    )}
                  </div>
                  <p className="mt-2 text-xs text-muted line-clamp-3">{outcome.draft_postmortem}</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => setPostMortemReview(outcome)}
                      className="rounded-lg border border-app px-2.5 py-1 text-xs font-medium text-app hover:bg-[hsl(var(--muted-2))]"
                    >
                      Review post-mortem
                    </button>
                    <TraceTriggerButton
                      label="Trace"
                      onClick={() =>
                        openDecisionTrace({
                          kind: "decision_outcome",
                          record: outcome as unknown as Record<string, unknown>,
                        })
                      }
                    />
                  </div>
                </div>
              ))}
            </div>
            {(data.decision_learning?.recent_finalized.items ?? []).length > 0 && (
              <div className="mt-4 space-y-2">
                <h3 className="text-xs font-semibold uppercase tracking-wide text-subtle">Recent finalized</h3>
                {(data.decision_learning?.recent_finalized.items ?? []).slice(0, 3).map(outcome => (
                  <div key={`final-outcome-${outcome.decision_outcome_id || outcome.object_uid || outcome.id}`} className="rounded-lg border border-app px-3 py-2 text-xs text-muted">
                    <div className="flex flex-wrap items-center gap-2">
                      {outcome.ticker && <span className="font-medium text-app">{outcome.ticker}</span>}
                      <span>{outcome.final_label_status}</span>
                      {outcome.finalized_at && <span>{outcome.finalized_at.slice(0, 10)}</span>}
                      <TraceTriggerButton
                        compact
                        label={`Trace finalized outcome ${outcome.ticker || ""}`}
                        className="ml-auto"
                        onClick={() =>
                          openDecisionTrace({
                            kind: "decision_outcome",
                            record: outcome as unknown as Record<string, unknown>,
                          })
                        }
                      />
                    </div>
                    <p className="mt-1 line-clamp-2">{outcome.final_postmortem || outcome.lessons_learned}</p>
                  </div>
                ))}
              </div>
            )}
          </section>
        ) : null}

        {/* Open Action Items */}
        {data.open_actions.count > 0 && (
          <section className="theme-surface flex min-h-0 max-h-[min(56rem,calc(100dvh-8rem))] flex-col overflow-hidden rounded-xl p-4 max-md:max-h-[min(40rem,calc(100dvh-12rem))]">
            <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
              <Play size={14} className="text-purple-500" />
              Internal Action Items
              <span className="ml-auto text-xs text-subtle">{data.open_actions.count} total</span>
            </h2>
            <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
              {data.open_actions.items.map(a => {
                const key = `action-${a.id}`
                const expanded = expandedIds.has(key)
                return (
                  <div key={a.id} className="rounded-lg border border-app px-3 py-2">
                    <div className="flex items-start gap-3 text-sm">
                      <span className={cn("text-xs px-1.5 py-0.5 rounded font-medium shrink-0 mt-0.5", URGENCY_COLORS[a.urgency] ?? URGENCY_COLORS.normal)}>
                        {a.urgency}
                      </span>
                      {a.ticker && (
                        <Link to={`/dossier/${encodeURIComponent(a.ticker)}`} state={{ from: "workspace" }} className="font-semibold text-app hover:underline shrink-0">
                          {a.ticker}
                        </Link>
                      )}
                      <p onClick={() => toggleExpanded(key)} className={cn("text-muted cursor-pointer", !expanded && "line-clamp-1")}>
                        {a.description}
                      </p>
                    </div>
                    <div className="flex items-center gap-1 mt-2">
                      <button
                        onClick={() => handleActionItem(a.id, "complete")}
                        disabled={processingIds.has(a.id)}
                        className="rounded px-2 py-1 text-xs font-medium text-green-700 bg-green-50 hover:bg-green-100 dark:text-green-400 dark:bg-green-950 disabled:opacity-50"
                      >
                        Propose Complete
                      </button>
                      <button
                        onClick={() => handleActionItem(a.id, "dismiss")}
                        disabled={processingIds.has(a.id)}
                        className="rounded px-2 py-1 text-xs font-medium text-gray-600 bg-gray-50 hover:bg-gray-100 dark:text-gray-400 dark:bg-gray-800 disabled:opacity-50"
                      >
                        Propose Dismiss
                      </button>
                    </div>
                  </div>
                )
              })}
            </div>
          </section>
        )}

        {builderDefinitionCount > 0 && (
          <section className="theme-surface flex min-h-0 max-h-[min(42rem,calc(100dvh-8rem))] flex-col overflow-hidden rounded-xl p-4">
            <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
              <Database size={14} className="text-violet-500" />
              Builder Definitions
              <span className="ml-auto text-xs text-subtle">{builderDefinitionCount} active</span>
            </h2>
            <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
              {[...builderMonitors, ...builderMissions].slice(0, 8).map(definition => {
                const id = String(definition.object_uid || definition.id || definition.monitor_id || definition.mission_id || definition.name)
                const kind = definition.monitor_id || definition.condition ? "Monitor" : "Mission"
                return (
                  <div key={id} className="rounded-lg border border-app/60 px-3 py-2 text-sm">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-medium text-app">{definition.name}</span>
                      <span className="text-xs uppercase tracking-wide text-subtle">{kind}</span>
                      <span className="rounded bg-violet-50 px-1.5 py-0.5 text-xs font-medium text-violet-800 dark:bg-violet-950 dark:text-violet-200">
                        safe mode
                      </span>
                    </div>
                    <p className="mt-1 text-xs text-muted">
                      {definition.condition || definition.mission_type || definition.trigger_type || "Review mission outputs"}
                    </p>
                    <p className="mt-1 text-[11px] text-subtle">Hits are recorded and review actions are staged before state changes.</p>
                  </div>
                )
              })}
            </div>
          </section>
        )}

        {/* Thesis surveillance / monitor hits */}
        {data.monitor_hits.count > 0 && (
          <section className="theme-surface flex min-h-0 max-h-[min(56rem,calc(100dvh-8rem))] flex-col overflow-hidden rounded-xl p-4 max-md:max-h-[min(40rem,calc(100dvh-12rem))]">
            <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
              <AlertTriangle size={14} className="text-amber-500" />
              Thesis Surveillance
              <span className="ml-auto text-xs text-subtle">{data.monitor_hits.count} open</span>
            </h2>
            <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
              {data.monitor_hits.items.map(hit => (
                <div key={hit.id} className="rounded-lg px-3 py-2 text-sm border border-app/60">
                  <div className="flex flex-wrap items-center gap-2">
                    {hit.ticker && (
                      <Link to={`/dossier/${encodeURIComponent(hit.ticker)}`} state={{ from: "workspace" }} className="font-semibold text-app hover:underline">
                        {hit.ticker}
                      </Link>
                    )}
                    <span className="text-xs uppercase tracking-wide text-subtle">{hit.entity_type.replace(/_/g, " ")}</span>
                    <span className="rounded bg-amber-50 px-1.5 py-0.5 text-xs font-medium text-amber-800 dark:bg-amber-950 dark:text-amber-200">
                      {hit.hit_type.replace(/_/g, " ")}
                    </span>
                    {hit.severity && <span className="text-xs text-subtle">{hit.severity}</span>}
                  </div>
                  <p className="mt-1 text-xs text-muted">{hit.entity_label || hit.entity_id}</p>
                  {hit.evidence && <p className="mt-1 text-xs text-subtle">{hit.evidence}</p>}
                  {hit.detected_at && <p className="mt-1 text-[11px] text-subtle">Detected {formatTime(hit.detected_at)}</p>}
                  <div className="mt-2">
                    <TraceTriggerButton
                      compact
                      label={`Trace monitor hit ${hit.ticker || hit.id}`}
                      onClick={() =>
                        openDecisionTrace({
                          kind: "monitor_hit",
                          record: hit as unknown as Record<string, unknown>,
                        })
                      }
                    />
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

      </div>

      {/* Empty state */}
      {!data.thesis_pressure.length &&
        !data.recommendations.latest_daily &&
        !data.recommendations.latest_weekly &&
        !data.recommendations.pending_actionable.count &&
        !data.recommendations.blocked_warnings.length &&
        !courseOfActions.pending.count &&
        !courseOfActions.recent.count &&
        !courseOfActions.comparisons.count &&
        !approvalSummaryInitialLoading &&
        !approvalSummaryError &&
        !approvalCount &&
        !data.open_actions.count &&
        !data.active_triggers.count &&
        builderDefinitionCount === 0 &&
        !data.monitor_hits.count &&
        !data.recent_workflow_runs.length &&
        !recentReportRuns.length &&
        optimizerAlertCount === 0 &&
        thesisClaimCount === 0 && (
        <div className="theme-surface rounded-xl p-8 text-center text-muted text-sm mt-4">
          No pending items. Run a workflow or chat with the agent to get started.
        </div>
      )}

      {data.source_health && (
        <div className="mt-6">
          <CollapsibleSourceHealth sourceHealth={data.source_health} />
        </div>
      )}
      <Dialog
        open={bulkDismissOpen}
        onOpenChange={open => {
          if (bulkDismissSubmitting) return
          setBulkDismissOpen(open)
          if (!open) setBulkDismissError(null)
        }}
        title="Dismiss All Pending Approvals"
        description="This rejects every currently pending approval proposal, including proposals not visible in the Workspace list."
        maxWidth="max-w-lg"
      >
        <div className="space-y-4">
          <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300">
            {BULK_DISMISS_APPROVAL_NOTE}
          </div>
          {bulkDismissError && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
              {bulkDismissError}
            </div>
          )}
          <div className="flex flex-wrap justify-end gap-2">
            <button
              type="button"
              onClick={() => {
                setBulkDismissOpen(false)
                setBulkDismissError(null)
              }}
              disabled={bulkDismissSubmitting}
              className="rounded-lg border border-app px-3 py-2 text-sm font-medium text-muted hover:text-app disabled:opacity-50"
            >
              Cancel
            </button>
            <ActionButton
              onClick={handleBulkDismissApprovals}
              loading={bulkDismissSubmitting}
              loadingText="Dismissing..."
              className="theme-button-destructive w-auto px-4"
            >
              Dismiss All
            </ActionButton>
          </div>
        </div>
      </Dialog>
      <Dialog
        open={approvalReview !== null}
        onOpenChange={open => {
          if (!open) {
            setApprovalReview(null)
            setApprovalNote("")
            setApprovalError(null)
            setApprovalDialogAction(null)
            clearApprovalDeepLink()
          }
        }}
        title="Review Approval"
        description="Review the staged change, then approve and apply it or reject the proposal."
        maxWidth="max-w-3xl"
      >
        {approvalReview && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <DecisionStateBadge state={approvalDecisionState(approvalReview)} />
              <BaseStateBadge
                state={approvalReview.base_state_status}
                message={approvalReview.base_state_message}
              />
              <EffectScopeBadge scope={approvalReview.effect_scope ?? "internal_state"} />
              <PolicyStateBadge state={approvalReview.policy_state ?? policyGateFromApproval(approvalReview)?.decision ?? "missing"} />
              <QualityStateBadge state={approvalReview.quality_state ?? "missing"} />
              <ApprovalSourceHealthBadge review={approvalReview.source_health_review} />
            </div>
            <div className="rounded-lg border border-app bg-[hsl(var(--muted-2))] p-3 text-xs text-muted">
              <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1">
                <span>{formatApprovalDisplayLabel(approvalReview.id, "Approval")}</span>
                <span>{approvalSubjectLabel(approvalReview)}</span>
                {approvalTickerLabel(approvalReview) && <span>{approvalTickerLabel(approvalReview)}</span>}
                <span>Application: {applicationLabel(approvalReview)}</span>
              </div>
              {approvalReasonLabel(approvalReview) && <p className="mb-2">{approvalReasonLabel(approvalReview)}</p>}
              <ApprovalProgressSummary approval={approvalReview} />
              {(() => {
                const record = approvalReview.proposed_change.record
                if (record && typeof record === "object" && !Array.isArray(record)) {
                  return <RiskBindingLine record={record as Record<string, unknown>} />
                }
                return null
              })()}
              <ApprovalChangeSummary approval={approvalReview} />
            </div>
            {approvalReview.base_state_status === "stale" && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
                {approvalReview.base_state_message || "The underlying state changed after this proposal was created."}
              </div>
            )}
            <ApprovalSourceHealthPanel review={approvalReview.source_health_review} />
            <div className="flex flex-wrap gap-2">
              <TraceTriggerButton
                label="View decision trace"
                onClick={() =>
                  openDecisionTrace({
                    kind: "approval",
                    record: approvalReview as unknown as Record<string, unknown>,
                  })
                }
              />
            </div>
            <div>
              <label htmlFor="approval-note" className="theme-field-label">
                Decision note
              </label>
              <textarea
                id="approval-note"
                value={approvalNote}
                onChange={e => setApprovalNote(e.target.value)}
                className="theme-input mt-1 min-h-[90px] w-full"
                placeholder="Required for approval. Optional for rejection."
              />
              <p className="theme-field-caption mt-1">Required for approval. Rejection notes are optional.</p>
            </div>
            {approvalError && (
              <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                {approvalError}
              </div>
            )}
            <div className="flex flex-col-reverse gap-2 sm:flex-row sm:flex-wrap sm:justify-end">
              <button
                type="button"
                onClick={() => {
                  setApprovalReview(null)
                  setApprovalNote("")
                  setApprovalError(null)
                  setApprovalDialogAction(null)
                }}
                className="w-full rounded-lg border border-app px-3 py-2.5 text-sm font-medium text-muted hover:text-app sm:w-auto sm:py-2"
              >
                Cancel
              </button>
              <ActionButton
                onClick={() => handleApproval(approvalReview, "reject", approvalNote)}
                loading={approvalDialogAction === "reject" && processingIds.has(approvalReview.id)}
                loadingText="Rejecting..."
                disabled={processingIds.has(approvalReview.id) || approvalReview.can_reject === false}
                className="theme-button-destructive w-full px-4 sm:w-auto"
              >
                Reject Proposal
              </ActionButton>
              {approvalReview.can_restage && (
                <ActionButton
                  onClick={() => handleRejectAndRestage(approvalReview, approvalNote)}
                  loading={approvalDialogAction === "restage" && processingIds.has(approvalReview.id)}
                  loadingText="Restaging..."
                  disabled={processingIds.has(approvalReview.id)}
                  className="w-full bg-amber-600 px-4 hover:bg-amber-700 sm:w-auto"
                >
                  Reject & Restage
                </ActionButton>
              )}
              {watchTriggerProposalFromApproval(approvalReview) && (
                <ActionButton
                  onClick={() => {
                    const trigger = watchTriggerProposalFromApproval(approvalReview)
                    if (!trigger) return
                    setTriggerEdit({ kind: "approval", approval: approvalReview, trigger })
                    setTriggerEditError(null)
                  }}
                  disabled={processingIds.has(approvalReview.id)}
                  className="w-full px-4 sm:w-auto"
                >
                  Edit Proposal
                </ActionButton>
              )}
              <ActionButton
                onClick={() => handleApproval(approvalReview, "approve", approvalNote)}
                loading={approvalDialogAction === "approve" && processingIds.has(approvalReview.id)}
                loadingText={approvalActionLabel(approvalReview) === "Record Approval" ? "Recording..." : "Applying..."}
                disabled={processingIds.has(approvalReview.id) || !approvalNote.trim() || approvalReview.can_approve === false}
                className="theme-button-success w-full px-4 sm:w-auto"
              >
                {approvalActionLabel(approvalReview)}
              </ActionButton>
            </div>
          </div>
        )}
      </Dialog>
      <WatchTriggerEditDialog
        open={triggerEdit !== null}
        onOpenChange={open => {
          if (!open) {
            setTriggerEdit(null)
            setTriggerEditError(null)
          }
        }}
        trigger={triggerEdit?.trigger ?? null}
        title={triggerEdit?.kind === "approval" ? "Edit Trigger Proposal" : "Replace Watch Trigger"}
        description={
          triggerEdit?.kind === "approval"
            ? "Create an edited replacement proposal and reject the original pending trigger proposal."
            : "Stage a replacement that cancels the current trigger and creates a new active trigger after approval."
        }
        submitLabel={triggerEdit?.kind === "approval" ? "Stage Edited Proposal" : "Stage Replacement"}
        loading={triggerEditSubmitting}
        error={triggerEditError}
        onSubmit={handleSubmitTriggerEdit}
      />
      <PostMortemReviewDialog
        open={postMortemReview !== null}
        onOpenChange={open => {
          if (!open) setPostMortemReview(null)
        }}
        outcome={postMortemReview}
        onFinalized={() => {
          void qc.invalidateQueries({ queryKey: ["workspace"] })
        }}
      />
    </div>
  )
}
