import { Link, useSearchParams } from "react-router-dom"
import { useQueryClient } from "@tanstack/react-query"
import { Bell, CheckCircle, AlertTriangle, Eye, Play, Clock, GitBranch, Database, FileText, X } from "lucide-react"
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
  type OptimizationAlert,
  type ThesisClaim,
  type PolicyGateReason,
  type PolicyGateResult,
  type ProvenanceSelector,
  type RecommendationRecord,
  type SourceHealth,
  type SourceHealthSource,
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
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { ProvenanceTraceDialog } from "@/components/shared/ProvenanceTraceDialog"
import { Dialog } from "@/components/shared/Dialog"
import { ApprovalChangeSummary } from "@/components/shared/ApprovalChangeSummary"
import { ApprovalProgressSummary } from "@/components/shared/ApprovalProgressSummary"
import { approvalActionLabel } from "@/components/shared/approvalProgress"
import { ActionButton } from "@/components/shared/FormControls"
import { formatApprovalDisplayLabel } from "@/components/shared/StagedProposalNotice"
import { WhatChangedPanel, type WhatChangedSummary } from "@/components/shared/WhatChangedPanel"
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
  recent_workflow_runs: WorkflowRun[]
  continuous_optimization?: {
    open_alert_count: number
    open_alerts: OptimizationAlert[]
  }
  thesis_claims?: {
    challenged_count: number
    items: ThesisClaim[]
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

function formatPnl(value: number | null | undefined): string {
  if (value == null) return "--"
  const sign = value >= 0 ? "+" : ""
  return `${sign}${value.toFixed(2)}%`
}

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

function OptimizationAlertsPanel({
  alerts,
  alertCount,
  processingIds,
  onDismiss,
  dismissError,
}: {
  alerts: OptimizationAlert[]
  alertCount: number
  processingIds: Set<number | string>
  onDismiss: (alert: OptimizationAlert) => void
  dismissError: string | null
}) {
  if (alertCount <= 0) return null
  return (
    <section className="theme-surface flex min-h-0 max-h-[min(56rem,calc(100dvh-8rem))] flex-col overflow-hidden rounded-xl p-4">
      <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
        <Bell size={14} className="text-purple-500" />
        Optimizer Alerts
        <Link to="/analyzer" className="ml-auto text-xs font-medium text-blue-600 hover:underline dark:text-blue-400">
          Open analyzer
        </Link>
      </h2>
      <p className="mb-3 text-xs text-subtle">{alertCount} open alert{alertCount !== 1 ? "s" : ""} with material action or risk changes.</p>
      {dismissError && (
        <div className="mb-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          Failed to dismiss alert: {dismissError}
        </div>
      )}
      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
        {alerts.map(alert => (
          <div key={alert.id} className="rounded-lg border border-app px-3 py-2 text-sm">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  {alert.ticker ? (
                    <Link to={`/dossier/${encodeURIComponent(alert.ticker)}`} state={{ from: "workspace" }} className="font-semibold text-app hover:underline">
                      {alert.ticker}
                    </Link>
                  ) : (
                    <span className="font-semibold text-app">PORTFOLIO</span>
                  )}
                  <span className={cn("rounded px-1.5 py-0.5 text-xs font-medium", alertSeverityClass(alert.severity))}>
                    {alert.severity}
                  </span>
                  <span className="text-xs text-subtle">{alert.alert_type.replace(/_/g, " ")}</span>
                </div>
                <p className="mt-1 text-xs text-muted line-clamp-2">{alert.change_summary}</p>
              </div>
              <button
                type="button"
                onClick={() => onDismiss(alert)}
                disabled={processingIds.has(`optimizer-alert-${alert.id}`)}
                className="rounded px-2 py-1 text-xs font-medium text-gray-600 bg-gray-50 hover:bg-gray-100 dark:text-gray-400 dark:bg-gray-800 disabled:opacity-50"
              >
                Dismiss
              </button>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
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

function RecentActivityPanel({
  workflowRuns,
  reportRuns,
  onViewWorkflowLineage,
}: {
  workflowRuns: WorkflowRun[]
  reportRuns: ReportRun[]
  onViewWorkflowLineage?: (runId: string) => void
}) {
  if (workflowRuns.length === 0 && reportRuns.length === 0) return null
  return (
    <section className="theme-surface rounded-xl p-4 lg:col-span-2">
      <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
        <Clock size={14} className="text-gray-500" />
        Recent Activity
      </h2>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {workflowRuns.length > 0 && (
          <div>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-subtle">Workflow Runs</h3>
            <div className="space-y-2">
              {workflowRuns.map((run, index) => {
                const ticker = workflowRunTicker(run)
                const runId = String(run.run_id ?? "").trim()
                return (
                  <div key={runId || `workflow-run-${index}`} className="flex items-center justify-between rounded-lg px-3 py-2 text-sm">
                    <div className="flex items-center gap-3">
                      <span className={cn("h-2 w-2 shrink-0 rounded-full", workflowStatusClass(run.status))} />
                      <span className="font-medium text-app">{workflowRunLabel(run)}</span>
                      {ticker && (
                        <Link to={`/dossier/${encodeURIComponent(ticker)}`} state={{ from: "workspace" }} className="text-blue-600 hover:underline dark:text-blue-400">
                          {ticker}
                        </Link>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-subtle">{workflowRunTime(run)}</span>
                      {runId && onViewWorkflowLineage && (
                        <button
                          type="button"
                          onClick={() => onViewWorkflowLineage(runId)}
                          className="theme-icon-button h-8 w-8"
                          aria-label={`View workflow ${runId} lineage`}
                          title="Lineage"
                        >
                          <GitBranch size={14} />
                        </button>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}
        {reportRuns.length > 0 && (
          <div>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-subtle">Report Runs</h3>
            <div className="space-y-2">
              {reportRuns.map((run, index) => (
                <div key={reportRunKey(run, index)} className="flex items-center justify-between rounded-lg px-3 py-2 text-sm">
                  <div className="flex min-w-0 items-center gap-3">
                    <FileText size={14} className="shrink-0 text-blue-500" />
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-medium text-app">{reportRunLabel(run)}</span>
                        <span className={cn("h-2 w-2 shrink-0 rounded-full", workflowStatusClass(run.status))} />
                        <span className="text-xs text-subtle">{run.status ?? "unknown"}</span>
                      </div>
                      {run.error && <p className="mt-0.5 truncate text-xs text-red-600 dark:text-red-400">{run.error}</p>}
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <span className="text-xs text-subtle">{reportRunTime(run)}</span>
                    {run.issue_url && (
                      <a href={run.issue_url} target="_blank" rel="noreferrer" className="text-xs font-medium text-blue-600 hover:underline dark:text-blue-400">
                        Issue
                      </a>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  )
}

function sourceHealthLabel(value: string | null | undefined): string {
  const normalized = String(value || "missing").replace(/_/g, " ")
  return normalized.charAt(0).toUpperCase() + normalized.slice(1)
}

function sourceHealthBadgeClass(source: Pick<SourceHealthSource, "status" | "required">): string {
  const status = String(source.status || "missing")
  if (source.required && ["stale", "failed", "missing"].includes(status)) return "theme-badge-error"
  if (status === "ok") return "theme-badge-success"
  if (status === "failed") return "theme-badge-error"
  if (status === "stale" || status === "degraded") return "theme-badge-warning"
  return "theme-badge-neutral"
}

function sourceHealthOverallClass(quality: string | null | undefined): string {
  const value = String(quality || "missing")
  if (value === "ok") return "theme-badge-success"
  if (value === "failed" || value === "missing") return "theme-badge-error"
  if (value === "stale" || value === "degraded") return "theme-badge-warning"
  return "theme-badge-neutral"
}

function sourceHealthTimestamp(source: SourceHealthSource): string {
  return formatTime(source.freshness_timestamp ?? source.as_of ?? source.fetched_at)
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

function SourceHealthPanel({ sourceHealth }: { sourceHealth: SourceHealth }) {
  const counts = sourceHealth.counts ?? {}
  const tierCounts = sourceHealth.tier_counts ?? {}
  const domains = sourceHealth.domains ?? []
  return (
    <section className="theme-surface mt-6 rounded-xl p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-app">
          <Database size={14} className="text-blue-500" />
          Source Health
        </h2>
        <span className={cn("theme-badge ml-auto", sourceHealthOverallClass(sourceHealth.overall_quality))}>
          {sourceHealthLabel(sourceHealth.overall_quality)}
        </span>
        <span className="text-xs text-subtle">Updated {formatTime(sourceHealth.generated_at)}</span>
      </div>
      <div className="mb-3 flex flex-wrap gap-2 text-xs text-muted">
        <span>{counts.total ?? 0} sources</span>
        <span>{counts.ok ?? 0} ok</span>
        <span>{counts.critical_stale ?? 0} critical stale</span>
        <span>{counts.critical_failed ?? 0} critical failed</span>
        <span>{counts.sla_breach ?? 0} SLA breach</span>
        <span>{counts.optional_degraded ?? 0} optional degraded</span>
        {(tierCounts.critical ?? 0) > 0 && <span>{tierCounts.critical} critical tier</span>}
      </div>
      {domains.length === 0 ? (
        <div className="rounded-lg border border-app px-3 py-2 text-sm text-muted">
          No source freshness records are available yet.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
          {domains.map(domain => (
            <div key={domain.domain} className="rounded-lg border border-app px-3 py-3">
              <div className="mb-2 flex items-center gap-2">
                <span className="text-sm font-semibold text-app">{domain.label}</span>
                <span className={cn("theme-badge ml-auto", sourceHealthOverallClass(domain.overall_quality))}>
                  {sourceHealthLabel(domain.overall_quality)}
                </span>
              </div>
              <div className="space-y-2">
                {domain.sources.map(source => (
                  <div
                    key={source.id}
                    className={cn(
                      "grid gap-2 rounded-md px-2 py-2 text-xs sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center",
                      source.required && ["stale", "failed", "missing"].includes(source.status)
                        ? "border border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-950/40"
                        : "bg-[hsl(var(--muted-2))]",
                    )}
                  >
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-semibold text-app">{source.source_name.replace(/_/g, " ")}</span>
                        <span className={cn("theme-badge", reliabilityTierBadgeClass(source.reliability_tier))}>
                          {reliabilityTierLabel(source.reliability_tier)}
                        </span>
                        <span className="text-subtle">{source.required ? "required" : "optional"}</span>
                        {source.sla_breach && <span className="text-red-600 dark:text-red-400">SLA breach</span>}
                      </div>
                      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-subtle">
                        <span>{sourceHealthTimestamp(source)}</span>
                        {source.detail && <span className="max-w-full truncate">{source.detail}</span>}
                      </div>
                    </div>
                    <span className={cn("theme-badge w-fit", sourceHealthBadgeClass(source))}>
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
  return Boolean(rec.policy_gate_decision) || ACTIONABLE_RECOMMENDATION_ACTIONS.has(rec.action)
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
  const [provenanceSelector, setProvenanceSelector] = useState<ProvenanceSelector | null>(null)
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
  const approvalHasMore = approvalSummaryData?.has_more ?? approvalItems.length < approvalCount
  const approvalSummaryInitialLoading = approvalSummary.isPending && !approvalSummaryData
  const approvalCountLabel = approvalSummaryInitialLoading
    ? "loading"
    : approvalHasMore
      ? `showing ${approvalItems.length} of ${approvalCount} total`
      : `${approvalCount} total`
  const approvalRecommendationCount =
    approvalSummaryData?.recommendation_approval_count ?? data.recommendations.pending_approval_count
  const courseOfActions = data.course_of_actions ?? {
    pending: { count: 0, items: [] },
    recent: { count: 0, items: [] },
    comparisons: { count: 0, items: [] },
    pending_approval_count: 0,
  }
  const approvalSummaryError = approvalSummary.error
  const regime = data.regime
  const portfolioRisk = data.portfolio?.risk
  const regimeInfo = regime?.signal ? REGIME_SIGNAL_MAP[regime.signal.toLowerCase()] : null
  const regimeSubtitle = [
    regime?.composite_score != null ? `Score: ${regime.composite_score}` : null,
    regime?.snapshot?.as_of ? `As of ${regime.snapshot.as_of}` : null,
    regime?.snapshot?.refresh_status && regime.snapshot.refresh_status !== "ok" ? `Refresh ${regime.snapshot.refresh_status}` : null,
    regime?.snapshot?.stale ? "Stale" : null,
  ].filter(Boolean).join(" · ")
  const optimizerAlerts = data.continuous_optimization?.open_alerts ?? []
  const optimizerAlertCount = data.continuous_optimization?.open_alert_count ?? optimizerAlerts.length
  const thesisClaimItems = data.thesis_claims?.items ?? []
  const thesisClaimCount = data.thesis_claims?.challenged_count ?? thesisClaimItems.length
  const recentReportRuns = data.recent_report_runs ?? []

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-app">Portfolio Commander</h1>
          <p className="mt-1 text-sm text-subtle">What changed, what matters, and what needs review.</p>
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

      {/* Top metrics row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard
          title="Market Regime"
          value={regime?.regime ?? "--"}
          subtitle={regimeSubtitle || undefined}
          signal={regimeInfo?.signal ?? null}
          signalLabel={regimeInfo?.label}
        />
        <MetricCard
          title="Positions"
          value={data.portfolio?.position_count ?? "--"}
          subtitle={[
            data.portfolio?.total_pnl_pct != null ? `P&L: ${formatPnl(data.portfolio.total_pnl_pct)}` : null,
            portfolioRisk?.average_risk_score != null ? `Avg risk ${formatRiskScore(portfolioRisk.average_risk_score)}` : null,
          ].filter(Boolean).join(" · ") || undefined}
          signal={portfolioRisk?.quality && portfolioRisk.quality !== "ok" ? "warning" : null}
          signalLabel={portfolioRisk?.risk_level ? String(portfolioRisk.risk_level) : undefined}
        />
        <MetricCard
          title="Pending Approvals"
          value={approvalSummaryInitialLoading ? "--" : approvalCount}
          signal={approvalCount > 0 ? "warning" : null}
          signalLabel={approvalCount > 0 ? "Needs Review" : undefined}
        />
        <MetricCard
          title="Courses Of Action"
          value={courseOfActions.pending.count}
          subtitle={[
            `${courseOfActions.pending_approval_count || approvalRecommendationCount} approval${(courseOfActions.pending_approval_count || approvalRecommendationCount) !== 1 ? "s" : ""}`,
            optimizerAlertCount > 0 ? `${optimizerAlertCount} optimizer alert${optimizerAlertCount !== 1 ? "s" : ""}` : null,
          ].filter(Boolean).join(" · ") || undefined}
          signal={data.recommendations.blocked_warnings.length > 0 || optimizerAlertCount > 0 ? "warning" : null}
          signalLabel={
            data.recommendations.blocked_warnings.length > 0
              ? "Blocked"
              : optimizerAlertCount > 0
                ? "Monitor Hits"
                : undefined
          }
        />
      </div>

      <WhatChangedPanel summary={data.what_changed} className="mb-6" from="workspace" />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {portfolioRisk && (
          <section className="theme-surface rounded-xl p-4 lg:col-span-2">
            <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
              <GitBranch size={14} className={portfolioRisk.quality === "ok" ? "text-blue-500" : "text-amber-500"} />
              Portfolio Risk
              <span className="ml-auto text-xs text-subtle">{portfolioRisk.as_of || portfolioRisk.computed_at}</span>
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div className="rounded-lg border border-app px-3 py-2">
                <p className="text-xs text-subtle">Average Risk</p>
                <p className="mt-1 text-lg font-semibold text-app">{formatRiskScore(portfolioRisk.average_risk_score)}</p>
              </div>
              <div className="rounded-lg border border-app px-3 py-2">
                <p className="text-xs text-subtle">Max Risk</p>
                <p className="mt-1 text-lg font-semibold text-app">{formatRiskScore(portfolioRisk.max_risk_score)}</p>
              </div>
              <div className="rounded-lg border border-app px-3 py-2">
                <p className="text-xs text-subtle">Quality</p>
                <div className="mt-1"><QualityStateBadge state={portfolioRisk.quality || "missing"} /></div>
              </div>
              <div className="rounded-lg border border-app px-3 py-2">
                <p className="text-xs text-subtle">Buckets</p>
                <p className="mt-1 text-sm font-medium text-app">
                  H {portfolioRisk.risk_buckets?.high ?? 0} · M {portfolioRisk.risk_buckets?.medium ?? 0} · L {portfolioRisk.risk_buckets?.low ?? 0}
                </p>
              </div>
            </div>
            {Array.isArray(portfolioRisk.top_contributors) && portfolioRisk.top_contributors.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2 text-xs">
                {portfolioRisk.top_contributors.slice(0, 5).map((row, idx) => (
                  <span key={`${String(row.ticker || "risk")}-${idx}`} className="rounded border border-app px-2 py-1 text-muted">
                    <span className="font-semibold text-app">{String(row.ticker || "Portfolio")}</span>
                    {" "}risk {formatRiskScore(typeof row.risk_score === "number" ? row.risk_score : null)}
                  </span>
                ))}
              </div>
            )}
          </section>
        )}

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
                </div>
              ))}
            </div>
          </section>
        )}

        <OptimizationAlertsPanel
          alerts={optimizerAlerts}
          alertCount={optimizerAlertCount}
          processingIds={processingIds}
          onDismiss={handleDismissOptimizerAlert}
          dismissError={optimizerDismissError}
        />

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
                    <PolicyStateBadge state={rec!.policy_state ?? rec!.policy_gate_decision ?? "missing"} />
                  </div>
                  <p className="mt-2 text-xs text-muted line-clamp-2">{rec!.rationale}</p>
                  <RiskBindingLine record={rec!} />
                  {recommendationNeedsPolicyGate(rec!) && <PolicyGatePanel gate={policyGateFromRecommendation(rec!)} />}
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
                      <PolicyStateBadge state={rec.policy_state ?? rec.policy_gate_decision ?? "missing"} />
                    </div>
                    <p className="mt-1 text-xs text-muted line-clamp-2">{rec.rationale}</p>
                    <RiskBindingLine record={rec} />
                    {recommendationNeedsPolicyGate(rec) && <PolicyGatePanel gate={policyGateFromRecommendation(rec)} />}
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        {/* Thesis Pressure */}
        {data.thesis_pressure.length > 0 && (
          <section className="theme-surface flex min-h-0 max-h-[min(56rem,calc(100dvh-8rem))] flex-col overflow-hidden rounded-xl p-4">
            <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
              <AlertTriangle size={14} className="text-amber-500" />
              Positions Under Pressure
            </h2>
            {pressureDismissError && (
              <div className="mb-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                Failed to clear pressure row: {pressureDismissError}
              </div>
            )}
            <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
              {data.thesis_pressure.map(tp => (
                <div
                  key={tp.pressure_key}
                  className="grid grid-cols-1 gap-2 rounded-lg px-3 py-2 text-sm transition-colors hover:bg-[hsl(var(--muted-2))] sm:grid-cols-[3rem_7.25rem_minmax(0,1fr)_4.5rem_2rem] sm:items-center"
                >
                  <Link
                    to={`/dossier/${encodeURIComponent(tp.ticker)}`}
                    state={{ from: "workspace" }}
                    className="font-semibold text-app hover:underline"
                  >
                    {tp.ticker}
                  </Link>
                  <span className={cn(
                    "w-fit rounded px-1.5 py-0.5 text-xs font-medium leading-4",
                    tp.action === "exit" || tp.action === "reduce"
                      ? "text-red-600 bg-red-50 dark:text-red-400 dark:bg-red-950"
                      : "text-amber-600 bg-amber-50 dark:text-amber-400 dark:bg-amber-950",
                  )}>
                    Evaluation: {tp.action}
                  </span>
                  {tp.risk_flag ? (
                    <span className="text-xs leading-5 text-red-500">{tp.risk_flag}</span>
                  ) : (
                    <span className="hidden sm:block" />
                  )}
                  <span className="text-xs text-subtle sm:text-right">{tp.confidence}</span>
                  <button
                    type="button"
                    onClick={() => handleDismissPressure(tp)}
                    disabled={processingIds.has(`pressure-${tp.pressure_key}`)}
                    className="theme-icon-button h-8 w-8 justify-self-start disabled:cursor-not-allowed disabled:opacity-50 sm:justify-self-end"
                    aria-label={`Clear ${tp.ticker} pressure row`}
                    title="Clear pressure row"
                  >
                    <X size={14} />
                  </button>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Pending Approvals */}
        {(approvalSummaryInitialLoading || approvalSummaryError || approvalCount > 0) && (
          <section className="theme-surface flex min-h-0 max-h-[min(56rem,calc(100dvh-8rem))] flex-col overflow-hidden rounded-xl p-4">
            <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
              <CheckCircle size={14} className="text-blue-500" />
              Pending Approvals
              <span className="ml-auto text-xs text-subtle">{approvalCountLabel}</span>
              {approvalCount > 0 && !approvalSummaryInitialLoading && !approvalSummaryError && (
                <button
                  type="button"
                  onClick={() => {
                    setBulkDismissOpen(true)
                    setBulkDismissError(null)
                  }}
                  disabled={bulkDismissSubmitting}
                  className="rounded px-2 py-1 text-xs font-medium text-red-700 bg-red-50 hover:bg-red-100 dark:text-red-300 dark:bg-red-950 dark:hover:bg-red-900 disabled:opacity-50"
                >
                  Dismiss all
                </button>
              )}
            </h2>
            <div className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
              {approvalSummaryInitialLoading && (
                <div className="rounded-lg border border-app px-3 py-2 text-sm text-muted">
                  Loading approvals...
                </div>
              )}
              {approvalSummaryError && (
                <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                  Failed to load approvals: {String(approvalSummaryError)}
                </div>
              )}
              {!approvalSummaryInitialLoading && !approvalSummaryError && approvalItems.map(a => {
                const key = `approval-${a.id}`
                const expanded = expandedIds.has(key)
                const gate = policyGateFromApproval(a)
                const displayTicker = approvalTickerLabel(a)
                const displayReason = approvalReasonLabel(a)
                return (
                  <div
                    key={a.id}
                    className="overflow-hidden rounded-lg border border-app px-3 py-3 text-sm"
                  >
                    <div className="grid gap-3 2xl:grid-cols-[minmax(0,1fr)_auto] 2xl:items-start">
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                          <DecisionStateBadge state={approvalDecisionState(a)} />
                          {displayTicker && (
                            <Link to={`/dossier/${encodeURIComponent(displayTicker)}`} state={{ from: "workspace" }} className="font-semibold text-app hover:underline">
                              {displayTicker}
                            </Link>
                          )}
                          <span className="min-w-0 break-words text-xs text-subtle">{approvalSubjectLabel(a)}</span>
                          {a.application_status && (
                            <span className="max-w-full truncate rounded border border-app px-1.5 py-0.5 text-[11px] text-subtle">
                              {applicationLabel(a)}
                            </span>
                          )}
                        </div>
                        <div className="mt-1 flex flex-wrap gap-2">
                          <BaseStateBadge state={a.base_state_status} message={a.base_state_message} />
                          <EffectScopeBadge scope={a.effect_scope ?? "internal_state"} />
                          <PolicyStateBadge state={a.policy_state ?? gate?.decision ?? "missing"} />
                          <QualityStateBadge state={a.quality_state ?? "missing"} />
                          <ApprovalSourceHealthBadge review={a.source_health_review} />
                        </div>
                        {displayReason && (
                          <p onClick={() => toggleExpanded(key)} className={cn("mt-0.5 cursor-pointer break-words text-xs text-muted", !expanded && "line-clamp-1")}>
                            {displayReason}
                          </p>
                        )}
                        {a.application_error && (
                          <p className="mt-1 break-words text-[11px] text-red-600 dark:text-red-400">
                            Application failed: {a.application_error}
                          </p>
                        )}
                        {approvalNeedsPolicyGate(a, gate) && (
                          <PolicyGatePanel gate={gate} />
                        )}
                        <ApprovalProgressSummary approval={a} compact />
                      </div>
                      <div className="flex flex-wrap items-center gap-2 2xl:justify-end">
                        <button
                          type="button"
                          onClick={() => setProvenanceSelector({ approval_id: a.id })}
                          className="theme-icon-button h-8 w-8 shrink-0"
                          aria-label={`View approval ${a.id} lineage`}
                          title="Lineage"
                        >
                          <GitBranch size={14} />
                        </button>
                        <button
                          type="button"
                          onClick={() => openApprovalReview(a)}
                          disabled={processingIds.has(a.id)}
                          className="inline-flex h-8 items-center justify-center whitespace-nowrap rounded px-2.5 text-xs font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 dark:text-blue-300 dark:bg-blue-950 dark:hover:bg-blue-900 disabled:opacity-50"
                          title="Review approval"
                        >
                          Review
                        </button>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </section>
        )}

        {/* Open Action Items */}
        {data.open_actions.count > 0 && (
          <section className="theme-surface flex min-h-0 max-h-[min(56rem,calc(100dvh-8rem))] flex-col overflow-hidden rounded-xl p-4">
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

        {/* Active Watch Triggers */}
        {data.active_triggers.count > 0 && (
          <section className="theme-surface flex min-h-0 max-h-[min(56rem,calc(100dvh-8rem))] flex-col overflow-hidden rounded-xl p-4">
            <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
              <Eye size={14} className="text-cyan-500" />
              Active Triggers
              <span className="ml-auto text-xs text-subtle">{data.active_triggers.count} total</span>
            </h2>
            <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
              {data.active_triggers.items.map(t => (
                <div key={t.id} className="rounded-lg px-3 py-2 text-sm">
                  <div className="grid gap-2 xl:grid-cols-[minmax(0,1fr)_auto] xl:items-start">
                    <div className="min-w-0">
                      <div className="flex items-center gap-3">
                        {t.ticker && (
                          <Link to={`/dossier/${encodeURIComponent(t.ticker)}`} state={{ from: "workspace" }} className="font-semibold text-app hover:underline shrink-0">
                            {t.ticker}
                          </Link>
                        )}
                        <span className="text-muted truncate">{t.condition}</span>
                        <span className="text-xs text-subtle shrink-0">{t.trigger_type.replace(/_/g, " ")}</span>
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-subtle">
                        <span>{t.status}</span>
                        {t.last_checked_at && <span>Checked {formatTime(t.last_checked_at)}</span>}
                        {t.last_evidence && <span className="truncate">{t.last_evidence}</span>}
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-2 xl:justify-end">
                      <button
                        type="button"
                        onClick={() => {
                          setTriggerEdit({ kind: "active", trigger: t })
                          setTriggerEditError(null)
                        }}
                        disabled={processingIds.has(t.id)}
                        className="rounded px-2 py-1 text-xs font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 dark:text-blue-300 dark:bg-blue-950 disabled:opacity-50"
                      >
                        Propose Edit
                      </button>
                      <button
                        type="button"
                        onClick={() => handleCancelTrigger(t.id)}
                        disabled={processingIds.has(t.id)}
                        className="rounded px-2 py-1 text-xs font-medium text-gray-600 bg-gray-50 hover:bg-gray-100 dark:text-gray-400 dark:bg-gray-800 disabled:opacity-50"
                      >
                        Propose Cancel
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Thesis surveillance / monitor hits */}
        {data.monitor_hits.count > 0 && (
          <section className="theme-surface flex min-h-0 max-h-[min(56rem,calc(100dvh-8rem))] flex-col overflow-hidden rounded-xl p-4">
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
                </div>
              ))}
            </div>
          </section>
        )}

        <RecentActivityPanel
          workflowRuns={data.recent_workflow_runs}
          reportRuns={recentReportRuns}
          onViewWorkflowLineage={runId => setProvenanceSelector({ workflow_run_id: runId })}
        />
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
        !data.monitor_hits.count &&
        !data.recent_workflow_runs.length &&
        !recentReportRuns.length &&
        optimizerAlertCount === 0 &&
        thesisClaimCount === 0 && (
        <div className="theme-surface rounded-xl p-8 text-center text-muted text-sm mt-4">
          No pending items. Run a workflow or chat with the agent to get started.
        </div>
      )}

      {data.source_health && <SourceHealthPanel sourceHealth={data.source_health} />}
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
            <div className="flex flex-wrap justify-end gap-2">
              <button
                type="button"
                onClick={() => {
                  setApprovalReview(null)
                  setApprovalNote("")
                  setApprovalError(null)
                  setApprovalDialogAction(null)
                }}
                className="rounded-lg border border-app px-3 py-2 text-sm font-medium text-muted hover:text-app"
              >
                Cancel
              </button>
              <ActionButton
                onClick={() => handleApproval(approvalReview, "reject", approvalNote)}
                loading={approvalDialogAction === "reject" && processingIds.has(approvalReview.id)}
                loadingText="Rejecting..."
                disabled={processingIds.has(approvalReview.id) || approvalReview.can_reject === false}
                className="theme-button-destructive w-auto px-4"
              >
                Reject Proposal
              </ActionButton>
              {approvalReview.can_restage && (
                <ActionButton
                  onClick={() => handleRejectAndRestage(approvalReview, approvalNote)}
                  loading={approvalDialogAction === "restage" && processingIds.has(approvalReview.id)}
                  loadingText="Restaging..."
                  disabled={processingIds.has(approvalReview.id)}
                  className="w-auto px-4 bg-amber-600 hover:bg-amber-700"
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
                  className="w-auto px-4"
                >
                  Edit Proposal
                </ActionButton>
              )}
              <ActionButton
                onClick={() => handleApproval(approvalReview, "approve", approvalNote)}
                loading={approvalDialogAction === "approve" && processingIds.has(approvalReview.id)}
                loadingText={approvalActionLabel(approvalReview) === "Record Approval" ? "Recording..." : "Applying..."}
                disabled={processingIds.has(approvalReview.id) || !approvalNote.trim() || approvalReview.can_approve === false}
                className="theme-button-success w-auto px-4"
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
      <ProvenanceTraceDialog
        open={provenanceSelector !== null}
        onOpenChange={open => {
          if (!open) setProvenanceSelector(null)
        }}
        selector={provenanceSelector}
      />
    </div>
  )
}
