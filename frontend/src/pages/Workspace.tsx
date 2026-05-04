import { Link } from "react-router-dom"
import { useQueryClient } from "@tanstack/react-query"
import { CheckCircle, AlertTriangle, Eye, Play, Clock, GitBranch } from "lucide-react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchWorkspace, approveItem, rejectItem, completeAction, dismissAction, refreshMarketSnapshots, type ProvenanceSelector } from "@/lib/api"
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { ProvenanceTraceDialog } from "@/components/shared/ProvenanceTraceDialog"
import { cn } from "@/lib/utils"
import { useState } from "react"

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
  } | null
  thesis_pressure: {
    ticker: string
    status: string
    action: string
    confidence: string
    risk_flag: string | null
    evaluated_at: string
  }[]
  pending_approvals: { count: number; items: Approval[] }
  recommendations: {
    latest_daily: Recommendation | null
    latest_weekly: Recommendation | null
    pending_actionable: { count: number; items: Recommendation[] }
    blocked_warnings: {
      report_type: string
      as_of: string
      critical_data_quality: string
      blocked_reasons: string[]
    }[]
    pending_approval_count: number
  }
  open_actions: { count: number; items: ActionItem[] }
  active_triggers: { count: number; items: Trigger[] }
  recent_workflow_runs: WorkflowRun[]
}

interface Approval {
  id: number
  entity_type: string
  action_id?: string | null
  ticker: string | null
  reason: string | null
  created_at: string
  application_status?: string | null
  application_attempts?: number | null
  source_type?: string | null
  source_id?: string | null
  proposed_change: Record<string, unknown>
}

interface PolicyGateReason {
  code?: string
  check?: string
  message?: string
  observed?: unknown
  limit?: unknown
}

interface PolicyGateResult {
  decision?: string
  review_required?: boolean
  failure_reasons?: PolicyGateReason[]
  warnings?: PolicyGateReason[]
  disclosures?: string[]
}

interface ActionItem {
  id: number
  ticker: string | null
  description: string
  action_type: string
  urgency: string
  created_at: string
}

interface Trigger {
  id: number
  ticker: string | null
  condition: string
  trigger_type: string
  status: string
  created_at: string
  last_checked_at: string | null
  last_evidence: string | null
}

interface WorkflowRun {
  run_id: string
  workflow_name: string
  ticker: string | null
  status: string
  started_at: string
  completed_at: string | null
}

interface Recommendation {
  id: number
  report_type: string
  as_of: string
  stance: string
  recommendation_status: string
  critical_data_quality: string
  action: string
  ticker: string | null
  instrument: string
  rationale: string
  confidence: number | null
  approval_status: string
  blocked_reasons_json?: string[]
  policy_gate_decision?: string | null
  policy_gate_review_required?: boolean | number | null
  policy_gate_failures_json?: PolicyGateReason[]
  policy_gate_warnings_json?: PolicyGateReason[]
  policy_gate_disclosures_json?: string[]
}

const REGIME_SIGNAL_MAP: Record<string, { signal: "success" | "warning" | "error"; label: string }> = {
  bullish: { signal: "success", label: "Bullish" },
  neutral: { signal: "warning", label: "Neutral" },
  transitional: { signal: "warning", label: "Transitional" },
  bearish: { signal: "error", label: "Bearish" },
  "risk-off": { signal: "error", label: "Risk-Off" },
  "risk-on": { signal: "success", label: "Risk-On" },
}

const URGENCY_COLORS: Record<string, string> = {
  urgent: "text-red-600 bg-red-50 dark:text-red-400 dark:bg-red-950",
  high: "text-orange-600 bg-orange-50 dark:text-orange-400 dark:bg-orange-950",
  normal: "text-blue-600 bg-blue-50 dark:text-blue-400 dark:bg-blue-950",
  low: "text-gray-600 bg-gray-50 dark:text-gray-400 dark:bg-gray-800",
}

const ACTIONABLE_RECOMMENDATION_ACTIONS = new Set(["buy", "sell", "reduce", "exit", "rebalance", "hedge"])
const FINANCIAL_ACTION_ITEM_TYPES = new Set(["enter", "exit", "resize", "hedge"])

function formatPnl(value: number | null | undefined): string {
  if (value == null) return "--"
  const sign = value >= 0 ? "+" : ""
  return `${sign}${value.toFixed(2)}%`
}

function formatTime(iso: string): string {
  try {
    const d = new Date(iso)
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })
  } catch {
    return iso
  }
}

function policyGateFromRecommendation(rec: Recommendation): PolicyGateResult | null {
  if (!rec.policy_gate_decision) return null
  return {
    decision: rec.policy_gate_decision,
    review_required: Boolean(rec.policy_gate_review_required),
    failure_reasons: rec.policy_gate_failures_json ?? [],
    warnings: rec.policy_gate_warnings_json ?? [],
    disclosures: rec.policy_gate_disclosures_json ?? [],
  }
}

function recommendationNeedsPolicyGate(rec: Recommendation): boolean {
  return Boolean(rec.policy_gate_decision) || ACTIONABLE_RECOMMENDATION_ACTIONS.has(rec.action)
}

function policyGateFromApproval(approval: Approval): PolicyGateResult | null {
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

function approvalNeedsPolicyGate(approval: Approval, gate: PolicyGateResult | null): boolean {
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

export function Workspace() {
  const qc = useQueryClient()
  const { data, isPending, error } = useApiQuery<WorkspaceData>(
    ["workspace"],
    fetchWorkspace,
    60_000,
  )

  const [processingIds, setProcessingIds] = useState<Set<number>>(new Set())
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())
  const [refreshError, setRefreshError] = useState<string | null>(null)
  const [provenanceSelector, setProvenanceSelector] = useState<ProvenanceSelector | null>(null)

  function toggleExpanded(key: string) {
    setExpandedIds(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  async function handleApproval(id: number, action: "approve" | "reject") {
    setProcessingIds(prev => new Set(prev).add(id))
    try {
      if (action === "approve") {
        await approveItem(id)
      } else {
        await rejectItem(id)
      }
      qc.invalidateQueries({ queryKey: ["workspace"] })
      qc.invalidateQueries({ queryKey: ["portfolio", "all_timeframes"] })
    } finally {
      setProcessingIds(prev => {
        const next = new Set(prev)
        next.delete(id)
        return next
      })
    }
  }

  async function handleActionItem(id: number, action: "complete" | "dismiss") {
    setProcessingIds(prev => new Set(prev).add(id))
    try {
      if (action === "complete") await completeAction(id)
      else await dismissAction(id)
      qc.invalidateQueries({ queryKey: ["workspace"] })
    } finally {
      setProcessingIds(prev => { const n = new Set(prev); n.delete(id); return n })
    }
  }

  if (isPending) return <LoadingSpinner message="Loading workspace..." />
  if (error) return <ErrorMessage message={String(error)} />
  if (!data) return null

  const regime = data.regime
  const regimeInfo = regime?.signal ? REGIME_SIGNAL_MAP[regime.signal.toLowerCase()] : null
  const regimeSubtitle = [
    regime?.composite_score != null ? `Score: ${regime.composite_score}` : null,
    regime?.snapshot?.as_of ? `As of ${regime.snapshot.as_of}` : null,
    regime?.snapshot?.stale ? "Stale" : null,
  ].filter(Boolean).join(" · ")

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold text-app">Workspace</h1>
        <RefreshButton
          queryKeys={[["workspace"]]}
          beforeRefetch={refreshMarketSnapshots}
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
          subtitle={data.portfolio?.total_pnl_pct != null ? `P&L: ${formatPnl(data.portfolio.total_pnl_pct)}` : undefined}
        />
        <MetricCard
          title="Pending Approvals"
          value={data.pending_approvals.count}
          signal={data.pending_approvals.count > 0 ? "warning" : null}
          signalLabel={data.pending_approvals.count > 0 ? "Needs Review" : undefined}
        />
        <MetricCard
          title="Recommendations"
          value={data.recommendations.pending_actionable.count}
          subtitle={`${data.recommendations.pending_approval_count} approval${data.recommendations.pending_approval_count !== 1 ? "s" : ""}`}
          signal={data.recommendations.blocked_warnings.length > 0 ? "warning" : null}
          signalLabel={data.recommendations.blocked_warnings.length > 0 ? "Blocked" : undefined}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recommendation Summary */}
        {(data.recommendations.latest_daily || data.recommendations.latest_weekly || data.recommendations.blocked_warnings.length > 0) && (
          <section className="theme-surface rounded-xl p-4 lg:col-span-2">
            <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
              <AlertTriangle size={14} className={data.recommendations.blocked_warnings.length ? "text-amber-500" : "text-blue-500"} />
              Recommendation Ledger
              <span className="ml-auto text-xs text-subtle">{data.recommendations.pending_actionable.count} pending action{data.recommendations.pending_actionable.count !== 1 ? "s" : ""}</span>
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[data.recommendations.latest_daily, data.recommendations.latest_weekly].filter(Boolean).map(rec => (
                <div key={`${rec!.report_type}-${rec!.id}`} className="rounded-lg border border-app px-3 py-2 text-sm">
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-medium text-app capitalize">{rec!.report_type}</span>
                    <span className="text-xs text-subtle">{rec!.as_of}</span>
                  </div>
                  <div className="mt-1 flex flex-wrap items-center gap-2">
                    <span className="text-xs px-1.5 py-0.5 rounded bg-[hsl(var(--muted-2))] text-muted">{rec!.stance}</span>
                    <span className="text-xs px-1.5 py-0.5 rounded bg-[hsl(var(--muted-2))] text-muted">{rec!.action.replace(/_/g, " ")}</span>
                    <span className={cn(
                      "text-xs px-1.5 py-0.5 rounded font-medium",
                      rec!.recommendation_status === "blocked" || rec!.critical_data_quality === "failed"
                        ? "text-amber-700 bg-amber-50 dark:text-amber-400 dark:bg-amber-950"
                        : "text-green-700 bg-green-50 dark:text-green-400 dark:bg-green-950",
                    )}>
                      {rec!.critical_data_quality}
                    </span>
                  </div>
                  <p className="mt-2 text-xs text-muted line-clamp-2">{rec!.rationale}</p>
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
                      <span className="font-semibold text-app">{rec.action.replace(/_/g, " ")}</span>
                      <span className="text-xs text-subtle">{rec.instrument}</span>
                      {rec.ticker && (
                        <Link to={`/dossier/${rec.ticker}`} state={{ from: "workspace" }} className="text-xs font-semibold text-blue-600 hover:underline dark:text-blue-400">
                          {rec.ticker}
                        </Link>
                      )}
                      <span className="ml-auto text-xs text-subtle">{rec.approval_status}</span>
                    </div>
                    <p className="mt-1 text-xs text-muted line-clamp-2">{rec.rationale}</p>
                    {recommendationNeedsPolicyGate(rec) && <PolicyGatePanel gate={policyGateFromRecommendation(rec)} />}
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        {/* Thesis Pressure */}
        {data.thesis_pressure.length > 0 && (
          <section className="theme-surface rounded-xl p-4">
            <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
              <AlertTriangle size={14} className="text-amber-500" />
              Positions Under Pressure
            </h2>
            <div className="space-y-2">
              {data.thesis_pressure.map(tp => (
                <Link
                  key={tp.ticker}
                  to={`/dossier/${tp.ticker}`}
                  state={{ from: "workspace" }}
                  className="flex items-center justify-between rounded-lg px-3 py-2 text-sm hover:bg-[hsl(var(--muted-2))] transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <span className="font-semibold text-app">{tp.ticker}</span>
                    <span className={cn(
                      "text-xs px-1.5 py-0.5 rounded font-medium",
                      tp.action === "exit" || tp.action === "reduce"
                        ? "text-red-600 bg-red-50 dark:text-red-400 dark:bg-red-950"
                        : "text-amber-600 bg-amber-50 dark:text-amber-400 dark:bg-amber-950",
                    )}>
                      {tp.action}
                    </span>
                    {tp.risk_flag && (
                      <span className="text-xs text-red-500">{tp.risk_flag}</span>
                    )}
                  </div>
                  <span className="text-xs text-subtle">{tp.confidence}</span>
                </Link>
              ))}
            </div>
          </section>
        )}

        {/* Pending Approvals */}
        {data.pending_approvals.count > 0 && (
          <section className="theme-surface rounded-xl p-4">
            <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
              <CheckCircle size={14} className="text-blue-500" />
              Pending Approvals
              <span className="ml-auto text-xs text-subtle">{data.pending_approvals.count} total</span>
            </h2>
            <div className="space-y-2 max-h-[400px] overflow-y-auto">
              {data.pending_approvals.items.map(a => {
                const key = `approval-${a.id}`
                const expanded = expandedIds.has(key)
                const gate = policyGateFromApproval(a)
                return (
                  <div
                    key={a.id}
                    className="rounded-lg px-3 py-2 text-sm border border-app"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          {a.ticker && (
                            <Link to={`/dossier/${a.ticker}`} state={{ from: "workspace" }} className="font-semibold text-app hover:underline">
                              {a.ticker}
                            </Link>
                          )}
                          <span className="text-xs text-subtle">{a.entity_type.replace(/_/g, " ")}</span>
                          {a.application_status && (
                            <span className="rounded border border-app px-1.5 py-0.5 text-[11px] text-subtle">
                              {a.application_status.replace(/_/g, " ")}
                            </span>
                          )}
                        </div>
                        {a.action_id && <p className="text-[11px] text-subtle mt-0.5">{a.action_id}</p>}
                        {a.reason && (
                          <p onClick={() => toggleExpanded(key)} className={cn("text-xs text-muted mt-0.5 cursor-pointer", !expanded && "line-clamp-1")}>
                            {a.reason}
                          </p>
                        )}
                        {(a.source_type || a.source_id) && (
                          <p className="text-[11px] text-subtle mt-1">
                            {[a.source_type, a.source_id].filter(Boolean).join(" · ")}
                            {a.application_attempts ? ` · attempts ${a.application_attempts}` : ""}
                          </p>
                        )}
                        {approvalNeedsPolicyGate(a, gate) && (
                          <PolicyGatePanel gate={gate} />
                        )}
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        <button
                          type="button"
                          onClick={() => setProvenanceSelector({ approval_id: a.id })}
                          className="theme-icon-button h-8 w-8"
                          aria-label={`View approval ${a.id} lineage`}
                          title="Lineage"
                        >
                          <GitBranch size={14} />
                        </button>
                        <button
                          onClick={() => handleApproval(a.id, "approve")}
                          disabled={processingIds.has(a.id)}
                          className="rounded px-2 py-1 text-xs font-medium text-green-700 bg-green-50 hover:bg-green-100 dark:text-green-400 dark:bg-green-950 dark:hover:bg-green-900 disabled:opacity-50"
                        >
                          Approve
                        </button>
                        <button
                          onClick={() => handleApproval(a.id, "reject")}
                          disabled={processingIds.has(a.id)}
                          className="rounded px-2 py-1 text-xs font-medium text-red-700 bg-red-50 hover:bg-red-100 dark:text-red-400 dark:bg-red-950 dark:hover:bg-red-900 disabled:opacity-50"
                        >
                          Reject
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
          <section className="theme-surface rounded-xl p-4">
            <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
              <Play size={14} className="text-purple-500" />
              Open Actions
              <span className="ml-auto text-xs text-subtle">{data.open_actions.count} total</span>
            </h2>
            <div className="space-y-2 max-h-[400px] overflow-y-auto">
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
                        <Link to={`/dossier/${a.ticker}`} state={{ from: "workspace" }} className="font-semibold text-app hover:underline shrink-0">
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
                        Complete
                      </button>
                      <button
                        onClick={() => handleActionItem(a.id, "dismiss")}
                        disabled={processingIds.has(a.id)}
                        className="rounded px-2 py-1 text-xs font-medium text-gray-600 bg-gray-50 hover:bg-gray-100 dark:text-gray-400 dark:bg-gray-800 disabled:opacity-50"
                      >
                        Dismiss
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
          <section className="theme-surface rounded-xl p-4">
            <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
              <Eye size={14} className="text-cyan-500" />
              Active Triggers
              <span className="ml-auto text-xs text-subtle">{data.active_triggers.count} total</span>
            </h2>
            <div className="space-y-2">
              {data.active_triggers.items.map(t => (
                <div key={t.id} className="rounded-lg px-3 py-2 text-sm">
                  <div className="flex items-center gap-3">
                    {t.ticker && (
                      <Link to={`/dossier/${t.ticker}`} state={{ from: "workspace" }} className="font-semibold text-app hover:underline shrink-0">
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
              ))}
            </div>
          </section>
        )}

        {/* Recent Workflow Runs */}
        {data.recent_workflow_runs.length > 0 && (
          <section className="theme-surface rounded-xl p-4 lg:col-span-2">
            <h2 className="text-sm font-semibold text-app mb-3 flex items-center gap-2">
              <Clock size={14} className="text-gray-500" />
              Recent Workflow Runs
            </h2>
            <div className="space-y-2">
              {data.recent_workflow_runs.map(run => (
                <div key={run.run_id} className="flex items-center justify-between rounded-lg px-3 py-2 text-sm">
                  <div className="flex items-center gap-3">
                    <span className={cn(
                      "h-2 w-2 rounded-full shrink-0",
                      run.status === "completed" ? "bg-green-500"
                        : run.status === "running" ? "bg-blue-500 animate-pulse"
                        : "bg-red-500",
                    )} />
                    <span className="font-medium text-app">{run.workflow_name.replace(/_/g, " ")}</span>
                    {run.ticker && (
                      <Link to={`/dossier/${run.ticker}`} state={{ from: "workspace" }} className="text-blue-600 hover:underline dark:text-blue-400">
                        {run.ticker}
                      </Link>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-subtle">{formatTime(run.started_at)}</span>
                    <button
                      type="button"
                      onClick={() => setProvenanceSelector({ workflow_run_id: run.run_id })}
                      className="theme-icon-button h-8 w-8"
                      aria-label={`View workflow ${run.run_id} lineage`}
                      title="Lineage"
                    >
                      <GitBranch size={14} />
                    </button>
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
        !data.pending_approvals.count &&
        !data.open_actions.count &&
        !data.active_triggers.count &&
        !data.recent_workflow_runs.length && (
        <div className="theme-surface rounded-xl p-8 text-center text-muted text-sm mt-4">
          No pending items. Run a workflow or chat with the agent to get started.
        </div>
      )}
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
