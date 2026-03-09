import { Link } from "react-router-dom"
import { useQueryClient } from "@tanstack/react-query"
import { CheckCircle, AlertTriangle, Eye, Play, Clock } from "lucide-react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchWorkspace, approveItem, rejectItem, completeAction, dismissAction } from "@/lib/api"
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { cn } from "@/lib/utils"
import { useState } from "react"

interface WorkspaceData {
  regime: {
    regime: string
    composite_score: number
    signal: string
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
  open_actions: { count: number; items: ActionItem[] }
  active_triggers: { count: number; items: Trigger[] }
  recent_workflow_runs: WorkflowRun[]
}

interface Approval {
  id: number
  entity_type: string
  ticker: string | null
  reason: string | null
  created_at: string
  proposed_change: Record<string, unknown>
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
  created_at: string
}

interface WorkflowRun {
  run_id: string
  workflow_name: string
  ticker: string | null
  status: string
  started_at: string
  completed_at: string | null
}

const REGIME_SIGNAL_MAP: Record<string, { signal: "success" | "warning" | "error"; label: string }> = {
  bullish: { signal: "success", label: "Bullish" },
  neutral: { signal: "warning", label: "Neutral" },
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

export function Workspace() {
  const qc = useQueryClient()
  const { data, isPending, error } = useApiQuery<WorkspaceData>(
    ["workspace"],
    fetchWorkspace,
    60_000,
  )

  const [processingIds, setProcessingIds] = useState<Set<number>>(new Set())
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())

  function toggleExpanded(key: string) {
    setExpandedIds(prev => { const n = new Set(prev); n.has(key) ? n.delete(key) : n.add(key); return n })
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

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold text-app">Workspace</h1>
        <RefreshButton queryKeys={[["workspace"]]} />
      </div>

      {/* Top metrics row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard
          title="Market Regime"
          value={regime?.regime ?? "--"}
          subtitle={regime?.composite_score != null ? `Score: ${regime.composite_score}` : undefined}
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
          title="Open Actions"
          value={data.open_actions.count}
          subtitle={`${data.active_triggers.count} active trigger${data.active_triggers.count !== 1 ? "s" : ""}`}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
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
                return (
                  <div
                    key={a.id}
                    className="rounded-lg px-3 py-2 text-sm border border-app"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          {a.ticker && (
                            <Link to={`/dossier/${a.ticker}`} className="font-semibold text-app hover:underline">
                              {a.ticker}
                            </Link>
                          )}
                          <span className="text-xs text-subtle">{a.entity_type.replace(/_/g, " ")}</span>
                        </div>
                        {a.reason && (
                          <p onClick={() => toggleExpanded(key)} className={cn("text-xs text-muted mt-0.5 cursor-pointer", !expanded && "line-clamp-1")}>
                            {a.reason}
                          </p>
                        )}
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
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
                        <Link to={`/dossier/${a.ticker}`} className="font-semibold text-app hover:underline shrink-0">
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
                <div key={t.id} className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm">
                  {t.ticker && (
                    <Link to={`/dossier/${t.ticker}`} className="font-semibold text-app hover:underline shrink-0">
                      {t.ticker}
                    </Link>
                  )}
                  <span className="text-muted truncate">{t.condition}</span>
                  <span className="text-xs text-subtle shrink-0">{t.trigger_type.replace(/_/g, " ")}</span>
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
                      <Link to={`/dossier/${run.ticker}`} className="text-blue-600 hover:underline dark:text-blue-400">
                        {run.ticker}
                      </Link>
                    )}
                  </div>
                  <span className="text-xs text-subtle">{formatTime(run.started_at)}</span>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>

      {/* Empty state */}
      {!data.thesis_pressure.length &&
        !data.pending_approvals.count &&
        !data.open_actions.count &&
        !data.active_triggers.count &&
        !data.recent_workflow_runs.length && (
        <div className="theme-surface rounded-xl p-8 text-center text-muted text-sm mt-4">
          No pending items. Run a workflow or chat with the agent to get started.
        </div>
      )}
    </div>
  )
}
