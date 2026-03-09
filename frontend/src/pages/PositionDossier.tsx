import { useState } from "react"
import { useParams, Link } from "react-router-dom"
import { useQueryClient } from "@tanstack/react-query"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchDossier, approveItem, rejectItem } from "@/lib/api"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { MarkdownRenderer } from "@/components/shared/MarkdownRenderer"
import { cn } from "@/lib/utils"

interface DossierData {
  ticker: string
  position: Record<string, unknown> | null
  thesis: {
    meta: ThesisMeta | null
    content: string | null
    status_history: StatusEntry[]
  }
  evaluations: Evaluation[]
  catalysts: Catalyst[]
  kill_conditions: KillCondition[]
  ontology_risk: Record<string, unknown> | null
  workflow_runs: WorkflowRun[]
  action_items: ActionItem[]
  watch_triggers: Trigger[]
  research_notes: ResearchNote[]
  pending_approvals: Approval[]
}

interface ThesisMeta {
  ticker: string
  status: string
  direction: string
  timeframe: string
  last_evaluated: string | null
}

interface StatusEntry { status: string; changed_at: string; reason: string | null }
interface Evaluation {
  id: number
  ticker: string
  thesis_status: string
  action: string
  confidence: string
  technical_read: string | null
  fundamental_read: string | null
  key_developments: string[] | null
  risk_flag: string | null
  evaluated_at: string
}
interface Catalyst { id: number; description: string; category: string; status: string; target_date: string | null; evidence: string | null }
interface KillCondition { id: number; condition: string; metric: string | null; threshold: string | null; status: string; triggered_at: string | null }
interface WorkflowRun { run_id: string; workflow_name: string; status: string; started_at: string; completed_at: string | null }
interface ActionItem { id: number; description: string; action_type: string; urgency: string; status: string; created_at: string }
interface Trigger { id: number; condition: string; trigger_type: string; status: string; created_at: string }
interface ResearchNote { id: number; title: string; content: string; note_type: string | null; created_at: string }
interface Approval { id: number; entity_type: string; reason: string | null; created_at: string; proposed_change: Record<string, unknown> }

const TABS = ["Thesis", "Catalysts", "Kill Conditions", "Evaluations", "Risk", "Research", "Workflows"] as const
type Tab = typeof TABS[number]

const STATUS_COLORS: Record<string, string> = {
  active: "text-green-700 bg-green-50 dark:text-green-400 dark:bg-green-950",
  under_review: "text-amber-700 bg-amber-50 dark:text-amber-400 dark:bg-amber-950",
  suspended: "text-red-700 bg-red-50 dark:text-red-400 dark:bg-red-950",
  closed: "text-gray-600 bg-gray-100 dark:text-gray-400 dark:bg-gray-800",
  pending: "text-blue-700 bg-blue-50 dark:text-blue-400 dark:bg-blue-950",
  played_out: "text-gray-600 bg-gray-100 dark:text-gray-400 dark:bg-gray-800",
  failed: "text-red-700 bg-red-50 dark:text-red-400 dark:bg-red-950",
  triggered: "text-red-700 bg-red-50 dark:text-red-400 dark:bg-red-950",
  retired: "text-gray-500 bg-gray-50 dark:text-gray-400 dark:bg-gray-800",
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })
  } catch {
    return iso
  }
}

export function PositionDossier() {
  const { ticker } = useParams<{ ticker: string }>()
  const [tab, setTab] = useState<Tab>("Thesis")
  const qc = useQueryClient()
  const [processingIds, setProcessingIds] = useState<Set<number>>(new Set())

  const { data, isLoading, error } = useApiQuery<DossierData>(
    ["dossier", ticker],
    () => fetchDossier(ticker!),
    60_000,
  )

  async function handleApproval(id: number, action: "approve" | "reject") {
    setProcessingIds(prev => new Set(prev).add(id))
    try {
      if (action === "approve") await approveItem(id)
      else await rejectItem(id)
      qc.invalidateQueries({ queryKey: ["dossier", ticker] })
    } finally {
      setProcessingIds(prev => { const n = new Set(prev); n.delete(id); return n })
    }
  }

  if (!ticker) return <ErrorMessage message="No ticker specified" />
  if (isLoading) return <LoadingSpinner message={`Loading dossier for ${ticker}...`} />
  if (error) return <ErrorMessage message={String(error)} />
  if (!data) return null

  const pos = data.position
  const meta = data.thesis?.meta

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <Link to="/" className="text-sm text-muted hover:text-app">&larr; Workspace</Link>
          <h1 className="text-2xl font-bold text-app">{data.ticker}</h1>
          {meta?.status && (
            <span className={cn("text-xs px-2 py-0.5 rounded font-medium", STATUS_COLORS[meta.status] ?? STATUS_COLORS.active)}>
              {meta.status.replace(/_/g, " ")}
            </span>
          )}
          {meta?.direction && <span className="text-sm text-muted">{meta.direction}</span>}
        </div>
        <RefreshButton queryKeys={[["dossier", ticker]]} />
      </div>

      {/* Position summary bar */}
      {pos && (
        <div className="theme-surface rounded-xl p-3 mb-4 flex flex-wrap gap-6 text-sm">
          {pos.shares != null && <div><span className="text-subtle">Shares</span> <span className="font-medium text-app ml-1">{String(pos.shares)}</span></div>}
          {pos.avg_cost != null && <div><span className="text-subtle">Avg Cost</span> <span className="font-medium text-app ml-1">${Number(pos.avg_cost).toFixed(2)}</span></div>}
          {pos.market_value != null && <div><span className="text-subtle">Mkt Value</span> <span className="font-medium text-app ml-1">${Number(pos.market_value).toLocaleString()}</span></div>}
          {pos.pnl_pct != null && (
            <div>
              <span className="text-subtle">P&L</span>
              <span className={cn("font-medium ml-1", Number(pos.pnl_pct) >= 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400")}>
                {Number(pos.pnl_pct) >= 0 ? "+" : ""}{Number(pos.pnl_pct).toFixed(2)}%
              </span>
            </div>
          )}
          {pos.weight != null && <div><span className="text-subtle">Weight</span> <span className="font-medium text-app ml-1">{Number(pos.weight).toFixed(1)}%</span></div>}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 mb-4 overflow-x-auto border-b border-app">
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={cn(
              "px-3 py-2 text-sm font-medium whitespace-nowrap border-b-2 transition-colors",
              tab === t
                ? "border-blue-500 text-blue-600 dark:text-blue-400"
                : "border-transparent text-muted hover:text-app",
            )}
          >
            {t}
            {t === "Catalysts" && data.catalysts.length > 0 && <span className="ml-1 text-xs text-subtle">({data.catalysts.length})</span>}
            {t === "Kill Conditions" && data.kill_conditions.length > 0 && <span className="ml-1 text-xs text-subtle">({data.kill_conditions.length})</span>}
            {t === "Evaluations" && data.evaluations.length > 0 && <span className="ml-1 text-xs text-subtle">({data.evaluations.length})</span>}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="theme-surface rounded-xl p-4">
        {tab === "Thesis" && <ThesisTab thesis={data.thesis} />}
        {tab === "Catalysts" && <CatalystsTab catalysts={data.catalysts} />}
        {tab === "Kill Conditions" && <KillConditionsTab conditions={data.kill_conditions} />}
        {tab === "Evaluations" && <EvaluationsTab evaluations={data.evaluations} />}
        {tab === "Risk" && <RiskTab ontology={data.ontology_risk} />}
        {tab === "Research" && <ResearchTab notes={data.research_notes} />}
        {tab === "Workflows" && <WorkflowsTab runs={data.workflow_runs} />}
      </div>

      {/* Pending Approvals for this ticker */}
      {data.pending_approvals.length > 0 && (
        <section className="mt-6 theme-surface rounded-xl p-4">
          <h2 className="text-sm font-semibold text-app mb-3">
            Pending Approvals ({data.pending_approvals.length})
          </h2>
          <div className="space-y-2">
            {data.pending_approvals.map(a => (
              <div key={a.id} className="flex items-center justify-between rounded-lg px-3 py-2 text-sm border border-app">
                <div className="min-w-0 flex-1">
                  <span className="text-xs text-subtle">{a.entity_type.replace(/_/g, " ")}</span>
                  {a.reason && <p className="text-xs text-muted truncate mt-0.5">{a.reason}</p>}
                </div>
                <div className="flex items-center gap-1 ml-3 shrink-0">
                  <button
                    onClick={() => handleApproval(a.id, "approve")}
                    disabled={processingIds.has(a.id)}
                    className="rounded px-2 py-1 text-xs font-medium text-green-700 bg-green-50 hover:bg-green-100 dark:text-green-400 dark:bg-green-950 disabled:opacity-50"
                  >
                    Approve
                  </button>
                  <button
                    onClick={() => handleApproval(a.id, "reject")}
                    disabled={processingIds.has(a.id)}
                    className="rounded px-2 py-1 text-xs font-medium text-red-700 bg-red-50 hover:bg-red-100 dark:text-red-400 dark:bg-red-950 disabled:opacity-50"
                  >
                    Reject
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Action Items + Triggers sidebar */}
      {(data.action_items.length > 0 || data.watch_triggers.length > 0) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
          {data.action_items.length > 0 && (
            <section className="theme-surface rounded-xl p-4">
              <h2 className="text-sm font-semibold text-app mb-3">Action Items</h2>
              <div className="space-y-2">
                {data.action_items.map(a => (
                  <div key={a.id} className="flex items-center gap-2 text-sm px-2 py-1.5">
                    <span className={cn("text-xs px-1.5 py-0.5 rounded font-medium shrink-0",
                      a.urgency === "urgent" ? "text-red-600 bg-red-50 dark:text-red-400 dark:bg-red-950" :
                      a.urgency === "high" ? "text-orange-600 bg-orange-50 dark:text-orange-400 dark:bg-orange-950" :
                      "text-blue-600 bg-blue-50 dark:text-blue-400 dark:bg-blue-950"
                    )}>{a.urgency}</span>
                    <span className="text-muted truncate">{a.description}</span>
                  </div>
                ))}
              </div>
            </section>
          )}
          {data.watch_triggers.length > 0 && (
            <section className="theme-surface rounded-xl p-4">
              <h2 className="text-sm font-semibold text-app mb-3">Watch Triggers</h2>
              <div className="space-y-2">
                {data.watch_triggers.map(t => (
                  <div key={t.id} className="flex items-center gap-2 text-sm px-2 py-1.5">
                    <span className="text-xs text-subtle shrink-0">{t.trigger_type.replace(/_/g, " ")}</span>
                    <span className="text-muted truncate">{t.condition}</span>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      )}
    </div>
  )
}

/* ---------- Sub-tab components ---------- */

function ThesisTab({ thesis }: { thesis: DossierData["thesis"] }) {
  if (!thesis.content && !thesis.meta) {
    return <p className="text-sm text-muted">No thesis on file for this position.</p>
  }
  return (
    <div>
      {thesis.meta && (
        <div className="flex flex-wrap gap-4 text-sm mb-4 pb-4 border-b border-app">
          <div><span className="text-subtle">Direction:</span> <span className="font-medium text-app">{thesis.meta.direction}</span></div>
          <div><span className="text-subtle">Timeframe:</span> <span className="font-medium text-app">{thesis.meta.timeframe}</span></div>
          {thesis.meta.last_evaluated && <div><span className="text-subtle">Last Evaluated:</span> <span className="font-medium text-app">{formatTime(thesis.meta.last_evaluated)}</span></div>}
        </div>
      )}
      {thesis.content && (
        <div className="prose prose-sm dark:prose-invert max-w-none">
          <MarkdownRenderer content={thesis.content} />
        </div>
      )}
      {thesis.status_history.length > 0 && (
        <div className="mt-4 pt-4 border-t border-app">
          <h3 className="text-xs font-semibold text-subtle uppercase mb-2">Status History</h3>
          <div className="space-y-1">
            {thesis.status_history.map((s, i) => (
              <div key={i} className="flex items-center gap-3 text-xs text-muted">
                <span className="text-subtle">{formatTime(s.changed_at)}</span>
                <span className={cn("px-1.5 py-0.5 rounded font-medium", STATUS_COLORS[s.status] ?? "")}>{s.status}</span>
                {s.reason && <span className="truncate">{s.reason}</span>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function CatalystsTab({ catalysts }: { catalysts: Catalyst[] }) {
  if (!catalysts.length) return <p className="text-sm text-muted">No catalysts tracked.</p>
  return (
    <div className="space-y-3">
      {catalysts.map(c => (
        <div key={c.id} className="rounded-lg border border-app px-4 py-3">
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm font-medium text-app">{c.description}</span>
            <span className={cn("text-xs px-1.5 py-0.5 rounded font-medium", STATUS_COLORS[c.status] ?? "")}>{c.status}</span>
          </div>
          <div className="flex gap-3 text-xs text-subtle">
            <span>{c.category}</span>
            {c.target_date && <span>Target: {c.target_date}</span>}
          </div>
          {c.evidence && <p className="text-xs text-muted mt-1">{c.evidence}</p>}
        </div>
      ))}
    </div>
  )
}

function KillConditionsTab({ conditions }: { conditions: KillCondition[] }) {
  if (!conditions.length) return <p className="text-sm text-muted">No kill conditions defined.</p>
  return (
    <div className="space-y-3">
      {conditions.map(k => (
        <div key={k.id} className={cn("rounded-lg border px-4 py-3", k.status === "triggered" ? "border-red-300 bg-red-50/50 dark:border-red-800 dark:bg-red-950/30" : "border-app")}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm font-medium text-app">{k.condition}</span>
            <span className={cn("text-xs px-1.5 py-0.5 rounded font-medium", STATUS_COLORS[k.status] ?? "")}>{k.status}</span>
          </div>
          <div className="flex gap-3 text-xs text-subtle">
            {k.metric && <span>Metric: {k.metric}</span>}
            {k.threshold && <span>Threshold: {k.threshold}</span>}
            {k.triggered_at && <span>Triggered: {formatTime(k.triggered_at)}</span>}
          </div>
        </div>
      ))}
    </div>
  )
}

function EvaluationsTab({ evaluations }: { evaluations: Evaluation[] }) {
  if (!evaluations.length) return <p className="text-sm text-muted">No evaluations recorded.</p>
  return (
    <div className="space-y-3">
      {evaluations.map(ev => (
        <div key={ev.id} className="rounded-lg border border-app px-4 py-3">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className={cn("text-xs px-1.5 py-0.5 rounded font-medium",
                ev.action === "hold" ? "text-green-600 bg-green-50 dark:text-green-400 dark:bg-green-950" :
                ev.action === "exit" || ev.action === "reduce" ? "text-red-600 bg-red-50 dark:text-red-400 dark:bg-red-950" :
                "text-blue-600 bg-blue-50 dark:text-blue-400 dark:bg-blue-950"
              )}>{ev.action}</span>
              <span className="text-xs text-subtle">{ev.confidence} confidence</span>
              {ev.risk_flag && <span className="text-xs text-red-500 font-medium">{ev.risk_flag}</span>}
            </div>
            <span className="text-xs text-subtle">{formatTime(ev.evaluated_at)}</span>
          </div>
          {ev.technical_read && <p className="text-xs text-muted"><span className="text-subtle">Technical:</span> {ev.technical_read}</p>}
          {ev.fundamental_read && <p className="text-xs text-muted mt-0.5"><span className="text-subtle">Fundamental:</span> {ev.fundamental_read}</p>}
          {ev.key_developments && ev.key_developments.length > 0 && (
            <ul className="mt-1 space-y-0.5">
              {ev.key_developments.map((d, i) => (
                <li key={i} className="text-xs text-muted pl-3 relative before:absolute before:left-0 before:top-[7px] before:h-1 before:w-1 before:rounded-full before:bg-gray-400">{d}</li>
              ))}
            </ul>
          )}
        </div>
      ))}
    </div>
  )
}

function RiskTab({ ontology }: { ontology: Record<string, unknown> | null }) {
  if (!ontology) return <p className="text-sm text-muted">No ontology risk data available.</p>
  return (
    <pre className="text-xs text-muted whitespace-pre-wrap overflow-auto max-h-[600px]">
      {JSON.stringify(ontology, null, 2)}
    </pre>
  )
}

function ResearchTab({ notes }: { notes: ResearchNote[] }) {
  if (!notes.length) return <p className="text-sm text-muted">No research notes.</p>
  return (
    <div className="space-y-4">
      {notes.map(n => (
        <div key={n.id} className="rounded-lg border border-app px-4 py-3">
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm font-semibold text-app">{n.title}</span>
            <span className="text-xs text-subtle">{formatTime(n.created_at)}</span>
          </div>
          {n.note_type && <span className="text-xs text-subtle">{n.note_type}</span>}
          <div className="prose prose-sm dark:prose-invert max-w-none mt-2">
            <MarkdownRenderer content={n.content} />
          </div>
        </div>
      ))}
    </div>
  )
}

function WorkflowsTab({ runs }: { runs: WorkflowRun[] }) {
  if (!runs.length) return <p className="text-sm text-muted">No workflow runs recorded.</p>
  return (
    <div className="space-y-2">
      {runs.map(run => (
        <div key={run.run_id} className="flex items-center justify-between rounded-lg border border-app px-4 py-3 text-sm">
          <div className="flex items-center gap-3">
            <span className={cn(
              "h-2 w-2 rounded-full shrink-0",
              run.status === "completed" ? "bg-green-500" : run.status === "running" ? "bg-blue-500 animate-pulse" : "bg-red-500",
            )} />
            <span className="font-medium text-app">{run.workflow_name.replace(/_/g, " ")}</span>
          </div>
          <div className="flex items-center gap-3 text-xs text-subtle">
            <span>{run.status}</span>
            <span>{formatTime(run.started_at)}</span>
          </div>
        </div>
      ))}
    </div>
  )
}
