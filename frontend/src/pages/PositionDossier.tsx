import { useEffect, useState } from "react"
import { useParams, useNavigate, useLocation } from "react-router-dom"
import { useQueryClient, useMutation, useQuery } from "@tanstack/react-query"
import { useApiQuery } from "@/hooks/useApiQuery"
import {
  fetchDossier,
  approveItem,
  rejectItem,
  updateThesisStatus,
  fetchThesisStatus,
  saveThesisContent,
  saveOverviewContent,
  runFinancials,
  completeAction,
  dismissAction,
  updateCatalystStatus,
  updateKillConditionStatus,
  createThesisClaim,
  updateThesisClaim,
  fetchOntologyRuns,
  queryOntology,
  runOntologyQueryAsync,
  type OntologyEvidence,
  type OntologyResponse,
  type SourceRequirement,
  type ThesisClaim,
  type ThesisClaimStatus,
  type ThesisStatus,
  type ThesisStatusValue,
} from "@/lib/api"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { MarkdownRenderer } from "@/components/shared/MarkdownRenderer"
import { MetricCard } from "@/components/shared/MetricCard"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { Dialog } from "@/components/shared/Dialog"
import { ActionButton, SegmentedControl, SelectInput, TextInput } from "@/components/shared/FormControls"
import { ThesisUpload } from "@/components/ThesisUpload"
import { OverviewUpload } from "@/components/OverviewUpload"
import { cn } from "@/lib/utils"

interface DossierData {
  ticker: string
  position: Record<string, unknown> | null
  overview_content: string | null
  overview_parsed: ParsedOverview | null
  thesis: {
    meta: ThesisMeta | null
    content: string | null
    status_history: StatusEntry[]
  }
  evaluations: Evaluation[]
  thesis_claims: ThesisClaim[]
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
interface Trigger { id: number; condition: string; trigger_type: string; status: string; created_at: string; last_checked_at: string | null; last_evidence: string | null }
interface ResearchNote { id: number; title: string; content: string; note_type: string | null; created_at: string }
interface Approval { id: number; entity_type: string; reason: string | null; created_at: string; proposed_change: Record<string, unknown> }

interface ParsedFinancialMetric { value: string | null; context: string }
interface DebtTranche { tranche: string; rate: string; maturity: string }
interface ParsedDebt { summary: string; tranches: DebtTranche[] }
interface ParsedFinancials {
  revenue_growth: ParsedFinancialMetric | null
  eps_growth: ParsedFinancialMetric | null
  debt: ParsedDebt | null
  reinvestment: string | null
}
interface SensitivityRow { factor: string; sensitivity: string; capacity: string }
interface PorterForce { force: string; rating: string; description: string }
interface OutlookPoint { label?: string; text: string }
interface OutlookSection { rating: string | null; points: (string | OutlookPoint)[] }
interface ParsedOverview {
  financials: ParsedFinancials | null
  sensitivity: SensitivityRow[] | null
  porters_five_forces: PorterForce[] | null
  supply_outlook: OutlookSection | null
  demand_outlook: OutlookSection | null
}

const BASE_TABS = ["Thesis", "Claims", "Catalysts", "Kill Conditions", "Evaluations", "Risk", "Research", "Workflows"] as const
type Tab = "Overview" | typeof BASE_TABS[number]

const STATUS_COLORS: Record<string, string> = {
  active: "text-green-700 bg-green-50 dark:text-green-400 dark:bg-green-950",
  under_review: "text-amber-700 bg-amber-50 dark:text-amber-400 dark:bg-amber-950",
  suspended: "text-red-700 bg-red-50 dark:text-red-400 dark:bg-red-950",
  closed: "text-gray-600 bg-gray-100 dark:text-gray-400 dark:bg-gray-800",
  pending: "text-blue-700 bg-blue-50 dark:text-blue-400 dark:bg-blue-950",
  played_out: "text-green-700 bg-green-50 dark:text-green-400 dark:bg-green-950",
  failed: "text-red-700 bg-red-50 dark:text-red-400 dark:bg-red-950",
  triggered: "text-red-700 bg-red-50 dark:text-red-400 dark:bg-red-950",
  supported: "text-green-700 bg-green-50 dark:text-green-400 dark:bg-green-950",
  challenged: "text-amber-700 bg-amber-50 dark:text-amber-400 dark:bg-amber-950",
  disconfirmed: "text-red-700 bg-red-50 dark:text-red-400 dark:bg-red-950",
  retired: "text-gray-500 bg-gray-50 dark:text-gray-400 dark:bg-gray-800",
  superseded: "text-gray-600 bg-gray-100 dark:text-gray-400 dark:bg-gray-800",
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
  const navigate = useNavigate()
  const location = useLocation()
  const from = (location.state as { from?: string })?.from
  const backTarget = from === "theses"
    ? { path: "/theses", label: "Theses" }
    : from === "portfolio"
    ? { path: "/", label: "Portfolio" }
    : { path: "/workspace", label: "Workspace" }
  const [tab, setTab] = useState<Tab>("Overview")
  const qc = useQueryClient()
  const [processingIds, setProcessingIds] = useState<Set<number>>(new Set())
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())
  const [statusDialogOpen, setStatusDialogOpen] = useState(false)
  const [newStatus, setNewStatus] = useState<ThesisStatusValue>("under_review")
  const [statusReason, setStatusReason] = useState("")

  const { data, isLoading, error } = useApiQuery<DossierData>(
    ["dossier", ticker],
    () => fetchDossier(ticker!),
    60_000,
  )

  const { data: thesisStatus } = useApiQuery<Record<string, string>>(
    ["thesis", "status"],
    fetchThesisStatus,
  )

  const statusMutation = useMutation({
    mutationFn: () => updateThesisStatus(ticker!, newStatus, statusReason),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dossier", ticker] })
      qc.invalidateQueries({ queryKey: ["thesis"] })
      setStatusDialogOpen(false)
      setStatusReason("")
    },
  })

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
      if (action === "approve") await approveItem(id)
      else await rejectItem(id)
      qc.invalidateQueries({ queryKey: ["dossier", ticker] })
    } finally {
      setProcessingIds(prev => { const n = new Set(prev); n.delete(id); return n })
    }
  }

  async function handleActionItem(id: number, action: "complete" | "dismiss") {
    setProcessingIds(prev => new Set(prev).add(id))
    try {
      if (action === "complete") await completeAction(id)
      else await dismissAction(id)
      qc.invalidateQueries({ queryKey: ["dossier", ticker] })
    } finally {
      setProcessingIds(prev => { const n = new Set(prev); n.delete(id); return n })
    }
  }

  if (!ticker) return <ErrorMessage message="No ticker specified" />
  if (isLoading) return <LoadingSpinner message={`Loading dossier for ${ticker}...`} />
  if (error) return <ErrorMessage message={String(error)} />
  if (!data) return null

  const isEquity = String(data.position?.asset ?? "") === "equity"
  const visibleTabs: Tab[] = isEquity ? ["Overview", ...BASE_TABS] : [...BASE_TABS]
  const activeTab: Tab = tab === "Overview" && !isEquity ? "Thesis" : tab
  const pos = data.position
  const meta = data.thesis?.meta

  return (
    <div>
      {/* Header */}
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0 flex-1 space-y-2">
          <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-2">
            <button type="button" onClick={() => navigate(backTarget.path)} className="shrink-0 text-sm text-muted hover:text-app">&larr; {backTarget.label}</button>
            <h1 className="text-2xl font-bold text-app">{data.ticker}</h1>
            {meta?.status && (
              <span className={cn("shrink-0 text-xs px-2 py-0.5 rounded font-medium", STATUS_COLORS[meta.status] ?? STATUS_COLORS.active)}>
                {meta.status.replace(/_/g, " ")}
              </span>
            )}
            {data.position?.direction != null && <span className="text-sm text-muted">{String(data.position.direction)}</span>}
          </div>
          <div className="flex max-w-full flex-wrap items-center gap-2">
            <ThesisUpload ticker={ticker!} status={(thesisStatus?.[ticker!] ?? "missing") as ThesisStatus} />
            {isEquity && <OverviewUpload ticker={ticker!} hasContent={!!data.overview_content} />}
          </div>
        </div>
        <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto sm:justify-end">
          {meta && (
            <button
              type="button"
              onClick={() => {
                setNewStatus(meta.status === "active" ? "under_review" : "active")
                setStatusDialogOpen(true)
              }}
              className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors"
            >
              Change Status
            </button>
          )}
          <RefreshButton queryKeys={[["dossier", ticker]]} />
        </div>
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
      <div className="mb-4 flex w-full max-w-full gap-1 overflow-x-auto overscroll-x-contain border-b border-app [-webkit-overflow-scrolling:touch]">
        {visibleTabs.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={cn(
              "shrink-0 px-3 py-2 text-sm font-medium whitespace-nowrap border-b-2 transition-colors",
              activeTab === t
                ? "border-blue-500 text-blue-600 dark:text-blue-400"
                : "border-transparent text-muted hover:text-app",
            )}
          >
            {t}
            {t === "Claims" && data.thesis_claims.length > 0 && <span className="ml-1 text-xs text-subtle">({data.thesis_claims.length})</span>}
            {t === "Catalysts" && data.catalysts.length > 0 && <span className="ml-1 text-xs text-subtle">({data.catalysts.length})</span>}
            {t === "Kill Conditions" && data.kill_conditions.length > 0 && <span className="ml-1 text-xs text-subtle">({data.kill_conditions.length})</span>}
            {t === "Evaluations" && data.evaluations.length > 0 && <span className="ml-1 text-xs text-subtle">({data.evaluations.length})</span>}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="theme-surface rounded-xl p-4">
        {activeTab === "Overview" && <OverviewTab content={data.overview_content} parsed={data.overview_parsed} ticker={data.ticker} />}
        {activeTab === "Thesis" && <ThesisTab thesis={data.thesis} ticker={data.ticker} position={data.position} />}
        {activeTab === "Claims" && <ClaimsTab claims={data.thesis_claims} catalysts={data.catalysts} conditions={data.kill_conditions} ticker={ticker!} />}
        {activeTab === "Catalysts" && <CatalystsTab catalysts={data.catalysts} ticker={ticker!} />}
        {activeTab === "Kill Conditions" && <KillConditionsTab conditions={data.kill_conditions} ticker={ticker!} />}
        {activeTab === "Evaluations" && <EvaluationsTab evaluations={data.evaluations} />}
        {activeTab === "Risk" && <RiskTab ticker={data.ticker} />}
        {activeTab === "Research" && <ResearchTab notes={data.research_notes} />}
        {activeTab === "Workflows" && <WorkflowsTab runs={data.workflow_runs} />}
      </div>

      {/* Pending Approvals for this ticker */}
      {data.pending_approvals.length > 0 && (
        <section className="mt-6 theme-surface rounded-xl p-4">
          <h2 className="text-sm font-semibold text-app mb-3">
            Pending Approvals ({data.pending_approvals.length})
          </h2>
          <div className="space-y-2 max-h-[400px] overflow-y-auto">
            {data.pending_approvals.map(a => {
              const key = `approval-${a.id}`
              const expanded = expandedIds.has(key)
              return (
                <div key={a.id} className="rounded-lg px-3 py-2 text-sm border border-app">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <span className="text-xs text-subtle">{a.entity_type.replace(/_/g, " ")}</span>
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
                </div>
              )
            })}
          </div>
        </section>
      )}

      {/* Action Items + Triggers sidebar */}
      {(data.action_items.length > 0 || data.watch_triggers.length > 0) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
          {data.action_items.length > 0 && (
            <section className="theme-surface rounded-xl p-4">
              <h2 className="text-sm font-semibold text-app mb-3">Action Items</h2>
              <div className="space-y-2 max-h-[400px] overflow-y-auto">
                {data.action_items.map(a => {
                  const key = `action-${a.id}`
                  const expanded = expandedIds.has(key)
                  return (
                    <div key={a.id} className="rounded-lg border border-app px-3 py-2">
                      <div className="flex items-start gap-2 text-sm">
                        <span className={cn("text-xs px-1.5 py-0.5 rounded font-medium shrink-0 mt-0.5",
                          a.urgency === "urgent" ? "text-red-600 bg-red-50 dark:text-red-400 dark:bg-red-950" :
                          a.urgency === "high" ? "text-orange-600 bg-orange-50 dark:text-orange-400 dark:bg-orange-950" :
                          "text-blue-600 bg-blue-50 dark:text-blue-400 dark:bg-blue-950"
                        )}>{a.urgency}</span>
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
          {data.watch_triggers.length > 0 && (
            <section className="theme-surface rounded-xl p-4">
              <h2 className="text-sm font-semibold text-app mb-3">Watch Triggers</h2>
              <div className="space-y-2">
                {data.watch_triggers.map(t => (
                  <div key={t.id} className="text-sm px-2 py-1.5">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-subtle shrink-0">{t.trigger_type.replace(/_/g, " ")}</span>
                      <span className="text-muted truncate">{t.condition}</span>
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
        </div>
      )}

      {/* Status change dialog */}
      <Dialog
        open={statusDialogOpen}
        onOpenChange={setStatusDialogOpen}
        title="Change Thesis Status"
        description={`Update the status for ${ticker}`}
      >
        <div className="space-y-4">
          <SelectInput
            label="New Status"
            value={newStatus}
            onChange={v => setNewStatus(v as ThesisStatusValue)}
            options={[
              { value: "active", label: "Active" },
              { value: "under_review", label: "Under Review" },
              { value: "invalidated", label: "Invalidated" },
            ]}
          />
          <TextInput
            label="Reason"
            value={statusReason}
            onChange={setStatusReason}
            placeholder="Why is the status changing?"
          />
          {statusMutation.isError && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {String(statusMutation.error)}
            </div>
          )}
          <ActionButton
            onClick={() => statusMutation.mutate()}
            loading={statusMutation.isPending}
            loadingText="Updating..."
          >
            Update Status
          </ActionButton>
        </div>
      </Dialog>
    </div>
  )
}

/* ---------- Sub-tab components ---------- */

/* ---------- Overview sub-section components ---------- */

type FinancialViewMode = "annual" | "quarterly"

type FinancialRow = {
  period_label?: string
  period_end?: string
  value?: number | null
  yoy_growth?: number | null
  form?: string
  filed?: string
  filing_url?: string
}

function formatPct(v: unknown): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "N/A"
  return `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`
}

function formatRevenue(v: unknown): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "N/A"
  if (Math.abs(v) >= 1e9) return `$${(v / 1e9).toFixed(2)}B`
  if (Math.abs(v) >= 1e6) return `$${(v / 1e6).toFixed(2)}M`
  return `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
}

function formatEps(v: unknown): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "N/A"
  return `${v >= 0 ? "+" : ""}${v.toFixed(3)}`
}

const revenueHistoryCols: ColumnDef[] = [
  { key: "period_label", header: "Period" },
  { key: "period_end", header: "Period End" },
  { key: "value_str", header: "Value" },
  { key: "yoy_str", header: "YoY", colorFn: (_, row) => {
    const g = row.yoy_growth as number | null
    if (typeof g !== "number") return ""
    return g >= 0 ? "#16a34a" : "#dc2626"
  }},
  { key: "filing_info", header: "Filing" },
]

function mapFinancialRows(rows: FinancialRow[], valueFmt: (v: unknown) => string) {
  return rows.map(r => ({
    period_label: r.period_label ?? "N/A",
    period_end: r.period_end ?? "N/A",
    value_str: valueFmt(r.value),
    yoy_growth: r.yoy_growth,
    yoy_str: formatPct(r.yoy_growth),
    filing_info: [r.form, r.filed].filter(Boolean).join(" · "),
  }))
}

const debtCols: ColumnDef[] = [
  { key: "tranche", header: "Tranche" },
  { key: "rate", header: "Rate" },
  { key: "maturity", header: "Maturity" },
]

function FinancialsSection({ ticker, parsed }: { ticker: string; parsed: ParsedFinancials | null }) {
  const [view, setView] = useState<FinancialViewMode>("annual")

  const { data: rawData, isLoading, error } = useApiQuery<Record<string, unknown>>(
    ["financials-overview-v9", ticker],
    () => runFinancials({ ticker }),
    300_000,
  )

  const metrics = (rawData?.metrics ?? {}) as Record<string, unknown>
  const annual = (rawData?.annual ?? {}) as Record<string, unknown>
  const quarterly = (rawData?.quarterly ?? {}) as Record<string, unknown>
  const revenueRows = (view === "annual" ? annual.revenue : quarterly.revenue) as FinancialRow[] | undefined
  const epsRows = (view === "annual" ? annual.eps : quarterly.eps) as FinancialRow[] | undefined

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-app uppercase tracking-wide">Financials</h3>

      {/* Metric cards from live EDGAR data */}
      {isLoading && <LoadingSpinner message="Loading SEC EDGAR financials..." />}
      {error && <p className="text-xs text-red-500">Live financials unavailable: {String(error)}</p>}
      {rawData && (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard title="3Y Revenue CAGR" value={formatPct(metrics.revenue_cagr_3y)} />
            <MetricCard title="3Y EPS CAGR" value={formatPct(metrics.eps_cagr_3y)} />
            <MetricCard title="Avg YoY Revenue (3Q)" value={formatPct(metrics.avg_yoy_revenue_growth_3q)} />
            <MetricCard title="Avg YoY EPS (3Q)" value={formatPct(metrics.avg_yoy_eps_growth_3q)} />
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs text-subtle">History</span>
            <SegmentedControl
              options={[
                { value: "annual" as const, label: "Annual" },
                { value: "quarterly" as const, label: "Quarterly" },
              ]}
              value={view}
              onChange={setView}
              size="sm"
            />
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <DataTable columns={revenueHistoryCols} rows={mapFinancialRows(revenueRows ?? [], formatRevenue)} maxHeight="300px" label="Revenue" />
            <DataTable columns={revenueHistoryCols} rows={mapFinancialRows(epsRows ?? [], formatEps)} maxHeight="300px" label="EPS" />
          </div>
        </>
      )}

      {/* Fallback: show parsed markdown values if EDGAR unavailable */}
      {!rawData && !isLoading && parsed && (
        <div className="space-y-2 text-sm text-muted">
          {parsed.revenue_growth && <p><span className="text-subtle font-medium">Revenue Growth:</span> {parsed.revenue_growth.context}</p>}
          {parsed.eps_growth && <p><span className="text-subtle font-medium">EPS Growth:</span> {parsed.eps_growth.context}</p>}
        </div>
      )}

      {/* Debt & reinvestment from parsed markdown */}
      {parsed?.debt && (
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-subtle uppercase">Debt</h4>
          <p className="text-sm text-muted">{parsed.debt.summary}</p>
          {parsed.debt.tranches.length > 0 && (
            <DataTable columns={debtCols} rows={parsed.debt.tranches.map(t => ({ ...t }))} maxHeight="200px" />
          )}
        </div>
      )}
      {parsed?.reinvestment && (
        <div className="space-y-1">
          <h4 className="text-xs font-semibold text-subtle uppercase">Reinvestment Costs</h4>
          <p className="text-sm text-muted">{parsed.reinvestment}</p>
        </div>
      )}
    </div>
  )
}

const PORTER_RATING_CONFIG: Record<string, { width: string; bg: string; text: string }> = {
  "Low": { width: "20%", bg: "bg-green-500", text: "text-green-700 dark:text-green-400" },
  "Low-Medium": { width: "35%", bg: "bg-lime-500", text: "text-lime-700 dark:text-lime-400" },
  "Medium": { width: "50%", bg: "bg-yellow-500", text: "text-yellow-700 dark:text-yellow-400" },
  "Medium-High": { width: "70%", bg: "bg-orange-500", text: "text-orange-700 dark:text-orange-400" },
  "High": { width: "90%", bg: "bg-red-500", text: "text-red-700 dark:text-red-400" },
}

function PortersForcesSection({ forces }: { forces: PorterForce[] }) {
  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-app uppercase tracking-wide">Porter&apos;s Five Forces</h3>
      <div className="space-y-4">
        {forces.map(f => {
          const cfg = PORTER_RATING_CONFIG[f.rating] ?? PORTER_RATING_CONFIG["Medium"]
          return (
            <div key={f.force}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium text-app">{f.force}</span>
                <span className={cn("text-xs font-semibold", cfg.text)}>{f.rating}</span>
              </div>
              <div className="h-2 w-full rounded-full bg-gray-200 dark:bg-gray-700">
                <div className={cn("h-2 rounded-full transition-all", cfg.bg)} style={{ width: cfg.width }} />
              </div>
              <p className="text-xs text-muted mt-1">{f.description}</p>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function sensitivityColor(val: unknown): string {
  const s = String(val).toLowerCase()
  if (s === "low") return "#16a34a"
  if (s === "low-medium") return "#65a30d"
  if (s === "medium") return "#ca8a04"
  if (s === "medium-high") return "#ea580c"
  if (s === "high") return "#dc2626"
  return ""
}

const sensitivityCols: ColumnDef[] = [
  { key: "factor", header: "Factor" },
  { key: "sensitivity", header: "Sensitivity", colorFn: (val) => sensitivityColor(val) },
  { key: "capacity", header: "Capacity to Deal" },
]

function SensitivitySection({ rows }: { rows: SensitivityRow[] }) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-app uppercase tracking-wide">Sensitivity to Extrinsic Factors</h3>
      <DataTable columns={sensitivityCols} rows={rows.map(r => ({ ...r }))} maxHeight="400px" />
    </div>
  )
}

const OUTLOOK_BADGE: Record<string, string> = {
  Strong: "text-green-700 bg-green-50 dark:text-green-400 dark:bg-green-950",
  Medium: "text-yellow-700 bg-yellow-50 dark:text-yellow-400 dark:bg-yellow-950",
  Weak: "text-red-700 bg-red-50 dark:text-red-400 dark:bg-red-950",
}

function OutlookSection({ supply, demand }: { supply: OutlookSection | null; demand: OutlookSection | null }) {
  const renderPoints = (points: (string | OutlookPoint)[]) =>
    points.map((p, i) => {
      if (typeof p === "string") {
        return <li key={i} className="text-sm text-muted pl-3 relative before:absolute before:left-0 before:top-[9px] before:h-1 before:w-1 before:rounded-full before:bg-gray-400">{p}</li>
      }
      return (
        <li key={i} className="text-sm text-muted pl-3 relative before:absolute before:left-0 before:top-[9px] before:h-1 before:w-1 before:rounded-full before:bg-gray-400">
          {p.label && <span className="font-medium text-app">{p.label}: </span>}
          {p.text}
        </li>
      )
    })

  const renderSection = (title: string, data: OutlookSection) => (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <h4 className="text-xs font-semibold text-subtle uppercase">{title}</h4>
        {data.rating && (
          <span className={cn("text-xs px-2 py-0.5 rounded font-medium", OUTLOOK_BADGE[data.rating] ?? "")}>{data.rating}</span>
        )}
      </div>
      <ul className="space-y-1.5">{renderPoints(data.points)}</ul>
    </div>
  )

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-app uppercase tracking-wide">Supply &amp; Demand Outlook</h3>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {supply && renderSection("Supply Outlook", supply)}
        {demand && renderSection("Demand Outlook", demand)}
      </div>
    </div>
  )
}

/* ---------- OverviewTab ---------- */

function OverviewTab({ content, parsed, ticker }: { content: string | null; parsed: ParsedOverview | null; ticker: string }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState("")
  const qc = useQueryClient()
  const saveMutation = useMutation({
    mutationFn: () => saveOverviewContent(ticker, draft),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dossier", ticker] })
      setEditing(false)
    },
  })

  const startEdit = () => {
    setDraft(content ?? "")
    setEditing(true)
  }

  if (!content) {
    return (
      <div>
        <p className="text-sm text-muted mb-3">No overview on file for this position.</p>
        <button type="button" onClick={startEdit} className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors">
          Write Overview
        </button>
        {editing && (
          <div className="mt-3">
            <textarea
              value={draft}
              onChange={e => setDraft(e.target.value)}
              className="w-full min-h-[300px] rounded-lg border border-app bg-transparent p-3 text-sm text-app font-mono focus:outline-none focus:ring-1 focus:ring-blue-500"
              placeholder={"# TICKER Overview\n## Financials\n## Sensitivity to Extrinsic Factors\n## Industry"}
            />
            {saveMutation.isError && <p className="text-xs text-red-600 mt-1">{String(saveMutation.error)}</p>}
            <div className="flex gap-2 mt-2">
              <ActionButton onClick={() => saveMutation.mutate()} loading={saveMutation.isPending} loadingText="Saving...">Save</ActionButton>
              <button type="button" onClick={() => setEditing(false)} className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors">Cancel</button>
            </div>
          </div>
        )}
      </div>
    )
  }

  if (editing) {
    return (
      <div>
        <textarea
          value={draft}
          onChange={e => setDraft(e.target.value)}
          className="w-full min-h-[400px] rounded-lg border border-app bg-transparent p-3 text-sm text-app font-mono focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        {saveMutation.isError && <p className="text-xs text-red-600 mt-1">{String(saveMutation.error)}</p>}
        <div className="flex gap-2 mt-2">
          <ActionButton onClick={() => saveMutation.mutate()} loading={saveMutation.isPending} loadingText="Saving...">Save</ActionButton>
          <button type="button" onClick={() => setEditing(false)} className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors">Cancel</button>
        </div>
      </div>
    )
  }

  // Structured read view
  if (parsed) {
    return (
      <div>
        <div className="flex justify-end mb-4">
          <button type="button" onClick={startEdit} className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors">
            Edit
          </button>
        </div>
        <div className="space-y-8">
          <FinancialsSection ticker={ticker} parsed={parsed.financials} />
          {parsed.porters_five_forces && <PortersForcesSection forces={parsed.porters_five_forces} />}
          {parsed.sensitivity && <SensitivitySection rows={parsed.sensitivity} />}
          {(parsed.supply_outlook || parsed.demand_outlook) && (
            <OutlookSection supply={parsed.supply_outlook} demand={parsed.demand_outlook} />
          )}
        </div>
      </div>
    )
  }

  // Fallback: render raw markdown if parsing failed
  return (
    <div>
      <div className="flex justify-end mb-2">
        <button type="button" onClick={startEdit} className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors">
          Edit
        </button>
      </div>
      <div className="prose prose-sm dark:prose-invert max-w-none">
        <MarkdownRenderer content={content} />
      </div>
    </div>
  )
}

function ThesisTab({ thesis, ticker, position }: { thesis: DossierData["thesis"]; ticker: string; position: Record<string, unknown> | null }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState("")
  const qc = useQueryClient()
  const saveMutation = useMutation({
    mutationFn: () => saveThesisContent(ticker, draft),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dossier", ticker] })
      qc.invalidateQueries({ queryKey: ["thesis"] })
      setEditing(false)
    },
  })

  const startEdit = () => {
    setDraft(thesis.content ?? "")
    setEditing(true)
  }

  if (!thesis.content && !thesis.meta) {
    return (
      <div>
        <p className="text-sm text-muted mb-3">No thesis on file for this position.</p>
        <button type="button" onClick={startEdit} className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors">
          Write Thesis
        </button>
        {editing && (
          <div className="mt-3">
            <textarea
              value={draft}
              onChange={e => setDraft(e.target.value)}
              className="w-full min-h-[300px] rounded-lg border border-app bg-transparent p-3 text-sm text-app font-mono focus:outline-none focus:ring-1 focus:ring-blue-500"
              placeholder="# TICKER&#10;## Thesis&#10;## Key Catalysts&#10;## Risk Factors"
            />
            {saveMutation.isError && <p className="text-xs text-red-600 mt-1">{String(saveMutation.error)}</p>}
            <div className="flex gap-2 mt-2">
              <ActionButton onClick={() => saveMutation.mutate()} loading={saveMutation.isPending} loadingText="Saving...">Save</ActionButton>
              <button type="button" onClick={() => setEditing(false)} className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors">Cancel</button>
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div>
      {(position || thesis.meta) && (
        <div className="flex flex-wrap gap-4 text-sm mb-4 pb-4 border-b border-app">
          {position?.direction != null && <div><span className="text-subtle">Direction:</span> <span className="font-medium text-app">{String(position.direction)}</span></div>}
          {thesis.meta?.last_evaluated && <div><span className="text-subtle">Last Evaluated:</span> <span className="font-medium text-app">{formatTime(thesis.meta.last_evaluated)}</span></div>}
        </div>
      )}
      {editing ? (
        <div>
          <textarea
            value={draft}
            onChange={e => setDraft(e.target.value)}
            className="w-full min-h-[400px] rounded-lg border border-app bg-transparent p-3 text-sm text-app font-mono focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
          {saveMutation.isError && <p className="text-xs text-red-600 mt-1">{String(saveMutation.error)}</p>}
          <div className="flex gap-2 mt-2">
            <ActionButton onClick={() => saveMutation.mutate()} loading={saveMutation.isPending} loadingText="Saving...">Save</ActionButton>
            <button type="button" onClick={() => setEditing(false)} className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors">Cancel</button>
          </div>
        </div>
      ) : (
        <>
          <div className="flex justify-end mb-2">
            <button type="button" onClick={startEdit} className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors">
              Edit
            </button>
          </div>
          {thesis.content && (
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <MarkdownRenderer content={thesis.content} />
            </div>
          )}
        </>
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

type SourceRequirementDraft = Omit<SourceRequirement, "freshness_days"> & { freshness_days: string }
interface ClaimDraft {
  id: number | null
  claim: string
  expected_evidence: string
  disconfirming_evidence: string
  source_requirements: SourceRequirementDraft[]
  cadence: string
  confidence: string
  status: ThesisClaimStatus
  linked_catalyst_ids: number[]
  linked_kill_condition_ids: number[]
}

const CLAIM_STATUSES: ThesisClaimStatus[] = ["active", "supported", "challenged", "disconfirmed", "retired"]

function blankClaimDraft(): ClaimDraft {
  return {
    id: null,
    claim: "",
    expected_evidence: "",
    disconfirming_evidence: "",
    source_requirements: [],
    cadence: "",
    confidence: "",
    status: "active",
    linked_catalyst_ids: [],
    linked_kill_condition_ids: [],
  }
}

function sourceRequirementsForClaim(claim: ThesisClaim): SourceRequirement[] {
  return claim.source_requirements ?? claim.source_requirements_json ?? []
}

function claimToDraft(claim: ThesisClaim): ClaimDraft {
  return {
    id: claim.id,
    claim: claim.claim,
    expected_evidence: claim.expected_evidence ?? "",
    disconfirming_evidence: claim.disconfirming_evidence ?? "",
    source_requirements: sourceRequirementsForClaim(claim).map(req => ({
      type: req.type || "custom",
      description: req.description || "",
      required: req.required !== false,
      freshness_days: req.freshness_days == null ? "" : String(req.freshness_days),
    })),
    cadence: claim.cadence ?? "",
    confidence: claim.confidence == null ? "" : String(claim.confidence),
    status: claim.status,
    linked_catalyst_ids: claim.linked_catalyst_ids ?? claim.linked_catalyst_ids_json ?? [],
    linked_kill_condition_ids: claim.linked_kill_condition_ids ?? claim.linked_kill_condition_ids_json ?? [],
  }
}

function draftToPayload(draft: ClaimDraft, ticker: string) {
  const confidenceText = draft.confidence.trim()
  const confidence = confidenceText === "" ? null : Number(confidenceText)
  return {
    ticker,
    claim: draft.claim.trim(),
    expected_evidence: draft.expected_evidence.trim() || null,
    disconfirming_evidence: draft.disconfirming_evidence.trim() || null,
    source_requirements: draft.source_requirements
      .filter(req => req.type.trim() || req.description.trim())
      .map(req => ({
        type: req.type.trim() || "custom",
        description: req.description.trim() || req.type.trim() || "custom",
        required: req.required,
        freshness_days: req.freshness_days.trim() === "" ? null : Number(req.freshness_days),
      })),
    cadence: draft.cadence.trim() || null,
    confidence: Number.isFinite(confidence) ? confidence : null,
    status: draft.status,
    linked_catalyst_ids: draft.linked_catalyst_ids,
    linked_kill_condition_ids: draft.linked_kill_condition_ids,
  }
}

function ClaimsTab({
  claims,
  catalysts,
  conditions,
  ticker,
}: {
  claims: ThesisClaim[]
  catalysts: Catalyst[]
  conditions: KillCondition[]
  ticker: string
}) {
  const [draft, setDraft] = useState<ClaimDraft | null>(null)
  const qc = useQueryClient()
  const mutation = useMutation({
    mutationFn: (next: ClaimDraft) => {
      const payload = draftToPayload(next, ticker)
      if (next.id) return updateThesisClaim(next.id, payload)
      return createThesisClaim({ ...payload, ticker, claim: payload.claim })
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dossier", ticker] })
      qc.invalidateQueries({ queryKey: ["thesis"] })
      setDraft(null)
    },
  })

  function updateDraft(patch: Partial<ClaimDraft>) {
    setDraft(prev => prev ? { ...prev, ...patch } : prev)
  }

  function updateSourceRequirement(index: number, patch: Partial<SourceRequirementDraft>) {
    setDraft(prev => {
      if (!prev) return prev
      const source_requirements = prev.source_requirements.map((req, i) => i === index ? { ...req, ...patch } : req)
      return { ...prev, source_requirements }
    })
  }

  function toggleLinked(kind: "catalyst" | "kill", id: number) {
    setDraft(prev => {
      if (!prev) return prev
      const key = kind === "catalyst" ? "linked_catalyst_ids" : "linked_kill_condition_ids"
      const current = new Set(prev[key])
      if (current.has(id)) current.delete(id)
      else current.add(id)
      return { ...prev, [key]: Array.from(current) }
    })
  }

  function addSourceRequirement() {
    setDraft(prev => prev ? {
      ...prev,
      source_requirements: [
        ...prev.source_requirements,
        { type: "custom", description: "", required: true, freshness_days: "" },
      ],
    } : prev)
  }

  function removeSourceRequirement(index: number) {
    setDraft(prev => prev ? {
      ...prev,
      source_requirements: prev.source_requirements.filter((_, i) => i !== index),
    } : prev)
  }

  const catalystLabelById = new Map(catalysts.map(c => [c.id, c.description.split(": ", 1)[0]]))
  const conditionLabelById = new Map(conditions.map(k => [k.id, k.condition.split(": ", 1)[0]]))

  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <button
          type="button"
          onClick={() => setDraft(blankClaimDraft())}
          className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors"
        >
          Add Claim
        </button>
      </div>

      {draft && (
        <div className="rounded-lg border border-app p-4 space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <SelectInput
              label="Status"
              value={draft.status}
              onChange={v => updateDraft({ status: v as ThesisClaimStatus })}
              options={CLAIM_STATUSES.map(status => ({ value: status, label: status.replace(/_/g, " ") }))}
            />
            <TextInput label="Cadence" value={draft.cadence} onChange={v => updateDraft({ cadence: v })} placeholder="weekly" />
            <TextInput label="Confidence" type="number" value={draft.confidence} onChange={v => updateDraft({ confidence: v })} placeholder="0.70" />
          </div>
          <div>
            <label className="mb-1.5 block text-sm text-muted">Claim</label>
            <textarea
              value={draft.claim}
              onChange={e => updateDraft({ claim: e.target.value })}
              className="theme-input min-h-[80px] w-full"
            />
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            <div>
              <label className="mb-1.5 block text-sm text-muted">Expected evidence</label>
              <textarea
                value={draft.expected_evidence}
                onChange={e => updateDraft({ expected_evidence: e.target.value })}
                className="theme-input min-h-[90px] w-full"
              />
            </div>
            <div>
              <label className="mb-1.5 block text-sm text-muted">Disconfirming evidence</label>
              <textarea
                value={draft.disconfirming_evidence}
                onChange={e => updateDraft({ disconfirming_evidence: e.target.value })}
                className="theme-input min-h-[90px] w-full"
              />
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold uppercase text-subtle">Source Requirements</h3>
              <button type="button" onClick={addSourceRequirement} className="text-xs font-medium text-blue-600 dark:text-blue-400">
                Add source
              </button>
            </div>
            {draft.source_requirements.map((req, index) => (
              <div key={index} className="grid grid-cols-1 md:grid-cols-[1fr_2fr_auto_auto_auto] gap-2 items-end">
                <TextInput label="Type" value={req.type} onChange={v => updateSourceRequirement(index, { type: v })} />
                <TextInput label="Description" value={req.description} onChange={v => updateSourceRequirement(index, { description: v })} />
                <TextInput label="Freshness days" type="number" value={req.freshness_days} onChange={v => updateSourceRequirement(index, { freshness_days: v })} />
                <label className="flex items-center gap-2 pb-2 text-sm text-muted">
                  <input type="checkbox" checked={req.required} onChange={e => updateSourceRequirement(index, { required: e.target.checked })} />
                  Required
                </label>
                <button type="button" onClick={() => removeSourceRequirement(index)} className="rounded border border-app px-2 py-2 text-xs text-muted hover:text-app">
                  Remove
                </button>
              </div>
            ))}
            {!draft.source_requirements.length && <p className="text-xs text-subtle">No source requirements.</p>}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            <LinkCheckboxes
              title="Linked Catalysts"
              items={catalysts.map(c => ({ id: c.id, label: c.description }))}
              selectedIds={draft.linked_catalyst_ids}
              onToggle={id => toggleLinked("catalyst", id)}
            />
            <LinkCheckboxes
              title="Linked Kill Conditions"
              items={conditions.map(k => ({ id: k.id, label: k.condition }))}
              selectedIds={draft.linked_kill_condition_ids}
              onToggle={id => toggleLinked("kill", id)}
            />
          </div>

          {mutation.isError && <p className="text-xs text-red-600">{errorMessage(mutation.error)}</p>}
          <div className="flex gap-2">
            <ActionButton
              onClick={() => draft.claim.trim() && mutation.mutate(draft)}
              loading={mutation.isPending}
              loadingText="Saving..."
              disabled={!draft.claim.trim()}
              className="w-auto px-4"
            >
              Save Claim
            </ActionButton>
            <button type="button" onClick={() => setDraft(null)} className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app">
              Cancel
            </button>
          </div>
        </div>
      )}

      {!claims.length && !draft && <p className="text-sm text-muted">No thesis claims tracked.</p>}
      {claims.map(claim => {
        const sourceRequirements = sourceRequirementsForClaim(claim)
        const linkedCatalysts = (claim.linked_catalyst_ids ?? claim.linked_catalyst_ids_json ?? [])
          .map(id => catalystLabelById.get(id))
          .filter(Boolean)
        const linkedConditions = (claim.linked_kill_condition_ids ?? claim.linked_kill_condition_ids_json ?? [])
          .map(id => conditionLabelById.get(id))
          .filter(Boolean)
        return (
          <div key={claim.id} className="rounded-lg border border-app px-4 py-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-sm font-medium text-app">{claim.claim}</p>
                <div className="mt-1 flex flex-wrap gap-2 text-xs text-subtle">
                  <span className={cn("px-1.5 py-0.5 rounded font-medium", STATUS_COLORS[claim.status] ?? "")}>{claim.status.replace(/_/g, " ")}</span>
                  {claim.cadence && <span>{claim.cadence}</span>}
                  {claim.confidence != null && <span>{formatConfidence(claim.confidence)} confidence</span>}
                </div>
              </div>
              <button type="button" onClick={() => setDraft(claimToDraft(claim))} className="rounded border border-app px-2 py-1 text-xs text-muted hover:text-app">
                Edit
              </button>
            </div>
            <div className="mt-3 grid grid-cols-1 lg:grid-cols-2 gap-3 text-xs">
              {claim.expected_evidence && <div><p className="font-semibold text-subtle uppercase">Expected Evidence</p><p className="mt-1 text-muted">{claim.expected_evidence}</p></div>}
              {claim.disconfirming_evidence && <div><p className="font-semibold text-subtle uppercase">Disconfirming Evidence</p><p className="mt-1 text-muted">{claim.disconfirming_evidence}</p></div>}
            </div>
            {sourceRequirements.length > 0 && (
              <div className="mt-3">
                <p className="text-xs font-semibold text-subtle uppercase">Sources</p>
                <div className="mt-1 flex flex-wrap gap-1.5">
                  {sourceRequirements.map((req, index) => (
                    <span key={index} className="rounded border border-app px-2 py-1 text-xs text-muted">
                      {req.type}: {req.description}{req.required ? "" : " (optional)"}{req.freshness_days != null ? `, ${req.freshness_days}d` : ""}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {(linkedCatalysts.length > 0 || linkedConditions.length > 0) && (
              <div className="mt-3 flex flex-wrap gap-2 text-xs text-subtle">
                {linkedCatalysts.length > 0 && <span>Catalysts: {linkedCatalysts.join("; ")}</span>}
                {linkedConditions.length > 0 && <span>Kill conditions: {linkedConditions.join("; ")}</span>}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function LinkCheckboxes({
  title,
  items,
  selectedIds,
  onToggle,
}: {
  title: string
  items: { id: number; label: string }[]
  selectedIds: number[]
  onToggle: (id: number) => void
}) {
  return (
    <div className="rounded-lg border border-app p-3">
      <p className="mb-2 text-xs font-semibold uppercase text-subtle">{title}</p>
      {items.length ? (
        <div className="max-h-40 space-y-1 overflow-y-auto">
          {items.map(item => (
            <label key={item.id} className="flex items-start gap-2 text-xs text-muted">
              <input
                type="checkbox"
                checked={selectedIds.includes(item.id)}
                onChange={() => onToggle(item.id)}
                className="mt-0.5"
              />
              <span>{item.label}</span>
            </label>
          ))}
        </div>
      ) : (
        <p className="text-xs text-subtle">None available.</p>
      )}
    </div>
  )
}

function CatalystsTab({ catalysts, ticker }: { catalysts: Catalyst[]; ticker: string }) {
  const [openMenuId, setOpenMenuId] = useState<number | null>(null)
  const qc = useQueryClient()
  const mutation = useMutation({
    mutationFn: ({ id, status }: { id: number; status: string }) => updateCatalystStatus(id, status),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dossier", ticker] })
      setOpenMenuId(null)
    },
  })

  if (!catalysts.length) return <p className="text-sm text-muted">No catalysts tracked.</p>
  const statusOptions = ["pending", "played_out", "failed", "superseded"]
  return (
    <div className="space-y-3">
      {catalysts.map(c => (
        <div key={c.id} className="rounded-lg border border-app px-4 py-3">
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm font-medium text-app">{c.description}</span>
            <button
              type="button"
              onClick={() => setOpenMenuId(openMenuId === c.id ? null : c.id)}
              className={cn("text-xs px-1.5 py-0.5 rounded font-medium cursor-pointer hover:ring-1 hover:ring-blue-300", STATUS_COLORS[c.status] ?? "")}
            >
              {c.status.replace(/_/g, " ")}
            </button>
          </div>
          {openMenuId === c.id && (
            <div className="flex flex-wrap gap-1.5 mt-2 mb-1">
              {statusOptions.filter(s => s !== c.status).map(s => (
                <button
                  key={s}
                  type="button"
                  onClick={() => mutation.mutate({ id: c.id, status: s })}
                  disabled={mutation.isPending}
                  className={cn("text-xs px-1.5 py-0.5 rounded font-medium transition-colors hover:ring-1 hover:ring-gray-300", STATUS_COLORS[s] ?? "")}
                >
                  {s.replace(/_/g, " ")}
                </button>
              ))}
            </div>
          )}
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

function KillConditionsTab({ conditions, ticker }: { conditions: KillCondition[]; ticker: string }) {
  const [openMenuId, setOpenMenuId] = useState<number | null>(null)
  const qc = useQueryClient()
  const mutation = useMutation({
    mutationFn: ({ id, status }: { id: number; status: string }) => updateKillConditionStatus(id, status),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dossier", ticker] })
      setOpenMenuId(null)
    },
  })

  if (!conditions.length) return <p className="text-sm text-muted">No kill conditions defined.</p>
  const statusOptions = ["active", "triggered", "retired"]
  return (
    <div className="space-y-3">
      {conditions.map(k => (
        <div key={k.id} className={cn("rounded-lg border px-4 py-3", k.status === "triggered" ? "border-red-300 bg-red-50/50 dark:border-red-800 dark:bg-red-950/30" : "border-app")}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm font-medium text-app">{k.condition}</span>
            <button
              type="button"
              onClick={() => setOpenMenuId(openMenuId === k.id ? null : k.id)}
              className={cn("text-xs px-1.5 py-0.5 rounded font-medium cursor-pointer hover:ring-1 hover:ring-blue-300", STATUS_COLORS[k.status] ?? "")}
            >
              {k.status}
            </button>
          </div>
          {openMenuId === k.id && (
            <div className="flex flex-wrap gap-1.5 mt-2 mb-1">
              {statusOptions.filter(s => s !== k.status).map(s => (
                <button
                  key={s}
                  type="button"
                  onClick={() => mutation.mutate({ id: k.id, status: s })}
                  disabled={mutation.isPending}
                  className={cn("text-xs px-1.5 py-0.5 rounded font-medium transition-colors hover:ring-1 hover:ring-gray-300", STATUS_COLORS[s] ?? "")}
                >
                  {s}
                </button>
              ))}
            </div>
          )}
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

function formatDateTime(value: string | null | undefined): string {
  if (!value) return "-"
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return value
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  })
}

function formatNumber(value: unknown, digits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-"
  return value.toFixed(digits)
}

function formatConfidence(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-"
  return `${Math.round(value * 100)}%`
}

function formatMetricValue(value: unknown): string {
  if (typeof value === "number" && Number.isFinite(value)) return value.toFixed(2)
  if (value == null || value === "") return "-"
  return String(value)
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message
  return String(error)
}

function riskLevelClass(level: unknown): string {
  const s = String(level ?? "").toLowerCase()
  if (s === "high") return "text-red-700 bg-red-50 dark:text-red-400 dark:bg-red-950"
  if (s === "medium") return "text-amber-700 bg-amber-50 dark:text-amber-400 dark:bg-amber-950"
  if (s === "low") return "text-green-700 bg-green-50 dark:text-green-400 dark:bg-green-950"
  return "text-gray-600 bg-gray-100 dark:text-gray-400 dark:bg-gray-800"
}

function moduleStatusClass(status: unknown): string {
  const s = String(status ?? "error").toLowerCase()
  if (s === "ok") return "text-green-700 bg-green-50 dark:text-green-400 dark:bg-green-950"
  if (s === "partial") return "text-amber-700 bg-amber-50 dark:text-amber-400 dark:bg-amber-950"
  return "text-red-700 bg-red-50 dark:text-red-400 dark:bg-red-950"
}

function evidenceTitle(ev: OntologyEvidence): string {
  return ev.name || ev.component || ev.source || "Risk driver"
}

function RiskTab({ ticker }: { ticker: string }) {
  const qc = useQueryClient()
  const [elapsed, setElapsed] = useState(0)

  const runsQuery = useQuery({
    queryKey: ["ontology-runs", "latest"],
    queryFn: () => fetchOntologyRuns(1),
    staleTime: 60 * 1000,
    retry: 1,
  })
  const latestRun = runsQuery.data?.runs?.[0]

  const cachedRiskQuery = useQuery({
    queryKey: ["ontology-risk", ticker, latestRun?.run_id],
    queryFn: () => queryOntology({
      filters: { tickers: [ticker] },
      run_id: latestRun!.run_id,
      include_graph: false,
      refresh_snapshot: false,
      page: 1,
      page_size: 1,
    }),
    enabled: Boolean(latestRun?.run_id),
    staleTime: 60 * 1000,
    retry: 1,
  })

  const refreshMutation = useMutation({
    mutationFn: () => runOntologyQueryAsync({
      filters: { tickers: [ticker] },
      timeframe: "Daily",
      include_graph: false,
      refresh_snapshot: true,
      page: 1,
      page_size: 1,
    }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["ontology-runs"] })
    },
  })

  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    if (!refreshMutation.isPending) {
      setElapsed(0)
      return
    }
    const start = Date.now()
    const id = window.setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000)
    return () => window.clearInterval(id)
  }, [refreshMutation.isPending])
  /* eslint-enable react-hooks/set-state-in-effect */

  const ontology: OntologyResponse | undefined = refreshMutation.data ?? cachedRiskQuery.data
  const rows = Array.isArray(ontology?.results) ? ontology.results : []
  const tickerUpper = ticker.toUpperCase()
  const row = rows.find(r => String(r.ticker ?? "").toUpperCase() === tickerUpper) ?? null
  const evidence = Array.isArray(row?.evidence) ? row.evidence : []
  const sourceStatus = ontology?.source_status ?? {}
  const moduleRows = Object.entries(sourceStatus).sort(([a], [b]) => a.localeCompare(b))
  const moduleIssueCount = moduleRows.filter(([, state]) => String(state?.status ?? "error").toLowerCase() !== "ok").length
  const requiredHealth = latestRun?.required_modules_ok == null
    ? moduleRows.length
      ? moduleIssueCount === 0 ? "OK" : `${moduleIssueCount} issue${moduleIssueCount === 1 ? "" : "s"}`
      : "-"
    : latestRun.required_modules_ok ? "OK" : "Degraded"
  const primaryError = refreshMutation.error ?? (!ontology ? cachedRiskQuery.error ?? runsQuery.error : null)
  const noCachedRun = !runsQuery.isLoading && !latestRun && !refreshMutation.data

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-sm font-semibold text-app">Ontology Risk</h2>
          <p className="text-xs text-subtle">Snapshot {formatDateTime(ontology?.as_of ?? latestRun?.as_of)}</p>
        </div>
        <ActionButton
          onClick={() => refreshMutation.mutate()}
          loading={refreshMutation.isPending}
          loadingText="Refreshing..."
          className="w-auto px-3 py-1.5"
        >
          Refresh Risk
        </ActionButton>
      </div>

      {refreshMutation.isPending && <LoadingSpinner message={`Refreshing ontology risk... (${elapsed}s elapsed)`} />}
      {!ontology && runsQuery.isLoading && <LoadingSpinner message="Loading cached ontology risk..." />}
      {!ontology && latestRun && cachedRiskQuery.isLoading && <LoadingSpinner message="Loading cached ontology risk..." />}

      {primaryError && <ErrorMessage message={errorMessage(primaryError)} />}
      {refreshMutation.isSuccess && (
        <div className="rounded-lg border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700 dark:border-green-900 dark:bg-green-950 dark:text-green-400">
          Risk snapshot refreshed.
        </div>
      )}

      {noCachedRun && (
        <div className="rounded-lg border border-app px-4 py-3">
          <p className="text-sm font-medium text-app">No cached ontology snapshot available.</p>
          <p className="mt-1 text-xs text-muted">Use Refresh Risk to build a fresh snapshot.</p>
        </div>
      )}

      {ontology && !row && !cachedRiskQuery.isLoading && (
        <div className="rounded-lg border border-app px-4 py-3">
          <p className="text-sm font-medium text-app">No cached ontology risk for {tickerUpper}.</p>
          <p className="mt-1 text-xs text-muted">Use Refresh Risk to rebuild the risk snapshot.</p>
        </div>
      )}

      {row && (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
            <div className="rounded-lg border border-app px-4 py-3">
              <p className="text-xs text-subtle">Risk Level</p>
              <span className={cn("mt-2 inline-flex rounded px-2 py-0.5 text-xs font-medium", riskLevelClass(row.risk_level))}>
                {String(row.risk_level ?? "unknown")}
              </span>
            </div>
            <div className="rounded-lg border border-app px-4 py-3">
              <p className="text-xs text-subtle">Risk Score</p>
              <p className="mt-1 text-xl font-semibold text-app">{formatNumber(row.risk_score)}</p>
            </div>
            <div className="rounded-lg border border-app px-4 py-3">
              <p className="text-xs text-subtle">Required Modules</p>
              <p className="mt-1 text-xl font-semibold text-app">{requiredHealth}</p>
            </div>
            <div className="rounded-lg border border-app px-4 py-3">
              <p className="text-xs text-subtle">Confidence</p>
              <p className="mt-1 text-xl font-semibold text-app">{formatConfidence(ontology?.aggregate?.confidence)}</p>
            </div>
          </div>

          <div className="rounded-lg border border-app px-4 py-3">
            <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-subtle">
              <span>{row.asset ?? "unknown"} asset</span>
              <span>{row.direction ?? "unknown"}</span>
              <span>{row.sector ?? "Unknown sector"}</span>
            </div>
            <h3 className="mb-2 text-sm font-semibold text-app">Top Drivers</h3>
            {evidence.length ? (
              <div className="space-y-2">
                {evidence.map((ev, i) => (
                  <div key={`${evidenceTitle(ev)}-${i}`} className="rounded border border-app px-3 py-2">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="text-sm font-medium text-app">{evidenceTitle(ev)}</span>
                      <span className="text-xs text-subtle">Contribution {formatMetricValue(ev.contribution)}</span>
                    </div>
                    <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted">
                      <span>{ev.source ?? "unknown source"}</span>
                      <span>{ev.direction ?? "unknown direction"}</span>
                      <span>Value {formatMetricValue(ev.value)}</span>
                    </div>
                    {ev.threshold && <p className="mt-1 text-xs text-subtle">{ev.threshold}</p>}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted">No evidence drivers available.</p>
            )}
          </div>

          <div className="rounded-lg border border-app px-4 py-3">
            <h3 className="mb-2 text-sm font-semibold text-app">Module Health</h3>
            {moduleRows.length ? (
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {moduleRows.map(([module, state]) => (
                  <div key={module} className="flex items-center justify-between gap-3 rounded border border-app px-3 py-2 text-sm">
                    <span className="min-w-0 truncate text-muted">{module}</span>
                    <span className={cn("shrink-0 rounded px-1.5 py-0.5 text-xs font-medium", moduleStatusClass(state?.status))}>
                      {state?.status ?? "error"}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted">No module status available.</p>
            )}
          </div>
        </>
      )}

      {ontology && (
        <details className="rounded-lg border border-app px-4 py-3">
          <summary className="cursor-pointer text-sm font-medium text-app">Raw ontology JSON</summary>
          <pre className="mt-3 max-h-[500px] overflow-auto whitespace-pre-wrap text-xs text-muted">
            {JSON.stringify(ontology, null, 2)}
          </pre>
        </details>
      )}
    </div>
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
