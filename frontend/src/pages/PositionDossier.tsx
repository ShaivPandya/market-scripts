import { useEffect, useState } from "react"
import { useParams, useNavigate, useLocation } from "react-router-dom"
import { useQueryClient, useMutation, useQuery } from "@tanstack/react-query"
import { useApiQuery } from "@/hooks/useApiQuery"
import {
  fetchDossier,
  fetchApprovalSummary,
  approveItem,
  rejectItem,
  rejectAndRestageApproval,
  updateThesisStatus,
  fetchThesisStatus,
  saveThesisContent,
  saveOverviewContent,
  saveManagementQualityContent,
  completeAction,
  dismissAction,
  updateCatalystStatus,
  updateKillConditionStatus,
  createThesisClaim,
  updateThesisClaim,
  fetchPositionRiskLatest,
  refreshPositionRisk,
  fetchPositionValuation,
  type ApprovalRecord,
  type PositionRiskEvidence,
  type PositionRiskSnapshot,
  type SourceRequirement,
  type StagedMutationResponse,
  type ThesisClaim,
  type ThesisClaimStatus,
  type ThesisStatus,
  type ThesisStatusValue,
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
import { MarkdownRenderer } from "@/components/shared/MarkdownRenderer"
import { Dialog } from "@/components/shared/Dialog"
import { formatApprovalDisplayLabel, StagedProposalNotice } from "@/components/shared/StagedProposalNotice"
import { ApprovalChangeSummary } from "@/components/shared/ApprovalChangeSummary"
import { ActionButton, SelectInput, TextInput } from "@/components/shared/FormControls"
import { EquityOverviewReadView } from "@/components/overview/EquityOverviewReadView"
import { PositionValuationTab } from "@/components/valuation/PositionValuationTab"
import type {
  ManagementQualityAssessment,
  ManagementQualityBullet,
  ParsedManagementQuality,
} from "@/lib/managementQualityTypes"
import type { ParsedOverview } from "@/lib/overviewTypes"
import {
  DecisionStateBadge,
  EffectScopeBadge,
  BaseStateBadge,
  PolicyStateBadge,
  QualityStateBadge,
} from "@/components/shared/DecisionStateBadge"
import { approvalDecisionState } from "@/lib/decisionState"
import { ThesisUpload } from "@/components/ThesisUpload"
import { OverviewUpload } from "@/components/OverviewUpload"
import { ManagementQualityUpload } from "@/components/ManagementQualityUpload"
import { cn } from "@/lib/utils"
import { cleanDossierDisplayText } from "@/lib/dossierText"

interface DossierData {
  ticker: string
  position: Record<string, unknown> | null
  overview_content: string | null
  overview_parsed: ParsedOverview | null
  management_quality: {
    content: string | null
    parsed: ParsedManagementQuality | null
    assessment?: ManagementQualityAssessment | null
  }
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
  pending_approvals: ApprovalRecord[]
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
type EntityId = number | string
interface Catalyst { id: EntityId; description: string | null; category: string | null; status: string | null; target_date: string | null; evidence: string | null }
interface KillCondition { id: EntityId; condition: string | null; metric: string | null; threshold: string | null; status: string | null; triggered_at: string | null }
interface WorkflowRun { run_id: string | null; workflow_name: string | null; status: string | null; started_at: string | null; completed_at: string | null }
interface ActionItem { id: number | string; description: string; action_type: string; urgency: string; status: string; created_at: string }
interface Trigger { id: number; condition: string; trigger_type: string; status: string; created_at: string; last_checked_at: string | null; last_evidence: string | null }

const BASE_TABS = ["Thesis", "Claims", "Catalysts", "Kill Conditions", "Evaluations", "Risk", "Workflows"] as const
type Tab = "Overview" | "Management Quality" | "Valuation" | typeof BASE_TABS[number]

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

function formatTime(iso: string | null | undefined): string {
  const value = String(iso ?? "").trim()
  if (!value) return "Unknown time"
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })
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

function workflowStatusClass(status: string | null | undefined): string {
  const value = String(status ?? "").toLowerCase()
  if (["completed", "succeeded", "success", "ok"].includes(value)) return "bg-green-500"
  if (["running", "started", "queued"].includes(value)) return "bg-blue-500 animate-pulse"
  if (["failed", "error"].includes(value)) return "bg-red-500"
  return "bg-gray-400"
}

function textOrFallback(value: unknown, fallback: string): string {
  const text = String(value ?? "").trim()
  return text || fallback
}

function statusOrFallback(value: unknown, fallback: string): string {
  return textOrFallback(value, fallback)
}

function subjectLabel(entityType?: string | null): string {
  return String(entityType || "proposal").replace(/_/g, " ")
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
  const [processingIds, setProcessingIds] = useState<Set<number | string>>(new Set())
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())
  const [statusDialogOpen, setStatusDialogOpen] = useState(false)
  const [newStatus, setNewStatus] = useState<ThesisStatusValue>("under_review")
  const [statusReason, setStatusReason] = useState("")
  const [lastProposal, setLastProposal] = useState<StagedMutationResponse | null>(null)
  const [approvalReview, setApprovalReview] = useState<{ approval: ApprovalRecord; action: "approve" | "reject" } | null>(null)
  const [approvalNote, setApprovalNote] = useState("")
  const [approvalError, setApprovalError] = useState<string | null>(null)

  const { data, isLoading, error } = useApiQuery<DossierData>(
    ["dossier", ticker],
    () => fetchDossier(ticker!),
    60_000,
  )
  const approvalSummary = useApiQuery(
    approvalSummaryQueryKey({ status: "pending", ticker, limit: 50 }),
    () => fetchApprovalSummary({ status: "pending", ticker, limit: 50 }),
    30_000,
  )

  const { data: thesisStatus } = useApiQuery<Record<string, string>>(
    ["thesis", "status"],
    fetchThesisStatus,
  )

  const isEquity =
    String(data?.position?.asset ?? "") === "equity" &&
    String(data?.position?.instrument_type ?? "security") !== "future"

  useEffect(() => {
    if (!ticker || !isEquity) return
    void qc.prefetchQuery({
      queryKey: ["valuation", ticker],
      queryFn: () => fetchPositionValuation(ticker),
      staleTime: 300_000,
    })
  }, [isEquity, qc, ticker])

  const statusMutation = useMutation({
    mutationFn: () => updateThesisStatus(ticker!, newStatus, statusReason),
    onSuccess: result => {
      setLastProposal(result as StagedMutationResponse)
      void invalidateApprovalSummaries(qc)
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

  function openApprovalReview(approval: ApprovalRecord, action: "approve" | "reject") {
    setApprovalReview({ approval, action })
    setApprovalNote("")
    setApprovalError(null)
  }

  async function handleApproval(approval: ApprovalRecord, action: "approve" | "reject", note?: string) {
    setProcessingIds(prev => new Set(prev).add(approval.id))
    setApprovalError(null)
    try {
      let resolved: ApprovalRecord
      if (action === "approve") {
        const trimmed = String(note || "").trim()
        if (!trimmed) {
          setApprovalError("Approval note is required before applying an internal state change.")
          return
        }
        resolved = await approveItem(approval.id, trimmed)
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
      setProcessingIds(prev => { const n = new Set(prev); n.delete(approval.id); return n })
    }
  }

  async function handleRejectAndRestage(approval: ApprovalRecord, note?: string) {
    setProcessingIds(prev => new Set(prev).add(approval.id))
    setApprovalError(null)
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
      setProcessingIds(prev => { const n = new Set(prev); n.delete(approval.id); return n })
    }
  }

  async function handleActionItem(id: number | string, action: "complete" | "dismiss") {
    setProcessingIds(prev => new Set(prev).add(id))
    try {
      const result = action === "complete" ? await completeAction(id) : await dismissAction(id)
      setLastProposal(result)
      void invalidateApprovalSummaries(qc)
      void qc.invalidateQueries({ queryKey: ["dossier", ticker] })
    } finally {
      setProcessingIds(prev => { const n = new Set(prev); n.delete(id); return n })
    }
  }

  if (!ticker) return <ErrorMessage message="No ticker specified" />
  if (isLoading) return <LoadingSpinner message={`Loading dossier for ${ticker}...`} />
  if (error) return <ErrorMessage message={String(error)} />
  if (!data) return null

  const approvalSummaryData = approvalSummary.data
  const approvalCount = approvalSummaryData?.count ?? 0
  const approvalItems = approvalSummaryData?.items ?? []
  const approvalSummaryInitialLoading = approvalSummary.isPending && !approvalSummaryData
  const approvalSummaryError = approvalSummary.error
  const visibleTabs: Tab[] = isEquity ? ["Overview", "Management Quality", "Valuation", ...BASE_TABS] : [...BASE_TABS]
  const activeTab: Tab = (tab === "Overview" || tab === "Management Quality" || tab === "Valuation") && !isEquity ? "Thesis" : tab
  const pos = data.position
  const thesisClaims = Array.isArray(data.thesis_claims) ? data.thesis_claims : []
  const catalysts = Array.isArray(data.catalysts) ? data.catalysts : []
  const killConditions = Array.isArray(data.kill_conditions) ? data.kill_conditions : []
  const evaluations = Array.isArray(data.evaluations) ? data.evaluations : []
  const workflowRuns = Array.isArray(data.workflow_runs) ? data.workflow_runs : []
  const positionQuantity = pos?.quantity ?? pos?.shares
  const positionQuantityLabel = pos?.instrument_type === "future" ? "Contracts" : "Quantity"
  const hasPositionSummary = Boolean(
    pos && (
      positionQuantity != null ||
      (pos.instrument_type === "future" && pos.contract_multiplier != null) ||
      pos.group_name != null ||
      pos.avg_cost != null ||
      pos.market_value != null ||
      pos.pnl_pct != null ||
      pos.weight != null
    ),
  )
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
            {isEquity && <ManagementQualityUpload ticker={ticker!} hasContent={!!data.management_quality?.content} />}
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
              Propose Status Change
            </button>
          )}
          <RefreshButton queryKeys={[["dossier", ticker]]} />
        </div>
      </div>

      {/* Position summary bar */}
      {hasPositionSummary && pos && (
        <div className="theme-surface rounded-xl p-3 mb-4 flex flex-wrap gap-6 text-sm">
          {positionQuantity != null && <div><span className="text-subtle">{positionQuantityLabel}</span> <span className="font-medium text-app ml-1">{String(positionQuantity)}</span></div>}
          {pos.instrument_type === "future" && pos.contract_multiplier != null && <div><span className="text-subtle">Multiplier</span> <span className="font-medium text-app ml-1">{String(pos.contract_multiplier)}</span></div>}
          {pos.group_name != null && <div><span className="text-subtle">Group</span> <span className="font-medium text-app ml-1">{String(pos.group_name)}{pos.group_conviction != null ? ` (${String(pos.group_conviction)})` : ""}</span></div>}
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

      {lastProposal && (
        <StagedProposalNotice proposal={lastProposal} className="mb-4 rounded-xl px-4 py-3">
          staged for {subjectLabel(lastProposal.entity_type)}. It will not change app state until approved and applied.
        </StagedProposalNotice>
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
            {t === "Claims" && thesisClaims.length > 0 && <span className="ml-1 text-xs text-subtle">({thesisClaims.length})</span>}
            {t === "Catalysts" && catalysts.length > 0 && <span className="ml-1 text-xs text-subtle">({catalysts.length})</span>}
            {t === "Kill Conditions" && killConditions.length > 0 && <span className="ml-1 text-xs text-subtle">({killConditions.length})</span>}
            {t === "Evaluations" && evaluations.length > 0 && <span className="ml-1 text-xs text-subtle">({evaluations.length})</span>}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="theme-surface rounded-xl p-4">
        {activeTab === "Overview" && <OverviewTab content={data.overview_content} parsed={data.overview_parsed} ticker={data.ticker} />}
        {activeTab === "Management Quality" && (
          <ManagementQualityTab
            content={data.management_quality?.content ?? null}
            parsed={data.management_quality?.parsed ?? null}
            ticker={data.ticker}
          />
        )}
        {activeTab === "Valuation" && <PositionValuationTab ticker={data.ticker} />}
        {activeTab === "Thesis" && <ThesisTab thesis={data.thesis} ticker={data.ticker} position={data.position} />}
        {activeTab === "Claims" && <ClaimsTab claims={thesisClaims} catalysts={catalysts} conditions={killConditions} ticker={ticker!} />}
        {activeTab === "Catalysts" && <CatalystsTab catalysts={catalysts} ticker={ticker!} />}
        {activeTab === "Kill Conditions" && <KillConditionsTab conditions={killConditions} ticker={ticker!} />}
        {activeTab === "Evaluations" && <EvaluationsTab evaluations={evaluations} />}
        {activeTab === "Risk" && <RiskTab ticker={data.ticker} />}
        {activeTab === "Workflows" && <WorkflowsTab runs={workflowRuns} />}
      </div>

      {/* Pending Approvals for this ticker */}
      {(approvalSummaryInitialLoading || approvalSummaryError || approvalCount > 0) && (
        <section className="mt-6 theme-surface rounded-xl p-4">
          <h2 className="text-sm font-semibold text-app mb-3">
            Pending Approvals ({approvalSummaryInitialLoading ? "loading" : approvalCount})
          </h2>
          <div className="space-y-2 max-h-[400px] overflow-y-auto">
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
              return (
                <div key={a.id} className="rounded-lg px-3 py-2 text-sm border border-app">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <span className="text-xs text-subtle">{subjectLabel(a.entity_type)}</span>
                      <div className="mt-1 flex flex-wrap gap-2">
                        <DecisionStateBadge state={approvalDecisionState(a)} />
                        <BaseStateBadge state={a.base_state_status} message={a.base_state_message} />
                        <EffectScopeBadge scope={a.effect_scope ?? "internal_state"} />
                        <PolicyStateBadge state={a.policy_state ?? a.policy_gate?.decision ?? "missing"} />
                        <QualityStateBadge state={a.quality_state ?? "missing"} />
                      </div>
                      {a.reason && (
                        <p onClick={() => toggleExpanded(key)} className={cn("text-xs text-muted mt-0.5 cursor-pointer", !expanded && "line-clamp-1")}>
                          {a.reason}
                        </p>
                      )}
                      {a.application_error && (
                        <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">Application failed: {a.application_error}</p>
                      )}
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      <button
                        onClick={() => openApprovalReview(a, "approve")}
                        disabled={processingIds.has(a.id) || a.can_approve === false}
                        className="rounded px-2 py-1 text-xs font-medium text-green-700 bg-green-50 hover:bg-green-100 dark:text-green-400 dark:bg-green-950 disabled:opacity-50"
                        title={a.base_state_status === "stale" ? a.base_state_message || "The underlying state changed." : "Review and apply internal state change"}
                      >
                        {a.can_retry_apply ? "Retry Apply" : "Approve & Apply"}
                      </button>
                      {a.can_restage && (
                        <button
                          type="button"
                          onClick={() => handleRejectAndRestage(a)}
                          disabled={processingIds.has(a.id)}
                          className="rounded px-2 py-1 text-xs font-medium text-amber-700 bg-amber-50 hover:bg-amber-100 dark:text-amber-300 dark:bg-amber-950 disabled:opacity-50"
                          title="Reject this stale proposal and create a fresh proposal from current state"
                        >
                          Reject & Restage
                        </button>
                      )}
                      <button
                        onClick={() => openApprovalReview(a, "reject")}
                        disabled={processingIds.has(a.id) || a.can_reject === false}
                        className="rounded px-2 py-1 text-xs font-medium text-red-700 bg-red-50 hover:bg-red-100 dark:text-red-400 dark:bg-red-950 disabled:opacity-50"
                      >
                        Reject Proposal
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
              <h2 className="text-sm font-semibold text-app mb-3">Internal Action Items</h2>
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

      <Dialog
        open={approvalReview !== null}
        onOpenChange={open => {
          if (!open) {
            setApprovalReview(null)
            setApprovalNote("")
            setApprovalError(null)
          }
        }}
        title={approvalReview?.action === "approve" ? "Approve And Apply Internal State" : "Reject Proposal"}
        description={
          approvalReview?.action === "approve"
            ? "Approval applies the staged internal state change."
            : "Rejecting leaves the proposal in audit history and does not apply the staged change."
        }
        maxWidth="max-w-3xl"
      >
        {approvalReview && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <DecisionStateBadge state={approvalDecisionState(approvalReview.approval)} />
              <BaseStateBadge
                state={approvalReview.approval.base_state_status}
                message={approvalReview.approval.base_state_message}
              />
              <EffectScopeBadge scope={approvalReview.approval.effect_scope ?? "internal_state"} />
              <PolicyStateBadge state={approvalReview.approval.policy_state ?? approvalReview.approval.policy_gate?.decision ?? "missing"} />
              <QualityStateBadge state={approvalReview.approval.quality_state ?? "missing"} />
            </div>
            <div className="rounded-lg border border-app bg-[hsl(var(--muted-2))] p-3 text-xs text-muted">
              <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1">
                <span>{formatApprovalDisplayLabel(approvalReview.approval.id, "Approval")}</span>
                <span>{subjectLabel(approvalReview.approval.entity_type)}</span>
                <span>
                  Application: {approvalReview.approval.base_state_status === "stale" ? "state changed" : approvalReview.approval.application_status || "pending"}
                </span>
              </div>
              {approvalReview.approval.reason && <p className="mb-2">{approvalReview.approval.reason}</p>}
              <ApprovalChangeSummary approval={approvalReview.approval} />
            </div>
            {approvalReview.approval.base_state_status === "stale" && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
                {approvalReview.approval.base_state_message || "The underlying state changed after this proposal was created."}
              </div>
            )}
            <div>
              <label htmlFor="dossier-approval-note" className="theme-field-label">
                {approvalReview.action === "approve" ? "Approval note" : "Rejection note"}
              </label>
              <textarea
                id="dossier-approval-note"
                value={approvalNote}
                onChange={e => setApprovalNote(e.target.value)}
                className="theme-input mt-1 min-h-[90px] w-full"
                placeholder={approvalReview.action === "approve" ? "State why this internal change is approved." : "Optional reason for rejecting this proposal."}
              />
              {approvalReview.action === "approve" && (
                <p className="theme-field-caption mt-1">Required. Approval applies app state only.</p>
              )}
            </div>
            {approvalError && (
              <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                {approvalError}
              </div>
            )}
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setApprovalReview(null)}
                className="rounded-lg border border-app px-3 py-2 text-sm font-medium text-muted hover:text-app"
              >
                Cancel
              </button>
              <ActionButton
                onClick={() => handleApproval(approvalReview.approval, approvalReview.action, approvalNote)}
                loading={processingIds.has(approvalReview.approval.id)}
                loadingText={approvalReview.action === "approve" ? "Applying..." : "Rejecting..."}
                disabled={
                  approvalReview.action === "approve" &&
                  (!approvalNote.trim() || approvalReview.approval.can_approve === false)
                }
                className={cn(
                  "w-auto px-4",
                  approvalReview.action === "approve" ? "theme-button-success" : "theme-button-destructive",
                )}
              >
                {approvalReview.action === "approve" ? "Approve And Apply Internal State" : "Reject Proposal"}
              </ActionButton>
              {approvalReview.approval.can_restage && (
                <ActionButton
                  onClick={() => handleRejectAndRestage(approvalReview.approval, approvalNote)}
                  loading={processingIds.has(approvalReview.approval.id)}
                  loadingText="Restaging..."
                  className="w-auto px-4 bg-amber-600 hover:bg-amber-700"
                >
                  Reject & Restage
                </ActionButton>
              )}
            </div>
          </div>
        )}
      </Dialog>

      {/* Status change dialog */}
      <Dialog
        open={statusDialogOpen}
        onOpenChange={setStatusDialogOpen}
        title="Propose Thesis Status Change"
        description={`Stage a status change proposal for ${ticker}. Approval is required before internal state changes.`}
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
            loadingText="Staging proposal..."
          >
            Propose Status Change
          </ActionButton>
        </div>
      </Dialog>
    </div>
  )
}

/* ---------- Sub-tab components ---------- */

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
              <ActionButton onClick={() => saveMutation.mutate()} loading={saveMutation.isPending} loadingText="Saving...">Save Overview</ActionButton>
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
          <ActionButton onClick={() => saveMutation.mutate()} loading={saveMutation.isPending} loadingText="Saving...">Save Overview</ActionButton>
          <button type="button" onClick={() => setEditing(false)} className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors">Cancel</button>
        </div>
      </div>
    )
  }

  return (
    <div>
      <div className="mb-4 flex justify-end">
        <button type="button" onClick={startEdit} className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors">
          Edit
        </button>
      </div>
      <EquityOverviewReadView content={content} parsed={parsed} ticker={ticker} />
    </div>
  )
}

function ManagementRatingBadge({ value }: { value?: string | null }) {
  const rating = cleanDossierDisplayText(value) || "Insufficient evidence"
  const normalized = rating.toLowerCase()
  const className =
    normalized.includes("strong") || normalized.includes("handled well")
      ? "border-green-200 bg-green-50 text-green-700 dark:border-green-900 dark:bg-green-950 dark:text-green-300"
      : normalized.includes("weak") || normalized.includes("poor") || normalized.includes("handled poorly")
        ? "border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300"
        : normalized.includes("mixed") || normalized.includes("too early")
          ? "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300"
          : "border-app bg-[hsl(var(--muted-2))] text-muted"

  return (
    <span className={cn("inline-flex shrink-0 items-center rounded border px-2 py-0.5 text-xs font-semibold", className)}>
      {rating}
    </span>
  )
}

function cleanedManagementBullets(items?: ManagementQualityBullet[] | null): ManagementQualityBullet[] {
  return (items || [])
    .map(item => {
      const textWithoutInlineResponse = item.text.replace(
        /\s*(?:\*\*)?Response(?:\*\*)?:\s*(?:[*_`~]+)?\s*(Handled well|Mixed|Handled poorly|Too early)(?:\s*[\u2014\u2013-]\s*.+)?$/i,
        "",
      )
      const responseRating = cleanDossierDisplayText(item.response_rating)
      const responseText = cleanDossierDisplayText(item.response_text)
      return {
        ...item,
        title: cleanDossierDisplayText(item.title) || null,
        text: cleanDossierDisplayText(textWithoutInlineResponse),
        response_rating: responseRating || undefined,
        response_text: responseText || null,
      }
    })
    .filter(item => {
      const title = (item.title || "").trim()
      const text = item.text.trim()
      if (title || !["", "-", "--", "\u2014", "\u2013"].includes(text)) return true
      return Boolean(item.response_rating || item.response_text)
    })
}

function ManagementBulletList({ items }: { items: ManagementQualityBullet[] }) {
  const visibleItems = cleanedManagementBullets(items)
  if (!visibleItems.length) return null

  return (
    <div className="space-y-2">
      {visibleItems.map((item, index) => (
        <div key={`${item.title || "item"}-${index}`} className="rounded-lg border border-app px-3 py-2 text-sm">
          <div className="flex flex-wrap items-center gap-2">
            {item.title && <h4 className="font-semibold text-app">{item.title}</h4>}
            {item.response_rating && <ManagementRatingBadge value={item.response_rating} />}
          </div>
          {item.text && <p className="mt-1 text-muted">{item.text}</p>}
          {item.response_text && <p className="mt-1 text-muted">Response: {item.response_text}</p>}
        </div>
      ))}
    </div>
  )
}

function ManagementQualityTab({
  content,
  parsed,
  ticker,
}: {
  content: string | null
  parsed: ParsedManagementQuality | null
  ticker: string
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState("")
  const [proposal, setProposal] = useState<StagedMutationResponse | null>(null)
  const qc = useQueryClient()
  const saveMutation = useMutation({
    mutationFn: () => saveManagementQualityContent(ticker, draft),
    onSuccess: result => {
      setProposal(result as StagedMutationResponse)
      void invalidateApprovalSummaries(qc)
      qc.invalidateQueries({ queryKey: ["dossier", ticker] })
      setEditing(false)
    },
  })

  const startEdit = () => {
    setDraft(content ?? "")
    setEditing(true)
  }

  const summary = parsed?.summary ?? null
  const summaryCards = [
    { label: "Overall", rating: summary?.overall_rating, text: summary?.bottom_line },
    { label: "Owner Mindset", rating: summary?.owner_mindset?.rating, text: summary?.owner_mindset?.text },
    {
      label: "Business Value",
      rating: summary?.business_value_understanding?.rating,
      text: summary?.business_value_understanding?.text,
    },
    { label: "Follow-through", rating: summary?.follow_through?.rating, text: summary?.follow_through?.text },
  ]
  const accomplishments = cleanedManagementBullets(parsed?.accomplishments)
  const setbacks = cleanedManagementBullets(parsed?.setbacks)

  if (!content) {
    return (
      <div>
        <StagedProposalNotice proposal={proposal} className="mb-3 text-xs" />
        <p className="mb-3 text-sm text-muted">No management quality assessment on file for this position.</p>
        <button type="button" onClick={startEdit} className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors">
          Draft Management Quality Proposal
        </button>
        {editing && (
          <div className="mt-3">
            <textarea
              value={draft}
              onChange={e => setDraft(e.target.value)}
              className="w-full min-h-[320px] rounded-lg border border-app bg-transparent p-3 text-sm text-app font-mono focus:outline-none focus:ring-1 focus:ring-blue-500"
              placeholder={"# TICKER Management Quality\n\n## Executive Summary\n\n## Management Scorecard\n\n## Most Impressive Accomplishments\n\n## Biggest Setbacks and Responses"}
            />
            {saveMutation.isError && <p className="mt-1 text-xs text-red-600">{String(saveMutation.error)}</p>}
            <div className="mt-2 flex gap-2">
              <ActionButton onClick={() => saveMutation.mutate()} loading={saveMutation.isPending} loadingText="Staging proposal...">Submit Proposed Assessment</ActionButton>
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
        <StagedProposalNotice proposal={proposal} className="mb-3 text-xs" />
        <textarea
          value={draft}
          onChange={e => setDraft(e.target.value)}
          className="w-full min-h-[420px] rounded-lg border border-app bg-transparent p-3 text-sm text-app font-mono focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        {saveMutation.isError && <p className="mt-1 text-xs text-red-600">{String(saveMutation.error)}</p>}
        <div className="mt-2 flex gap-2">
          <ActionButton onClick={() => saveMutation.mutate()} loading={saveMutation.isPending} loadingText="Staging proposal...">Submit Proposed Assessment</ActionButton>
          <button type="button" onClick={() => setEditing(false)} className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors">Cancel</button>
        </div>
      </div>
    )
  }

  return (
    <div>
      <StagedProposalNotice proposal={proposal} className="mb-3 text-xs" />
      <div className="mb-4 flex justify-end">
        <button type="button" onClick={startEdit} className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors">
          Edit
        </button>
      </div>

      {summary && (
        <section className="mb-5">
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-4">
            {summaryCards.map(card => (
              <div key={card.label} className="rounded-lg border border-app px-3 py-3">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <h3 className="text-xs font-semibold uppercase text-subtle">{card.label}</h3>
                  <ManagementRatingBadge value={card.rating} />
                </div>
                {card.text && <p className="text-sm leading-6 text-muted">{card.text}</p>}
              </div>
            ))}
          </div>
        </section>
      )}

      {parsed?.scorecard && (
        <section className="mb-5">
          <h3 className="mb-2 text-sm font-semibold text-app">Management Scorecard</h3>
          <div className="overflow-x-auto rounded-lg border border-app">
            <table className="min-w-full text-left text-sm">
              <thead className="border-b border-app text-xs uppercase text-subtle">
                <tr>
                  <th className="px-3 py-2 font-semibold">Question</th>
                  <th className="px-3 py-2 font-semibold">Rating</th>
                  <th className="px-3 py-2 font-semibold">Evidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[hsl(var(--border))]">
                {parsed.scorecard.map(row => (
                  <tr key={row.question}>
                    <td className="px-3 py-2 text-app">{row.question}</td>
                    <td className="px-3 py-2"><ManagementRatingBadge value={row.rating} /></td>
                    <td className="px-3 py-2 text-muted">{row.evidence}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {accomplishments.length > 0 && (
        <section className="mb-5">
          <h3 className="mb-2 text-sm font-semibold text-app">Most Impressive Accomplishments</h3>
          <ManagementBulletList items={accomplishments} />
        </section>
      )}

      {setbacks.length > 0 && (
        <section className="mb-5">
          <h3 className="mb-2 text-sm font-semibold text-app">Biggest Setbacks And Responses</h3>
          <ManagementBulletList items={setbacks} />
        </section>
      )}

      <section className="border-t border-app pt-4">
        <h3 className="mb-2 text-sm font-semibold text-app">Full Assessment</h3>
        <div className="prose prose-sm dark:prose-invert max-w-none">
          <MarkdownRenderer content={content} />
        </div>
      </section>
    </div>
  )
}

function ThesisTab({ thesis, ticker, position }: { thesis: DossierData["thesis"]; ticker: string; position: Record<string, unknown> | null }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState("")
  const [proposal, setProposal] = useState<StagedMutationResponse | null>(null)
  const qc = useQueryClient()
  const saveMutation = useMutation({
    mutationFn: () => saveThesisContent(ticker, draft),
    onSuccess: result => {
      setProposal(result as StagedMutationResponse)
      void invalidateApprovalSummaries(qc)
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
        <StagedProposalNotice proposal={proposal} className="mb-3 text-xs" />
        <p className="text-sm text-muted mb-3">No thesis on file for this position.</p>
        <button type="button" onClick={startEdit} className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors">
          Draft Thesis Proposal
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
              <ActionButton onClick={() => saveMutation.mutate()} loading={saveMutation.isPending} loadingText="Staging proposal...">Submit Proposed Thesis</ActionButton>
              <button type="button" onClick={() => setEditing(false)} className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors">Cancel</button>
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div>
      <StagedProposalNotice proposal={proposal} className="mb-3 text-xs" />
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
            <ActionButton onClick={() => saveMutation.mutate()} loading={saveMutation.isPending} loadingText="Staging proposal...">Submit Proposed Thesis</ActionButton>
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
  const [proposal, setProposal] = useState<StagedMutationResponse | null>(null)
  const qc = useQueryClient()
  const mutation = useMutation({
    mutationFn: (next: ClaimDraft) => {
      const payload = draftToPayload(next, ticker)
      if (next.id) return updateThesisClaim(next.id, payload)
      return createThesisClaim({ ...payload, ticker, claim: payload.claim })
    },
    onSuccess: result => {
      setProposal(result as StagedMutationResponse)
      void invalidateApprovalSummaries(qc)
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

  const catalystLabelById = new Map(catalysts.map(c => [c.id, textOrFallback(c.description, "Untitled catalyst").split(": ", 1)[0]]))
  const conditionLabelById = new Map(conditions.map(k => [k.id, textOrFallback(k.condition, "Untitled condition").split(": ", 1)[0]]))

  return (
    <div className="space-y-4">
      <StagedProposalNotice proposal={proposal} className="text-xs" />
      <div className="flex justify-end">
        <button
          type="button"
          onClick={() => setDraft(blankClaimDraft())}
          className="rounded-lg border border-app px-3 py-1.5 text-sm font-medium text-muted hover:text-app transition-colors"
        >
          Draft Claim Proposal
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
              items={catalysts.flatMap(c => typeof c.id === "number" ? [{ id: c.id, label: textOrFallback(c.description, "Untitled catalyst") }] : [])}
              selectedIds={draft.linked_catalyst_ids}
              onToggle={id => toggleLinked("catalyst", id)}
            />
            <LinkCheckboxes
              title="Linked Kill Conditions"
              items={conditions.flatMap(k => typeof k.id === "number" ? [{ id: k.id, label: textOrFallback(k.condition, "Untitled condition") }] : [])}
              selectedIds={draft.linked_kill_condition_ids}
              onToggle={id => toggleLinked("kill", id)}
            />
          </div>

          {mutation.isError && <p className="text-xs text-red-600">{errorMessage(mutation.error)}</p>}
          <div className="flex gap-2">
            <ActionButton
              onClick={() => draft.claim.trim() && mutation.mutate(draft)}
              loading={mutation.isPending}
              loadingText="Staging proposal..."
              disabled={!draft.claim.trim()}
              className="w-auto px-4"
            >
              Submit Proposed Claim
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
  const [openMenuId, setOpenMenuId] = useState<EntityId | null>(null)
  const [proposal, setProposal] = useState<StagedMutationResponse | null>(null)
  const qc = useQueryClient()
  const mutation = useMutation({
    mutationFn: ({ id, status }: { id: EntityId; status: string }) => updateCatalystStatus(id, status),
    onSuccess: result => {
      setProposal(result as StagedMutationResponse)
      void invalidateApprovalSummaries(qc)
      qc.invalidateQueries({ queryKey: ["dossier", ticker] })
      setOpenMenuId(null)
    },
  })

  if (!catalysts.length) return <p className="text-sm text-muted">No catalysts tracked.</p>
  const statusOptions = ["pending", "played_out", "failed", "superseded"]
  return (
    <div className="space-y-3">
      <StagedProposalNotice proposal={proposal} className="text-xs" />
      {catalysts.map(c => {
        const status = statusOrFallback(c.status, "pending")
        return (
          <div key={c.id} className="rounded-lg border border-app px-4 py-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm font-medium text-app">{textOrFallback(c.description, "Untitled catalyst")}</span>
              <button
                type="button"
                onClick={() => setOpenMenuId(openMenuId === c.id ? null : c.id)}
                className={cn("text-xs px-1.5 py-0.5 rounded font-medium cursor-pointer hover:ring-1 hover:ring-blue-300", STATUS_COLORS[status] ?? "")}
              >
                {status.replace(/_/g, " ")}
              </button>
            </div>
            {openMenuId === c.id && (
              <div className="flex flex-wrap gap-1.5 mt-2 mb-1">
                {statusOptions.filter(s => s !== status).map(s => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => mutation.mutate({ id: c.id, status: s })}
                    disabled={mutation.isPending}
                    className={cn("text-xs px-1.5 py-0.5 rounded font-medium transition-colors hover:ring-1 hover:ring-gray-300", STATUS_COLORS[s] ?? "")}
                  >
                    Propose {s.replace(/_/g, " ")}
                  </button>
                ))}
              </div>
            )}
            <div className="flex gap-3 text-xs text-subtle">
              <span>{textOrFallback(c.category, "uncategorized")}</span>
              {c.target_date && <span>Target: {c.target_date}</span>}
            </div>
            {c.evidence && <p className="text-xs text-muted mt-1">{c.evidence}</p>}
          </div>
        )
      })}
    </div>
  )
}

function KillConditionsTab({ conditions, ticker }: { conditions: KillCondition[]; ticker: string }) {
  const [openMenuId, setOpenMenuId] = useState<EntityId | null>(null)
  const [proposal, setProposal] = useState<StagedMutationResponse | null>(null)
  const qc = useQueryClient()
  const mutation = useMutation({
    mutationFn: ({ id, status }: { id: EntityId; status: string }) => updateKillConditionStatus(id, status),
    onSuccess: result => {
      setProposal(result as StagedMutationResponse)
      void invalidateApprovalSummaries(qc)
      qc.invalidateQueries({ queryKey: ["dossier", ticker] })
      setOpenMenuId(null)
    },
  })

  if (!conditions.length) return <p className="text-sm text-muted">No kill conditions defined.</p>
  const statusOptions = ["active", "triggered", "retired"]
  return (
    <div className="space-y-3">
      <StagedProposalNotice proposal={proposal} className="text-xs" />
      {conditions.map(k => {
        const status = statusOrFallback(k.status, "active")
        return (
          <div key={k.id} className={cn("rounded-lg border px-4 py-3", status === "triggered" ? "border-red-300 bg-red-50/50 dark:border-red-800 dark:bg-red-950/30" : "border-app")}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm font-medium text-app">{textOrFallback(k.condition, "Untitled condition")}</span>
              <button
                type="button"
                onClick={() => setOpenMenuId(openMenuId === k.id ? null : k.id)}
                className={cn("text-xs px-1.5 py-0.5 rounded font-medium cursor-pointer hover:ring-1 hover:ring-blue-300", STATUS_COLORS[status] ?? "")}
              >
                {status}
              </button>
            </div>
            {openMenuId === k.id && (
              <div className="flex flex-wrap gap-1.5 mt-2 mb-1">
                {statusOptions.filter(s => s !== status).map(s => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => mutation.mutate({ id: k.id, status: s })}
                    disabled={mutation.isPending}
                    className={cn("text-xs px-1.5 py-0.5 rounded font-medium transition-colors hover:ring-1 hover:ring-gray-300", STATUS_COLORS[s] ?? "")}
                  >
                    Propose {s}
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
        )
      })}
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
              <DecisionStateBadge state="analysis" />
              <span className={cn("text-xs px-1.5 py-0.5 rounded font-medium",
                ev.action === "hold" ? "text-green-600 bg-green-50 dark:text-green-400 dark:bg-green-950" :
                ev.action === "exit" || ev.action === "reduce" ? "text-red-600 bg-red-50 dark:text-red-400 dark:bg-red-950" :
                "text-blue-600 bg-blue-50 dark:text-blue-400 dark:bg-blue-950"
              )}>Evaluation: {ev.action}</span>
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
  const dateOnly = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value)
  if (dateOnly) {
    const [, year, month, day] = dateOnly
    return new Date(Number(year), Number(month) - 1, Number(day)).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    })
  }
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
  if (s === "partial" || s === "stale") return "text-amber-700 bg-amber-50 dark:text-amber-400 dark:bg-amber-950"
  return "text-red-700 bg-red-50 dark:text-red-400 dark:bg-red-950"
}

function evidenceTitle(ev: PositionRiskEvidence): string {
  return ev.name || ev.component || ev.source || "Risk driver"
}

function refreshCacheStatus(snapshot: PositionRiskSnapshot | null | undefined): string | null {
  const meta = snapshot?._meta
  if (!meta || typeof meta !== "object" || Array.isArray(meta)) return null
  const status = (meta as Record<string, unknown>).cache_status
  return typeof status === "string" ? status : null
}

function RiskTab({ ticker }: { ticker: string }) {
  const qc = useQueryClient()
  const [elapsed, setElapsed] = useState(0)
  const tickerUpper = ticker.toUpperCase()
  const riskQueryKey = ["position-risk", tickerUpper] as const

  const latestRiskQuery = useQuery({
    queryKey: riskQueryKey,
    queryFn: () => fetchPositionRiskLatest(tickerUpper),
    staleTime: 60 * 1000,
    retry: 1,
  })

  const refreshMutation = useMutation({
    mutationFn: () => refreshPositionRisk(tickerUpper),
    onSuccess: snapshot => {
      qc.setQueryData<PositionRiskSnapshot | null>(riskQueryKey, snapshot)
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

  const snapshot: PositionRiskSnapshot | null | undefined = refreshMutation.data ?? latestRiskQuery.data
  const evidence = Array.isArray(snapshot?.evidence) ? snapshot.evidence : []
  const sourceStatus = snapshot?.source_status ?? {}
  const moduleRows = Object.entries(sourceStatus).sort(([a], [b]) => a.localeCompare(b))
  const requiredIssues = moduleRows.filter(([, state]) =>
    state?.required && (String(state?.status ?? "error").toLowerCase() !== "ok" || state.accepted === false)
  )
  const optionalIssues = moduleRows.filter(([, state]) =>
    !state?.required && (String(state?.status ?? "error").toLowerCase() !== "ok" || state.accepted === false)
  )
  const requiredHealth = requiredIssues.length === 0
    ? "OK"
    : `${requiredIssues.length} issue${requiredIssues.length === 1 ? "" : "s"}`
  const primaryError = refreshMutation.error ?? (!snapshot ? latestRiskQuery.error : null)
  const noSnapshot = !latestRiskQuery.isLoading && !snapshot && !refreshMutation.data
  const qualityState = snapshot?.quality === "ok" ? "ok" : snapshot ? "degraded" : "missing"
  const marketAsOf = snapshot?.market_snapshot_as_of ?? snapshot?.as_of
  const cacheStatus = refreshCacheStatus(refreshMutation.data)

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-sm font-semibold text-app">Risk Analysis Snapshot</h2>
          <p className="text-xs text-subtle">
            {snapshot ? `Computed ${formatDateTime(snapshot.computed_at)}` : "No persisted snapshot"}
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            <DecisionStateBadge state="analysis" />
            <EffectScopeBadge scope="read_only" />
            <QualityStateBadge state={qualityState} />
          </div>
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

      {refreshMutation.isPending && <LoadingSpinner message={`Refreshing risk... (${elapsed}s elapsed)`} />}
      {!snapshot && latestRiskQuery.isLoading && <LoadingSpinner message="Loading latest risk snapshot..." />}

      {primaryError && <ErrorMessage message={errorMessage(primaryError)} />}
      {refreshMutation.isSuccess && (
        <div className="rounded-lg border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700 dark:border-green-900 dark:bg-green-950 dark:text-green-400">
          {cacheStatus === "hit"
            ? `Risk cache is still fresh; using market snapshot from ${formatDateTime(marketAsOf)}.`
            : `Risk refreshed using market snapshot from ${formatDateTime(marketAsOf)}.`}
        </div>
      )}

      {snapshot && (requiredIssues.length > 0 || optionalIssues.length > 0) && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300">
          {requiredIssues.length > 0
            ? `${requiredIssues.length} required module${requiredIssues.length === 1 ? "" : "s"} degraded.`
            : `${optionalIssues.length} optional module${optionalIssues.length === 1 ? "" : "s"} unavailable.`}
          {" "}Score is shown with reduced confidence.
        </div>
      )}

      {noSnapshot && (
        <div className="rounded-lg border border-app px-4 py-3">
          <p className="text-sm font-medium text-app">No risk snapshot yet for {tickerUpper}.</p>
          <p className="mt-1 text-xs text-muted">Use Refresh Risk to compute a position risk snapshot.</p>
        </div>
      )}

      {snapshot && (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
            <div className="rounded-lg border border-app px-4 py-3">
              <p className="text-xs text-subtle">Risk Level</p>
              <span className={cn("mt-2 inline-flex rounded px-2 py-0.5 text-xs font-medium", riskLevelClass(snapshot.risk_level))}>
                {String(snapshot.risk_level ?? "unknown")}
              </span>
            </div>
            <div className="rounded-lg border border-app px-4 py-3">
              <p className="text-xs text-subtle">Risk Score</p>
              <p className="mt-1 text-xl font-semibold text-app">{formatNumber(snapshot.risk_score)}</p>
            </div>
            <div className="rounded-lg border border-app px-4 py-3">
              <p className="text-xs text-subtle">Required Modules</p>
              <p className="mt-1 text-xl font-semibold text-app">{requiredHealth}</p>
            </div>
            <div className="rounded-lg border border-app px-4 py-3">
              <p className="text-xs text-subtle">Confidence</p>
              <p className="mt-1 text-xl font-semibold text-app">{formatConfidence(snapshot.confidence)}</p>
            </div>
          </div>

          <div className="rounded-lg border border-app px-4 py-3">
            <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-subtle">
              <span>{snapshot.asset ?? "unknown"} asset</span>
              <span>{snapshot.direction ?? "unknown"}</span>
              {snapshot.position?.group_name != null && <span>Group {String(snapshot.position.group_name)}{snapshot.position.group_conviction != null ? ` (${String(snapshot.position.group_conviction)})` : ""}</span>}
              <span>{snapshot.sector ?? "Unknown sector"}</span>
              <span>Market snapshot {formatDateTime(marketAsOf)}</span>
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
                  <div key={module} className="rounded border border-app px-3 py-2 text-sm">
                    <div className="flex items-center justify-between gap-3">
                      <span className="min-w-0 truncate text-muted">{module}</span>
                      <span className={cn("shrink-0 rounded px-1.5 py-0.5 text-xs font-medium", moduleStatusClass(state?.status))}>
                        {state?.status ?? "error"}
                      </span>
                    </div>
                    <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-subtle">
                      <span>{state?.required ? "required" : "optional"}</span>
                      {state?.freshness?.observed_as_of_date && <span>as of {state.freshness.observed_as_of_date}</span>}
                      {state?.freshness?.fresh === false && <span>stale</span>}
                      {state?.fallback_used && <span>fallback used</span>}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted">No module status available.</p>
            )}
          </div>
        </>
      )}

      {snapshot && (
        <details className="rounded-lg border border-app px-4 py-3">
          <summary className="cursor-pointer text-sm font-medium text-app">Raw risk JSON</summary>
          <pre className="mt-3 max-h-[500px] overflow-auto whitespace-pre-wrap text-xs text-muted">
            {JSON.stringify(snapshot, null, 2)}
          </pre>
        </details>
      )}
    </div>
  )
}

function WorkflowsTab({ runs }: { runs: WorkflowRun[] }) {
  if (!runs.length) return <p className="text-sm text-muted">No workflow runs recorded.</p>
  return (
    <div className="space-y-2">
      {runs.map(run => (
        <div key={run.run_id || `${workflowRunLabel(run)}-${run.started_at ?? run.completed_at ?? "unknown"}`} className="flex items-center justify-between rounded-lg border border-app px-4 py-3 text-sm">
          <div className="flex items-center gap-3">
            <span className={cn("h-2 w-2 shrink-0 rounded-full", workflowStatusClass(run.status))} />
            <span className="font-medium text-app">{workflowRunLabel(run)}</span>
          </div>
          <div className="flex items-center gap-3 text-xs text-subtle">
            <span>{run.status ?? "unknown"}</span>
            <span>{formatTime(run.started_at ?? run.completed_at)}</span>
          </div>
        </div>
      ))}
    </div>
  )
}
