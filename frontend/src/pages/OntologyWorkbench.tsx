import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { GitBranch, Info } from "lucide-react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { useApiQuery } from "@/hooks/useApiQuery"
import {
  fetchOntologyRuns,
  createMissionDefinition,
  createMonitorDefinition,
  disableMissionDefinition,
  disableMonitorDefinition,
  fetchMonitorBuilderDefinitions,
  previewMonitorDefinition,
  runMonitorBuilderDefinitions,
  runOntologyQueryAsync,
  type MissionDefinitionRecord,
  type MonitorDefinitionBody,
  type MonitorDefinitionRecord,
  type OntologyEvidence,
  type OntologyResponse,
  type OntologyRunSummary,
  type ProvenanceSelector,
} from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { ActionButton, SelectInput, SegmentedControl, TextInput, Toggle } from "@/components/shared/FormControls"
import { ProvenanceTraceDialog } from "@/components/shared/ProvenanceTraceDialog"
import { DecisionStateBadge, EffectScopeBadge, QualityStateBadge } from "@/components/shared/DecisionStateBadge"

type Intent = "auto" | "portfolio_risk_exposure" | "positions_in_deteriorating_macro" | "entity_context"
type Timeframe = "This Week" | "Daily" | "Weekly" | "Monthly"
type TemporalMode = "current" | "historical"

const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

function formatTimestampLabel(rawValue: string | undefined): string {
  const raw = String(rawValue ?? "").trim()
  if (!raw) return "N/A"

  const isoMatch = raw.match(
    /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:(Z)|([+-]\d{2}:\d{2}))?$/,
  )
  if (isoMatch) {
    const [, year, month, day, hour, minute, second, zulu, offset] = isoMatch
    const monthName = MONTH_NAMES[Math.max(0, Number(month) - 1)] ?? month
    const tzLabel = zulu ? " UTC" : offset ? ` UTC${offset}` : ""
    return `${monthName} ${Number(day)}, ${year} ${hour}:${minute}:${second}${tzLabel}`
  }

  const parsed = new Date(raw)
  if (Number.isNaN(parsed.getTime())) return raw
  return parsed.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  })
}

function formatRunOptionLabel(run: OntologyRunSummary): string {
  const asOf = formatTimestampLabel(run.as_of)
  const created = formatTimestampLabel(run.created_at)
  const health = run.required_modules_ok ? "healthy" : "degraded"
  return `${asOf} | created ${created} | ${health}`
}

function parseCsv(text: string): string[] | undefined {
  const vals = text
    .split(",")
    .map(v => v.trim())
    .filter(Boolean)
  return vals.length > 0 ? vals : undefined
}

function datetimeLocalToIso(rawValue: string): string | undefined {
  const raw = rawValue.trim()
  if (!raw) return undefined
  const parsed = new Date(raw)
  if (Number.isNaN(parsed.getTime())) return undefined
  return parsed.toISOString()
}

function summarizeEvidence(evidence: OntologyEvidence[] | undefined): string {
  if (!Array.isArray(evidence) || evidence.length === 0) return "—"
  return evidence
    .slice(0, 2)
    .map(e => {
      const base = [e.source, e.name].filter(Boolean).join(": ")
      if (typeof e.contribution === "number") return `${base} (${e.contribution.toFixed(2)})`
      return base || "signal"
    })
    .join(" | ")
}

const RESULT_COLUMNS: ColumnDef[] = [
  { key: "ticker", header: "Ticker" },
  { key: "asset", header: "Asset" },
  { key: "direction", header: "Direction" },
  { key: "sector", header: "Sector" },
  { key: "risk_score", header: "Risk Score", format: v => typeof v === "number" ? v.toFixed(3) : "0.000" },
  { key: "risk_level", header: "Risk Level" },
  { key: "evidence", header: "Top Evidence", format: v => summarizeEvidence(v as OntologyEvidence[] | undefined) },
]

const STATUS_COLUMNS: ColumnDef[] = [
  { key: "module", header: "Module" },
  {
    key: "status",
    header: "Status",
    colorFn: v => v === "ok" ? "#00c853; font-weight: bold" : "#ff1744; font-weight: bold",
  },
  { key: "detail", header: "Detail" },
]

const BUILDER_TEMPLATES = [
  { value: "thesis_monitor", label: "Thesis monitor", triggerType: "fundamental_news", condition: "Watch for thesis-changing evidence" },
  { value: "risk_mission", label: "Risk mission", triggerType: "macro", condition: "Review risk posture when macro score deteriorates" },
  { value: "catalyst_tracker", label: "Catalyst tracker", triggerType: "news_event", condition: "Track catalyst evidence from source-backed news" },
  { value: "price_threshold", label: "Price threshold", triggerType: "price_level", condition: "Alert when price crosses threshold" },
]

function definitionId(row: MonitorDefinitionRecord | MissionDefinitionRecord): string {
  const typed = row as MonitorDefinitionRecord & MissionDefinitionRecord
  return String(row.object_uid || row.id || typed.monitor_id || typed.mission_id || "")
}

function MonitorMissionBuilder() {
  const queryClient = useQueryClient()
  const [kind, setKind] = useState<"monitor" | "mission">("monitor")
  const [templateId, setTemplateId] = useState("thesis_monitor")
  const [name, setName] = useState("Thesis Evidence Monitor")
  const [ticker, setTicker] = useState("")
  const [condition, setCondition] = useState(BUILDER_TEMPLATES[0].condition)
  const [threshold, setThreshold] = useState("")
  const [sourceName, setSourceName] = useState("trusted_news")
  const [cadence, setCadence] = useState("hourly")
  const [builderError, setBuilderError] = useState<string | null>(null)
  const [preview, setPreview] = useState<Record<string, unknown> | null>(null)

  const definitionsQuery = useApiQuery(["monitor-builder-definitions"], () => fetchMonitorBuilderDefinitions({ status: "active" }), 60 * 1000)

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ["monitor-builder-definitions"] })
  const createMutation = useMutation({
    mutationFn: (body: MonitorDefinitionBody) =>
      kind === "monitor"
        ? createMonitorDefinition(body)
        : createMissionDefinition({
            name: body.name,
            description: body.description,
            template_id: body.template_id,
            mission_type: body.template_id === "risk_mission" ? "risk_review" : "monitor_review",
            scope: body.scope,
            schedule: body.cadence,
            source_requirements: body.source_requirements,
            thresholds: body.thresholds,
            output_policy: body.output_policy,
            approval_behavior: "hit_only_then_human_review",
            reason: body.reason,
          }),
    onSuccess: () => invalidate(),
  })
  const previewMutation = useMutation({
    mutationFn: previewMonitorDefinition,
    onSuccess: result => setPreview(result),
  })
  const runMutation = useMutation({ mutationFn: runMonitorBuilderDefinitions })
  const disableMonitorMutation = useMutation({ mutationFn: (id: string | number) => disableMonitorDefinition(id), onSuccess: () => invalidate() })
  const disableMissionMutation = useMutation({ mutationFn: (id: string | number) => disableMissionDefinition(id), onSuccess: () => invalidate() })

  const selectedTemplate = BUILDER_TEMPLATES.find(item => item.value === templateId) ?? BUILDER_TEMPLATES[0]

  function applyTemplate(value: string) {
    const template = BUILDER_TEMPLATES.find(item => item.value === value) ?? BUILDER_TEMPLATES[0]
    setTemplateId(template.value)
    setCondition(template.condition)
    if (!name.trim() || BUILDER_TEMPLATES.some(item => name === item.label)) setName(template.label)
  }

  function buildBody(): MonitorDefinitionBody {
    const thresholdNumber = Number.parseFloat(threshold)
    return {
      name: name.trim(),
      template_id: templateId,
      scope: { ticker: ticker.trim().toUpperCase() || undefined },
      trigger_type: selectedTemplate.triggerType,
      condition: condition.trim(),
      definition: {
        type: selectedTemplate.triggerType,
        ticker: ticker.trim().toUpperCase() || undefined,
        operator: ">=",
        threshold: Number.isFinite(thresholdNumber) ? thresholdNumber : undefined,
      },
      thresholds: Number.isFinite(thresholdNumber) ? { primary: thresholdNumber } : {},
      source_requirements: sourceName.trim()
        ? [{ source_name: sourceName.trim(), required: true, freshness_days: 7 }]
        : [],
      cadence: { label: cadence },
      severity: "medium",
      output_policy: { safe_mode: true, creates_monitor_hit: true, stages_review_action: true },
      approval_behavior: "hit_only_then_human_review",
      reason: `Create ${kind} from ${selectedTemplate.label} template`,
    }
  }

  async function submit() {
    setBuilderError(null)
    setPreview(null)
    const body = buildBody()
    if (!body.name || !body.condition) {
      setBuilderError("Name and condition are required.")
      return
    }
    await createMutation.mutateAsync(body)
  }

  async function previewCurrent() {
    setBuilderError(null)
    if (kind !== "monitor") {
      setBuilderError("Preview is currently available for monitor definitions. Mission runs are safe-mode scheduled jobs.")
      return
    }
    await previewMutation.mutateAsync(buildBody())
  }

  const monitors = definitionsQuery.data?.monitors ?? []
  const missions = definitionsQuery.data?.missions ?? []

  return (
    <section className="theme-surface mb-6 rounded-xl p-4" aria-label="Low-code monitor and mission builder">
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-sm font-semibold text-app">Monitor And Mission Builder</h2>
          <p className="mt-1 text-xs text-subtle">
            Configure source-backed monitors and missions. Runs record hits and stage review work; no user-visible state changes before approval.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <DecisionStateBadge state="proposal" />
          <EffectScopeBadge scope="internal_state" />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <SegmentedControl
          value={kind}
          onChange={value => setKind(value as "monitor" | "mission")}
          options={[
            { value: "monitor", label: "Monitor" },
            { value: "mission", label: "Mission" },
          ]}
        />
        <SelectInput
          id="builder-template"
          label="Template"
          value={templateId}
          onChange={applyTemplate}
          options={BUILDER_TEMPLATES.map(item => ({ value: item.value, label: item.label }))}
        />
        <TextInput id="builder-cadence" label="Cadence" value={cadence} onChange={setCadence} placeholder="hourly" />
        <TextInput id="builder-name" label="Name" value={name} onChange={setName} placeholder="Monitor name" />
        <TextInput id="builder-ticker" label="Ticker Scope" value={ticker} onChange={setTicker} uppercase placeholder="Optional" />
        <TextInput id="builder-threshold" label="Primary Threshold" value={threshold} onChange={setThreshold} type="number" placeholder="Optional" />
        <TextInput id="builder-source" label="Required Source" value={sourceName} onChange={setSourceName} placeholder="trusted_news" />
        <TextInput
          id="builder-condition"
          label="Condition"
          value={condition}
          onChange={setCondition}
          className="md:col-span-2"
          placeholder="Condition to evaluate"
        />
      </div>

      {builderError && <p className="mt-3 text-xs text-negative">{builderError}</p>}
      {createMutation.isSuccess && (
        <p className="mt-3 text-xs text-subtle">Definition staged for approval. Review it from Workspace before applying.</p>
      )}
      {preview && (
        <pre className="mt-3 max-h-44 overflow-auto rounded-lg border border-app bg-card-muted p-3 text-xs text-muted">
          {JSON.stringify(preview, null, 2)}
        </pre>
      )}

      <div className="mt-4 flex flex-wrap gap-2">
        <ActionButton onClick={submit} loading={createMutation.isPending} loadingText="Staging..." className="w-auto px-4">
          Stage Definition
        </ActionButton>
        <button type="button" onClick={previewCurrent} className="theme-button-secondary rounded-lg px-3 py-2 text-sm font-medium">
          Preview Monitor
        </button>
        <button
          type="button"
          onClick={() => runMutation.mutate({ source: "manual" })}
          className="theme-button-secondary rounded-lg px-3 py-2 text-sm font-medium"
        >
          Run Active Definitions
        </button>
      </div>

      <div className="mt-5 grid gap-3 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-subtle">Active Monitors</h3>
          <div className="space-y-2">
            {monitors.length === 0 && <p className="text-xs text-subtle">No active monitors.</p>}
            {monitors.slice(0, 5).map(row => (
              <div key={definitionId(row)} className="rounded-lg border border-app px-3 py-2 text-xs">
                <div className="flex items-center justify-between gap-3">
                  <span className="font-medium text-app">{row.name}</span>
                  <button
                    type="button"
                    onClick={() => disableMonitorMutation.mutate(definitionId(row))}
                    className="text-subtle hover:text-app"
                  >
                    Propose Disable
                  </button>
                </div>
                <p className="mt-1 text-subtle">{row.condition}</p>
              </div>
            ))}
          </div>
        </div>
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-subtle">Active Missions</h3>
          <div className="space-y-2">
            {missions.length === 0 && <p className="text-xs text-subtle">No active missions.</p>}
            {missions.slice(0, 5).map(row => (
              <div key={definitionId(row)} className="rounded-lg border border-app px-3 py-2 text-xs">
                <div className="flex items-center justify-between gap-3">
                  <span className="font-medium text-app">{row.name}</span>
                  <button
                    type="button"
                    onClick={() => disableMissionMutation.mutate(definitionId(row))}
                    className="text-subtle hover:text-app"
                  >
                    Propose Disable
                  </button>
                </div>
                <p className="mt-1 text-subtle">{row.mission_type || "monitor_review"} | {row.status}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

export function OntologyWorkbench() {
  const [showInfo, setShowInfo] = useState(false)
  const [query, setQuery] = useState("")
  const [intent, setIntent] = useState<Intent>("auto")
  const [timeframe, setTimeframe] = useState<Timeframe>("Daily")
  const [tickers, setTickers] = useState("")
  const [sectors, setSectors] = useState("")
  const [assets, setAssets] = useState("")
  const [minRiskScore, setMinRiskScore] = useState("")
  const [pageSize, setPageSize] = useState("25")
  const [page, setPage] = useState(1)
  const [runId, setRunId] = useState("")
  const [temporalMode, setTemporalMode] = useState<TemporalMode>("current")
  const [asOf, setAsOf] = useState("")
  const [txAsOf, setTxAsOf] = useState("")
  const [includeHistory, setIncludeHistory] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)
  const [cachedResult, setCachedResult] = useState<OntologyResponse | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [provenanceSelector, setProvenanceSelector] = useState<ProvenanceSelector | null>(null)
  const runsListId = "ontology-run-id-suggestions"

  const {
    data: runsData,
    isLoading: runsLoading,
    error: runsError,
  } = useApiQuery(["ontology-runs"], () => fetchOntologyRuns(150), 60 * 1000)

  const mutation = useMutation({
    mutationFn: (body: Parameters<typeof runOntologyQueryAsync>[0]) => {
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller
      return runOntologyQueryAsync(body, controller.signal)
    },
    onSuccess: result => setCachedResult((result as OntologyResponse) ?? null),
  })

  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    if (!mutation.isPending) {
      setElapsed(0)
      return
    }
    const start = Date.now()
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000)
    return () => clearInterval(id)
  }, [mutation.isPending])
  /* eslint-enable react-hooks/set-state-in-effect */

  const handleCancel = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
  }, [])

  const handleIntentChange = useCallback((v: string) => setIntent(v as Intent), [])
  const handleTimeframeChange = useCallback((v: string) => setTimeframe(v as Timeframe), [])
  const handleTemporalModeChange = useCallback((v: TemporalMode) => {
    setTemporalMode(v)
    setFormError(null)
  }, [])

  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    setPage(1)
  }, [query, intent, timeframe, tickers, sectors, assets, minRiskScore, pageSize, runId, temporalMode, asOf, txAsOf, includeHistory])
  /* eslint-enable react-hooks/set-state-in-effect */

  const submitQuery = useCallback((nextPage: number) => {
    let temporalAsOf: string | undefined
    let temporalTxAsOf: string | undefined
    if (temporalMode === "historical") {
      if (runId.trim()) {
        setFormError("Clear Snapshot Run ID before running a historical ontology query.")
        return
      }
      temporalAsOf = datetimeLocalToIso(asOf)
      if (!temporalAsOf) {
        setFormError("Choose a valid As of timestamp before running a historical ontology query.")
        return
      }
      temporalTxAsOf = txAsOf.trim() ? datetimeLocalToIso(txAsOf) : undefined
      if (txAsOf.trim() && !temporalTxAsOf) {
        setFormError("Choose a valid Tx as of timestamp or leave it blank.")
        return
      }
    }

    const filters: Record<string, unknown> = {}
    const parsedTickers = parseCsv(tickers)
    const parsedSectors = parseCsv(sectors)
    const parsedAssets = parseCsv(assets)
    if (parsedTickers) filters.tickers = parsedTickers
    if (parsedSectors) filters.sectors = parsedSectors
    if (parsedAssets) filters.assets = parsedAssets

    const parsedMinRisk = Number.parseFloat(minRiskScore)
    if (Number.isFinite(parsedMinRisk)) filters.min_risk_score = parsedMinRisk

    const parsedPageSize = Number.parseInt(pageSize, 10)
    const safePageSize = Number.isFinite(parsedPageSize) ? Math.min(Math.max(parsedPageSize, 1), 100) : 25
    const safePage = Math.max(1, nextPage)

    setPage(safePage)
    setFormError(null)
    mutation.mutate({
      query: query.trim() || undefined,
      intent: intent === "auto" ? undefined : intent,
      filters: Object.keys(filters).length > 0 ? filters : undefined,
      timeframe,
      run_id: runId.trim() || undefined,
      as_of: temporalAsOf,
      tx_as_of: temporalTxAsOf,
      include_history: temporalMode === "historical" ? includeHistory : undefined,
      include_graph: false,
      refresh_snapshot: false,
      page: safePage,
      page_size: safePageSize,
    })
  }, [query, intent, timeframe, tickers, sectors, assets, minRiskScore, pageSize, runId, temporalMode, asOf, txAsOf, includeHistory, mutation])

  const handleSubmit = useCallback(() => submitQuery(page), [page, submitQuery])

  const data = (mutation.data as OntologyResponse | undefined) ?? cachedResult
  const rows = useMemo(() => (Array.isArray(data?.results) ? data.results : []), [data])
  const pagination = data?._meta?.pagination
  const temporalMeta = data?._meta?.temporal
  const currentPage = pagination?.page ?? page
  const runOptions = useMemo(() => {
    const items = runsData?.runs
    return Array.isArray(items) ? items : []
  }, [runsData?.runs])
  const statusRows = useMemo(
    () =>
      Object.entries(data?.source_status ?? {}).map(([module, state]) => ({
        module,
        status: state?.status || "error",
        detail: state?.detail || "—",
      })),
    [data?.source_status],
  )

  return (
    <div>
      <div className="mb-6">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Ontology Workbench</h1>
          <button
            onClick={() => setShowInfo(v => !v)}
            className="text-gray-300 hover:text-gray-500 transition-colors"
            title="What is this?"
          >
            <Info size={16} />
          </button>
        </div>
        <p className="text-sm text-gray-400 mt-0.5">
          Query materialized portfolio-linked risk snapshots with natural language and structured filters.
        </p>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <DecisionStateBadge state="analysis" />
          <EffectScopeBadge scope="read_only" />
          <span className="text-xs text-gray-500">Risk evidence only. Results do not create proposals or apply portfolio changes.</span>
        </div>
        {showInfo && (
          <p className="text-xs text-gray-500 mt-2 max-w-xl leading-relaxed">
            Links each portfolio position to macro conditions, sector dynamics, and cross-asset signals via a
            materialized semantic/risk graph. Filter by ticker, sector, asset class, or risk threshold to see which
            holdings carry elevated risk and the evidence behind each score. Use snapshot run IDs to compare risk
            states across time.
          </p>
        )}
      </div>

      <MonitorMissionBuilder />

      <div className="theme-surface mb-6 grid grid-cols-1 gap-3 rounded-xl p-4 md:grid-cols-3">
        <TextInput
          label="Natural Language Query"
          value={query}
          onChange={setQuery}
          placeholder="Which positions are in deteriorating macro conditions?"
          className="md:col-span-3"
        />
        <SelectInput
          label="Intent (optional)"
          value={intent}
          onChange={handleIntentChange}
          options={[
            { value: "auto", label: "Auto (from query)" },
            { value: "portfolio_risk_exposure", label: "Portfolio Risk Exposure" },
            { value: "positions_in_deteriorating_macro", label: "Deteriorating Macro" },
            { value: "entity_context", label: "Entity Context" },
          ]}
        />
        <SelectInput
          label="Timeframe"
          value={timeframe}
          onChange={handleTimeframeChange}
          options={[
            { value: "This Week", label: "This Week" },
            { value: "Daily", label: "Daily" },
            { value: "Weekly", label: "Weekly" },
            { value: "Monthly", label: "Monthly" },
          ]}
        />
        <div>
          <label className="mb-1.5 block text-sm text-muted">Snapshot Run ID (optional)</label>
          <input
            type="text"
            value={runId}
            onChange={e => setRunId(e.target.value)}
            placeholder="Type to search runs..."
            list={runsListId}
            className="theme-input w-full"
          />
          <datalist id={runsListId}>
            {runOptions.map(run => (
              <option key={run.run_id} value={run.run_id}>
                {formatRunOptionLabel(run)}
              </option>
            ))}
          </datalist>
          <p className="mt-1 text-[11px] text-subtle">
            {runsLoading && "Loading recent snapshots..."}
            {!runsLoading && !runsError && `Type to search ${runOptions.length} recent snapshot run IDs.`}
            {!runsLoading && runsError && "Could not load run suggestions. Manual run ID entry still works."}
          </p>
        </div>
        <div className="space-y-1.5 md:col-span-3">
          <span className="theme-field-label">Temporal Mode</span>
          <SegmentedControl
            value={temporalMode}
            onChange={handleTemporalModeChange}
            options={[
              { value: "current", label: "Current" },
              { value: "historical", label: "Historical" },
            ]}
          />
          <p className="theme-field-caption">
            Current reads the latest temporal read model. Historical reads require an as-of timestamp and cannot be combined with snapshot replay.
          </p>
        </div>
        {temporalMode === "historical" && (
          <>
            <TextInput
              id="ontology-as-of"
              label="As of"
              value={asOf}
              onChange={setAsOf}
              type="datetime-local"
              errorText={!asOf.trim() && formError?.includes("As of") ? formError : undefined}
            />
            <TextInput
              id="ontology-tx-as-of"
              label="Tx as of"
              value={txAsOf}
              onChange={setTxAsOf}
              type="datetime-local"
              helperText="Optional transaction-time cutoff."
              errorText={formError?.includes("Tx as of") ? formError : undefined}
            />
            <div className="flex items-end">
              <Toggle
                label="Include history"
                checked={includeHistory}
                onChange={setIncludeHistory}
                description="Return historical rows when the backend has version history."
              />
            </div>
          </>
        )}
        <TextInput
          label="Tickers (CSV)"
          value={tickers}
          onChange={setTickers}
          placeholder="AAPL, MSFT"
          uppercase
        />
        <TextInput
          label="Sectors (CSV)"
          value={sectors}
          onChange={setSectors}
          placeholder="Information Technology, Energy"
        />
        <TextInput
          label="Assets (CSV)"
          value={assets}
          onChange={setAssets}
          placeholder="equity, fx, commodity"
        />
        <TextInput
          label="Min Risk Score"
          value={minRiskScore}
          onChange={setMinRiskScore}
          placeholder="0.6"
          type="number"
        />
        <TextInput
          label="Page Size"
          value={pageSize}
          onChange={setPageSize}
          placeholder="25"
          type="number"
        />
        <div className="flex items-end">
          <button
            type="button"
            onClick={handleSubmit}
            disabled={mutation.isPending}
            className="theme-button-secondary h-10 w-full rounded-lg px-4 text-sm font-medium md:w-auto"
          >
            {mutation.isPending ? "Querying..." : "Run Query"}
          </button>
        </div>
      </div>

      {formError && <ErrorMessage message={formError} />}
      {mutation.isPending && (
        <div className="flex items-center gap-4">
          <LoadingSpinner message={`Running ontology query... (${elapsed}s elapsed)`} />
          <button
            type="button"
            onClick={handleCancel}
            className="theme-button-secondary rounded-lg px-3 py-1.5 text-xs font-medium"
          >
            Cancel
          </button>
        </div>
      )}
      {mutation.isError && <ErrorMessage message={String(mutation.error)} />}

      {data && (
        <>
          <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard title="Risk Analysis Snapshot" value={formatTimestampLabel(data.run_id)} />
            <MetricCard title="As Of" value={formatTimestampLabel(data.as_of)} />
            <MetricCard
              title="Confidence"
              value={typeof data.aggregate?.confidence === "number" ? `${(data.aggregate.confidence * 100).toFixed(1)}%` : "N/A"}
            />
            <MetricCard title="Positions" value={String(data.aggregate?.position_count ?? rows.length)} />
          </div>
          <div className="mb-4 flex flex-wrap gap-2">
            <DecisionStateBadge state="analysis" />
            <EffectScopeBadge scope="read_only" />
            <QualityStateBadge state={statusRows.some(row => row.status !== "ok") ? "degraded" : "ok"} />
          </div>

          {temporalMeta && (
            <section className="mb-6 rounded-lg border border-app bg-card-muted p-4" aria-label="Temporal query context">
              <h2 className="mb-3 text-xs font-semibold tracking-widest uppercase text-gray-400">Temporal Context</h2>
              <dl className="grid grid-cols-1 gap-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
                <div>
                  <dt className="text-xs font-medium uppercase tracking-wide text-subtle">Mode</dt>
                  <dd className="mt-1 text-app">{temporalMeta.mode || "N/A"}</dd>
                </div>
                <div>
                  <dt className="text-xs font-medium uppercase tracking-wide text-subtle">As of</dt>
                  <dd className="mt-1 text-app">{formatTimestampLabel(temporalMeta.as_of ?? undefined)}</dd>
                </div>
                <div>
                  <dt className="text-xs font-medium uppercase tracking-wide text-subtle">Tx as of</dt>
                  <dd className="mt-1 text-app">{formatTimestampLabel(temporalMeta.tx_as_of ?? undefined)}</dd>
                </div>
                <div>
                  <dt className="text-xs font-medium uppercase tracking-wide text-subtle">History Included</dt>
                  <dd className="mt-1 text-app">{temporalMeta.include_history ? "Yes" : "No"}</dd>
                </div>
              </dl>
            </section>
          )}

          {data.run_id && (
            <div className="mb-6 flex justify-end">
              <button
                type="button"
                onClick={() => setProvenanceSelector({ ontology_run_id: String(data.run_id) })}
                className="theme-button-secondary inline-flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-medium"
              >
                <GitBranch size={14} />
                Lineage
              </button>
            </div>
          )}

          <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
            <MetricCard title="High Risk" value={String(data.aggregate?.risk_buckets?.high ?? 0)} />
            <MetricCard title="Medium Risk" value={String(data.aggregate?.risk_buckets?.medium ?? 0)} />
            <MetricCard title="Low Risk" value={String(data.aggregate?.risk_buckets?.low ?? 0)} />
          </div>

          <section className="mb-8">
            <div className="mb-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400">Risk Analysis Results</h2>
              {pagination && (
                <div className="flex items-center gap-3 text-xs text-subtle">
                  <span>
                    {pagination.total_results && pagination.returned_results
                      ? `${(pagination.page! - 1) * pagination.page_size! + 1}-${(pagination.page! - 1) * pagination.page_size! + pagination.returned_results} of ${pagination.total_results}`
                      : `0 of ${pagination.total_results ?? 0}`}
                  </span>
                  <button
                    type="button"
                    onClick={() => submitQuery(Math.max(1, currentPage - 1))}
                    disabled={mutation.isPending || !pagination.has_prev}
                    className="theme-button-secondary rounded-lg px-3 py-1.5 text-xs font-medium disabled:opacity-50"
                  >
                    Previous
                  </button>
                  <span>
                    Page {currentPage}
                    {pagination.total_pages ? ` / ${pagination.total_pages}` : ""}
                  </span>
                  <button
                    type="button"
                    onClick={() => submitQuery(currentPage + 1)}
                    disabled={mutation.isPending || !pagination.has_next}
                    className="theme-button-secondary rounded-lg px-3 py-1.5 text-xs font-medium disabled:opacity-50"
                  >
                    Next
                  </button>
                </div>
              )}
            </div>
            <DataTable columns={RESULT_COLUMNS} rows={rows as unknown as Record<string, unknown>[]} responsiveCards />
          </section>

          <section>
            <h2 className="mb-3 text-xs font-semibold tracking-widest uppercase text-gray-400">Source Health And Staleness</h2>
            <DataTable columns={STATUS_COLUMNS} rows={statusRows} responsiveCards />
          </section>
        </>
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
