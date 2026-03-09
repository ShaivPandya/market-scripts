import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Info } from "lucide-react"
import { useMutation } from "@tanstack/react-query"
import { runOntologyQueryAsync } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { SelectInput, TextInput } from "@/components/shared/FormControls"

type Intent = "auto" | "portfolio_risk_exposure" | "positions_in_deteriorating_macro" | "entity_context"
type Timeframe = "This Week" | "Daily" | "Weekly" | "Monthly"

interface OntologyEvidence {
  source?: string
  name?: string
  contribution?: number
}

interface OntologyRow {
  ticker?: string
  asset?: string
  direction?: string
  sector?: string
  risk_score?: number
  risk_level?: string
  evidence?: OntologyEvidence[]
}

interface OntologyResponse {
  run_id: string
  as_of: string
  source_status: Record<string, { status?: string; detail?: string }>
  results: OntologyRow[]
  aggregate: {
    position_count?: number
    confidence?: number
    risk_buckets?: { high?: number; medium?: number; low?: number }
  }
}

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

function parseCsv(text: string): string[] | undefined {
  const vals = text
    .split(",")
    .map(v => v.trim())
    .filter(Boolean)
  return vals.length > 0 ? vals : undefined
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

export function OntologyWorkbench() {
  const [showInfo, setShowInfo] = useState(false)
  const [query, setQuery] = useState("")
  const [intent, setIntent] = useState<Intent>("auto")
  const [timeframe, setTimeframe] = useState<Timeframe>("Daily")
  const [tickers, setTickers] = useState("")
  const [sectors, setSectors] = useState("")
  const [assets, setAssets] = useState("")
  const [minRiskScore, setMinRiskScore] = useState("")
  const [maxResults, setMaxResults] = useState("")
  const [runId, setRunId] = useState("")
  const [cachedResult, setCachedResult] = useState<OntologyResponse | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const [elapsed, setElapsed] = useState(0)

  const mutation = useMutation({
    mutationFn: (body: Parameters<typeof runOntologyQueryAsync>[0]) => {
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller
      return runOntologyQueryAsync(body, controller.signal)
    },
    onSuccess: result => setCachedResult((result as OntologyResponse) ?? null),
  })

  useEffect(() => {
    if (!mutation.isPending) {
      setElapsed(0)
      return
    }
    const start = Date.now()
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000)
    return () => clearInterval(id)
  }, [mutation.isPending])

  const handleCancel = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
  }, [])

  const handleIntentChange = useCallback((v: string) => setIntent(v as Intent), [])
  const handleTimeframeChange = useCallback((v: string) => setTimeframe(v as Timeframe), [])

  const handleSubmit = useCallback(() => {
    const filters: Record<string, unknown> = {}
    const parsedTickers = parseCsv(tickers)
    const parsedSectors = parseCsv(sectors)
    const parsedAssets = parseCsv(assets)
    if (parsedTickers) filters.tickers = parsedTickers
    if (parsedSectors) filters.sectors = parsedSectors
    if (parsedAssets) filters.assets = parsedAssets

    const parsedMinRisk = Number.parseFloat(minRiskScore)
    if (Number.isFinite(parsedMinRisk)) filters.min_risk_score = parsedMinRisk

    const parsedMaxResults = Number.parseInt(maxResults, 10)
    if (Number.isFinite(parsedMaxResults) && parsedMaxResults > 0) filters.max_results = parsedMaxResults

    mutation.mutate({
      query: query.trim() || undefined,
      intent: intent === "auto" ? undefined : intent,
      filters: Object.keys(filters).length > 0 ? filters : undefined,
      timeframe,
      run_id: runId.trim() || undefined,
      include_graph: false,
      refresh_snapshot: false,
    })
  }, [query, intent, timeframe, tickers, sectors, assets, minRiskScore, maxResults, runId, mutation])

  const data = (mutation.data as OntologyResponse | undefined) ?? cachedResult
  const rows = useMemo(() => (Array.isArray(data?.results) ? data.results : []), [data?.results])
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
          Query portfolio-linked risk snapshots with natural language and structured filters.
        </p>
        {showInfo && (
          <p className="text-xs text-gray-500 mt-2 max-w-xl leading-relaxed">
            Links each portfolio position to macro conditions, sector dynamics, and cross-asset signals via a knowledge
            graph. Filter by ticker, sector, asset class, or risk threshold to see which holdings carry elevated risk
            and the evidence behind each score. Use snapshot run IDs to compare risk states across time.
          </p>
        )}
      </div>

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
        <TextInput
          label="Snapshot Run ID (optional)"
          value={runId}
          onChange={setRunId}
          placeholder="2026-03-08T14:30:12.123456+00:00"
        />
        <TextInput
          label="Tickers (CSV)"
          value={tickers}
          onChange={setTickers}
          placeholder="AAPL, MSFT"
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
          label="Max Results"
          value={maxResults}
          onChange={setMaxResults}
          placeholder="20"
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
            <MetricCard title="Snapshot Run" value={formatTimestampLabel(data.run_id)} />
            <MetricCard title="As Of" value={formatTimestampLabel(data.as_of)} />
            <MetricCard
              title="Confidence"
              value={typeof data.aggregate?.confidence === "number" ? `${(data.aggregate.confidence * 100).toFixed(1)}%` : "N/A"}
            />
            <MetricCard title="Positions" value={String(data.aggregate?.position_count ?? rows.length)} />
          </div>

          <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
            <MetricCard title="High Risk" value={String(data.aggregate?.risk_buckets?.high ?? 0)} />
            <MetricCard title="Medium Risk" value={String(data.aggregate?.risk_buckets?.medium ?? 0)} />
            <MetricCard title="Low Risk" value={String(data.aggregate?.risk_buckets?.low ?? 0)} />
          </div>

          <section className="mb-8">
            <h2 className="mb-3 text-xs font-semibold tracking-widest uppercase text-gray-400">Position Results</h2>
            <DataTable columns={RESULT_COLUMNS} rows={rows as Record<string, unknown>[]} />
          </section>

          <section>
            <h2 className="mb-3 text-xs font-semibold tracking-widest uppercase text-gray-400">Source Status</h2>
            <DataTable columns={STATUS_COLUMNS} rows={statusRows} />
          </section>
        </>
      )}
    </div>
  )
}
