import { useMemo, useState } from "react"
import { useRegisterScreenContext } from "@/contexts/ScreenContext"
import { useQuery } from "@tanstack/react-query"
import { Info } from "lucide-react"
import { fetchSignalAggregator } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { MetricCard } from "@/components/shared/MetricCard"
import { TimeSeriesChart, type DataPoint } from "@/components/shared/TimeSeriesChart"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { SelectInput, TextInput } from "@/components/shared/FormControls"

const DEFAULT_INSTRUMENTS = "SP500,NASDAQ,RUSSELL,US10Y,EUR"
const LOOKBACK_OPTIONS = [52, 104, 156, 260]

const FACTOR_DIRECTION: Record<string, "Contrarian" | "Same-Dir"> = {
  vix: "Contrarian",
  breadth: "Contrarian",
  sector: "Contrarian",
  momentum: "Contrarian",
  liquidity: "Same-Dir",
}

interface RegimeSummary {
  label: string
  score: number
  confidence: number
  history_percentile?: number | null
}

interface FactorRow {
  key: string
  status: string
  score: number | null
  weight: number
  contribution: number
  highlights?: Record<string, unknown>
}

interface HistoryPoint {
  date: string
  score: number
  label: string
  factors?: Record<string, number>
}

interface HistoryEpisode {
  regime: string
  start_date: string
  end_date: string
  duration_weeks: number
  avg_score: number
}

interface ForwardOutlook {
  label: string
  detail: string
  basis: string
}

interface SignalAggregatorResponse {
  status: string
  as_of: string
  regime: RegimeSummary
  forward_outlook?: ForwardOutlook
  factors: FactorRow[]
  failed_modules: string[]
  module_status: Record<string, { status?: string; detail?: string }>
  history: {
    series: HistoryPoint[]
    episodes: HistoryEpisode[]
  }
  _meta?: {
    snapshot?: {
      as_of?: string | null
      stale?: boolean
      refresh_status?: string
      error?: string | null
    }
  }
}

function signalForRegime(label?: string): "success" | "warning" | "error" | "info" {
  const l = String(label || "").toLowerCase()
  if (l === "risk-on") return "success"
  if (l === "risk-off") return "error"
  return "warning"
}

function summarizeHighlights(highlights?: Record<string, unknown>): string {
  if (!highlights) return "—"
  const entries = Object.entries(highlights)
    .filter(([, v]) => v != null && v !== "")
    .slice(0, 3)
  if (!entries.length) return "—"
  return entries
    .map(([k, v]) => {
      if (typeof v === "number") return `${k}=${v.toFixed(2)}`
      return `${k}=${String(v)}`
    })
    .join(" · ")
}

export function SignalAggregator() {
  const [showInfo, setShowInfo] = useState(false)
  const [lookbackInput, setLookbackInput] = useState("156")
  const [instrumentInput, setInstrumentInput] = useState(DEFAULT_INSTRUMENTS)
  const [appliedLookback, setAppliedLookback] = useState(156)
  const [appliedInstruments, setAppliedInstruments] = useState(DEFAULT_INSTRUMENTS)
  const [hasAppliedFilters, setHasAppliedFilters] = useState(false)

  const { data, isLoading, error } = useQuery<SignalAggregatorResponse>({
    queryKey: ["signal-aggregator", appliedLookback, appliedInstruments],
    queryFn: () =>
      fetchSignalAggregator({
        lookback_weeks: appliedLookback,
        positioning_instruments: appliedInstruments,
      }),
    staleTime: 30 * 60 * 1000,
    retry: 1,
    enabled: hasAppliedFilters,
  })

  const factors = useMemo(() => data?.factors ?? [], [data?.factors])
  const historySeries = useMemo(() => data?.history?.series ?? [], [data?.history?.series])
  const historyEpisodes = useMemo(() => data?.history?.episodes ?? [], [data?.history?.episodes])
  const failedModules = data?.failed_modules ?? []
  const isDegraded = data?.status === "degraded" || failedModules.length > 0

  // Register screen context for agent chat
  const screenCtx = useMemo(() => {
    if (!data) return null
    const regime = data.regime
    const metrics: Record<string, string> = {}
    if (regime?.label) metrics["Regime"] = regime.label
    if (typeof regime?.score === "number") metrics["Composite Score"] = regime.score.toFixed(2)
    if (typeof regime?.confidence === "number") metrics["Confidence"] = `${(regime.confidence * 100).toFixed(0)}%`
    if (typeof regime?.history_percentile === "number") metrics["History Percentile"] = `${regime.history_percentile.toFixed(1)}%`
    const topF = [...(data.factors ?? [])]
      .filter(f => typeof f.contribution === "number")
      .sort((a, b) => (b.contribution ?? 0) - (a.contribution ?? 0))
      .slice(0, 3)
    if (topF.length > 0) {
      metrics["Top Factors"] = topF.map(f => `${f.key}(${f.contribution.toFixed(2)})`).join(", ")
    }
    if (data.forward_outlook) {
      metrics["Forward Outlook"] = `${data.forward_outlook.label} — ${data.forward_outlook.detail}`
    }
    if (failedModules.length > 0) {
      metrics["Degraded Modules"] = failedModules.join(", ")
    }
    return {
      pageName: "Signal Aggregator",
      metrics,
      filters: { lookback: `${appliedLookback} weeks`, instruments: appliedInstruments },
      summary: `Regime: ${regime?.label ?? "unknown"}, Score: ${regime?.score?.toFixed(2) ?? "N/A"}, Status: ${data.status}`,
      correspondingTools: ["get_signal_aggregator"],
    }
  }, [data, failedModules, appliedLookback, appliedInstruments])
  useRegisterScreenContext(screenCtx)

  const factorColumns: ColumnDef[] = [
    { key: "factor", header: "Factor" },
    {
      key: "direction",
      header: "Direction",
      colorFn: v => {
        if (v === "Contrarian") return "#7c3aed; font-weight: bold"
        if (v === "Same-Dir") return "#0284c7; font-weight: bold"
        return ""
      },
    },
    {
      key: "status",
      header: "Status",
      colorFn: v => {
        if (v === "ok") return "#00c853; font-weight: bold"
        return "#ff1744; font-weight: bold"
      },
    },
    {
      key: "score",
      header: "Score",
      format: v => (typeof v === "number" ? v.toFixed(2) : "N/A"),
    },
    {
      key: "weight",
      header: "Weight",
      format: v => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "0.0%"),
    },
    {
      key: "contribution",
      header: "Contribution",
      format: v => (typeof v === "number" ? v.toFixed(2) : "0.00"),
    },
    { key: "highlights", header: "Highlights" },
  ]

  const moduleColumns: ColumnDef[] = [
    { key: "module", header: "Module" },
    {
      key: "status",
      header: "Status",
      colorFn: v => (v === "ok" ? "#00c853; font-weight: bold" : "#ff1744; font-weight: bold"),
    },
    { key: "detail", header: "Detail" },
  ]

  const episodeColumns: ColumnDef[] = [
    { key: "regime", header: "Regime" },
    { key: "start_date", header: "Start" },
    { key: "end_date", header: "End" },
    { key: "duration_weeks", header: "Duration (W)" },
    { key: "avg_score", header: "Avg Score", format: v => (typeof v === "number" ? v.toFixed(2) : "N/A") },
  ]

  const backtestColumns: ColumnDef[] = [
    { key: "factor", header: "Factor" },
    {
      key: "direction",
      header: "Direction",
      colorFn: v => {
        if (v === "Contrarian") return "#7c3aed; font-weight: bold"
        if (v === "Same-Dir") return "#0284c7; font-weight: bold"
        return ""
      },
    },
    { key: "weight", header: "Weight" },
    {
      key: "spread",
      header: "Q5-Q1 Spread (4W)",
      colorFn: v => {
        const n = typeof v === "number" ? v : parseFloat(String(v))
        if (isNaN(n)) return ""
        return n < 0 ? "#7c3aed; font-weight: bold" : "#0284c7; font-weight: bold"
      },
      format: v => {
        const n = typeof v === "number" ? v : parseFloat(String(v))
        if (isNaN(n)) return "N/A"
        return `${n >= 0 ? "+" : ""}${n.toFixed(2)}%`
      },
    },
    { key: "interpretation", header: "Interpretation" },
  ]

  const backtestRows: Record<string, unknown>[] = [
    { factor: "VIX", direction: "Contrarian", weight: "20%", spread: -1.74, interpretation: "High fear → higher returns" },
    { factor: "BREADTH", direction: "Contrarian", weight: "20%", spread: -1.83, interpretation: "Poor breadth → mean reversion" },
    { factor: "LIQUIDITY", direction: "Same-Dir", weight: "35%", spread: 1.13, interpretation: "Tight liquidity → lower returns" },
    { factor: "SECTOR", direction: "Contrarian", weight: "15%", spread: -0.55, interpretation: "Weak contrarian signal" },
    { factor: "MOMENTUM", direction: "Contrarian", weight: "10%", spread: -1.23, interpretation: "Moderate contrarian signal" },
  ]

  const factorRows = factors.map(f => ({
    factor: String(f.key).toUpperCase(),
    direction: FACTOR_DIRECTION[f.key] ?? "—",
    status: f.status,
    score: f.score,
    weight: f.weight,
    contribution: f.contribution,
    highlights: summarizeHighlights(f.highlights),
  }))

  const MODULE_LABELS: Record<string, string> = {
    vix_term_structure: "VIX Term Structure",
    market_breadth: "Market Breadth",
    top50_breadth: "Top 50 Breadth",
    sector_metrics: "Sector Metrics",
    momentum: "Momentum",
    liquidity: "Liquidity",
  }

  const moduleRows = Object.entries(data?.module_status ?? {}).map(([module, state]) => ({
    module: MODULE_LABELS[module] ?? module.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()),
    status: state?.status || "error",
    detail: state?.detail || "—",
  }))
  const episodeRows: Record<string, unknown>[] = historyEpisodes.map(e => ({ ...e }))

  const chartData: DataPoint[] = historySeries.map((p: HistoryPoint) => ({
    date: p.date,
    value: p.score,
  }))

  const topFactors = [...factors]
    .filter(f => typeof f.contribution === "number")
    .sort((a, b) => (b.contribution || 0) - (a.contribution || 0))
    .slice(0, 3)

  const regime = data?.regime
  const snapshot = data?._meta?.snapshot

  function applyControls() {
    const parsed = Number.parseInt(lookbackInput, 10)
    const bounded = Number.isFinite(parsed) ? Math.max(26, Math.min(parsed, 520)) : 156
    setAppliedLookback(bounded)
    setLookbackInput(String(bounded))
    setAppliedInstruments(instrumentInput.trim() || DEFAULT_INSTRUMENTS)
    setHasAppliedFilters(true)
  }

  return (
    <div>
      <div className="mb-6 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Signal Aggregator</h1>
            <button
              onClick={() => setShowInfo(v => !v)}
              className="text-gray-300 hover:text-gray-500 transition-colors"
              title="What is this?"
            >
              <Info size={16} />
            </button>
          </div>
          <p className="text-sm text-gray-400 mt-0.5">Unified cross-module regime synthesis</p>
          {showInfo && (
            <p className="text-xs text-gray-500 mt-2 max-w-xl leading-relaxed">
              Combines signals from VIX term structure, market breadth, sector rotation, momentum, and macro
              liquidity into a single composite regime score. Most factors are <strong>contrarian</strong> — elevated
              stress historically precedes higher forward returns (mean reversion). Liquidity is the
              exception: it is <strong>same-direction</strong>, where tight conditions genuinely predict lower returns.
              The Forward Outlook translates the composite into a predictive label based on 10-year backtested spreads.
            </p>
          )}
          {snapshot && (
            <p className={`mt-2 text-xs ${snapshot.stale ? "text-amber-600" : "text-gray-400"}`}>
              As of {snapshot.as_of ?? "unknown"}
              {snapshot.stale ? " · stale" : ""}
              {snapshot.refresh_status && snapshot.refresh_status !== "ok" ? ` · refresh ${snapshot.refresh_status}` : ""}
              {snapshot.error ? ` · ${snapshot.error}` : ""}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <RefreshButton queryKeys={[["signal-aggregator", appliedLookback, appliedInstruments]]} />
        </div>
      </div>

      <div className="theme-surface mb-6 grid grid-cols-1 gap-3 rounded-xl p-4 md:grid-cols-3">
        <SelectInput
          label="History Lookback"
          value={lookbackInput}
          onChange={setLookbackInput}
          options={LOOKBACK_OPTIONS.map(v => ({ value: String(v), label: `${v} weeks` }))}
        />
        <TextInput
          label="Positioning Instruments"
          value={instrumentInput}
          onChange={setInstrumentInput}
          placeholder={DEFAULT_INSTRUMENTS}
        />
        <div className="flex items-end">
          <button
            onClick={applyControls}
            className="theme-button-secondary h-10 rounded-lg px-4 text-sm font-medium"
          >
            Apply
          </button>
        </div>
      </div>

      {isLoading && <LoadingSpinner message="Aggregating module signals..." />}
      {!isLoading && error && <ErrorMessage message={String(error)} />}
      {!hasAppliedFilters && !isLoading && (
        <div className="rounded-xl border border-app bg-card p-4 text-sm text-gray-500">
          Choose parameters and press Apply to fetch signal aggregator data.
        </div>
      )}

      {data && !isLoading && (
        <>
          {isDegraded && (
            <div className="mb-5 rounded-xl border border-yellow-200 bg-yellow-50 px-4 py-3 text-sm text-yellow-800">
              Partial data: {failedModules.length > 0 ? failedModules.join(", ") : "some factors"} failed. Composite
              score was reweighted using available signals.
            </div>
          )}

          <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              title="Regime"
              value={String(regime?.label || "N/A").toUpperCase()}
              signal={signalForRegime(regime?.label)}
              signalLabel={String(regime?.label || "unknown").toUpperCase()}
            />
            <MetricCard title="Composite Score" value={typeof regime?.score === "number" ? regime.score.toFixed(2) : "N/A"} />
            <MetricCard
              title="Confidence"
              value={typeof regime?.confidence === "number" ? `${(regime.confidence * 100).toFixed(1)}%` : "N/A"}
            />
            <MetricCard
              title="History Percentile"
              value={typeof regime?.history_percentile === "number" ? `${regime.history_percentile.toFixed(1)}%` : "N/A"}
              subtitle={`As of ${data.as_of}`}
            />
          </div>

          {data.forward_outlook && (
            <div
              className={`mb-5 rounded-xl border px-4 py-3 text-sm ${
                data.forward_outlook.label === "opportunity"
                  ? "border-green-200 bg-green-50 text-green-800"
                  : data.forward_outlook.label === "complacent"
                    ? "border-amber-200 bg-amber-50 text-amber-800"
                    : "border-blue-200 bg-blue-50 text-blue-800"
              }`}
            >
              <span className="font-semibold">
                Forward Outlook: {data.forward_outlook.label.toUpperCase()}
              </span>
              {" — "}
              {data.forward_outlook.detail}
              <span className="ml-2 text-xs opacity-70">
                ({data.forward_outlook.basis})
              </span>
            </div>
          )}

          {topFactors.length > 0 && (
            <div className="mb-6 grid grid-cols-1 gap-3 md:grid-cols-3">
              {topFactors.map(f => (
                <MetricCard
                  key={f.key}
                  title={`Top Contribution: ${String(f.key).toUpperCase()}`}
                  value={`${(f.contribution || 0).toFixed(2)}`}
                  subtitle={`Score ${typeof f.score === "number" ? f.score.toFixed(2) : "N/A"} · W ${(f.weight * 100).toFixed(1)}% · ${FACTOR_DIRECTION[f.key] ?? "—"}`}
                />
              ))}
            </div>
          )}

          <section className="mb-8">
            <h2 className="mb-3 text-xs font-semibold tracking-widest uppercase text-gray-400">Factor Contributions</h2>
            <DataTable columns={factorColumns} rows={factorRows} />
          </section>

          <section className="mb-8">
            <h2 className="mb-3 text-xs font-semibold tracking-widest uppercase text-gray-400">Historical Regime Score</h2>
            <div className="rounded-xl border border-app bg-card p-3">
              <TimeSeriesChart data={chartData} height={240} timeframe="Weekly" />
            </div>
          </section>

          <section className="mb-8">
            <h2 className="mb-3 text-xs font-semibold tracking-widest uppercase text-gray-400">Historical Episodes</h2>
            <DataTable columns={episodeColumns} rows={episodeRows} />
          </section>

          <section className="mb-8">
            <h2 className="mb-3 text-xs font-semibold tracking-widest uppercase text-gray-400">Module Status</h2>
            <DataTable columns={moduleColumns} rows={moduleRows} />
          </section>

          <section className="mt-10 border-t border-app pt-6">
            <h2 className="mb-3 text-xs font-semibold tracking-widest uppercase text-gray-400">
              Backtest Evidence (10-Year)
            </h2>
            <div className="mb-4 rounded-xl border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-800">
              Based on 523 weekly observations (2016-2026). The composite score is a{" "}
              <strong>contrarian</strong> indicator: &ldquo;risk-off&rdquo; (high stress) historically
              precedes the highest 4-week forward returns. Liquidity is the exception &mdash;
              tight conditions are directionally negative for returns.
            </div>
            <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
              <MetricCard
                title="Risk-On (Score < 40)"
                value="+1.07%"
                subtitle="4-wk fwd return spread"
                signal="success"
                signalLabel="LOW STRESS"
              />
              <MetricCard
                title="Transitional (40-65)"
                value="+2.45%"
                subtitle="4-wk fwd return spread"
                signal="warning"
                signalLabel="MIXED"
              />
              <MetricCard
                title="Risk-Off (Score ≥ 65)"
                value="+10.70%"
                subtitle="4-wk fwd return spread"
                signal="error"
                signalLabel="HIGH STRESS"
              />
            </div>
            <DataTable columns={backtestColumns} rows={backtestRows} />
          </section>
        </>
      )}
    </div>
  )
}
