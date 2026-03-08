import { useMemo, useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchSignalAggregator } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { MetricCard } from "@/components/shared/MetricCard"
import { TimeSeriesChart, type DataPoint } from "@/components/shared/TimeSeriesChart"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { SelectInput, TextInput } from "@/components/shared/FormControls"

const DEFAULT_INSTRUMENTS = "SP500,NASDAQ,RUSSELL,US10Y,EUR"
const LOOKBACK_OPTIONS = [52, 104, 156, 260]

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

interface SignalAggregatorResponse {
  status: string
  as_of: string
  regime: RegimeSummary
  factors: FactorRow[]
  failed_modules: string[]
  module_status: Record<string, { status?: string; detail?: string }>
  history: {
    series: HistoryPoint[]
    episodes: HistoryEpisode[]
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
  const [lookbackInput, setLookbackInput] = useState("156")
  const [instrumentInput, setInstrumentInput] = useState(DEFAULT_INSTRUMENTS)
  const [appliedLookback, setAppliedLookback] = useState(156)
  const [appliedInstruments, setAppliedInstruments] = useState(DEFAULT_INSTRUMENTS)

  const { data, isLoading, error } = useApiQuery<SignalAggregatorResponse>(
    ["signal-aggregator", appliedLookback, appliedInstruments],
    () =>
      fetchSignalAggregator({
        lookback_weeks: appliedLookback,
        positioning_instruments: appliedInstruments,
      }),
  )

  const factors = useMemo(() => data?.factors ?? [], [data?.factors])
  const historySeries = useMemo(() => data?.history?.series ?? [], [data?.history?.series])
  const historyEpisodes = useMemo(() => data?.history?.episodes ?? [], [data?.history?.episodes])
  const failedModules = data?.failed_modules ?? []
  const isDegraded = data?.status === "degraded" || failedModules.length > 0

  const factorColumns: ColumnDef[] = [
    { key: "factor", header: "Factor" },
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

  const factorRows = factors.map(f => ({
    factor: String(f.key).toUpperCase(),
    status: f.status,
    score: f.score,
    weight: f.weight,
    contribution: f.contribution,
    highlights: summarizeHighlights(f.highlights),
  }))

  const moduleRows = Object.entries(data?.module_status ?? {}).map(([module, state]) => ({
    module,
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

  function applyControls() {
    const parsed = Number.parseInt(lookbackInput, 10)
    const bounded = Number.isFinite(parsed) ? Math.max(26, Math.min(parsed, 520)) : 156
    setAppliedLookback(bounded)
    setLookbackInput(String(bounded))
    setAppliedInstruments(instrumentInput.trim() || DEFAULT_INSTRUMENTS)
  }

  return (
    <div>
      <div className="mb-6 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Signal Aggregator</h1>
          <p className="text-sm text-gray-400 mt-0.5">Unified cross-module regime synthesis</p>
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

          {topFactors.length > 0 && (
            <div className="mb-6 grid grid-cols-1 gap-3 md:grid-cols-3">
              {topFactors.map(f => (
                <MetricCard
                  key={f.key}
                  title={`Top Contribution: ${String(f.key).toUpperCase()}`}
                  value={`${(f.contribution || 0).toFixed(2)}`}
                  subtitle={`Score ${typeof f.score === "number" ? f.score.toFixed(2) : "N/A"} · W ${(f.weight * 100).toFixed(1)}%`}
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

          <section>
            <h2 className="mb-3 text-xs font-semibold tracking-widest uppercase text-gray-400">Module Status</h2>
            <DataTable columns={moduleColumns} rows={moduleRows} />
          </section>
        </>
      )}
    </div>
  )
}
