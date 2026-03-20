import { useMemo, useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchCommodityResearch } from "@/lib/api"
import { colorPositiveNegative } from "@/lib/colors"
import { fmtNum, fmtPct } from "@/lib/utils"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { MetricCard } from "@/components/shared/MetricCard"
import { SegmentedControl } from "@/components/shared/FormControls"
import { LoadingSpinner } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { TimeSeriesChart } from "@/components/shared/TimeSeriesChart"

/* ── Types ──────────────────────────────────────────────────────────────────── */

interface CommodityFactor {
  score: number | null
  weight: number
  contribution: number
  label?: string
  source?: string
  proxy?: boolean
}

interface CommodityIdea {
  commodity: string
  ticker: string
  sector: string
  spot_price: number | null
  returns: { "1m": number | null; "3m": number | null; "12m": number | null }
  factors: Record<string, CommodityFactor>
  composite_score: number | null
  direction: "long" | "short" | "watchlist"
  confidence: "high" | "medium" | "low"
  rationale: string[]
  data_quality: Record<string, string>
  price_series: Array<{ date: string; value: number }>
}

interface CommodityResearchResponse {
  status: string
  timestamp: string
  macro_regime: { label: string | null; score: number | null; forward_outlook: string | null }
  ideas: CommodityIdea[]
  summary: {
    top_long: { commodity: string; score: number } | null
    top_short: { commodity: string; score: number } | null
    strongest_tailwind: { commodity: string; macro_score: number } | null
    strongest_headwind: { commodity: string; macro_score: number } | null
    data_health: { ok: number; degraded: number; missing: number }
  }
  methodology_note: string
}

type Direction = "all" | "long" | "short" | "watchlist"
type Confidence = "all" | "high" | "medium" | "low"

/* ── Constants ──────────────────────────────────────────────────────────────── */

const DIRECTION_OPTIONS: { value: Direction; label: string }[] = [
  { value: "all", label: "All" },
  { value: "long", label: "Long" },
  { value: "short", label: "Short" },
  { value: "watchlist", label: "Watchlist" },
]

const CONFIDENCE_OPTIONS: { value: Confidence; label: string }[] = [
  { value: "all", label: "All" },
  { value: "high", label: "High" },
  { value: "medium", label: "Medium" },
  { value: "low", label: "Low" },
]

const FACTOR_LABELS: Record<string, string> = {
  momentum: "Momentum",
  relative_value: "Relative Value",
  macro: "Macro Alignment",
  supply_demand: "Supply/Demand Proxy",
  velocity: "Velocity",
}

const FACTOR_COLORS: Record<string, string> = {
  momentum: "#3b82f6",
  relative_value: "#8b5cf6",
  macro: "#f59e0b",
  supply_demand: "#10b981",
  velocity: "#6366f1",
}

const DQ_BADGE: Record<string, { bg: string; text: string }> = {
  ok: { bg: "bg-green-100 text-green-800", text: "OK" },
  stale: { bg: "bg-yellow-100 text-yellow-800", text: "Stale" },
  "n/a": { bg: "bg-gray-100 text-gray-500", text: "N/A" },
  missing: { bg: "bg-red-100 text-red-800", text: "Missing" },
  error: { bg: "bg-red-100 text-red-800", text: "Error" },
}

/* ── Table columns ──────────────────────────────────────────────────────────── */

const TABLE_COLUMNS: ColumnDef[] = [
  { key: "commodity", header: "Commodity" },
  { key: "sector", header: "Sector" },
  {
    key: "spot_price",
    header: "Spot",
    format: v => (typeof v === "number" ? fmtNum(v) : "N/A"),
  },
  {
    key: "return_1m",
    header: "1M Ret",
    format: v => fmtPct(v as number),
    colorFn: v => colorPositiveNegative(v),
  },
  {
    key: "return_3m",
    header: "3M Ret",
    format: v => fmtPct(v as number),
    colorFn: v => colorPositiveNegative(v),
  },
  {
    key: "return_12m",
    header: "12M Ret",
    format: v => fmtPct(v as number),
    colorFn: v => colorPositiveNegative(v),
  },
  {
    key: "composite_score",
    header: "Score",
    format: v => (typeof v === "number" ? fmtNum(v, 1) : "N/A"),
  },
  {
    key: "direction",
    header: "Direction",
    colorFn: v =>
      v === "long"
        ? "#00c853; font-weight: bold"
        : v === "short"
          ? "#ff1744; font-weight: bold"
          : "gray",
    format: v => String(v ?? "").toUpperCase(),
  },
  {
    key: "confidence",
    header: "Confidence",
    colorFn: v =>
      v === "high" ? "#00c853" : v === "medium" ? "#ffc107" : "gray",
    format: v => {
      const s = String(v ?? "")
      return s.charAt(0).toUpperCase() + s.slice(1)
    },
  },
]

/* ── Component ──────────────────────────────────────────────────────────────── */

export function CommodityResearch() {
  const [directionFilter, setDirectionFilter] = useState<Direction>("all")
  const [confidenceFilter, setConfidenceFilter] = useState<Confidence>("all")
  const [selectedCommodity, setSelectedCommodity] = useState<string | null>(null)

  const { data, isLoading, error } = useApiQuery<CommodityResearchResponse>(
    ["commodity-research"],
    () => fetchCommodityResearch(),
  )

  /* Filtered + flattened rows for DataTable */
  const { filteredIdeas, tableRows } = useMemo(() => {
    let ideas = data?.ideas ?? []
    if (directionFilter !== "all") ideas = ideas.filter(i => i.direction === directionFilter)
    if (confidenceFilter !== "all") ideas = ideas.filter(i => i.confidence === confidenceFilter)

    const rows = ideas.map(i => ({
      commodity: i.commodity,
      sector: i.sector,
      spot_price: i.spot_price,
      return_1m: i.returns["1m"],
      return_3m: i.returns["3m"],
      return_12m: i.returns["12m"],
      composite_score: i.composite_score,
      direction: i.direction,
      confidence: i.confidence,
    }))

    return { filteredIdeas: ideas, tableRows: rows }
  }, [data?.ideas, directionFilter, confidenceFilter])

  const selectedIdea = useMemo(
    () => data?.ideas.find(i => i.commodity === selectedCommodity) ?? null,
    [data?.ideas, selectedCommodity],
  )

  /* ── Render ─────────────────────────────────────────────────────────────── */

  return (
    <>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-semibold text-app">Commodity Research</h1>
          <span className="px-1.5 py-0.5 rounded text-xs font-semibold bg-yellow-100 text-yellow-800 border border-yellow-300">Beta</span>
        </div>
        <RefreshButton queryKeys={[["commodity-research"]]} />
      </div>

      {/* Proxy disclaimer */}
      <div className="mb-4 rounded-xl border border-amber-300/40 bg-amber-50/60 px-4 py-2 text-xs text-amber-800 dark:border-amber-400/20 dark:bg-amber-950/30 dark:text-amber-300">
        Scores are proxy-based composites derived from price momentum, curve shape, macro regime, and cross-sectional rank. Supply/demand estimates are heuristic.
      </div>

      {/* Filters */}
      <div className="mb-6 flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-muted">Direction</span>
          <SegmentedControl options={DIRECTION_OPTIONS} value={directionFilter} onChange={setDirectionFilter} size="sm" />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-muted">Confidence</span>
          <SegmentedControl options={CONFIDENCE_OPTIONS} value={confidenceFilter} onChange={setConfidenceFilter} size="sm" />
        </div>
      </div>

      {/* Loading / Error */}
      {isLoading && <LoadingSpinner />}
      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900/40 dark:bg-red-950/30 dark:text-red-300">
          Failed to load commodity research data. {String(error)}
        </div>
      )}

      {data && (
        <>
          {/* Summary cards */}
          <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              title="Top Long Idea"
              value={data.summary.top_long?.commodity ?? "None"}
              subtitle={data.summary.top_long ? `Score: ${fmtNum(data.summary.top_long.score, 1)}` : undefined}
              signal={data.summary.top_long ? "success" : null}
              signalLabel={data.summary.top_long ? "Long" : undefined}
            />
            <MetricCard
              title="Top Short Idea"
              value={data.summary.top_short?.commodity ?? "None"}
              subtitle={data.summary.top_short ? `Score: ${fmtNum(data.summary.top_short.score, 1)}` : undefined}
              signal={data.summary.top_short ? "error" : null}
              signalLabel={data.summary.top_short ? "Short" : undefined}
            />
            <MetricCard
              title="Macro Regime"
              value={data.macro_regime.label ?? "N/A"}
              subtitle={data.macro_regime.score != null ? `Composite: ${fmtNum(data.macro_regime.score, 1)}` : undefined}
              signal={
                data.macro_regime.label === "risk-off" ? "error" :
                data.macro_regime.label === "transitional" ? "warning" :
                data.macro_regime.label === "risk-on" ? "success" : null
              }
              signalLabel={data.macro_regime.forward_outlook ?? undefined}
            />
            <MetricCard
              title="Data Health"
              value={`${data.summary.data_health.ok} / ${data.ideas.length}`}
              subtitle={`${data.summary.data_health.degraded} degraded, ${data.summary.data_health.missing} missing`}
              signal={data.summary.data_health.missing > 0 ? "error" : data.summary.data_health.degraded > 0 ? "warning" : "success"}
              signalLabel={data.status === "ok" ? "Healthy" : "Degraded"}
            />
          </div>

          {/* Ideas table */}
          <div className="mb-6">
            <DataTable
              label={`Ranked Ideas (${filteredIdeas.length})`}
              columns={TABLE_COLUMNS}
              rows={tableRows}
              onRowClick={row => setSelectedCommodity(row.commodity as string)}
            />
          </div>

          {/* Detail panel */}
          {selectedIdea && (
            <div className="rounded-xl border border-app bg-card p-5">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-app">
                  {selectedIdea.commodity}
                  <span className="ml-2 text-sm font-normal text-muted">{selectedIdea.ticker}</span>
                </h2>
                <button
                  type="button"
                  onClick={() => setSelectedCommodity(null)}
                  className="rounded-lg px-2 py-1 text-xs text-muted hover:bg-muted-surface"
                >
                  Close
                </button>
              </div>

              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                {/* Left: Factor breakdown + rationale */}
                <div>
                  <h3 className="mb-3 text-sm font-semibold text-muted">Factor Breakdown</h3>
                  <div className="space-y-2">
                    {Object.entries(selectedIdea.factors).map(([key, factor]) => {
                      const maxContrib = Math.max(
                        ...Object.values(selectedIdea.factors).map(f => Math.abs(f.contribution)),
                        0.01,
                      )
                      const barWidth = factor.score != null
                        ? Math.round((Math.abs(factor.contribution) / maxContrib) * 100)
                        : 0

                      return (
                        <div key={key}>
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-medium text-app">
                              {FACTOR_LABELS[key] ?? key}
                              {factor.proxy && (
                                <span className="ml-1 rounded bg-amber-100 px-1 py-0.5 text-[10px] text-amber-700 dark:bg-amber-900/40 dark:text-amber-400">
                                  proxy
                                </span>
                              )}
                            </span>
                            <span className="text-muted">
                              {factor.score != null ? (factor.score * 100).toFixed(0) : "N/A"}
                              <span className="ml-1 text-subtle">({(factor.weight * 100).toFixed(0)}%)</span>
                            </span>
                          </div>
                          <div className="mt-0.5 h-2 w-full rounded-full bg-muted-surface">
                            <div
                              className="h-2 rounded-full transition-all duration-300"
                              style={{
                                width: `${barWidth}%`,
                                backgroundColor: FACTOR_COLORS[key] ?? "#6b7280",
                              }}
                            />
                          </div>
                        </div>
                      )
                    })}
                  </div>

                  {/* Rationale */}
                  <h3 className="mb-2 mt-5 text-sm font-semibold text-muted">Rationale</h3>
                  <ul className="list-disc space-y-1 pl-4 text-sm text-app">
                    {selectedIdea.rationale.map((bullet, i) => (
                      <li key={i}>{bullet}</li>
                    ))}
                  </ul>

                  {/* Data quality */}
                  <h3 className="mb-2 mt-5 text-sm font-semibold text-muted">Data Quality</h3>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(selectedIdea.data_quality).map(([source, status]) => {
                      const badge = DQ_BADGE[status] ?? DQ_BADGE.error
                      return (
                        <span key={source} className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium ${badge.bg}`}>
                          {source.replace(/_/g, " ")}: {badge.text}
                        </span>
                      )
                    })}
                  </div>
                </div>

                {/* Right: Price chart */}
                <div>
                  <h3 className="mb-3 text-sm font-semibold text-muted">90-Day Price</h3>
                  <TimeSeriesChart
                    data={selectedIdea.price_series}
                    height={220}
                    timeframe="Daily"
                    tooltipFormatter={v => fmtNum(v)}
                  />

                  {/* Return summary */}
                  <div className="mt-4 grid grid-cols-3 gap-3">
                    {(["1m", "3m", "12m"] as const).map(period => {
                      const val = selectedIdea.returns[period]
                      return (
                        <div key={period} className="rounded-lg bg-muted-surface p-2 text-center">
                          <p className="text-[10px] font-medium uppercase text-muted">{period}</p>
                          <p
                            className="text-sm font-semibold"
                            style={{ color: val != null ? (val >= 0 ? "#00c853" : "#ff1744") : "gray" }}
                          >
                            {fmtPct(val)}
                          </p>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </>
  )
}
