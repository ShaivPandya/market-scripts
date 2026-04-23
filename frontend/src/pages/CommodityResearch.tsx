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

interface CommodityFactor {
  score: number | null
  contribution: number
  display_label: string
  category: string
  source: string
  included_in_composite: boolean
  configured_weight: number
  effective_weight: number
  quality: string
}

interface CommodityIdea {
  commodity: string
  ticker: string
  sector: string
  spot_price: number | null
  returns: { "1m": number | null; "3m": number | null; "12m": number | null }
  factors: Record<string, CommodityFactor>
  composite_score: number | null
  observed_composite_score: number | null
  coverage_ratio: number
  direction: "long" | "short" | "watchlist"
  confidence: "high" | "medium" | "low"
  rationale: string[]
  data_quality: Record<string, string>
  price_series: Array<{ date: string; value: number }>
}

interface CommodityResearchResponse {
  schema_version: number
  status: string
  timestamp: string
  methodology: {
    name: string
    note: string
    ranking_mode: string
  }
  macro_overlay: {
    label: string | null
    score: number | null
    forward_outlook: string | null
    as_of: string | null
    status: string
    quality: string
    confidence?: number | null
    history_percentile?: number | null
  }
  ideas: CommodityIdea[]
  summary: {
    top_long: { commodity: string; score: number } | null
    top_short: { commodity: string; score: number } | null
    data_health: { ok: number; degraded: number; missing: number }
  }
}

type Direction = "all" | "long" | "short" | "watchlist"
type Confidence = "all" | "high" | "medium" | "low"

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

const FACTOR_ORDER = [
  "trend",
  "relative_strength",
  "acceleration",
  "curve_structure",
  "market_stress_overlay",
]

const FACTOR_COLORS: Record<string, string> = {
  trend: "#2563eb",
  relative_strength: "#0f766e",
  acceleration: "#ea580c",
  curve_structure: "#7c3aed",
  market_stress_overlay: "#b45309",
}

const DQ_BADGE: Record<string, { bg: string; text: string }> = {
  ok: { bg: "bg-green-100 text-green-800", text: "OK" },
  degraded: { bg: "bg-amber-100 text-amber-800", text: "Degraded" },
  stale: { bg: "bg-yellow-100 text-yellow-800", text: "Stale" },
  "n/a": { bg: "bg-gray-100 text-gray-500", text: "N/A" },
  missing: { bg: "bg-red-100 text-red-800", text: "Missing" },
  error: { bg: "bg-red-100 text-red-800", text: "Error" },
}

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

function titleCase(value: string | null | undefined) {
  if (!value) return "N/A"
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, ch => ch.toUpperCase())
}

function formatCoverageRatio(value: number | null | undefined) {
  if (typeof value !== "number") return "N/A"
  return `${fmtNum(value * 100, 0)}%`
}

function factorBarWidth(factor: CommodityFactor, maxContribution: number) {
  if (factor.score == null) return 0
  if (!factor.included_in_composite) return Math.round(Math.max(8, Math.min(100, factor.score)))
  return Math.round((Math.abs(factor.contribution) / maxContribution) * 100)
}

export function CommodityResearch() {
  const [directionFilter, setDirectionFilter] = useState<Direction>("all")
  const [confidenceFilter, setConfidenceFilter] = useState<Confidence>("all")
  const [selectedCommodity, setSelectedCommodity] = useState<string | null>(null)

  const { data, isLoading, error } = useApiQuery<CommodityResearchResponse>(
    ["commodity-research"],
    () => fetchCommodityResearch(),
  )

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

  const orderedFactors = useMemo(() => {
    if (!selectedIdea) return []
    return FACTOR_ORDER
      .map(key => [key, selectedIdea.factors[key]] as const)
      .filter(([, factor]) => factor != null)
  }, [selectedIdea])

  const maxContribution = useMemo(() => {
    if (!selectedIdea) return 1
    const ranked = Object.values(selectedIdea.factors)
      .filter(f => f.included_in_composite)
      .map(f => Math.abs(f.contribution))
    return Math.max(...ranked, 1)
  }, [selectedIdea])

  return (
    <>
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-semibold text-app">Commodity Proxy Screener</h1>
          <span className="rounded border border-yellow-300 bg-yellow-100 px-1.5 py-0.5 text-xs font-semibold text-yellow-800">Beta</span>
        </div>
        <RefreshButton queryKeys={[["commodity-research"]]} />
      </div>

      {data && (
        <div className="mb-4 rounded-xl border border-amber-300/40 bg-amber-50/60 px-4 py-3 text-sm text-amber-900 dark:border-amber-400/20 dark:bg-amber-950/30 dark:text-amber-300">
          <p className="font-semibold">{data.methodology.name}</p>
          <p className="mt-1 text-xs">{data.methodology.note}</p>
          <p className="mt-1 text-[11px] uppercase tracking-[0.12em] text-amber-700 dark:text-amber-400">
            {data.methodology.ranking_mode}
          </p>
        </div>
      )}

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

      {isLoading && <LoadingSpinner />}
      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900/40 dark:bg-red-950/30 dark:text-red-300">
          Failed to load commodity proxy screener data. {String(error)}
        </div>
      )}

      {data && (
        <>
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
              title="Macro Overlay"
              value={data.macro_overlay.label ?? "N/A"}
              subtitle={data.macro_overlay.score != null ? `Score: ${fmtNum(data.macro_overlay.score, 1)}` : undefined}
              signal={
                data.macro_overlay.label === "risk-off" ? "warning"
                : data.macro_overlay.label === "transitional" ? "warning"
                : data.macro_overlay.label === "risk-on" ? "success"
                : null
              }
              signalLabel={`${titleCase(data.macro_overlay.forward_outlook)} · ${titleCase(data.macro_overlay.quality)}`}
            />
            <MetricCard
              title="Data Health"
              value={`${data.summary.data_health.ok} / ${data.ideas.length}`}
              subtitle={`${data.summary.data_health.degraded} degraded, ${data.summary.data_health.missing} missing`}
              signal={data.summary.data_health.missing > 0 ? "error" : data.summary.data_health.degraded > 0 ? "warning" : "success"}
              signalLabel={data.status === "ok" ? "Healthy" : "Degraded"}
            />
          </div>

          <div className="mb-6">
            <DataTable
              label={`Ranked Ideas (${filteredIdeas.length})`}
              columns={TABLE_COLUMNS}
              rows={tableRows}
              onRowClick={row => setSelectedCommodity(row.commodity as string)}
            />
          </div>

          {selectedIdea && (
            <div className="rounded-xl border border-app bg-card p-5">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-app">
                    {selectedIdea.commodity}
                    <span className="ml-2 text-sm font-normal text-muted">{selectedIdea.ticker}</span>
                  </h2>
                  <div className="mt-2 flex flex-wrap gap-2 text-[11px]">
                    <span className="rounded bg-muted-surface px-2 py-0.5 text-muted">
                      Score: {selectedIdea.composite_score != null ? fmtNum(selectedIdea.composite_score, 1) : "N/A"}
                    </span>
                    <span className="rounded bg-muted-surface px-2 py-0.5 text-muted">
                      Observed: {selectedIdea.observed_composite_score != null ? fmtNum(selectedIdea.observed_composite_score, 1) : "N/A"}
                    </span>
                    <span className="rounded bg-muted-surface px-2 py-0.5 text-muted">
                      Coverage: {formatCoverageRatio(selectedIdea.coverage_ratio)}
                    </span>
                    <span className="rounded bg-muted-surface px-2 py-0.5 text-muted">
                      Confidence: {titleCase(selectedIdea.confidence)}
                    </span>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setSelectedCommodity(null)}
                  className="rounded-lg px-2 py-1 text-xs text-muted hover:bg-muted-surface"
                >
                  Close
                </button>
              </div>

              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <div>
                  <h3 className="mb-3 text-sm font-semibold text-muted">Factor Breakdown</h3>
                  <div className="space-y-3">
                    {orderedFactors.map(([key, factor]) => (
                      <div key={key}>
                        <div className="flex items-start justify-between gap-3 text-xs">
                          <div>
                            <div className="font-medium text-app">{factor.display_label}</div>
                            <div className="mt-1 flex flex-wrap gap-1">
                              <span className="rounded bg-muted-surface px-1.5 py-0.5 text-[10px] text-muted">
                                {factor.source}
                              </span>
                              <span className="rounded bg-muted-surface px-1.5 py-0.5 text-[10px] text-muted">
                                {factor.included_in_composite ? "ranked" : "overlay"}
                              </span>
                              <span className={`rounded px-1.5 py-0.5 text-[10px] ${DQ_BADGE[factor.quality]?.bg ?? DQ_BADGE.error.bg}`}>
                                {DQ_BADGE[factor.quality]?.text ?? "Error"}
                              </span>
                            </div>
                          </div>
                          <div className="text-right text-muted">
                            <div>{factor.score != null ? fmtNum(factor.score, 1) : "N/A"}</div>
                            <div className="text-[10px]">
                              cfg {fmtNum(factor.configured_weight * 100, 0)}%
                              {factor.included_in_composite && ` · eff ${fmtNum(factor.effective_weight * 100, 0)}%`}
                            </div>
                          </div>
                        </div>
                        <div className="mt-1 h-2 w-full rounded-full bg-muted-surface">
                          <div
                            className="h-2 rounded-full transition-all duration-300"
                            style={{
                              width: `${factorBarWidth(factor, maxContribution)}%`,
                              backgroundColor: FACTOR_COLORS[key] ?? "#6b7280",
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  <h3 className="mb-2 mt-5 text-sm font-semibold text-muted">Rationale</h3>
                  <ul className="list-disc space-y-1 pl-4 text-sm text-app">
                    {selectedIdea.rationale.map((bullet, index) => (
                      <li key={index}>{bullet}</li>
                    ))}
                  </ul>

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

                <div>
                  <h3 className="mb-3 text-sm font-semibold text-muted">90-Day Price</h3>
                  <TimeSeriesChart
                    data={selectedIdea.price_series}
                    height={220}
                    timeframe="Daily"
                    tooltipFormatter={value => fmtNum(value)}
                  />

                  <div className="mt-4 grid grid-cols-3 gap-3">
                    {(["1m", "3m", "12m"] as const).map(period => {
                      const value = selectedIdea.returns[period]
                      return (
                        <div key={period} className="rounded-lg bg-muted-surface p-2 text-center">
                          <p className="text-[10px] font-medium uppercase text-muted">{period}</p>
                          <p
                            className="text-sm font-semibold"
                            style={{ color: value != null ? (value >= 0 ? "#00c853" : "#ff1744") : "gray" }}
                          >
                            {fmtPct(value)}
                          </p>
                        </div>
                      )
                    })}
                  </div>

                  <div className="mt-5 rounded-xl border border-app bg-muted-surface/60 p-4">
                    <h3 className="text-sm font-semibold text-app">Macro Overlay</h3>
                    <p className="mt-2 text-sm text-app">
                      {data.macro_overlay.label ? titleCase(data.macro_overlay.label) : "N/A"}
                    </p>
                    <p className="mt-1 text-xs text-muted">
                      Score {data.macro_overlay.score != null ? fmtNum(data.macro_overlay.score, 1) : "N/A"} ·
                      Outlook {titleCase(data.macro_overlay.forward_outlook)} ·
                      Quality {titleCase(data.macro_overlay.quality)}
                    </p>
                    {data.macro_overlay.as_of && (
                      <p className="mt-1 text-[11px] text-subtle">As of {data.macro_overlay.as_of}</p>
                    )}
                    <p className="mt-2 text-xs text-muted">
                      Overlay data is informational only and is excluded from the ranked composite.
                    </p>
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
