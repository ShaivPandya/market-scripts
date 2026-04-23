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

interface CommodityFundamentalInputs {
  coverage_status: "unavailable" | "partial" | "available"
  coverage_note: string
  available_inputs: string[]
}

interface CommodityProxySignals {
  proxy_score: number | null
  observed_proxy_score: number | null
  proxy_coverage_ratio: number
  bias: "bullish" | "bearish" | "neutral"
  signal_conviction: "high" | "medium" | "low"
  factors: Record<string, CommodityFactor>
  rationale: string[]
  data_quality: Record<string, string>
}

interface CommodityResult {
  commodity: string
  ticker: string
  sector: string
  spot_price: number | null
  returns: { "1m": number | null; "3m": number | null; "12m": number | null }
  price_series: Array<{ date: string; value: number }>
  proxy_signals: CommodityProxySignals
  fundamental_inputs: CommodityFundamentalInputs
}

interface CommodityResearchResponse {
  schema_version: number
  status: string
  timestamp: string
  methodology: {
    proxy_signals: {
      name: string
      note: string
      limitations: string
      ranking_mode: string
    }
    fundamental_inputs: {
      coverage_policy: string
      current_status: string
    }
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
  commodities: CommodityResult[]
  summary: {
    strongest_bullish_bias: { commodity: string; proxy_score: number } | null
    strongest_bearish_bias: { commodity: string; proxy_score: number } | null
    proxy_data_health: { ok: number; degraded: number; missing: number }
    fundamental_coverage: Record<string, CommodityFundamentalInputs>
  }
}

type Bias = "all" | "bullish" | "bearish" | "neutral"
type SignalConviction = "all" | "high" | "medium" | "low"

const BIAS_OPTIONS: { value: Bias; label: string }[] = [
  { value: "all", label: "All" },
  { value: "bullish", label: "Bullish" },
  { value: "bearish", label: "Bearish" },
  { value: "neutral", label: "Neutral" },
]

const CONVICTION_OPTIONS: { value: SignalConviction; label: string }[] = [
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

const COVERAGE_BADGE: Record<string, { bg: string; text: string }> = {
  unavailable: { bg: "bg-slate-100 text-slate-700", text: "Unavailable" },
  partial: { bg: "bg-amber-100 text-amber-800", text: "Partial" },
  available: { bg: "bg-green-100 text-green-800", text: "Available" },
}

const TABLE_COLUMNS: ColumnDef[] = [
  { key: "commodity", header: "Commodity" },
  { key: "sector", header: "Family" },
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
    key: "proxy_score",
    header: "Proxy Score",
    format: v => (typeof v === "number" ? fmtNum(v, 1) : "N/A"),
  },
  {
    key: "bias",
    header: "Bias",
    colorFn: v =>
      v === "bullish"
        ? "#00c853; font-weight: bold"
        : v === "bearish"
          ? "#ff1744; font-weight: bold"
          : "gray",
    format: v => titleCase(String(v ?? "")),
  },
  {
    key: "signal_conviction",
    header: "Signal Conviction",
    colorFn: v =>
      v === "high" ? "#00c853" : v === "medium" ? "#ffc107" : "gray",
    format: v => titleCase(String(v ?? "")),
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
  const [biasFilter, setBiasFilter] = useState<Bias>("all")
  const [convictionFilter, setConvictionFilter] = useState<SignalConviction>("all")
  const [selectedCommodity, setSelectedCommodity] = useState<string | null>(null)

  const { data, isLoading, error } = useApiQuery<CommodityResearchResponse>(
    ["commodity-research", "v3"],
    () => fetchCommodityResearch(),
  )

  const { filteredCommodities, tableRows } = useMemo(() => {
    let commodities = data?.commodities ?? []
    if (biasFilter !== "all") commodities = commodities.filter(i => i.proxy_signals.bias === biasFilter)
    if (convictionFilter !== "all") {
      commodities = commodities.filter(i => i.proxy_signals.signal_conviction === convictionFilter)
    }

    const rows = commodities.map(i => ({
      commodity: i.commodity,
      sector: i.sector,
      spot_price: i.spot_price,
      return_1m: i.returns["1m"],
      return_3m: i.returns["3m"],
      return_12m: i.returns["12m"],
      proxy_score: i.proxy_signals.proxy_score,
      bias: i.proxy_signals.bias,
      signal_conviction: i.proxy_signals.signal_conviction,
    }))

    return { filteredCommodities: commodities, tableRows: rows }
  }, [biasFilter, convictionFilter, data?.commodities])

  const selectedResult = useMemo(
    () => data?.commodities.find(i => i.commodity === selectedCommodity) ?? null,
    [data?.commodities, selectedCommodity],
  )

  const orderedFactors = useMemo(() => {
    if (!selectedResult) return []
    return FACTOR_ORDER
      .map(key => [key, selectedResult.proxy_signals.factors[key]] as const)
      .filter(([, factor]) => factor != null)
  }, [selectedResult])

  const maxContribution = useMemo(() => {
    if (!selectedResult) return 1
    const ranked = Object.values(selectedResult.proxy_signals.factors)
      .filter(f => f.included_in_composite)
      .map(f => Math.abs(f.contribution))
    return Math.max(...ranked, 1)
  }, [selectedResult])

  return (
    <>
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-semibold text-app">Commodity Proxy Screener</h1>
          <span className="rounded border border-yellow-300 bg-yellow-100 px-1.5 py-0.5 text-xs font-semibold text-yellow-800">Beta</span>
        </div>
        <RefreshButton queryKeys={[["commodity-research", "v3"]]} />
      </div>

      {data && (
        <div className="mb-5 grid grid-cols-1 gap-3 xl:grid-cols-2">
          <div className="rounded-xl border border-amber-300/40 bg-amber-50/60 px-4 py-3 text-sm text-amber-900 dark:border-amber-400/20 dark:bg-amber-950/30 dark:text-amber-300">
            <p className="font-semibold">{data.methodology.proxy_signals.name}</p>
            <p className="mt-1 text-xs">{data.methodology.proxy_signals.note}</p>
            <p className="mt-2 text-xs text-amber-800 dark:text-amber-400">
              {data.methodology.proxy_signals.limitations}
            </p>
            <p className="mt-2 text-[11px] uppercase tracking-[0.12em] text-amber-700 dark:text-amber-400">
              {data.methodology.proxy_signals.ranking_mode}
            </p>
          </div>

          <div className="rounded-xl border border-slate-300/50 bg-slate-50/80 px-4 py-3 text-sm text-slate-900 dark:border-slate-700 dark:bg-slate-900/40 dark:text-slate-200">
            <p className="font-semibold">Fundamental Inputs</p>
            <p className="mt-1 text-xs">{data.methodology.fundamental_inputs.coverage_policy}</p>
            <p className="mt-2 text-xs text-slate-700 dark:text-slate-300">
              {data.methodology.fundamental_inputs.current_status}
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              {Object.entries(data.summary.fundamental_coverage).map(([family, coverage]) => {
                const badge = COVERAGE_BADGE[coverage.coverage_status] ?? COVERAGE_BADGE.unavailable
                return (
                  <span
                    key={family}
                    className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium ${badge.bg}`}
                  >
                    {titleCase(family)}: {badge.text}
                  </span>
                )
              })}
            </div>
          </div>
        </div>
      )}

      <div className="mb-6 flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-muted">Bias</span>
          <SegmentedControl options={BIAS_OPTIONS} value={biasFilter} onChange={setBiasFilter} size="sm" />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-muted">Signal Conviction</span>
          <SegmentedControl
            options={CONVICTION_OPTIONS}
            value={convictionFilter}
            onChange={setConvictionFilter}
            size="sm"
          />
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
              title="Strongest Bullish Bias"
              value={data.summary.strongest_bullish_bias?.commodity ?? "None"}
              subtitle={
                data.summary.strongest_bullish_bias
                  ? `Proxy Score: ${fmtNum(data.summary.strongest_bullish_bias.proxy_score, 1)}`
                  : undefined
              }
              signal={data.summary.strongest_bullish_bias ? "success" : null}
              signalLabel={data.summary.strongest_bullish_bias ? "Bullish" : undefined}
            />
            <MetricCard
              title="Strongest Bearish Bias"
              value={data.summary.strongest_bearish_bias?.commodity ?? "None"}
              subtitle={
                data.summary.strongest_bearish_bias
                  ? `Proxy Score: ${fmtNum(data.summary.strongest_bearish_bias.proxy_score, 1)}`
                  : undefined
              }
              signal={data.summary.strongest_bearish_bias ? "error" : null}
              signalLabel={data.summary.strongest_bearish_bias ? "Bearish" : undefined}
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
              title="Proxy Data Health"
              value={`${data.summary.proxy_data_health.ok} / ${data.commodities.length}`}
              subtitle={`${data.summary.proxy_data_health.degraded} degraded, ${data.summary.proxy_data_health.missing} missing`}
              signal={
                data.summary.proxy_data_health.missing > 0 ? "error"
                : data.summary.proxy_data_health.degraded > 0 ? "warning"
                : "success"
              }
              signalLabel={data.status === "ok" ? "Healthy" : "Degraded"}
            />
          </div>

          <div className="mb-6">
            <DataTable
              label={`Ranked Commodities (${filteredCommodities.length})`}
              columns={TABLE_COLUMNS}
              rows={tableRows}
              onRowClick={row => setSelectedCommodity(row.commodity as string)}
            />
          </div>

          {selectedResult && (
            <div className="rounded-xl border border-app bg-card p-5">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-app">
                    {selectedResult.commodity}
                    <span className="ml-2 text-sm font-normal text-muted">{selectedResult.ticker}</span>
                  </h2>
                  <div className="mt-2 flex flex-wrap gap-2 text-[11px]">
                    <span className="rounded bg-muted-surface px-2 py-0.5 text-muted">
                      Proxy Score: {selectedResult.proxy_signals.proxy_score != null ? fmtNum(selectedResult.proxy_signals.proxy_score, 1) : "N/A"}
                    </span>
                    <span className="rounded bg-muted-surface px-2 py-0.5 text-muted">
                      Observed Proxy: {selectedResult.proxy_signals.observed_proxy_score != null ? fmtNum(selectedResult.proxy_signals.observed_proxy_score, 1) : "N/A"}
                    </span>
                    <span className="rounded bg-muted-surface px-2 py-0.5 text-muted">
                      Coverage: {formatCoverageRatio(selectedResult.proxy_signals.proxy_coverage_ratio)}
                    </span>
                    <span className="rounded bg-muted-surface px-2 py-0.5 text-muted">
                      Bias: {titleCase(selectedResult.proxy_signals.bias)}
                    </span>
                    <span className="rounded bg-muted-surface px-2 py-0.5 text-muted">
                      Signal Conviction: {titleCase(selectedResult.proxy_signals.signal_conviction)}
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

              <section>
                <div className="mb-3">
                  <h3 className="text-sm font-semibold text-muted">Proxy / Technical Signals</h3>
                  <p className="mt-1 text-xs text-subtle">
                    Proxy score reflects price-based and market-structure signals only. It is not a combined fundamentals score.
                  </p>
                </div>

                <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                  <div>
                    <h4 className="mb-3 text-sm font-semibold text-muted">Factor Breakdown</h4>
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
                                <span
                                  className={`rounded px-1.5 py-0.5 text-[10px] ${DQ_BADGE[factor.quality]?.bg ?? DQ_BADGE.error.bg}`}
                                >
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

                    <h4 className="mb-2 mt-5 text-sm font-semibold text-muted">Rationale</h4>
                    <ul className="list-disc space-y-1 pl-4 text-sm text-app">
                      {selectedResult.proxy_signals.rationale.map((bullet, index) => (
                        <li key={index}>{bullet}</li>
                      ))}
                    </ul>

                    <h4 className="mb-2 mt-5 text-sm font-semibold text-muted">Proxy Data Quality</h4>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(selectedResult.proxy_signals.data_quality).map(([source, status]) => {
                        const badge = DQ_BADGE[status] ?? DQ_BADGE.error
                        return (
                          <span
                            key={source}
                            className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium ${badge.bg}`}
                          >
                            {source.replace(/_/g, " ")}: {badge.text}
                          </span>
                        )
                      })}
                    </div>
                  </div>

                  <div>
                    <h4 className="mb-3 text-sm font-semibold text-muted">90-Day Price</h4>
                    <TimeSeriesChart
                      data={selectedResult.price_series}
                      height={220}
                      timeframe="Daily"
                      tooltipFormatter={value => fmtNum(value)}
                    />

                    <div className="mt-4 grid grid-cols-3 gap-3">
                      {(["1m", "3m", "12m"] as const).map(period => {
                        const value = selectedResult.returns[period]
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
                      <h4 className="text-sm font-semibold text-app">Macro Overlay</h4>
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
                        Overlay data is informational only and is excluded from the ranked proxy score.
                      </p>
                    </div>
                  </div>
                </div>
              </section>

              <section className="mt-6 rounded-xl border border-app bg-muted-surface/40 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold text-app">Fundamental Inputs</h3>
                    <p className="mt-1 text-xs text-subtle">
                      This section only shows real commodity-specific fundamentals. Synthetic placeholders are intentionally excluded.
                    </p>
                  </div>
                  <span
                    className={`inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium ${
                      COVERAGE_BADGE[selectedResult.fundamental_inputs.coverage_status]?.bg ?? COVERAGE_BADGE.unavailable.bg
                    }`}
                  >
                    {COVERAGE_BADGE[selectedResult.fundamental_inputs.coverage_status]?.text ?? "Unavailable"}
                  </span>
                </div>

                <p className="mt-3 text-sm text-app">{selectedResult.fundamental_inputs.coverage_note}</p>

                {selectedResult.fundamental_inputs.available_inputs.length > 0 ? (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {selectedResult.fundamental_inputs.available_inputs.map(input => (
                      <span key={input} className="rounded bg-card px-2 py-0.5 text-xs text-muted">
                        {input}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="mt-3 text-xs text-muted">
                    No real fundamental inputs are currently available for this commodity family in this screener.
                  </p>
                )}
              </section>
            </div>
          )}
        </>
      )}
    </>
  )
}
