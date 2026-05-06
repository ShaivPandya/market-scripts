import { useMemo, useState } from "react"
import { Link } from "react-router-dom"
import { useRegisterScreenContext } from "@/contexts/ScreenContext"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchPortfolioAllTimeframes } from "@/lib/api"
import { TimeSeriesChart, calcReturn, type DataPoint, type SeriesDef } from "@/components/shared/TimeSeriesChart"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { SegmentedControl } from "@/components/shared/FormControls"
import { PortfolioEditor } from "@/components/PortfolioEditor"
import { Notice } from "@/components/shared/Notice"
import { PageHeader } from "@/components/shared/PageHeader"
import { ChartTile } from "@/components/shared/ChartTile"

const TIMEFRAMES = ["This Week", "Daily", "Weekly", "Monthly"] as const
const TIMEFRAME_OPTIONS = TIMEFRAMES.map(tf => ({
  value: tf,
  label: tf === "This Week" ? "Past Week" : tf,
}))
const VIEW_MODES = ["Grid", "Unified"] as const
const POSITION_COLORS = [
  "#2563eb",
  "#16a34a",
  "#dc2626",
  "#9333ea",
  "#f97316",
  "#0891b2",
  "#db2777",
  "#65a30d",
  "#7c3aed",
  "#0f766e",
  "#b45309",
  "#475569",
]
type Timeframe = typeof TIMEFRAMES[number]
type ViewMode = typeof VIEW_MODES[number]
type TimeframePayload = {
  positions?: Record<string, DataPoint[]>
  position_order?: string[]
  warning?: string
}
type PortfolioAllTimeframesResponse = {
  timeframes?: Partial<Record<Timeframe, TimeframePayload>>
  holdings?: Array<{ ticker?: string | null; role?: string | null }>
  warning?: string
}

interface UnifiedPositionSummary {
  ticker: string
  seriesKey: string
  color: string
  baseValue: number
  returnPct: number
}

interface UnifiedPerformance {
  chartData: Record<string, unknown>[]
  sortedPositions: UnifiedPositionSummary[]
  series: SeriesDef[]
}

function timeframeLabel(timeframe: Timeframe): string {
  return timeframe === "This Week" ? "Past Week" : timeframe
}

function isFiniteNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

function formatPercent(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`
}

function colorForPosition(index: number): string {
  return POSITION_COLORS[index % POSITION_COLORS.length]
}

function buildUnifiedPerformance(
  order: string[],
  positions: Record<string, DataPoint[]>,
): UnifiedPerformance {
  const comparablePositions = order.flatMap((ticker, index) => {
    const series = positions[ticker]
    if (!Array.isArray(series) || series.length === 0) return []

    const values = series
      .map(point => point.value)
      .filter(isFiniteNumber)
    if (values.length < 2 || values[0] === 0) return []

    return [{
      ticker,
      seriesKey: `position_${index}`,
      color: colorForPosition(index),
      baseValue: values[0],
      returnPct: (values[values.length - 1] / values[0] - 1) * 100,
    }]
  })

  const rowsByDate = new Map<string, Record<string, unknown>>()
  for (const position of comparablePositions) {
    const series = positions[position.ticker] ?? []
    for (const point of series) {
      const row = rowsByDate.get(point.date) ?? { date: point.date }
      row[position.seriesKey] = isFiniteNumber(point.value)
        ? (point.value / position.baseValue - 1) * 100
        : null
      rowsByDate.set(point.date, row)
    }
  }

  const chartData = Array.from(rowsByDate.values())
    .sort((a, b) => String(a.date).localeCompare(String(b.date)))
    .map(row => {
      for (const position of comparablePositions) {
        if (!(position.seriesKey in row)) row[position.seriesKey] = null
      }
      return row
    })

  const sortedPositions = [...comparablePositions]
    .sort((a, b) => b.returnPct - a.returnPct)

  return {
    chartData,
    sortedPositions,
    series: comparablePositions.map(position => ({
      key: position.seriesKey,
      name: position.ticker,
      color: position.color,
      strokeWidth: 1.8,
    })),
  }
}

export function PortfolioDashboard() {
  const [timeframe, setTimeframe] = useState<Timeframe>("This Week")
  const [viewMode, setViewMode] = useState<ViewMode>("Grid")
  const [editOpen, setEditOpen] = useState(false)

  const { data, isLoading, error } = useApiQuery<PortfolioAllTimeframesResponse>(
    ["portfolio", "all_timeframes"],
    fetchPortfolioAllTimeframes,
  )
  const timeframeData = data?.timeframes?.[timeframe]
  const loadError = error
    ? String(error)
    : !data
    ? "Failed to load portfolio data."
    : !timeframeData
    ? `No ${timeframeLabel(timeframe)} portfolio data returned.`
    : null
  const activePortfolioTickers = useMemo(() => {
    if (!Array.isArray(data?.holdings)) return null

    const tickers = new Set(
      data.holdings
        .filter(holding => String(holding.role ?? "position").toLowerCase() !== "hedge")
        .map(holding => String(holding.ticker ?? "").trim().toUpperCase())
        .filter(Boolean),
    )

    return tickers.size > 0 ? tickers : null
  }, [data?.holdings])
  const positions: Record<string, DataPoint[]> = useMemo(
    () => {
      const rawPositions = timeframeData?.positions ?? {}
      if (!activePortfolioTickers) return rawPositions

      return Object.fromEntries(
        Object.entries(rawPositions).filter(([ticker]) => activePortfolioTickers.has(ticker.toUpperCase())),
      )
    },
    [activePortfolioTickers, timeframeData?.positions],
  )
  const order: string[] = useMemo(
    () => {
      const rawOrder = timeframeData?.position_order ?? Object.keys(positions)
      if (!activePortfolioTickers) return rawOrder
      return rawOrder.filter(ticker => activePortfolioTickers.has(ticker.toUpperCase()))
    },
    [activePortfolioTickers, positions, timeframeData?.position_order],
  )
  const warning = timeframeData?.warning ?? data?.warning
  const hasSeries = order.some(ticker => {
    const series = positions[ticker]
    return Array.isArray(series) && series.length > 0
  })

  // Register screen context for agent chat
  const screenCtx = useMemo(() => {
    if (!timeframeData || order.length === 0) return null
    const metrics: Record<string, string> = {
      "Positions": `${order.length} tickers`,
      "Timeframe": timeframeLabel(timeframe),
      "View": viewMode,
    }
    const returns = order.slice(0, 10).map(ticker => {
      const series = positions[ticker]
      if (!series || series.length === 0) return null
      const ret = calcReturn(series)
      return ret != null ? `${ticker}: ${ret >= 0 ? "+" : ""}${ret.toFixed(2)}%` : null
    }).filter(Boolean)
    if (returns.length > 0) metrics["Returns"] = returns.join(", ")
    return {
      pageName: "Portfolio Dashboard",
      metrics,
      filters: { timeframe: timeframeLabel(timeframe), view: viewMode },
      summary: `Portfolio with ${order.length} positions, viewing ${timeframeLabel(timeframe)} in ${viewMode} mode`,
      correspondingTools: ["get_portfolio"],
    }
  }, [timeframeData, order, positions, timeframe, viewMode])
  useRegisterScreenContext(screenCtx)

  const unifiedPerformance = useMemo(
    () => buildUnifiedPerformance(order, positions),
    [order, positions],
  )
  const hasUnifiedSeries = unifiedPerformance.sortedPositions.length > 0 && unifiedPerformance.chartData.length > 0

  return (
    <div>
      <PageHeader
        title="Portfolio Dashboard"
        subtitle="Portfolio positions organized as an adaptive chart grid and comparable performance view."
        actions={(
          <>
            <button
              type="button"
              onClick={() => setEditOpen(true)}
              className="theme-button-base theme-button-secondary px-4"
            >
              Edit Portfolio
            </button>
            <RefreshButton queryKeys={[["portfolio", "all_timeframes"], ["thesis", "status"]]} />
          </>
        )}
      />

      <div className="mb-6 flex flex-wrap items-center gap-3">
        <SegmentedControl
          options={TIMEFRAME_OPTIONS}
          value={timeframe}
          onChange={setTimeframe}
        />
        <SegmentedControl
          options={VIEW_MODES.map(mode => ({ value: mode, label: mode }))}
          value={viewMode}
          onChange={setViewMode}
        />
      </div>

      {isLoading && <LoadingSpinner message="Fetching portfolio data..." />}
      {!isLoading && loadError && (
        <ErrorMessage message={loadError} />
      )}
      {!isLoading && !error && warning && (
        <Notice tone="warning" className="mb-4">{warning}</Notice>
      )}
      {!isLoading && !error && timeframeData && !hasSeries && (
        <Notice tone="info">No price series available.</Notice>
      )}

      {timeframeData && !isLoading && hasSeries && viewMode === "Grid" && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {order.map(ticker => {
            const series = positions[ticker]
            if (!series || series.length === 0) return null
            const ret = calcReturn(series)
            return (
              <ChartTile
                key={ticker}
                title={ticker}
                href={`/dossier/${encodeURIComponent(ticker)}`}
                meta={ret != null ? (
                  <span className={`text-xs font-semibold ${ret >= 0 ? "text-positive" : "text-negative"}`}>
                    {ret >= 0 ? "+" : ""}{ret.toFixed(2)}%
                  </span>
                ) : undefined}
              >
                <TimeSeriesChart data={series} height={160} timeframe={timeframe} />
              </ChartTile>
            )
          })}
        </div>
      )}

      {timeframeData && !isLoading && hasSeries && viewMode === "Unified" && !hasUnifiedSeries && (
        <Notice tone="info">No comparable price series available.</Notice>
      )}

      {timeframeData && !isLoading && hasSeries && viewMode === "Unified" && hasUnifiedSeries && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {unifiedPerformance.sortedPositions.map(position => (
              <Link
                key={position.ticker}
                to={`/dossier/${encodeURIComponent(position.ticker)}`}
                className="rounded-xl border border-app bg-card px-4 py-3 shadow-sm transition-colors hover:border-strong hover:bg-card-muted"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span
                        className="h-2.5 w-2.5 shrink-0 rounded-full"
                        style={{ backgroundColor: position.color }}
                      />
                      <span className="truncate text-sm font-semibold text-app">{position.ticker}</span>
                    </div>
                  </div>
                  <span className={`shrink-0 text-sm font-semibold tabular-nums ${position.returnPct >= 0 ? "text-positive" : "text-negative"}`}>
                    {formatPercent(position.returnPct)}
                  </span>
                </div>
              </Link>
            ))}
          </div>

          <div className="theme-surface p-4 sm:p-5">
            <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
              <div>
                <h2 className="section-title">Unified Performance</h2>
                <p className="mt-1 text-sm text-muted">{timeframeLabel(timeframe)} raw price return, normalized to 0%.</p>
              </div>
              <span className="text-xs font-medium text-subtle">{unifiedPerformance.sortedPositions.length} positions</span>
            </div>
            <TimeSeriesChart
              multiData={unifiedPerformance.chartData}
              series={unifiedPerformance.series}
              height={360}
              timeframe={timeframe}
              zeroLine
              yFormatter={formatPercent}
              tooltipFormatter={formatPercent}
              tooltipSortByValueDesc
              yAxisOrientation="right"
              toggleableLegend
            />
          </div>
        </div>
      )}

      <PortfolioEditor open={editOpen} onOpenChange={setEditOpen} />
    </div>
  )
}
