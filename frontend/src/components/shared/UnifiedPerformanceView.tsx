import { useMemo } from "react"
import { Link } from "react-router-dom"
import { Notice } from "@/components/shared/Notice"
import { TimeSeriesChart, type DataPoint, type SeriesDef } from "@/components/shared/TimeSeriesChart"

const SERIES_COLORS = [
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

interface UnifiedSeriesSummary {
  name: string
  seriesKey: string
  color: string
  baseValue: number
  returnPct: number
}

interface UnifiedPerformance {
  chartData: Record<string, unknown>[]
  sortedSeries: UnifiedSeriesSummary[]
  series: SeriesDef[]
}

interface UnifiedPerformanceViewProps {
  order: string[]
  seriesByName: Record<string, DataPoint[]>
  timeframe: string
  timeframeLabel: string
  itemLabel: string
  getHref?: (name: string) => string | undefined
}

function isFiniteNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

function formatPercent(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`
}

function colorForSeries(index: number): string {
  return SERIES_COLORS[index % SERIES_COLORS.length]
}

function buildUnifiedPerformance(
  order: string[],
  seriesByName: Record<string, DataPoint[]>,
): UnifiedPerformance {
  const comparableSeries = order.flatMap((name, index) => {
    const points = seriesByName[name]
    if (!Array.isArray(points) || points.length === 0) return []

    const values = points
      .map(point => point.value)
      .filter(isFiniteNumber)
    if (values.length < 2 || values[0] === 0) return []

    return [{
      name,
      seriesKey: `series_${index}`,
      color: colorForSeries(index),
      baseValue: values[0],
      returnPct: (values[values.length - 1] / values[0] - 1) * 100,
    }]
  })

  const rowsByDate = new Map<string, Record<string, unknown>>()
  for (const item of comparableSeries) {
    const points = seriesByName[item.name] ?? []
    for (const point of points) {
      const row = rowsByDate.get(point.date) ?? { date: point.date }
      row[item.seriesKey] = isFiniteNumber(point.value)
        ? (point.value / item.baseValue - 1) * 100
        : null
      rowsByDate.set(point.date, row)
    }
  }

  const chartData = Array.from(rowsByDate.values())
    .sort((a, b) => String(a.date).localeCompare(String(b.date)))
    .map(row => {
      for (const item of comparableSeries) {
        if (!(item.seriesKey in row)) row[item.seriesKey] = null
      }
      return row
    })

  return {
    chartData,
    sortedSeries: [...comparableSeries].sort((a, b) => b.returnPct - a.returnPct),
    series: comparableSeries.map(item => ({
      key: item.seriesKey,
      name: item.name,
      color: item.color,
      strokeWidth: 1.8,
    })),
  }
}

export function UnifiedPerformanceView({
  order,
  seriesByName,
  timeframe,
  timeframeLabel,
  itemLabel,
  getHref,
}: UnifiedPerformanceViewProps) {
  const unifiedPerformance = useMemo(
    () => buildUnifiedPerformance(order, seriesByName),
    [order, seriesByName],
  )
  const hasUnifiedSeries = unifiedPerformance.sortedSeries.length > 0 && unifiedPerformance.chartData.length > 0

  if (!hasUnifiedSeries) {
    return <Notice tone="info">No comparable price series available.</Notice>
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {unifiedPerformance.sortedSeries.map(item => {
          const href = getHref?.(item.name)
          const content = (
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span
                    className="h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="truncate text-sm font-semibold text-app">{item.name}</span>
                </div>
              </div>
              <span className={`shrink-0 text-sm font-semibold tabular-nums ${item.returnPct >= 0 ? "text-positive" : "text-negative"}`}>
                {formatPercent(item.returnPct)}
              </span>
            </div>
          )
          const className = "rounded-xl border border-app bg-card px-4 py-3 shadow-sm"
          const interactiveClassName = `${className} transition-colors hover:border-strong hover:bg-card-muted`

          return href ? (
            <Link key={item.name} to={href} className={interactiveClassName}>
              {content}
            </Link>
          ) : (
            <div key={item.name} className={className}>
              {content}
            </div>
          )
        })}
      </div>

      <div className="theme-surface p-4 sm:p-5">
        <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
          <div>
            <h2 className="section-title">Unified Performance</h2>
            <p className="mt-1 text-sm text-muted">{timeframeLabel} raw price return, normalized to 0%.</p>
          </div>
          <span className="text-xs font-medium text-subtle">{unifiedPerformance.sortedSeries.length} {itemLabel}</span>
        </div>
        <TimeSeriesChart
          multiData={unifiedPerformance.chartData}
          series={unifiedPerformance.series}
          height={360}
          timeframe={timeframe}
          zeroLine
          yFormatter={formatPercent}
          tooltipFormatter={formatPercent}
          yAxisOrientation="right"
          toggleableLegend
        />
      </div>
    </div>
  )
}
