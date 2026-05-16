import { useMemo, useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { ChevronDown, Sparkles } from "lucide-react"
import {
  dashboardTimeframeStaleTime,
  useDashboardTimeframePrefetch,
} from "@/hooks/useDashboardTimeframePrefetch"
import { useApiQuery } from "@/hooks/useApiQuery"
import { useSessionAiOverview } from "@/hooks/useSessionAiOverview"
import { fetchSectorMetrics, fetchSectorMetricsSeries, analyzeSectorMetrics, refreshMarketSnapshots } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { colorPositiveNegative } from "@/lib/colors"
import { SegmentedControl } from "@/components/shared/FormControls"
import { ChartTile } from "@/components/shared/ChartTile"
import { TimeSeriesChart, calcReturn, type DataPoint } from "@/components/shared/TimeSeriesChart"
import { Notice } from "@/components/shared/Notice"
import { UnifiedPerformanceView } from "@/components/shared/UnifiedPerformanceView"
import { UNIFIED_VIEW_MODES, type UnifiedViewMode } from "@/lib/unifiedViewMode"

const fmtPp = (v: unknown) => v != null ? `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(2)}pp` : "N/A"
const fmtPct = (v: unknown) => v != null ? `${Number(v).toFixed(1)}%` : "N/A"
const TIMEFRAMES = ["This Week", "Daily", "Weekly", "Monthly"] as const
const TIMEFRAME_OPTIONS = TIMEFRAMES.map(tf => ({
  value: tf,
  label: tf === "This Week" ? "Past Week" : tf,
}))
const PERFORMANCE_MODES = ["ETF Returns", "vs SPY"] as const
type Timeframe = typeof TIMEFRAMES[number]
type PerformanceMode = typeof PERFORMANCE_MODES[number]

type SectorMetricsSeriesResponse = {
  sector_prices?: Record<string, DataPoint[]>
  sector_relative_prices?: Record<string, DataPoint[]>
  sector_order?: string[]
  benchmark?: string
  timeframe?: string
  timestamp?: string
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

function normalizeReturnSeries(series: DataPoint[]): DataPoint[] {
  const base = series.map(point => point.value).find(isFiniteNumber)
  if (!base || base === 0) return []

  return series.map(point => ({
    date: point.date,
    value: isFiniteNumber(point.value) ? (point.value / base - 1) * 100 : null,
  }))
}

const columns: ColumnDef[] = [
  { key: "Sector", header: "Sector" },
  { key: "Weight_Now", header: "Weight Now", format: fmtPct },
  { key: "Chg_1M_pp", header: "1M Chg (pp)", colorFn: colorPositiveNegative, format: fmtPp },
  { key: "Chg_3M_pp", header: "3M Chg (pp)", colorFn: colorPositiveNegative, format: fmtPp },
  { key: "Chg_6M_pp", header: "6M Chg (pp)", colorFn: colorPositiveNegative, format: fmtPp },
  { key: "RelPerf_1M_pp", header: "Rel Perf 1M", colorFn: colorPositiveNegative, format: fmtPp },
  { key: "RelPerf_3M_pp", header: "Rel Perf 3M", colorFn: colorPositiveNegative, format: fmtPp },
  { key: "RelPerf_6M_pp", header: "Rel Perf 6M", colorFn: colorPositiveNegative, format: fmtPp },
  { key: "RelPerf_12M_pp", header: "Rel Perf 12M", colorFn: colorPositiveNegative, format: fmtPp },
  { key: "Pct_Above_200DMA", header: "% Above 200DMA", format: fmtPct },
]

export function SectorMetrics() {
  const [timeframe, setTimeframe] = useState<Timeframe>("This Week")
  const [performanceMode, setPerformanceMode] = useState<PerformanceMode>("ETF Returns")
  const [viewMode, setViewMode] = useState<UnifiedViewMode>("Grid")
  const { analysis: persistedAnalysis, isOpen, setIsOpen, setAnalysis: setPersistedAnalysis } = useSessionAiOverview("ai-overview:sector-metrics")
  const mutation = useMutation({
    mutationFn: analyzeSectorMetrics,
    onSuccess: data => {
      const analysis = typeof data?.analysis === "string" ? data.analysis : null
      if (analysis) setPersistedAnalysis(analysis)
    },
  })

  const { data, isLoading, error } = useApiQuery(
    ["sector-metrics"],
    fetchSectorMetrics,
    60 * 60 * 1000,
  )
  const {
    data: seriesData,
    isLoading: isSeriesLoading,
    error: seriesError,
    isSuccess: isSeriesSuccess,
  } = useApiQuery<SectorMetricsSeriesResponse>(
    ["sector-metrics-series", timeframe],
    () => fetchSectorMetricsSeries(timeframe),
    dashboardTimeframeStaleTime(timeframe),
  )
  useDashboardTimeframePrefetch({
    queryKeyRoot: "sector-metrics-series",
    timeframes: TIMEFRAMES,
    activeTimeframe: timeframe,
    isReady: isSeriesSuccess,
    fetchTimeframe: fetchSectorMetricsSeries,
  })

  const rows = (data?.weights_df ?? []) as Record<string, unknown>[]
  const liveAnalysis = typeof mutation.data?.analysis === "string" ? mutation.data.analysis : null
  const analysisText = liveAnalysis ?? persistedAnalysis
  const showPanel = Boolean(analysisText || mutation.isPending || mutation.isError)
  const refreshQueryKeys = useMemo(
    () => [["sector-metrics"], ["sector-metrics-series", timeframe]],
    [timeframe],
  )
  const sectorPrices = seriesData?.sector_prices ?? {}
  const sectorRelativePrices = seriesData?.sector_relative_prices ?? {}
  const order = seriesData?.sector_order ?? Object.keys(sectorPrices)
  const benchmark = seriesData?.benchmark ?? "SPY"
  const isRelativeMode = performanceMode === "vs SPY"
  const activeSeries = isRelativeMode ? sectorRelativePrices : sectorPrices
  const hasChartSeries = order.some(name => {
    const series = activeSeries[name]
    return Array.isArray(series) && series.length > 0
  })
  const chartDescription = isRelativeMode
    ? `${timeframeLabel(timeframe)} sector ETF return versus ${benchmark}, normalized to 0%.`
    : `${timeframeLabel(timeframe)} sector ETF price return, normalized to 0%.`
  const chartEmptyMessage = isRelativeMode
    ? `No ${benchmark}-relative sector series available.`
    : "No sector ETF price series available."

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Sector Metrics</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              mutation.mutate({
                rows,
                timestamp: typeof data?.timestamp === "string" ? data.timestamp : null,
              })
              setIsOpen(true)
            }}
            disabled={mutation.isPending || rows.length === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-lg bg-blue-50 text-blue-600 border border-blue-200 hover:bg-blue-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Sparkles size={14} />
            AI Overview
          </button>
          <RefreshButton queryKeys={refreshQueryKeys} beforeRefetch={refreshMarketSnapshots} />
        </div>
      </div>

      {showPanel && (
        <div className="mb-6 rounded-xl border border-blue-200 bg-white overflow-hidden">
          <button
            onClick={() => setIsOpen(o => !o)}
            className="w-full flex items-center justify-between px-4 py-3 bg-blue-50 hover:bg-blue-100 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Sparkles size={14} className="text-blue-500" />
              <span className="text-sm font-semibold text-blue-700">AI Overview</span>
            </div>
            <ChevronDown
              size={16}
              className={`text-blue-500 transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`}
            />
          </button>

          {isOpen && (
            <div className="px-4 py-4">
              {mutation.isPending && (
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                  Analyzing sector data...
                </div>
              )}
              {mutation.isError && (
                <p className="text-sm text-red-600">
                  {String(mutation.error) || "Analysis failed. Please try again."}
                </p>
              )}
              {analysisText && (
                <p className="whitespace-pre-wrap text-sm text-gray-700 leading-relaxed">
                  {analysisText}
                </p>
              )}
            </div>
          )}
        </div>
      )}

      <div className="mb-6 flex flex-wrap items-center gap-3">
        <SegmentedControl
          options={TIMEFRAME_OPTIONS}
          value={timeframe}
          onChange={setTimeframe}
        />
        <SegmentedControl
          options={PERFORMANCE_MODES.map(mode => ({ value: mode, label: mode }))}
          value={performanceMode}
          onChange={setPerformanceMode}
        />
        <SegmentedControl
          options={UNIFIED_VIEW_MODES.map(mode => ({ value: mode, label: mode }))}
          value={viewMode}
          onChange={setViewMode}
        />
      </div>

      {isSeriesLoading && <LoadingSpinner message="Fetching sector chart data..." />}
      {!isSeriesLoading && seriesError && <ErrorMessage message={String(seriesError)} />}
      {seriesData && !isSeriesLoading && !seriesError && !hasChartSeries && (
        <Notice tone="info">{chartEmptyMessage}</Notice>
      )}
      {seriesData && !isSeriesLoading && !seriesError && hasChartSeries && viewMode === "Grid" && (
        <div className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {order.map(sector => {
            const series = activeSeries[sector]
            if (!series || series.length === 0) return null
            const ret = calcReturn(series)
            const chartData = isRelativeMode ? normalizeReturnSeries(series) : series
            return (
              <ChartTile
                key={sector}
                title={sector}
                subtitle={isRelativeMode ? `Relative to ${benchmark}` : undefined}
                meta={ret != null ? (
                  <span className={`text-xs font-semibold ${ret >= 0 ? "text-positive" : "text-negative"}`}>
                    {formatPercent(ret)}
                  </span>
                ) : undefined}
              >
                <TimeSeriesChart
                  data={chartData}
                  height={160}
                  timeframe={timeframe}
                  zeroLine={isRelativeMode}
                  yFormatter={isRelativeMode ? formatPercent : undefined}
                  tooltipFormatter={isRelativeMode ? formatPercent : undefined}
                />
              </ChartTile>
            )
          })}
        </div>
      )}
      {seriesData && !isSeriesLoading && !seriesError && hasChartSeries && viewMode === "Unified" && (
        <div className="mb-6">
          <UnifiedPerformanceView
            order={order}
            seriesByName={activeSeries}
            timeframe={timeframe}
            timeframeLabel={timeframeLabel(timeframe)}
            itemLabel="sectors"
            title={isRelativeMode ? `Unified Performance vs ${benchmark}` : "Unified Sector Performance"}
            description={chartDescription}
            emptyMessage={chartEmptyMessage}
          />
        </div>
      )}

      {isLoading && <LoadingSpinner message="Fetching sector metrics..." />}
      {!isLoading && (error || !data) && <ErrorMessage message={String(error) || "Failed to load"} />}

      {data && !isLoading && (
        <>
          {data.timestamp && (
            <p className="text-xs text-gray-400 mb-4">As of: {new Date(data.timestamp as string).toLocaleString()}</p>
          )}
          <DataTable columns={columns} rows={rows} />
        </>
      )}
    </div>
  )
}
