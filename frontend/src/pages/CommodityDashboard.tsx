import { useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchCommodities } from "@/lib/api"
import { TimeSeriesChart, calcReturn, type DataPoint } from "@/components/shared/TimeSeriesChart"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { SegmentedControl } from "@/components/shared/FormControls"
import { PageHeader } from "@/components/shared/PageHeader"
import { ChartTile } from "@/components/shared/ChartTile"
import { Notice } from "@/components/shared/Notice"
import { UnifiedPerformanceView } from "@/components/shared/UnifiedPerformanceView"
import { UNIFIED_VIEW_MODES, type UnifiedViewMode } from "@/lib/unifiedViewMode"

const TIMEFRAMES = ["This Week", "Daily", "Weekly", "Monthly"] as const
const TIMEFRAME_OPTIONS = TIMEFRAMES.map(tf => ({
  value: tf,
  label: tf === "This Week" ? "Past Week" : tf,
}))
type Timeframe = typeof TIMEFRAMES[number]

function timeframeLabel(timeframe: Timeframe): string {
  return timeframe === "This Week" ? "Past Week" : timeframe
}

export function CommodityDashboard() {
  const [timeframe, setTimeframe] = useState<Timeframe>("This Week")
  const [viewMode, setViewMode] = useState<UnifiedViewMode>("Grid")
  const { data, isLoading, error } = useApiQuery(
    ["commodities", timeframe],
    () => fetchCommodities(timeframe),
  )

  const commodities: Record<string, DataPoint[]> = data?.commodities ?? {}
  const order: string[] = data?.commodity_order ?? Object.keys(commodities)
  const hasSeries = order.some(name => {
    const series = commodities[name]
    return Array.isArray(series) && series.length > 0
  })

  return (
    <div>
      <PageHeader
        title="Commodity Dashboard"
        subtitle="Cross-commodity trend snapshots in a compact chart grid and comparable performance view."
        actions={<RefreshButton queryKeys={[["commodities", timeframe]]} />}
      />
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <SegmentedControl
          options={TIMEFRAME_OPTIONS}
          value={timeframe}
          onChange={setTimeframe}
        />
        <SegmentedControl
          options={UNIFIED_VIEW_MODES.map(mode => ({ value: mode, label: mode }))}
          value={viewMode}
          onChange={setViewMode}
        />
      </div>
      {isLoading && <LoadingSpinner />}
      {!isLoading && error && <ErrorMessage message={String(error)} />}
      {data && !isLoading && !error && !hasSeries && (
        <Notice tone="info">No price series available.</Notice>
      )}
      {data && !isLoading && hasSeries && viewMode === "Grid" && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {order.map(name => {
            const series = commodities[name]
            if (!series || series.length === 0) return null
            const ret = calcReturn(series)
            return (
              <ChartTile
                key={name}
                title={name}
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
      {data && !isLoading && hasSeries && viewMode === "Unified" && (
        <UnifiedPerformanceView
          order={order}
          seriesByName={commodities}
          timeframe={timeframe}
          timeframeLabel={timeframeLabel(timeframe)}
          itemLabel="commodities"
        />
      )}
    </div>
  )
}
