import { useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchCommodities } from "@/lib/api"
import { TimeSeriesChart, calcReturn, type DataPoint } from "@/components/shared/TimeSeriesChart"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { SegmentedControl } from "@/components/shared/FormControls"
import { PageHeader } from "@/components/shared/PageHeader"
import { ChartTile } from "@/components/shared/ChartTile"

const TIMEFRAMES = ["This Week", "Daily", "Weekly", "Monthly"] as const
type Timeframe = typeof TIMEFRAMES[number]

export function CommodityDashboard() {
  const [timeframe, setTimeframe] = useState<Timeframe>("This Week")
  const { data, isLoading, error } = useApiQuery(
    ["commodities", timeframe],
    () => fetchCommodities(timeframe),
  )

  const commodities: Record<string, DataPoint[]> = data?.commodities ?? {}
  const order: string[] = data?.commodity_order ?? Object.keys(commodities)

  return (
    <div>
      <PageHeader
        title="Commodity Dashboard"
        subtitle="Cross-commodity trend snapshots with restrained chrome and comparable chart scales."
        actions={<RefreshButton queryKeys={[["commodities", timeframe]]} />}
      />
      <div className="mb-6">
        <SegmentedControl
          options={TIMEFRAMES.map(tf => ({ value: tf, label: tf }))}
          value={timeframe}
          onChange={setTimeframe}
        />
      </div>
      {isLoading && <LoadingSpinner />}
      {!isLoading && error && <ErrorMessage message={String(error)} />}
      {data && !isLoading && (
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
    </div>
  )
}
