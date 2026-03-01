import { useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchIndexDashboard } from "@/lib/api"
import { TimeSeriesChart, type DataPoint } from "@/components/shared/TimeSeriesChart"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"

const TIMEFRAMES = ["This Week", "Daily", "Weekly", "Monthly"] as const
type Timeframe = typeof TIMEFRAMES[number]
const DEFAULT_INDEX_ORDER = ["S&P 500", "NASDAQ", "Russell 2000", "STOXX 600", "DAX", "Nikkei 225"]

export function IndexDashboard() {
  const [timeframe, setTimeframe] = useState<Timeframe>("This Week")
  const { data, isLoading, error } = useApiQuery(
    ["index-dashboard", timeframe],
    () => fetchIndexDashboard(timeframe),
  )

  const indices: Record<string, DataPoint[]> = data?.indices ?? {}
  const order: string[] = data?.index_order ?? [
    ...DEFAULT_INDEX_ORDER.filter(name => indices[name]?.length),
    ...Object.keys(indices).filter(name => !DEFAULT_INDEX_ORDER.includes(name)),
  ]

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold">Index Dashboard</h1>
        <RefreshButton queryKeys={[["index-dashboard", timeframe]]} />
      </div>
      <div className="flex gap-2 mb-6">
        {TIMEFRAMES.map(tf => (
          <button
            key={tf}
            onClick={() => setTimeframe(tf)}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              timeframe === tf ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"
            }`}
          >
            {tf}
          </button>
        ))}
      </div>
      {isLoading && <LoadingSpinner />}
      {!isLoading && error && <ErrorMessage message={String(error)} />}
      {data && !isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {order.map(name => {
            const series = indices[name]
            if (!series || series.length === 0) return null
            return (
              <div key={name} className="rounded-lg border bg-white p-4 shadow-sm">
                <p className="text-sm font-semibold text-gray-700 mb-2">{name}</p>
                <TimeSeriesChart data={series} height={160} />
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
