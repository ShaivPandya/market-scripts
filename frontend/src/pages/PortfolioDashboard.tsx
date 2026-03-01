import { useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchPortfolio } from "@/lib/api"
import { TimeSeriesChart, type DataPoint } from "@/components/shared/TimeSeriesChart"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"

const TIMEFRAMES = ["This Week", "Daily", "Weekly", "Monthly"] as const
type Timeframe = typeof TIMEFRAMES[number]

export function PortfolioDashboard() {
  const [timeframe, setTimeframe] = useState<Timeframe>("This Week")

  const { data, isLoading, error } = useApiQuery(
    ["portfolio", timeframe],
    () => fetchPortfolio(timeframe),
  )

  const positions: Record<string, DataPoint[]> = data?.positions ?? {}
  const order: string[] = data?.position_order ?? Object.keys(positions)

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold">Portfolio Dashboard</h1>
        <RefreshButton queryKeys={[["portfolio", timeframe]]} />
      </div>

      <div className="flex gap-2 mb-6">
        {TIMEFRAMES.map(tf => (
          <button
            key={tf}
            onClick={() => setTimeframe(tf)}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              timeframe === tf
                ? "bg-blue-600 text-white"
                : "bg-gray-100 text-gray-700 hover:bg-gray-200"
            }`}
          >
            {tf}
          </button>
        ))}
      </div>

      {isLoading && <LoadingSpinner message="Fetching portfolio data..." />}
      {!isLoading && (error || !data) && <ErrorMessage message={String(error) || "Failed to load"} />}

      {data && !isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {order.map(ticker => {
            const series = positions[ticker]
            if (!series || series.length === 0) return null
            return (
              <div key={ticker} className="rounded-lg border bg-white p-4 shadow-sm">
                <p className="text-sm font-semibold text-gray-700 mb-2">{ticker}</p>
                <TimeSeriesChart data={series} height={160} />
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
