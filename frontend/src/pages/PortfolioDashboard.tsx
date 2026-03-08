import { useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchPortfolioAllTimeframes, fetchThesisStatus, type ThesisStatus } from "@/lib/api"
import { ThesisUpload } from "@/components/ThesisUpload"
import { TimeSeriesChart, calcReturn, type DataPoint } from "@/components/shared/TimeSeriesChart"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { SegmentedControl } from "@/components/shared/FormControls"
import { PortfolioEditor } from "@/components/PortfolioEditor"

const TIMEFRAMES = ["This Week", "Daily", "Weekly", "Monthly"] as const
type Timeframe = typeof TIMEFRAMES[number]
type TimeframePayload = {
  positions?: Record<string, DataPoint[]>
  position_order?: string[]
}
type PortfolioAllTimeframesResponse = {
  timeframes?: Partial<Record<Timeframe, TimeframePayload>>
}

export function PortfolioDashboard() {
  const [timeframe, setTimeframe] = useState<Timeframe>("This Week")
  const [editOpen, setEditOpen] = useState(false)

  const { data, isLoading, error } = useApiQuery<PortfolioAllTimeframesResponse>(
    ["portfolio", "all_timeframes"],
    fetchPortfolioAllTimeframes,
  )
  const { data: thesisStatus } = useApiQuery<Record<string, ThesisStatus>>(["thesis", "status"], fetchThesisStatus)

  const timeframeData = data?.timeframes?.[timeframe]
  const positions: Record<string, DataPoint[]> = timeframeData?.positions ?? {}
  const order: string[] = timeframeData?.position_order ?? Object.keys(positions)

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Portfolio Dashboard</h1>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setEditOpen(true)}
            className="rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm font-medium text-gray-600 hover:bg-gray-50 transition-colors"
          >
            Edit Portfolio
          </button>
          <RefreshButton queryKeys={[["portfolio", "all_timeframes"], ["thesis", "status"]]} />
        </div>
      </div>

      <div className="mb-6">
        <SegmentedControl
          options={TIMEFRAMES.map(tf => ({ value: tf, label: tf }))}
          value={timeframe}
          onChange={setTimeframe}
        />
      </div>

      {isLoading && <LoadingSpinner message="Fetching portfolio data..." />}
      {!isLoading && (error || !data || !timeframeData) && (
        <ErrorMessage message={String(error) || "Failed to load"} />
      )}

      {timeframeData && !isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {order.map(ticker => {
            const series = positions[ticker]
            if (!series || series.length === 0) return null
            const ret = calcReturn(series)
            return (
              <div key={ticker} className="rounded-xl border bg-white p-4 shadow-sm">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-semibold text-gray-700">{ticker}</p>
                    <ThesisUpload ticker={ticker} status={thesisStatus?.[ticker] ?? "missing"} />
                  </div>
                  {ret != null && (
                    <span className={`text-xs font-medium ${ret >= 0 ? "text-green-600" : "text-red-600"}`}>
                      {ret >= 0 ? "+" : ""}{ret.toFixed(2)}%
                    </span>
                  )}
                </div>
                <TimeSeriesChart data={series} height={160} timeframe={timeframe} />
              </div>
            )
          })}
        </div>
      )}

      <PortfolioEditor open={editOpen} onOpenChange={setEditOpen} />
    </div>
  )
}
