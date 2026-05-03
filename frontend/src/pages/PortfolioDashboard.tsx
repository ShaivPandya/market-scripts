import { useMemo, useState } from "react"
import { useRegisterScreenContext } from "@/contexts/ScreenContext"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchPortfolioAllTimeframes } from "@/lib/api"
import { TimeSeriesChart, calcReturn, type DataPoint } from "@/components/shared/TimeSeriesChart"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { SegmentedControl } from "@/components/shared/FormControls"
import { PortfolioEditor } from "@/components/PortfolioEditor"
import { Notice } from "@/components/shared/Notice"
import { PageHeader } from "@/components/shared/PageHeader"
import { ChartTile } from "@/components/shared/ChartTile"

const TIMEFRAMES = ["This Week", "Daily", "Weekly", "Monthly"] as const
type Timeframe = typeof TIMEFRAMES[number]
type TimeframePayload = {
  positions?: Record<string, DataPoint[]>
  position_order?: string[]
  warning?: string
}
type PortfolioAllTimeframesResponse = {
  timeframes?: Partial<Record<Timeframe, TimeframePayload>>
  warning?: string
}

export function PortfolioDashboard() {
  const [timeframe, setTimeframe] = useState<Timeframe>("This Week")
  const [editOpen, setEditOpen] = useState(false)

  const { data, isLoading, error } = useApiQuery<PortfolioAllTimeframesResponse>(
    ["portfolio", "all_timeframes"],
    fetchPortfolioAllTimeframes,
  )
  const timeframeData = data?.timeframes?.[timeframe]
  const positions: Record<string, DataPoint[]> = useMemo(
    () => timeframeData?.positions ?? {},
    [timeframeData?.positions],
  )
  const order: string[] = useMemo(
    () => timeframeData?.position_order ?? Object.keys(positions),
    [positions, timeframeData?.position_order],
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
      "Timeframe": timeframe,
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
      filters: { timeframe },
      summary: `Portfolio with ${order.length} positions, viewing ${timeframe}`,
      correspondingTools: ["get_portfolio"],
    }
  }, [timeframeData, order, positions, timeframe])
  useRegisterScreenContext(screenCtx)

  return (
    <div>
      <PageHeader
        title="Portfolio Dashboard"
        subtitle="Portfolio positions organized as an adaptive chart grid with quick access to each dossier."
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
      {!isLoading && !error && warning && (
        <Notice tone="warning" className="mb-4">{warning}</Notice>
      )}
      {!isLoading && !error && timeframeData && !hasSeries && (
        <Notice tone="info">No price series available.</Notice>
      )}

      {timeframeData && !isLoading && hasSeries && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {order.map(ticker => {
            const series = positions[ticker]
            if (!series || series.length === 0) return null
            const ret = calcReturn(series)
            return (
              <ChartTile
                key={ticker}
                title={ticker}
                href={`/dossier/${ticker}`}
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

      <PortfolioEditor open={editOpen} onOpenChange={setEditOpen} />
    </div>
  )
}
