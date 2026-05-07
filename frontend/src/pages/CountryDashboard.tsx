import { useState } from "react"
import { useDashboardTimeframePrefetch } from "@/hooks/useDashboardTimeframePrefetch"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchCountryDashboard } from "@/lib/api"
import { TimeSeriesChart, type DataPoint } from "@/components/shared/TimeSeriesChart"
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { SegmentedControl } from "@/components/shared/FormControls"
import { PageHeader } from "@/components/shared/PageHeader"
import { ChartTile } from "@/components/shared/ChartTile"

const METRICS = ["Inflation", "Unemployment", "GDP"] as const
type Metric = typeof METRICS[number]
const COUNTRY_DASHBOARD_STALE_TIME_MS = 24 * 60 * 60 * 1000
const countryDashboardStaleTime = () => COUNTRY_DASHBOARD_STALE_TIME_MS

const SOURCE_DISPLAY: Record<string, string> = {
  fred: "FRED", statcan_wds: "Statcan", ons: "ONS",
  eurostat: "Eurostat", snb: "SNB", oecd: "OECD",
}

export function CountryDashboard() {
  const [metric, setMetric] = useState<Metric>("Inflation")
  const [currentTimestamp] = useState(() => Date.now())
  const { data, isLoading, error, isSuccess } = useApiQuery(
    ["country-dashboard", metric],
    () => fetchCountryDashboard(metric),
    COUNTRY_DASHBOARD_STALE_TIME_MS,
  )
  useDashboardTimeframePrefetch({
    queryKeyRoot: "country-dashboard",
    timeframes: METRICS,
    activeTimeframe: metric,
    isReady: isSuccess,
    fetchTimeframe: fetchCountryDashboard,
    staleTimeForTimeframe: countryDashboardStaleTime,
  })

  const METRIC_LABEL: Record<Metric, string> = {
    Inflation: "CPI YoY %",
    Unemployment: "Rate %",
    GDP: "Real GDP YoY %",
  }

  return (
    <div>
      <PageHeader
        title="Country Dashboard"
        subtitle="Cross-country macro comparisons with content-first cards that adapt cleanly between compact and wide layouts."
        actions={<RefreshButton queryKeys={[["country-dashboard", metric]]} />}
      />

      <div className="mb-6">
        <SegmentedControl
          options={METRICS.map(m => ({ value: m, label: m }))}
          value={metric}
          onChange={setMetric}
        />
      </div>

      {isLoading && <LoadingSpinner message={`Fetching ${metric} data...`} />}
      {!isLoading && (error || !data) && <ErrorMessage message={String(error) || "Failed to load"} />}

      {data && !isLoading && (
        <>
          <p className="mb-4 text-xs text-subtle">Showing: {METRIC_LABEL[metric]}</p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {(data.country_order as string[]).map(country => {
              const series: DataPoint[] | null = (data.countries as Record<string, DataPoint[] | null>)[country]
              const obsDate = (data.latest_observation_dates as Record<string, string>)[country]
              const source = SOURCE_DISPLAY[(data.series_used as Record<string, string>)[country]] ?? (data.series_used as Record<string, string>)[country]
              const errors = (data.errors as Record<string, string[]>)[country] ?? []

              if (!series || series.length === 0) {
                return (
                  <ChartTile key={country} title={country} subtitle="N/A" meta={source ? <span className="caption">{source}</span> : undefined}>
                    {errors[0] ? <p className="caption truncate" title={errors[0]}>{errors[0].slice(0, 100)}</p> : <MetricCard title="Status" value="No data" />}
                  </ChartTile>
                )
              }

              const latest = series[series.length - 1]?.value
              const prev = series.length > 1 ? series[series.length - 2]?.value : null
              const delta = prev != null && latest != null ? latest - prev : null
              const ageDays = obsDate ? Math.floor((currentTimestamp - new Date(obsDate).getTime()) / 86400000) : null

              return (
                <ChartTile
                  key={country}
                  title={country}
                  subtitle={latest != null ? `${latest.toFixed(1)}%` : "N/A"}
                  meta={source ? <span className="caption">{source}</span> : undefined}
                >
                  {delta != null ? (
                    <p className={`mb-2 text-sm font-medium ${
                      metric === "GDP"
                        ? delta >= 0 ? "text-positive" : "text-negative"
                        : delta <= 0 ? "text-positive" : "text-negative"
                    }`}>
                      {delta >= 0 ? "+" : ""}{delta.toFixed(1)}pp
                    </p>
                  ) : null}
                  {obsDate ? (
                    <p className="mb-3 text-xs text-subtle">
                      Latest: {new Date(obsDate).toLocaleDateString()}
                      {ageDays != null && ageDays > 180 && ` (stale: ${ageDays}d)`}
                    </p>
                  ) : null}
                  <TimeSeriesChart data={series} height={150} />
                </ChartTile>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}
