import { useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchCountryDashboard } from "@/lib/api"
import { TimeSeriesChart, type DataPoint } from "@/components/shared/TimeSeriesChart"
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"

const METRICS = ["Inflation", "Unemployment", "GDP"] as const
type Metric = typeof METRICS[number]

const SOURCE_DISPLAY: Record<string, string> = {
  fred: "FRED", statcan_wds: "Statcan", ons: "ONS",
  eurostat: "Eurostat", snb: "SNB", oecd: "OECD",
}

export function CountryDashboard() {
  const [metric, setMetric] = useState<Metric>("Inflation")
  const { data, isLoading, error } = useApiQuery(
    ["country-dashboard", metric],
    () => fetchCountryDashboard(metric),
    60 * 60 * 1000, // 1h stale time
  )

  const METRIC_LABEL: Record<Metric, string> = {
    Inflation: "CPI YoY %",
    Unemployment: "Rate %",
    GDP: "Real GDP YoY %",
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold">Country Dashboard</h1>
        <RefreshButton queryKeys={[["country-dashboard", metric]]} />
      </div>

      <div className="flex gap-2 mb-4">
        {METRICS.map(m => (
          <button key={m} onClick={() => setMetric(m)}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${metric === m ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"}`}>
            {m}
          </button>
        ))}
      </div>

      {isLoading && <LoadingSpinner message={`Fetching ${metric} data...`} />}
      {!isLoading && (error || !data) && <ErrorMessage message={String(error) || "Failed to load"} />}

      {data && !isLoading && (
        <>
          <p className="text-xs text-gray-400 mb-4">Showing: {METRIC_LABEL[metric]}</p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {(data.country_order as string[]).map(country => {
              const series: DataPoint[] | null = (data.countries as Record<string, DataPoint[] | null>)[country]
              const obsDate = (data.latest_observation_dates as Record<string, string>)[country]
              const source = SOURCE_DISPLAY[(data.series_used as Record<string, string>)[country]] ?? (data.series_used as Record<string, string>)[country]
              const errors = (data.errors as Record<string, string[]>)[country] ?? []

              if (!series || series.length === 0) {
                return (
                  <div key={country} className="rounded-lg border bg-white p-4 shadow-sm">
                    <MetricCard title={country} value="N/A" />
                    {errors[0] && <p className="text-xs text-gray-400 mt-1 truncate" title={errors[0]}>{errors[0].slice(0, 100)}</p>}
                  </div>
                )
              }

              const latest = series[series.length - 1]?.value
              const prev = series.length > 1 ? series[series.length - 2]?.value : null
              const delta = prev != null && latest != null ? latest - prev : null
              const ageDays = obsDate ? Math.floor((Date.now() - new Date(obsDate).getTime()) / 86400000) : null

              return (
                <div key={country} className="rounded-lg border bg-white p-4 shadow-sm">
                  <div className="flex justify-between items-start mb-1">
                    <p className="text-sm font-semibold text-gray-700">{country}</p>
                    {source && <span className="text-xs text-gray-400">{source}</span>}
                  </div>
                  <p className="text-2xl font-bold text-gray-900">
                    {latest != null ? `${latest.toFixed(1)}%` : "N/A"}
                  </p>
                  {delta != null && (
                    <p className={`text-sm ${
                      metric === "GDP"
                        ? delta >= 0 ? "text-green-600" : "text-red-600"
                        : delta <= 0 ? "text-green-600" : "text-red-600"
                    }`}>
                      {delta >= 0 ? "+" : ""}{delta.toFixed(1)}pp
                    </p>
                  )}
                  {obsDate && (
                    <p className="text-xs text-gray-400 mt-1">
                      Latest: {new Date(obsDate).toLocaleDateString()}
                      {ageDays != null && ageDays > 180 && ` (stale: ${ageDays}d)`}
                    </p>
                  )}
                  <TimeSeriesChart data={series} height={150} />
                </div>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}
