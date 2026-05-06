import { useState, useMemo } from "react"
import { useMutation } from "@tanstack/react-query"
import { ChevronDown, Sparkles } from "lucide-react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { useSessionAiOverview } from "@/hooks/useSessionAiOverview"
import { fetchLaborMarket, analyzeLaborMarket } from "@/lib/api"
import { TimeSeriesChart, type DataPoint } from "@/components/shared/TimeSeriesChart"
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"

type Timeframe = "1Y" | "3Y" | "5Y" | "Max"

const TIMEFRAME_DAYS: Record<Timeframe, number | null> = {
  "1Y": 365,
  "3Y": 365 * 3,
  "5Y": 365 * 5,
  "Max": null,
}

const SERIES_ORDER = [
  "initial_claims",
  "continuing_claims",
  "median_weeks_unemployed",
  "weekly_hours",
  "wage_growth",
  "job_openings",
] as const

function filterByTimeframe(dates: string[], values: number[], days: number | null): DataPoint[] {
  if (!dates || !values) return []
  const cutoff = days ? new Date(Date.now() - days * 24 * 60 * 60 * 1000) : null
  const points: DataPoint[] = []
  for (let i = 0; i < dates.length; i++) {
    if (cutoff && new Date(dates[i]) < cutoff) continue
    points.push({ date: dates[i], value: values[i] })
  }
  return points
}

function fmtValue(value: number | null, unit: string): string {
  if (value == null) return "N/A"
  if (unit === "%") return `${value.toFixed(1)}%`
  if (unit === "hours") return `${value.toFixed(1)}h`
  if (unit === "weeks") return `${value.toFixed(1)} wks`
  if (unit === "thousands") {
    if (value >= 1000) return `${(value / 1000).toFixed(1)}M`
    return `${Math.round(value)}K`
  }
  return value.toFixed(2)
}

function fmtChange(change: number | null, unit: string): string | null {
  if (change == null) return null
  const sign = change >= 0 ? "+" : ""
  if (unit === "%") return `${sign}${change.toFixed(2)}pp`
  if (unit === "hours") return `${sign}${change.toFixed(2)}h`
  if (unit === "weeks") return `${sign}${change.toFixed(2)} wks`
  if (unit === "thousands") return `${sign}${Math.round(change)}K`
  return `${sign}${change.toFixed(2)}`
}

function yFormatter(unit: string) {
  return (v: number) => {
    if (unit === "%") return `${v.toFixed(1)}%`
    if (unit === "hours") return `${v.toFixed(1)}`
    if (unit === "weeks") return `${v.toFixed(1)}`
    if (unit === "thousands") {
      if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}M`
      return `${Math.round(v)}K`
    }
    return v.toFixed(1)
  }
}

export function LaborMarket() {
  const [timeframe, setTimeframe] = useState<Timeframe>("3Y")

  const { analysis: persistedAnalysis, isOpen, setIsOpen, setAnalysis: setPersistedAnalysis } =
    useSessionAiOverview("ai-overview:labor-market")

  const mutation = useMutation({
    mutationFn: analyzeLaborMarket,
    onSuccess: data => {
      const analysis = typeof data?.analysis === "string" ? data.analysis : null
      if (analysis) setPersistedAnalysis(analysis)
    },
  })

  const { data, isLoading, error } = useApiQuery(["labor-market"], () => fetchLaborMarket())

  const liveAnalysis = typeof mutation.data?.analysis === "string" ? mutation.data.analysis : null
  const analysisText = liveAnalysis ?? persistedAnalysis
  const showPanel = Boolean(analysisText || mutation.isPending || mutation.isError)

  const days = TIMEFRAME_DAYS[timeframe]

  const filteredSeries = useMemo(() => {
    if (!data?.series) return {}
    const result: Record<string, DataPoint[]> = {}
    for (const key of SERIES_ORDER) {
      const s = data.series[key]
      if (!s) continue
      result[key] = filterByTimeframe(s.dates, s.values, days)
    }
    return result
  }, [data, days])

  return (
    <div>
      <div className="flex items-start justify-between mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Labor Market</h1>
        <div className="flex items-center gap-3">
          <button
            onClick={() => {
              if (!data) return
              const series = data.series ?? {}
              const latest = data.latest ?? {}
              mutation.mutate({
                latest,
                series_labels: Object.fromEntries(
                  Object.entries(series).map(([k, v]) => [k, (v as { label: string }).label]),
                ),
                series_units: Object.fromEntries(
                  Object.entries(series).map(([k, v]) => [k, (v as { unit: string }).unit]),
                ),
                timestamp: data.timestamp ?? null,
              })
              setIsOpen(true)
            }}
            disabled={mutation.isPending || !data}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-lg bg-blue-50 text-blue-600 border border-blue-200 hover:bg-blue-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Sparkles size={14} />
            AI Overview
          </button>
          <RefreshButton
            clearBackendCache={false}
            beforeRefetch={() => fetchLaborMarket({ force_refresh: true })}
            queryKeys={[["labor-market"]]}
          />
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
                  Analyzing labor market data...
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

      {isLoading && <LoadingSpinner message="Fetching labor market data..." />}
      {!isLoading && (error || !data) && <ErrorMessage message={String(error) || "Failed to load"} />}

      {data && !isLoading && (
        <>
          {/* Metric cards */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
            {SERIES_ORDER.map(key => {
              const latest = (data.latest ?? {})[key] as { value: number | null; date: string; change: number | null } | undefined
              const series = (data.series ?? {})[key] as { label: string; unit: string } | undefined
              if (!series) return null
              const unit = series.unit
              const val = latest?.value ?? null
              const chg = latest ? fmtChange(latest.change, unit) : null
              return (
                <MetricCard
                  key={key}
                  title={series.label}
                  value={fmtValue(val, unit)}
                  subtitle={chg ? `${chg} prev` : undefined}
                />
              )
            })}
          </div>

          {/* Timeframe selector */}
          <div className="flex items-center gap-1 mb-5">
            {(["1Y", "3Y", "5Y", "Max"] as Timeframe[]).map(tf => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                  timeframe === tf
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
              >
                {tf}
              </button>
            ))}
          </div>

          {/* Charts grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {SERIES_ORDER.map(key => {
              const series = (data.series ?? {})[key] as { label: string; unit: string } | undefined
              const points = filteredSeries[key] ?? []
              if (!series) return null
              const unit = series.unit
              return (
                <div key={key} className="theme-surface rounded-xl p-4">
                  <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-3">
                    {series.label}
                  </p>
                  <TimeSeriesChart
                    data={points}
                    height={200}
                    yFormatter={yFormatter(unit)}
                    tooltipFormatter={yFormatter(unit)}
                    zeroLine={unit === "%"}
                    timeframe="Monthly"
                  />
                </div>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}
