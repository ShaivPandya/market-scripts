import { useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchBondDashboard } from "@/lib/api"
import { TimeSeriesChart, type DataPoint } from "@/components/shared/TimeSeriesChart"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { SegmentedControl } from "@/components/shared/FormControls"

const TENORS = ["2Y", "10Y", "30Y"] as const
type Tenor = typeof TENORS[number]

interface TenorData {
  series: DataPoint[]
  latest: number | null
  latest_date: string | null
  year_ago: number | null
  year_ago_date: string | null
  change_bps: number | null
}

interface CountryData {
  code: string
  name: string
  source: string
  tenors: Record<string, TenorData>
}

interface BondDashboardResponse {
  timestamp?: string
  lookback_days: number
  tenors: string[]
  country_order: string[]
  countries: Record<string, CountryData>
}

export function BondDashboard() {
  const [tenor, setTenor] = useState<Tenor>("10Y")
  const { data, isLoading, error } = useApiQuery<BondDashboardResponse>(
    ["bond-dashboard"],
    () => fetchBondDashboard(),
    5 * 60 * 1000,
  )

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Bond Dashboard</h1>
          {data?.timestamp && (
            <p className="text-xs text-gray-400 mt-0.5">
              Snapshot: {new Date(data.timestamp).toLocaleString()}
            </p>
          )}
        </div>
        <RefreshButton queryKeys={[["bond-dashboard"]]} />
      </div>

      <div className="mb-6">
        <SegmentedControl
          options={TENORS.map(t => ({ value: t, label: t }))}
          value={tenor}
          onChange={setTenor}
        />
      </div>

      {isLoading && <LoadingSpinner message="Fetching bond yield data..." />}
      {!isLoading && (error || !data) && <ErrorMessage message={String(error) || "Failed to load"} />}

      {data && !isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {(data.country_order ?? []).map(code => {
            const country = data.countries?.[code]
            if (!country) return null
            const td = country.tenors?.[tenor]
            const series = td?.series ?? []
            const latest = td?.latest
            const changeBps = td?.change_bps

            return (
              <div key={code} className="rounded-xl border bg-white p-4 shadow-sm">
                <div className="flex justify-between items-start mb-1">
                  <p className="text-sm font-semibold text-gray-700">{country.name}</p>
                  <span className="text-xs text-gray-400">{country.source}</span>
                </div>
                <p className="text-2xl font-bold text-gray-900">
                  {latest != null ? `${latest.toFixed(3)}%` : "N/A"}
                </p>
                {changeBps != null && (
                  <p className={`text-sm ${changeBps <= 0 ? "text-green-600" : "text-red-600"}`}>
                    {changeBps >= 0 ? "+" : ""}{changeBps.toFixed(1)} bps YoY
                  </p>
                )}
                {td?.latest_date && (
                  <p className="text-xs text-gray-400 mt-1">
                    Latest: {new Date(td.latest_date).toLocaleDateString()}
                  </p>
                )}
                {series.length > 0 ? (
                  <TimeSeriesChart
                    data={series}
                    height={160}
                    yFormatter={v => `${v.toFixed(1)}%`}
                    tooltipFormatter={v => `${v.toFixed(3)}%`}
                  />
                ) : (
                  <p className="text-sm text-gray-400 mt-4">No data available</p>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
