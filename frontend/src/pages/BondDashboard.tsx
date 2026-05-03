import { useState } from "react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchBondDashboard } from "@/lib/api"
import { TimeSeriesChart, type DataPoint } from "@/components/shared/TimeSeriesChart"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { SegmentedControl } from "@/components/shared/FormControls"
import { PageHeader } from "@/components/shared/PageHeader"
import { ChartTile } from "@/components/shared/ChartTile"

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
      <PageHeader
        title="Bond Dashboard"
        subtitle={data?.timestamp ? `Snapshot: ${new Date(data.timestamp).toLocaleString()}` : "Global sovereign yield snapshots across key maturities."}
        actions={<RefreshButton queryKeys={[["bond-dashboard"]]} />}
      />

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
              <ChartTile
                key={code}
                title={country.name}
                subtitle={latest != null ? `${latest.toFixed(3)}%` : "N/A"}
                meta={<span className="caption">{country.source}</span>}
              >
                {changeBps != null ? (
                  <p className={`mb-2 text-sm font-medium ${changeBps <= 0 ? "text-positive" : "text-negative"}`}>
                    {changeBps >= 0 ? "+" : ""}{changeBps.toFixed(1)} bps YoY
                  </p>
                ) : null}
                {td?.latest_date ? (
                  <p className="mb-3 text-xs text-subtle">
                    Latest: {new Date(td.latest_date).toLocaleDateString()}
                  </p>
                ) : null}
                {series.length > 0 ? (
                  <TimeSeriesChart
                    data={series}
                    height={160}
                    yFormatter={v => `${v.toFixed(1)}%`}
                    tooltipFormatter={v => `${v.toFixed(3)}%`}
                  />
                ) : (
                  <p className="body-copy pt-4">No data available.</p>
                )}
              </ChartTile>
            )
          })}
        </div>
      )}
    </div>
  )
}
