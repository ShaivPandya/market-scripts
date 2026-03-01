import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchEconomicGrowth } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { colorPositiveNegative, colorReturnVsBenchmark } from "@/lib/colors"

export function EconomicGrowth() {
  const { data, isLoading, error } = useApiQuery(
    ["economic-growth"],
    fetchEconomicGrowth,
  )

  if (isLoading) return <LoadingSpinner message="Fetching economic growth data..." />
  if (error || !data) return <ErrorMessage message={String(error) || "Failed to load"} />

  const periods: string[] = data.equity_periods ?? ["1-mo", "3-mo", "6-mo", "1-yr"]
  const currencyPeriods: string[] = data.currency_periods ?? ["1-mo", "3-mo", "6-mo"]

  const periodCols = (periods_: string[], colorFn: (v: unknown) => string): ColumnDef[] =>
    periods_.map(p => ({
      key: p,
      header: p,
      colorFn: colorFn,
    }))

  // Build rows from nested dict {name: {period: value}}
  function buildRows(
    dict: Record<string, Record<string, number | null>>,
    nameKey: string,
    periods_: string[],
  ) {
    return Object.entries(dict).map(([name, returns]) => {
      const row: Record<string, unknown> = { [nameKey]: name }
      periods_.forEach(p => {
        const val = returns[p]
        row[p] = val !== null && val !== undefined ? `${val >= 0 ? "+" : ""}${val.toFixed(1)}%` : "N/A"
      })
      return row
    })
  }

  const commodityRows = buildRows(data.commodities ?? {}, "Name", periods)
  const equityRows = buildRows(data.equities ?? {}, "Name", periods)
  const currencyRows = buildRows(data.currencies ?? {}, "Pair", currencyPeriods)

  const nameCol = (key: string): ColumnDef => ({ key, header: key, width: "160px" })

  return (
    <div>
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 tracking-tight">Economic Growth Dashboard</h1>
          {data.timestamp && (
            <p className="text-sm text-gray-400 mt-0.5">
              As of {new Date(data.timestamp).toLocaleString()}
            </p>
          )}
        </div>
        <RefreshButton queryKeys={[["economic-growth"]]} />
      </div>

      <section className="mb-8">
        <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">Commodities</h2>
        <DataTable
          columns={[nameCol("Name"), ...periodCols(periods, colorPositiveNegative)]}
          rows={commodityRows}
        />
      </section>

      <section className="mb-8">
        <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">Equities vs Benchmark</h2>
        <DataTable
          columns={[nameCol("Name"), ...periodCols(periods, colorReturnVsBenchmark)]}
          rows={equityRows}
        />
        <p className="text-xs text-gray-400 mt-1.5">
          (+) outperforming benchmark · (−) underperforming benchmark
        </p>
      </section>

      <section className="mb-8">
        <h2 className="text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3">Currencies</h2>
        <DataTable
          columns={[nameCol("Pair"), ...periodCols(currencyPeriods, colorPositiveNegative)]}
          rows={currencyRows}
        />
      </section>
    </div>
  )
}
