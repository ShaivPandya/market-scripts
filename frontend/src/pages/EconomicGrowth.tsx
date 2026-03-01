import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { ChevronDown, Sparkles } from "lucide-react"
import { useApiQuery } from "@/hooks/useApiQuery"
import { fetchEconomicGrowth, analyzeEconomicGrowth } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { RefreshButton } from "@/components/shared/RefreshButton"
import { colorPositiveNegative, colorReturnVsBenchmark } from "@/lib/colors"

export function EconomicGrowth() {
  const { data, isLoading, error } = useApiQuery(
    ["economic-growth"],
    fetchEconomicGrowth,
  )

  const [isOpen, setIsOpen] = useState(false)
  const mutation = useMutation({ mutationFn: analyzeEconomicGrowth })

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

  const showPanel = mutation.data || mutation.isPending || mutation.isError

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
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              mutation.mutate({
                commodities: data.commodities ?? {},
                equities: data.equities ?? {},
                currencies: data.currencies ?? {},
                equity_periods: periods,
                currency_periods: currencyPeriods,
              })
              setIsOpen(true)
            }}
            disabled={mutation.isPending}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-lg bg-indigo-50 text-indigo-700 border border-indigo-200 hover:bg-indigo-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Sparkles size={14} />
            AI Overview
          </button>
          <RefreshButton queryKeys={[["economic-growth"]]} />
        </div>
      </div>

      {showPanel && (
        <div className="mb-6 rounded-lg border border-indigo-200 bg-white overflow-hidden">
          <button
            onClick={() => setIsOpen(o => !o)}
            className="w-full flex items-center justify-between px-4 py-3 bg-indigo-50 hover:bg-indigo-100 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Sparkles size={14} className="text-indigo-600" />
              <span className="text-sm font-semibold text-indigo-800">AI Overview</span>
            </div>
            <ChevronDown
              size={16}
              className={`text-indigo-600 transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`}
            />
          </button>

          {isOpen && (
            <div className="px-4 py-4">
              {mutation.isPending && (
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <div className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
                  Analyzing market data...
                </div>
              )}
              {mutation.isError && (
                <p className="text-sm text-red-600">
                  {String(mutation.error) || "Analysis failed. Please try again."}
                </p>
              )}
              {mutation.data && (
                <p className="whitespace-pre-wrap text-sm text-gray-700 leading-relaxed">
                  {mutation.data.analysis}
                </p>
              )}
            </div>
          )}
        </div>
      )}

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
