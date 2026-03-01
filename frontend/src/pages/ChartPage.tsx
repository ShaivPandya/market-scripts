import { useState, useMemo } from "react"
import { useMutation } from "@tanstack/react-query"
import { runChart } from "@/lib/api"
import { TimeSeriesChart, type SeriesDef } from "@/components/shared/TimeSeriesChart"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"

const LOOKBACKS = ["3M", "1Y", "2Y", "5Y"] as const

const MA_COLUMNS = ["100D SMA", "150D SMA", "200D SMA", "40W SMA", "200W SMA", "10M SMA", "20M SMA"]
const ROC_COLUMNS = ["1M ROC", "3M ROC", "12M ROC"]

const PRICE_SERIES: SeriesDef[] = [
  { key: "Close",    color: "#1f77b4", strokeWidth: 2,   opacity: 1    },
  { key: "100D SMA", color: "#ff7f0e", strokeWidth: 1,   opacity: 0.75 },
  { key: "150D SMA", color: "#2ca02c", strokeWidth: 1,   opacity: 0.75 },
  { key: "200D SMA", color: "#d62728", strokeWidth: 1,   opacity: 0.75 },
  { key: "40W SMA",  color: "#9467bd", strokeWidth: 1,   opacity: 0.75 },
  { key: "200W SMA", color: "#8c564b", strokeWidth: 1,   opacity: 0.75 },
  { key: "10M SMA",  color: "#e377c2", strokeWidth: 1,   opacity: 0.75 },
  { key: "20M SMA",  color: "#7f7f7f", strokeWidth: 1,   opacity: 0.75 },
]

const ROC_SERIES: SeriesDef[] = [
  { key: "1M ROC",  color: "#1f77b4" },
  { key: "3M ROC",  color: "#ff7f0e" },
  { key: "12M ROC", color: "#2ca02c" },
]

function lookbackCutoff(lookback: string): Date {
  const now = new Date()
  switch (lookback) {
    case "3M": return new Date(now.getFullYear(), now.getMonth() - 3, now.getDate())
    case "1Y": return new Date(now.getFullYear() - 1, now.getMonth(), now.getDate())
    case "2Y": return new Date(now.getFullYear() - 2, now.getMonth(), now.getDate())
    default:   return new Date(0)
  }
}

export function ChartPage() {
  const [ticker, setTicker] = useState("SPY")
  const [lookback, setLookback] = useState<string>("2Y")

  const mutation = useMutation({ mutationFn: runChart })

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    // Always fetch the full 5Y dataset; timeframe filtering is done client-side
    mutation.mutate({ ticker: ticker.trim().toUpperCase(), lookback: "5Y" })
  }

  const data = mutation.data

  // Parse the full 5Y datasets once when data arrives
  const allPriceData = useMemo<Record<string, unknown>[]>(() => (
    (data?.price_data ?? []).map((r: Record<string, unknown>) => {
      const pt: Record<string, unknown> = {
        date: String(r["Date"] ?? r["date"] ?? r["index"] ?? ""),
        Close: r["Close"] != null ? Number(r["Close"]) : null,
      }
      for (const col of MA_COLUMNS) {
        const v = r[col]
        pt[col] = v != null && v !== "" ? Number(v) : null
      }
      return pt
    }).filter((d: Record<string, unknown>) => d.date)
  ), [data])

  const allRocData = useMemo<Record<string, unknown>[]>(() => (
    (data?.roc_data ?? []).map((r: Record<string, unknown>) => {
      const pt: Record<string, unknown> = {
        date: String(r["Date"] ?? r["date"] ?? r["index"] ?? ""),
      }
      for (const col of ROC_COLUMNS) {
        const v = r[col]
        pt[col] = v != null && v !== "" ? Number(v) : null
      }
      return pt
    }).filter((d: Record<string, unknown>) => d.date)
  ), [data])

  // Slice to the selected lookback client-side — instant, no re-fetch
  const cutoff = lookbackCutoff(lookback)
  const priceMultiData = useMemo(
    () => allPriceData.filter(d => new Date(String(d.date)) >= cutoff),
    [allPriceData, lookback] // eslint-disable-line react-hooks/exhaustive-deps
  )
  const rocMultiData = useMemo(
    () => allRocData.filter(d => new Date(String(d.date)) >= cutoff),
    [allRocData, lookback] // eslint-disable-line react-hooks/exhaustive-deps
  )

  const summaryRows: Record<string, unknown>[] = Array.isArray(data?.summary) ? data.summary : []
  const summaryCols: ColumnDef[] = summaryRows.length > 0
    ? Object.keys(summaryRows[0]).map(k => ({
        key: k,
        header: k,
        colorFn: k.toLowerCase().includes("bias")
          ? (v: unknown) => (String(v).toLowerCase() === "bullish" ? "green" : "red")
          : undefined,
      }))
    : []

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Chart</h1>

      <form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Ticker</label>
          <input
            type="text"
            value={ticker}
            onChange={e => setTicker(e.target.value)}
            placeholder="SPY"
            className="border rounded px-3 py-1.5 text-sm w-28 uppercase"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Lookback</label>
          <div className="flex gap-1">
            {LOOKBACKS.map(l => (
              <button
                key={l}
                type="button"
                onClick={() => setLookback(l)}
                className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${lookback === l ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"}`}
              >
                {l}
              </button>
            ))}
          </div>
        </div>
        <button
          type="submit"
          disabled={mutation.isPending}
          className="px-4 py-1.5 rounded bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
        >
          {mutation.isPending ? "Analyzing..." : "Analyze"}
        </button>
      </form>

      {mutation.isPending && <LoadingSpinner message="Fetching chart data..." />}
      {mutation.isError && <ErrorMessage message={String(mutation.error)} />}

      {data && !mutation.isPending && (
        <div className="space-y-6">
          {priceMultiData.length > 0 && (
            <div>
              <h2 className="text-base font-semibold mb-2">{ticker} — Price</h2>
              <TimeSeriesChart multiData={priceMultiData} series={PRICE_SERIES} height={280} />
            </div>
          )}
          {rocMultiData.length > 0 && (
            <div>
              <h2 className="text-base font-semibold mb-2">Rate of Change</h2>
              <TimeSeriesChart multiData={rocMultiData} series={ROC_SERIES} height={220} zeroLine />
            </div>
          )}
          {summaryRows.length > 0 && (
            <div>
              <h2 className="text-base font-semibold mb-2">Signal Summary</h2>
              <DataTable columns={summaryCols} rows={summaryRows} />
            </div>
          )}
        </div>
      )}

      {!data && !mutation.isPending && !mutation.isError && (
        <p className="text-gray-400 text-sm">Enter a ticker and click Analyze to view the chart.</p>
      )}
    </div>
  )
}
