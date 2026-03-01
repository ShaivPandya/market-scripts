import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { runChart } from "@/lib/api"
import { TimeSeriesChart, type DataPoint } from "@/components/shared/TimeSeriesChart"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { colorPositiveNegative } from "@/lib/colors"

const LOOKBACKS = ["3M", "1Y", "2Y", "5Y"] as const

export function ChartPage() {
  const [ticker, setTicker] = useState("SPY")
  const [lookback, setLookback] = useState<string>("2Y")

  const mutation = useMutation({ mutationFn: runChart })

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    mutation.mutate({ ticker: ticker.trim().toUpperCase(), lookback })
  }

  const data = mutation.data

  // Expect price_data as [{Date, Close, ...}] or [{date, value}]
  const priceData: DataPoint[] = (data?.price_data ?? []).map((r: Record<string, unknown>) => ({
    date: String(r["Date"] ?? r["date"] ?? r["index"] ?? ""),
    value: Number(r["Close"] ?? r["value"] ?? 0),
  })).filter((d: DataPoint) => d.date && d.value != null && !isNaN(d.value as number))

  const rocData: DataPoint[] = (data?.roc_data ?? []).map((r: Record<string, unknown>) => ({
    date: String(r["Date"] ?? r["date"] ?? r["index"] ?? ""),
    value: Number(r["ROC"] ?? r["roc"] ?? r["value"] ?? 0),
  })).filter((d: DataPoint) => d.date && d.value != null && !isNaN(d.value as number))

  const summaryRows: Record<string, unknown>[] = Array.isArray(data?.summary) ? data.summary : []
  const summaryCols: ColumnDef[] = summaryRows.length > 0
    ? Object.keys(summaryRows[0]).map(k => ({
        key: k,
        header: k,
        colorFn: k.toLowerCase().includes("return") || k.toLowerCase().includes("pct")
          ? colorPositiveNegative : undefined,
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
          {priceData.length > 0 && (
            <div>
              <h2 className="text-base font-semibold mb-2">{ticker} — Price</h2>
              <TimeSeriesChart data={priceData} height={280} />
            </div>
          )}
          {rocData.length > 0 && (
            <div>
              <h2 className="text-base font-semibold mb-2">Rate of Change</h2>
              <TimeSeriesChart data={rocData} height={180} zeroLine />
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
