import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { runFxModel, fetchFxModelPairs } from "@/lib/api"
import { useApiQuery } from "@/hooks/useApiQuery"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { TimeSeriesChart, type DataPoint } from "@/components/shared/TimeSeriesChart"
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { colorPositiveNegative } from "@/lib/colors"

const DEFAULT_PAIRS = ["USDCAD", "GBPUSD", "AUDUSD", "USDJPY", "EURUSD"]

export function FXModel() {
  const { data: pairsData } = useApiQuery(["fx-model-pairs"], fetchFxModelPairs)
  const availablePairs: string[] = pairsData?.pairs ?? DEFAULT_PAIRS

  const [pair, setPair] = useState("EURUSD")
  const [bootstrap, setBootstrap] = useState(1000)
  const [skipBis, setSkipBis] = useState(false)
  const [horizons, setHorizons] = useState("12,24")

  const mutation = useMutation({ mutationFn: runFxModel })

  function handleRun() {
    mutation.mutate({ pair, bootstrap, skip_bis: skipBis, horizons })
  }

  const data = mutation.data

  // Try to extract forecast rows and chart data from response
  const forecastRows: Record<string, unknown>[] = data?.forecast ?? data?.forecasts ?? []
  const forecastCols: ColumnDef[] = forecastRows.length > 0
    ? Object.keys(forecastRows[0]).map(k => ({
        key: k,
        header: k,
        colorFn: k.toLowerCase().includes("return") || k.toLowerCase().includes("pct")
          ? colorPositiveNegative : undefined,
        format: (v: unknown) =>
          typeof v === "number" ? (v >= 0 ? "+" : "") + v.toFixed(3) : String(v ?? "N/A"),
      }))
    : []

  // Confidence interval chart
  const ciData: DataPoint[] = (data?.ci_series ?? []).map((r: Record<string, unknown>) => ({
    date: String(r["date"] ?? r["horizon"] ?? ""),
    value: Number(r["p50"] ?? r["median"] ?? r["value"] ?? 0),
  }))

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">FX Model</h1>
      <p className="text-sm text-gray-500 mb-4">
        Multi-factor FX forecasting using FRED, IMF, BIS data
      </p>

      <div className="bg-gray-50 rounded-lg border border-gray-200 p-4 mb-6 space-y-4 max-w-md">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Currency Pair</label>
          <select value={pair} onChange={e => setPair(e.target.value)}
            className="border rounded px-2 py-1.5 text-sm w-full">
            {availablePairs.map(p => <option key={p}>{p}</option>)}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Bootstrap Draws: <strong>{bootstrap}</strong>
          </label>
          <input type="range" min={100} max={5000} step={100}
            value={bootstrap} onChange={e => setBootstrap(Number(e.target.value))}
            className="w-full" />
          <div className="flex justify-between text-xs text-gray-400"><span>100</span><span>5000</span></div>
        </div>

        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <input type="checkbox" checked={skipBis} onChange={e => setSkipBis(e.target.checked)} />
          Skip BIS data
        </label>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Horizons (months)</label>
          <input type="text" value={horizons} onChange={e => setHorizons(e.target.value)}
            className="border rounded px-2 py-1.5 text-sm w-full" placeholder="12,24" />
        </div>

        <button onClick={handleRun} disabled={mutation.isPending}
          className="w-full py-2 rounded bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-50">
          {mutation.isPending ? "Running Model (~60s)..." : "Run Model"}
        </button>
      </div>

      {mutation.isPending && <LoadingSpinner message="Running FX model (may take up to 60s)..." />}
      {mutation.isError && <ErrorMessage message={String(mutation.error)} />}

      {data && !mutation.isPending && (
        <div className="space-y-6">
          {/* Summary metrics */}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {Object.entries(data)
              .filter(([k, v]) => typeof v === "number" && !Array.isArray(v) && k !== "bootstrap_draws")
              .slice(0, 6)
              .map(([k, v]) => (
                <MetricCard
                  key={k}
                  title={k.replace(/_/g, " ")}
                  value={typeof v === "number" ? (v >= 0 ? "+" : "") + v.toFixed(4) : String(v)}
                />
              ))}
          </div>

          {forecastRows.length > 0 && (
            <div>
              <h2 className="text-base font-semibold mb-2">Forecast</h2>
              <DataTable columns={forecastCols} rows={forecastRows} />
            </div>
          )}

          {ciData.length > 0 && (
            <div>
              <h2 className="text-base font-semibold mb-2">Median Forecast Path</h2>
              <TimeSeriesChart data={ciData} height={220} zeroLine />
            </div>
          )}

          {forecastRows.length === 0 && ciData.length === 0 && (
            <div>
              <h2 className="text-base font-semibold mb-2">Raw Output</h2>
              <pre className="text-xs bg-gray-50 border rounded p-4 overflow-auto max-h-96">
                {JSON.stringify(data, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}

      {!data && !mutation.isPending && !mutation.isError && (
        <p className="text-gray-400 text-sm">Select a pair and click Run Model.</p>
      )}
    </div>
  )
}
