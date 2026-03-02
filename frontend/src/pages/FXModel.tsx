import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { runFxModel, fetchFxModelPairs } from "@/lib/api"
import { useApiQuery } from "@/hooks/useApiQuery"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { TimeSeriesChart, type DataPoint } from "@/components/shared/TimeSeriesChart"
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { SelectInput, SliderInput, Toggle, TextInput, ActionButton, ControlPanel } from "@/components/shared/FormControls"
import { colorPositiveNegative } from "@/lib/colors"

const DEFAULT_PAIRS = ["USDCAD", "GBPUSD", "AUDUSD", "USDJPY", "EURUSD"]

const FORECAST_COLUMN_LABELS: Record<string, string> = {
  horizon_months: "Horizon (Months)",
  spot_now: "Spot Now",
  point_level: "Point Forecast",
  expected_move_pct: "Expected Move (%)",
  q05: "P5",
  q50: "Median (P50)",
  q95: "P95",
  valuation_rer_z: "Valuation (RER Z)",
  r2: "R²",
  nobs: "Observations",
}

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
  const errorMessage = mutation.error instanceof Error ? mutation.error.message : String(mutation.error)

  function toFiniteNumberOrNull(v: unknown): number | null {
    const n = Number(v)
    return Number.isFinite(n) ? n : null
  }

  function formatSigned(v: unknown, decimals = 4): string {
    const n = toFiniteNumberOrNull(v)
    if (n == null) return "N/A"
    return `${n >= 0 ? "+" : ""}${n.toFixed(decimals)}`
  }

  // Try to extract forecast rows and chart data from response
  const forecastRows: Record<string, unknown>[] = data?.forecast ?? data?.forecasts ?? []
  const forecastCols: ColumnDef[] = forecastRows.length > 0
    ? Object.keys(forecastRows[0]).map(k => ({
        key: k,
        header: FORECAST_COLUMN_LABELS[k] ?? k.replace(/_/g, " "),
        colorFn: k.toLowerCase().includes("return") || k.toLowerCase().includes("pct")
          ? colorPositiveNegative : undefined,
        format: (v: unknown) =>
          typeof v === "number" ? (v >= 0 ? "+" : "") + v.toFixed(3) : String(v ?? "N/A"),
      }))
    : []

  // Confidence interval chart
  const ciData: DataPoint[] = (data?.ci_series ?? []).map((r: Record<string, unknown>) => ({
    date: String(r["date"] ?? r["horizon"] ?? ""),
    value: toFiniteNumberOrNull(r["p50"] ?? r["median"] ?? r["value"]),
  }))
  const driverBreakdown: Record<string, unknown>[] = data?.driver_breakdown ?? []

  return (
    <div>
      <div className="mb-6">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">FX Model</h1>
          <span className="px-1.5 py-0.5 rounded text-xs font-semibold bg-yellow-100 text-yellow-800 border border-yellow-300">Beta</span>
        </div>
        <p className="text-sm text-gray-400 mt-0.5">
          Multi-factor FX forecasting using FRED, IMF, BIS data
          {data?.latest_date && data?.feature_asof_date && (
            <span className="block mt-1 text-xs text-gray-400">
              Spot as of {String(data.latest_date)}; features as of {String(data.feature_asof_date)} (lag {String(data.feature_lag_months ?? 1)}m)
            </span>
          )}
        </p>
      </div>

      <ControlPanel>
        <SelectInput
          label="Currency Pair"
          value={pair}
          onChange={setPair}
          options={availablePairs.map(p => ({ value: p, label: p }))}
        />

        <SliderInput
          label="Bootstrap Draws"
          value={bootstrap}
          onChange={setBootstrap}
          min={100}
          max={5000}
          step={100}
          formatValue={v => v.toLocaleString()}
          minLabel="100"
          maxLabel="5,000"
        />

        <Toggle
          label="Skip BIS data"
          checked={skipBis}
          onChange={setSkipBis}
        />

        <TextInput
          label="Horizons (months)"
          value={horizons}
          onChange={setHorizons}
          placeholder="12,24"
        />

        <ActionButton onClick={handleRun} loading={mutation.isPending} loadingText="Running Model (~60s)...">
          Run Model
        </ActionButton>
      </ControlPanel>

      {mutation.isPending && <LoadingSpinner message="Running FX model (may take up to 60s)..." />}
      {mutation.isError && <ErrorMessage message={errorMessage} />}

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
              <TimeSeriesChart data={ciData} height={220} />
            </div>
          )}

          {driverBreakdown.length > 0 && (
            <div>
              <h2 className="text-base font-semibold mb-2">Drivers & Factors</h2>
              <div className="space-y-4">
                {driverBreakdown.map((block, idx) => {
                  const horizon = toFiniteNumberOrNull(block["horizon_months"])
                  const conclusion = typeof block["conclusion"] === "string" ? block["conclusion"] : ""
                  const drivers = Array.isArray(block["drivers"])
                    ? (block["drivers"] as Record<string, unknown>[])
                    : []
                  const maxAbsContribution = Math.max(
                    1e-9,
                    ...drivers.map(d => Math.abs(toFiniteNumberOrNull(d["contribution"]) ?? 0)),
                  )

                  return (
                    <div key={`${horizon ?? idx}-${idx}`} className="rounded-xl border border-gray-200 bg-white">
                      <div className="px-4 py-3 border-b border-gray-200">
                        <h3 className="text-sm font-semibold text-gray-900">
                          {horizon != null ? `${horizon}-Month Forecast Drivers` : "Forecast Drivers"}
                        </h3>
                      </div>

                      {drivers.length > 0 ? (
                        <div className="overflow-x-auto">
                          <table className="w-full text-sm border-collapse">
                            <thead className="bg-gray-50">
                              <tr>
                                <th className="px-3 py-2 text-left font-semibold text-gray-600 border-b border-gray-200">Feature</th>
                                <th className="px-3 py-2 text-right font-semibold text-gray-600 border-b border-gray-200">Coeff</th>
                                <th className="px-3 py-2 text-right font-semibold text-gray-600 border-b border-gray-200">Value</th>
                                <th className="px-3 py-2 text-right font-semibold text-gray-600 border-b border-gray-200">Contribution</th>
                                <th className="px-3 py-2 text-left font-semibold text-gray-600 border-b border-gray-200 min-w-[120px]">Bar</th>
                                <th className="px-3 py-2 text-left font-semibold text-gray-600 border-b border-gray-200 min-w-[260px]">Interpretation</th>
                              </tr>
                            </thead>
                            <tbody>
                              {drivers.map((d, i) => {
                                const contribution = toFiniteNumberOrNull(d["contribution"]) ?? 0
                                const barWidthPct = Math.max(2, Math.round(Math.abs(contribution) / maxAbsContribution * 100))
                                const positive = contribution >= 0
                                const label = typeof d["label"] === "string" ? d["label"] : String(d["name"] ?? "N/A")
                                const description = typeof d["description"] === "string" ? d["description"] : ""

                                return (
                                  <tr key={`${label}-${i}`} className="border-b border-gray-100">
                                    <td className="px-3 py-2 whitespace-nowrap">{label}</td>
                                    <td className="px-3 py-2 text-right font-mono whitespace-nowrap">{formatSigned(d["coefficient"], 4)}</td>
                                    <td className="px-3 py-2 text-right font-mono whitespace-nowrap">{formatSigned(d["value"], 4)}</td>
                                    <td className={`px-3 py-2 text-right font-mono whitespace-nowrap font-semibold ${positive ? "text-green-600" : "text-red-600"}`}>
                                      {formatSigned(d["contribution"], 5)}
                                    </td>
                                    <td className="px-3 py-2">
                                      <div className="h-2 w-full bg-gray-100 rounded">
                                        <div
                                          className={`h-2 rounded ${positive ? "bg-green-500" : "bg-red-500"}`}
                                          style={{ width: `${barWidthPct}%` }}
                                        />
                                      </div>
                                    </td>
                                    <td className="px-3 py-2 text-gray-700 whitespace-normal">{description || "N/A"}</td>
                                  </tr>
                                )
                              })}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <p className="px-4 py-3 text-sm text-gray-500">No driver data available.</p>
                      )}

                      <div className="px-4 py-3 border-t border-gray-100 bg-gray-50 text-xs text-gray-500">
                        Contribution = Coefficient x Feature Value (lagged; see as-of dates above)
                      </div>
                      {conclusion && (
                        <div className="px-4 py-3 border-t border-gray-100 bg-blue-50 text-sm text-blue-900">
                          {conclusion}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {forecastRows.length === 0 && ciData.length === 0 && driverBreakdown.length === 0 && (
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
