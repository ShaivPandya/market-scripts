import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { runPortfolioOptimizerAsync } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { MetricCard } from "@/components/shared/MetricCard"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { colorPositiveNegative } from "@/lib/colors"

function buildCols(rows: Record<string, unknown>[]): ColumnDef[] {
  if (rows.length === 0) return []
  return Object.keys(rows[0]).map(k => ({
    key: k,
    header: k,
    colorFn: k.toLowerCase().includes("weight") || k.toLowerCase().includes("pct")
      ? colorPositiveNegative : undefined,
    format: (v: unknown) =>
      typeof v === "number" ? `${v >= 0 ? "+" : ""}${v.toFixed(2)}` : String(v ?? "N/A"),
  }))
}

export function PortfolioOptimizer() {
  const [bookSize, setBookSize] = useState(100_000)
  const [targetLeverage, setTargetLeverage] = useState(2.0)

  const mutation = useMutation({ mutationFn: runPortfolioOptimizerAsync })

  function handleRun() {
    mutation.mutate({ book: bookSize, target_leverage: targetLeverage })
  }

  const data = mutation.data
  const weightsRows: Record<string, unknown>[] = data?.weights_df ?? []
  const hedgesRows: Record<string, unknown>[] = data?.hedges_df ?? []

  return (
    <div>
      <h1 className="text-2xl font-bold mb-2">Portfolio Optimizer</h1>
      <p className="text-sm text-gray-500 mb-6">Beta-neutral portfolio construction with volatility targeting</p>

      <div className="bg-gray-50 rounded-lg border border-gray-200 p-4 mb-6 space-y-4 max-w-md">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Book Size ($): <strong>{bookSize.toLocaleString()}</strong>
          </label>
          <input type="range" min={10_000} max={10_000_000} step={10_000}
            value={bookSize} onChange={e => setBookSize(Number(e.target.value))}
            className="w-full" />
          <div className="flex justify-between text-xs text-gray-400"><span>$10k</span><span>$10M</span></div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Target Gross Leverage: <strong>{targetLeverage.toFixed(1)}x</strong>
          </label>
          <input type="range" min={0.5} max={4.0} step={0.1}
            value={targetLeverage} onChange={e => setTargetLeverage(Number(e.target.value))}
            className="w-full" />
          <div className="flex justify-between text-xs text-gray-400"><span>0.5x</span><span>4.0x</span></div>
        </div>

        <div className="text-xs text-gray-400 space-y-0.5">
          <p className="font-medium text-gray-500">Constraints</p>
          <p>Total gross: 4.0x · FX: 2.0x · Commodities: 1.0x · Bonds: 3.0x</p>
          <p>Long max: +20% · Short max: −10% · Equity net: −50% to +100%</p>
        </div>

        <button onClick={handleRun} disabled={mutation.isPending}
          className="w-full py-2 rounded bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-50">
          {mutation.isPending ? "Optimizing (can take 1-3 min)..." : "Optimize Portfolio"}
        </button>
      </div>

      {mutation.isPending && <LoadingSpinner message="Running optimization..." />}
      {mutation.isError && <ErrorMessage message={String(mutation.error)} />}

      {data && !mutation.isPending && (
        <div className="space-y-6">
          {/* Summary metrics */}
          {(data.daily_vol != null || data.gross_leverage != null) && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {data.daily_vol != null && <MetricCard title="Daily Volatility" value={`${(Number(data.daily_vol) * 100).toFixed(2)}%`} />}
              {data.gross_leverage != null && <MetricCard title="Gross Leverage" value={`${Number(data.gross_leverage).toFixed(2)}x`} />}
              {data.equity_net != null && <MetricCard title="Equity Net" value={`${(Number(data.equity_net) * 100).toFixed(1)}%`} />}
              {data.net_beta_spy != null && <MetricCard title="Net Beta (SPY)" value={Number(data.net_beta_spy).toFixed(3)} />}
            </div>
          )}

          {weightsRows.length > 0 && (
            <div>
              <h2 className="text-base font-semibold mb-2">Portfolio Weights</h2>
              <DataTable columns={buildCols(weightsRows)} rows={weightsRows} />
            </div>
          )}

          {hedgesRows.length > 0 && (
            <div>
              <h2 className="text-base font-semibold mb-2">Hedge Positions</h2>
              <DataTable columns={buildCols(hedgesRows)} rows={hedgesRows} />
            </div>
          )}
        </div>
      )}

      {!data && !mutation.isPending && !mutation.isError && (
        <p className="text-gray-400 text-sm">Configure settings above and click Optimize Portfolio.</p>
      )}
    </div>
  )
}
