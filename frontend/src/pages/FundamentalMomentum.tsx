import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { runFundamentalMomentum } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { colorPositiveNegative, colorZscore } from "@/lib/colors"

const UNIVERSE_OPTIONS = [
  "S&P 500", "Russell 2000", "S&P 400",
  "XLB — Materials", "XLC — Communication Services", "XLE — Energy",
  "XLF — Financials", "XLI — Industrials", "XLK — Technology",
  "XLP — Consumer Staples", "XLRE — Real Estate", "XLU — Utilities",
  "XLV — Health Care", "XLY — Consumer Discretionary",
]

function buildCols(rows: Record<string, unknown>[]): ColumnDef[] {
  if (rows.length === 0) return []
  return Object.keys(rows[0]).map(k => ({
    key: k,
    header: k,
    colorFn: k.toLowerCase().includes("z") || k.toLowerCase().includes("score")
      ? colorZscore
      : k.toLowerCase().includes("pct") || k.toLowerCase().includes("yoy") || k.toLowerCase().includes("cagr")
        ? colorPositiveNegative
        : undefined,
    format: (v: unknown) =>
      v != null && typeof v === "number"
        ? `${v >= 0 ? "+" : ""}${v.toFixed(2)}`
        : String(v ?? "N/A"),
  }))
}

export function FundamentalMomentum() {
  const [screenType, setScreenType] = useState<"EPS" | "Revenue" | "Both">("Both")
  const [inputMode, setInputMode] = useState<"Universe" | "Custom Tickers">("Universe")
  const [universe, setUniverse] = useState("S&P 500")
  const [tickers, setTickers] = useState("")
  const [benchmark, setBenchmark] = useState("S&P 500")

  const mutation = useMutation({ mutationFn: runFundamentalMomentum })

  function handleRun() {
    mutation.mutate({ screen_type: screenType, universe, tickers, benchmark, input_mode: inputMode })
  }

  const epsRows: Record<string, unknown>[] = mutation.data?.eps?.results_df ?? []
  const revRows: Record<string, unknown>[] = mutation.data?.rev?.results_df ?? []

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Fundamental Momentum</h1>

      <div className="bg-gray-50 rounded-lg border border-gray-200 p-4 mb-6 space-y-4 max-w-lg">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Screen Type</label>
          <div className="flex gap-3">
            {(["EPS", "Revenue", "Both"] as const).map(t => (
              <label key={t} className="flex items-center gap-1.5 text-sm cursor-pointer">
                <input type="radio" checked={screenType === t} onChange={() => setScreenType(t)} />
                {t}
              </label>
            ))}
          </div>
        </div>

        <div className="flex gap-3">
          {(["Universe", "Custom Tickers"] as const).map(m => (
            <label key={m} className="flex items-center gap-1.5 text-sm cursor-pointer">
              <input type="radio" checked={inputMode === m} onChange={() => setInputMode(m)} />
              {m}
            </label>
          ))}
        </div>

        {inputMode === "Universe" ? (
          <select value={universe} onChange={e => setUniverse(e.target.value)}
            className="border rounded px-2 py-1.5 text-sm w-full">
            {UNIVERSE_OPTIONS.map(o => <option key={o}>{o}</option>)}
          </select>
        ) : (
          <input type="text" placeholder="AAPL, MSFT, GOOG" value={tickers}
            onChange={e => setTickers(e.target.value)}
            className="border rounded px-2 py-1.5 text-sm w-full" />
        )}

        <select value={benchmark} onChange={e => setBenchmark(e.target.value)}
          className="border rounded px-2 py-1.5 text-sm w-full">
          <option>S&amp;P 500</option>
          <option>Same as Input</option>
        </select>

        <button onClick={handleRun} disabled={mutation.isPending}
          className="w-full py-2 rounded bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-50">
          {mutation.isPending ? "Screening..." : "Run Screen"}
        </button>
      </div>

      {mutation.isPending && <LoadingSpinner message="Running fundamental momentum screen..." />}
      {mutation.isError && <ErrorMessage message={String(mutation.error)} />}

      {mutation.data && !mutation.isPending && (
        <div className="space-y-6">
          {epsRows.length > 0 && (
            <div>
              <h2 className="text-base font-semibold mb-2">EPS Momentum</h2>
              <DataTable columns={buildCols(epsRows)} rows={epsRows} />
            </div>
          )}
          {revRows.length > 0 && (
            <div>
              <h2 className="text-base font-semibold mb-2">Revenue Momentum</h2>
              <DataTable columns={buildCols(revRows)} rows={revRows} />
            </div>
          )}
          {epsRows.length === 0 && revRows.length === 0 && (
            <p className="text-gray-400">No results returned.</p>
          )}
        </div>
      )}

      {!mutation.data && !mutation.isPending && !mutation.isError && (
        <p className="text-gray-400 text-sm">Configure inputs above and click Run Screen.</p>
      )}
    </div>
  )
}
