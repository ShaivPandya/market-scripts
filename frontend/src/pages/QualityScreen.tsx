import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { runQualityScreen } from "@/lib/api"
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

const BENCHMARK_OPTIONS = ["S&P 500", "Same as Input"]

export function QualityScreen() {
  const [inputMode, setInputMode] = useState<"Universe" | "Custom Tickers">("Universe")
  const [universe, setUniverse] = useState("S&P 500")
  const [tickers, setTickers] = useState("")
  const [benchmark, setBenchmark] = useState("S&P 500")

  const mutation = useMutation({ mutationFn: runQualityScreen })

  function handleRun() {
    mutation.mutate({ universe, tickers, benchmark, input_mode: inputMode })
  }

  const rows: Record<string, unknown>[] = mutation.data?.results_df ?? []
  const columns: ColumnDef[] = rows.length > 0
    ? Object.keys(rows[0]).map(k => ({
        key: k,
        header: k,
        colorFn: k.toLowerCase().includes("z") || k.toLowerCase().includes("score")
          ? colorZscore
          : k.toLowerCase().includes("pct") ? colorPositiveNegative : undefined,
        format: (v: unknown) => v != null ? (typeof v === "number" ? (v >= 0 ? "+" : "") + v.toFixed(2) : String(v)) : "N/A",
      }))
    : []

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Quality Screen</h1>

      <div className="bg-gray-50 rounded-lg border border-gray-200 p-4 mb-6 space-y-4 max-w-lg">
        <div className="flex gap-3">
          {(["Universe", "Custom Tickers"] as const).map(m => (
            <label key={m} className="flex items-center gap-1.5 text-sm cursor-pointer">
              <input type="radio" checked={inputMode === m} onChange={() => setInputMode(m)} />
              {m}
            </label>
          ))}
        </div>

        {inputMode === "Universe" ? (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Universe</label>
            <select
              value={universe}
              onChange={e => setUniverse(e.target.value)}
              className="border rounded px-2 py-1.5 text-sm w-full"
            >
              {UNIVERSE_OPTIONS.map(o => <option key={o}>{o}</option>)}
            </select>
          </div>
        ) : (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Tickers</label>
            <input
              type="text"
              placeholder="AAPL, MSFT, GOOG"
              value={tickers}
              onChange={e => setTickers(e.target.value)}
              className="border rounded px-2 py-1.5 text-sm w-full"
            />
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Benchmark</label>
          <select
            value={benchmark}
            onChange={e => setBenchmark(e.target.value)}
            className="border rounded px-2 py-1.5 text-sm w-full"
          >
            {BENCHMARK_OPTIONS.map(o => <option key={o}>{o}</option>)}
          </select>
        </div>

        <button
          onClick={handleRun}
          disabled={mutation.isPending}
          className="w-full py-2 rounded bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
        >
          {mutation.isPending ? "Screening (~30s)..." : "Run Screen"}
        </button>
      </div>

      {mutation.isPending && <LoadingSpinner message="Running quality screen..." />}
      {mutation.isError && <ErrorMessage message={String(mutation.error)} />}

      {mutation.data && !mutation.isPending && (
        <>
          <div className="flex gap-6 text-sm text-gray-600 mb-4">
            <span>Input: <strong>{mutation.data.input_count ?? "—"}</strong></span>
            <span>Universe size: <strong>{mutation.data.universe_size ?? "—"}</strong></span>
            <span>Scored: <strong>{mutation.data.scored_count ?? rows.length}</strong></span>
            <span>Benchmark: <strong>{mutation.data.benchmark_name ?? benchmark}</strong></span>
          </div>
          {Array.isArray(mutation.data.failed) && mutation.data.failed.length > 0 && (
            <div className="mb-4 rounded border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
              <strong>Some tickers failed:</strong> {mutation.data.failed.join(", ")}
            </div>
          )}
          {rows.length > 0 ? (
            <DataTable columns={columns} rows={rows} />
          ) : (
            <p className="text-gray-400">No results returned.</p>
          )}
        </>
      )}

      {!mutation.data && !mutation.isPending && !mutation.isError && (
        <p className="text-gray-400 text-sm">Configure inputs above and click Run Screen.</p>
      )}
    </div>
  )
}
