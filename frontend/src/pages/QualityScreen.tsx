import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { runQualityScreen } from "@/lib/api"
import { DataTable, type ColumnDef } from "@/components/shared/DataTable"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { SegmentedControl, SelectInput, TextInput, ActionButton, ControlPanel } from "@/components/shared/FormControls"
import { colorPositiveNegative, colorZscore } from "@/lib/colors"

const UNIVERSE_OPTIONS = [
  "S&P 500", "Russell 2000", "S&P 400",
  "XLB — Materials", "XLC — Communication Services", "XLE — Energy",
  "XLF — Financials", "XLI — Industrials", "XLK — Technology",
  "XLP — Consumer Staples", "XLRE — Real Estate", "XLU — Utilities",
  "XLV — Health Care", "XLY — Consumer Discretionary",
]

const BENCHMARK_OPTIONS = ["S&P 500", "Same as Input", "Universes"]

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
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Quality Screen</h1>
      </div>

      <ControlPanel maxWidth="max-w-lg">
        <SegmentedControl
          options={[
            { value: "Universe" as const, label: "Universe" },
            { value: "Custom Tickers" as const, label: "Custom Tickers" },
          ]}
          value={inputMode}
          onChange={setInputMode}
        />

        {inputMode === "Universe" ? (
          <SelectInput
            label="Universe"
            value={universe}
            onChange={setUniverse}
            options={UNIVERSE_OPTIONS.map(o => ({ value: o, label: o }))}
          />
        ) : (
          <TextInput
            label="Tickers"
            value={tickers}
            onChange={setTickers}
            placeholder="AAPL, MSFT, GOOG"
          />
        )}

        <SelectInput
          label="Benchmark"
          value={benchmark}
          onChange={setBenchmark}
          options={BENCHMARK_OPTIONS.map(o => ({ value: o, label: o }))}
        />

        <ActionButton onClick={handleRun} loading={mutation.isPending} loadingText="Screening (~30s)...">
          Run Screen
        </ActionButton>
      </ControlPanel>

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
